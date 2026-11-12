"""Compare RI models on IMD best-track data with the same storm-level, time-ordered split.

usage: python3 compare_models.py OBSERVATIONS.json STORMS.json OUTDIR
Train 1982-2012, validation 2013-2018 (threshold choice), test 2019-2025 (reported once).
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss, fbeta_score,
                             precision_score, recall_score, f1_score, precision_recall_curve, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import prep

warnings.filterwarnings("ignore")
obs, storms, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
d = prep.build_rows(prep.load(obs), storms)
lab = d[d.wind_24.notna()].copy()
sp = prep.split(lab)
tr, va, te = sp["train"], sp["val"], sp["test"]
F = prep.FEATURES
SEED = 7


def pre(scale=True):
    steps = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([("x", make_pipeline(*steps), F)])


MODELS = {
    "Logistic regression": make_pipeline(pre(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)),
    "Random forest": make_pipeline(pre(False), RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
    "Gradient boosting": make_pipeline(pre(False), HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                        max_iter=300, class_weight="balanced", random_state=SEED)),
    "Neural network (MLP)": make_pipeline(pre(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                           early_stopping=True, max_iter=500, random_state=SEED)),
}


def balanced_fit(model, X, y):
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    name = model.steps[-1][0]
    try:
        return model.fit(X, y, **{f"{name}__sample_weight": w}) if "mlp" in name else model.fit(X, y)
    except TypeError:                       # older sklearn: oversample RI rows instead
        idx = np.concatenate([np.arange(len(y))] + [np.where(y == 1)[0]] * int(round(w.max()) - 1))
        return model.fit(X.iloc[idx], y.iloc[idx])


def pick_threshold(y, p):
    grid = np.round(np.arange(0.02, 0.98, 0.01), 2)
    return float(grid[int(np.argmax([fbeta_score(y, p >= t, beta=2, zero_division=0) for t in grid]))])


def storm_scores(df, flag):
    df = df.assign(f=flag)
    ri = df[df.RI == 1].storm_id.unique()
    caught = df[(df.RI == 1) & (df.f == 1)].storm_id.nunique()
    fa = df[(df.f == 1) & ~df.storm_id.isin(ri)].storm_id.nunique()
    return caught, len(ri), fa, df.storm_id.nunique() - len(ri)


def all_scores(df, p, th):
    y, f = df.RI.values, (p >= th).astype(int)
    c, n, fa, nn = storm_scores(df, f)
    return {"precision": precision_score(y, f, zero_division=0), "recall": recall_score(y, f), "F1": f1_score(y, f),
            "F2": fbeta_score(y, f, beta=2), "PR_AUC": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, np.clip(p, 0, 1)), "threshold": th, "flags": int(f.sum()),
            "false_alarm_times": int(((f == 1) & (y == 0)).sum()),
            "RI_storms_caught": f"{c} of {n}", "nonRI_storms_with_false_alarm": f"{fa} of {nn}"}


rows, probs = [], {}
# baselines
rule = (te.dV_past12 >= 10).astype(float).fillna(0).values
rows.append({"model": "Rule: +10 kt in last 12 h", **all_scores(te, rule, 0.5)})
rows.append({"model": "Climatology (always 5.8%)", **all_scores(te, np.full(len(te), tr.RI.mean()), 0.99)})
for name, m in MODELS.items():
    m = balanced_fit(clone(m), tr[F], tr.RI)
    th = pick_threshold(va.RI.values, m.predict_proba(va[F])[:, 1])
    p = m.predict_proba(te[F])[:, 1]
    probs[name] = (m, p, th)
    rows.append({"model": name, **all_scores(te, p, th)})
    print(f"{name:22s} done")
R = pd.DataFrame(rows)
R.to_csv(f"{out}/model_comparison_test.csv", index=False)

# grouped 5-fold CV on 1982-2018 storms (robustness)
dev = pd.concat([tr, va])
cv = []
for name, m in MODELS.items():
    aps, recs = [], []
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(dev, dev.RI, dev.storm_id)):
        mm = balanced_fit(clone(m), dev.iloc[a][F], dev.iloc[a].RI)
        pb = mm.predict_proba(dev.iloc[b][F])[:, 1]
        aps.append(average_precision_score(dev.iloc[b].RI, pb))
        recs.append(roc_auc_score(dev.iloc[b].RI, pb))
    cv.append({"model": name, "PR_AUC_mean": np.mean(aps), "PR_AUC_sd": np.std(aps),
               "ROC_AUC_mean": np.mean(recs), "ROC_AUC_sd": np.std(recs)})
CV = pd.DataFrame(cv)
CV.to_csv(f"{out}/model_cv_1982_2018.csv", index=False)

# ablation for logistic regression: drop one input group at a time
groups = {"satellite CI number": ["ci_no"], "recent intensity change": ["dV_past6", "dV_past12", "dV_past24"],
          "current intensity (wind, pressure)": ["wind", "pressure"], "position": ["lat", "lon"],
          "season and basin": ["day_of_year", "arabian_sea"]}
abl = []
base_lr = probs["Logistic regression"]
abl.append({"removed": "none (full model)", "PR_AUC": average_precision_score(te.RI, base_lr[1]),
            "recall": recall_score(te.RI, base_lr[1] >= base_lr[2]),
            "precision": precision_score(te.RI, base_lr[1] >= base_lr[2])})
for g, cols in groups.items():
    keep = [c for c in F if c not in cols]
    m = make_pipeline(ColumnTransformer([("x", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                             StandardScaler()), keep)]),
                      LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)).fit(tr, tr.RI)
    th = pick_threshold(va.RI.values, m.predict_proba(va)[:, 1])
    p = m.predict_proba(te)[:, 1]
    abl.append({"removed": g, "PR_AUC": average_precision_score(te.RI, p), "recall": recall_score(te.RI, p >= th),
                "precision": precision_score(te.RI, p >= th, zero_division=0)})
A = pd.DataFrame(abl)
A.to_csv(f"{out}/ablation_logistic.csv", index=False)

# permutation importance (test PR-AUC drop) for logistic regression and gradient boosting
imp = []
for name in ["Logistic regression", "Gradient boosting"]:
    m = probs[name][0]
    r = permutation_importance(m, te[F], te.RI, scoring="average_precision", n_repeats=30, random_state=SEED)
    for f_, mu, sd in zip(F, r.importances_mean, r.importances_std):
        imp.append({"model": name, "feature": f_, "PR_AUC_drop": mu, "sd": sd})
I = pd.DataFrame(imp)
I.to_csv(f"{out}/permutation_importance.csv", index=False)

# storm-level bootstrap CI for the chosen model (logistic regression)
rng = np.random.default_rng(SEED)
ids = te.storm_id.unique()
bs = {"recall": [], "precision": [], "PR_AUC": []}
p_lr, th_lr = base_lr[1], base_lr[2]
tt = te.assign(p=p_lr)
for _ in range(2000):
    s = rng.choice(ids, len(ids), replace=True)
    b = pd.concat([tt[tt.storm_id == x] for x in s])
    if b.RI.sum() == 0:
        continue
    f = b.p >= th_lr
    bs["recall"].append(recall_score(b.RI, f))
    bs["precision"].append(precision_score(b.RI, f, zero_division=0))
    bs["PR_AUC"].append(average_precision_score(b.RI, b.p))
CI = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
json.dump(CI, open(f"{out}/bootstrap_ci_logistic.json", "w"), indent=1)

# curves for figures
curves = {}
for name, (m, p, th) in probs.items():
    pr, rc, _ = precision_recall_curve(te.RI, p)
    fpr, tpr, _ = roc_curve(te.RI, p)
    curves[name] = {"pr": [pr.tolist(), rc.tolist()], "roc": [fpr.tolist(), tpr.tolist()],
                    "op": [float(precision_score(te.RI, p >= th)), float(recall_score(te.RI, p >= th))]}
# calibration (10 bins) for each model
from sklearn.calibration import calibration_curve
for name, (m, p, th) in probs.items():
    fy, mp = calibration_curve(te.RI, p, n_bins=10, strategy="quantile")
    curves[name]["calib"] = [mp.tolist(), fy.tolist()]
json.dump(curves, open(f"{out}/curves.json", "w"))

pd.set_option("display.width", 220)
print("\nTEST 2019-2025\n", R.round(3).to_string(index=False))
print("\nGROUPED 5-FOLD CV 1982-2018\n", CV.round(3).to_string(index=False))
print("\nABLATION (logistic)\n", A.round(3).to_string(index=False))
print("\nIMPORTANCE\n", I.sort_values(["model", "PR_AUC_drop"], ascending=[True, False]).round(4).to_string(index=False))
print("\nBOOTSTRAP 95% CI (logistic, by storm):", {k: [round(x, 3) for x in v] for k, v in CI.items()})
print("\ncounts: train", len(tr), tr.RI.sum(), "val", len(va), va.RI.sum(), "test", len(te), te.RI.sum())
