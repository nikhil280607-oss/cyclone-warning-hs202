"""Join INSAT-3DR features to IMD records and test whether they add skill for RI (6 case-study cyclones).

usage: python3 satellite_analysis.py OBS.json STORMS.json satellite_features.csv OUTDIR
"""
import sys, os, warnings, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import prep, model
warnings.filterwarnings("ignore")
obs, storms, satf, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
d = prep.build_rows(prep.load(obs), storms)
lab = d[d.wind_24.notna()]
sp = prep.split(lab)
m, th = model.fit(sp["train"], sp["val"])
d["p_imd"] = m.predict_proba(d[prep.FEATURES])[:, 1]
sat = pd.read_csv(satf)
sat["t"] = pd.to_datetime(sat.time_utc, utc=True)
rows = []
for _, r in sat.iterrows():
    g = d[(d.name == r.storm) & (d.year == r.t.year)]
    if g.empty:
        continue
    dt = (g.t - r.t).abs()
    j = dt.idxmin()
    if dt[j] > pd.Timedelta(hours=1.5) or pd.isna(g.loc[j, "wind_24"]):
        continue
    rows.append({**r.to_dict(), "imd_time": g.loc[j, "t"], "RI": int(g.loc[j, "RI"]), "p_imd": g.loc[j, "p_imd"],
                 "dV24_future": g.loc[j, "dV24"], "wind": g.loc[j, "wind"]})
J = pd.DataFrame(rows)
J.to_csv(f"{out}/satellite_joined.csv", index=False)
SAT = ["frac_lt200_200km", "frac_lt220_200km", "mean_bt_200km", "std_bt_200km", "frac_lt220_500km", "mean_bt_500km"]
summary = J.groupby("RI")[SAT + ["p_imd"]].mean().T
summary.columns = ["not RI", "RI next 24 h"]
print("samples", len(J), "RI samples", int(J.RI.sum()), "storms", J.storm.nunique())
print(summary.round(3).to_string())
# leave-one-storm-out: IMD model probability alone vs plus satellite features
res = {}
for name, cols in {"IMD model only": ["p_imd"], "IMD model + satellite": ["p_imd"] + SAT}.items():
    pred = np.zeros(len(J))
    for s in J.storm.unique():
        tr, te = J.storm != s, J.storm == s
        if J[tr].RI.nunique() < 2:
            pred[te] = J[tr].RI.mean(); continue
        mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)).fit(J.loc[tr, cols], J.loc[tr, "RI"])
        pred[te] = mm.predict_proba(J.loc[te, cols])[:, 1]
    res[name] = {"ROC_AUC": roc_auc_score(J.RI, pred), "PR_AUC": average_precision_score(J.RI, pred)}
    J[f"pred_{name}"] = pred
base = average_precision_score(J.RI, J.p_imd)
print("\nIMD model as-is on these samples: ROC", round(roc_auc_score(J.RI, J.p_imd), 3), "PR", round(base, 3))
print("leave-one-storm-out:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in res.items()})
json.dump({"n": len(J), "n_RI": int(J.RI.sum()), "means": summary.round(4).to_dict(), "loso": res,
           "imd_as_is": {"ROC_AUC": roc_auc_score(J.RI, J.p_imd), "PR_AUC": base}}, open(f"{out}/satellite_results.json", "w"), indent=1, default=float)
