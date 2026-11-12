"""Try extra inputs and an ensemble; choose ONLY on 2013-2018 validation, then report test once.

usage: python3 improve_model.py OBS.json STORMS.json COAST.geojson OUTDIR
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import prep

warnings.filterwarnings("ignore")
obs, storms, coast_f, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
d = prep.build_rows(prep.load(obs), storms)

# --- extra inputs, all known at time T ---
coast_pts = []
for f in json.load(open(coast_f))["features"]:
    g = f["geometry"]
    for p in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
        coast_pts.append(np.array(p[0])[::5])
coast_pts = np.vstack(coast_pts)


def km(lat1, lon1, lat2, lon2):
    r = np.radians
    a = np.sin(r(lat2 - lat1) / 2) ** 2 + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


parts = []
for sid, g in d.groupby("storm_id", sort=False):
    g = g.sort_values("t").copy()
    th = (g.t - g.t.iloc[0]).dt.total_seconds().values / 3600
    lat6 = np.interp(th - 6, th, g.lat.values, left=np.nan)
    lon6 = np.interp(th - 6, th, g.lon.values, left=np.nan)
    lat6[th < 6] = np.nan; lon6[th < 6] = np.nan
    g["speed_kmh"] = km(lat6, lon6, g.lat.values, g.lon.values) / 6
    g["north_kmh"] = (g.lat.values - lat6) * 111 / 6
    g["max_wind_so_far"] = g.wind.cummax()
    g["below_peak_so_far"] = g.max_wind_so_far - g.wind
    g["dist_coast_km"] = [km(a, b, coast_pts[:, 1], coast_pts[:, 0]).min() for a, b in zip(g.lat, g.lon)]
    parts.append(g)
d = pd.concat(parts)
lab = d[d.wind_24.notna()]
sp = prep.split(lab)
tr, va, te = sp["train"], sp["val"], sp["test"]
BASE = list(prep.FEATURES)
EXTRA = ["speed_kmh", "north_kmh", "below_peak_so_far", "dist_coast_km", "age_h"]


def lr(F, C=0.5):
    return make_pipeline(ColumnTransformer([("x", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                                StandardScaler()), F)]),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def rf(F):
    return make_pipeline(ColumnTransformer([("x", SimpleImputer(strategy="median", add_indicator=True), F)]),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=7))


def hgb(F):
    return make_pipeline(ColumnTransformer([("x", SimpleImputer(strategy="median", add_indicator=True), F)]),
                         HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                        class_weight="balanced", random_state=7))


cands = {}
for fname, F in {"base": BASE, "base+extra": BASE + EXTRA}.items():
    for mname, mk in {"LR": lr, "RF": rf, "HGB": hgb}.items():
        m = mk(F).fit(tr, tr.RI)
        cands[f"{mname} {fname}"] = (m, m.predict_proba(va)[:, 1], m.predict_proba(te)[:, 1])
# ensembles (average of probabilities, rank-normalised)
def rank(p):          # plain average of probabilities (deployable: no ranking over the test set)
    return p
for fname in ["base", "base+extra"]:
    v = np.mean([rank(cands[f"{k} {fname}"][1]) for k in ["LR", "RF"]], axis=0)
    t = np.mean([rank(cands[f"{k} {fname}"][2]) for k in ["LR", "RF"]], axis=0)
    cands[f"LR+RF ensemble {fname}"] = (None, v, t)
    v = np.mean([rank(cands[f"{k} {fname}"][1]) for k in ["LR", "RF", "HGB"]], axis=0)
    t = np.mean([rank(cands[f"{k} {fname}"][2]) for k in ["LR", "RF", "HGB"]], axis=0)
    cands[f"LR+RF+HGB ensemble {fname}"] = (None, v, t)

rows = []
grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
for name, (m, pv, pt) in cands.items():
    th = float(grid[int(np.argmax([fbeta_score(va.RI, pv >= g_, beta=2, zero_division=0) for g_ in grid]))])
    f = pt >= th
    tt = te.assign(f=f.astype(int))
    ri = tt[tt.RI == 1].storm_id.unique()
    rows.append({"model": name, "val_PR_AUC": average_precision_score(va.RI, pv), "val_ROC_AUC": roc_auc_score(va.RI, pv),
                 "test_PR_AUC": average_precision_score(te.RI, pt), "test_ROC_AUC": roc_auc_score(te.RI, pt),
                 "test_precision": precision_score(te.RI, f, zero_division=0), "test_recall": recall_score(te.RI, f),
                 "test_F2": fbeta_score(te.RI, f, beta=2), "threshold": th,
                 "RI_storms_caught": tt[(tt.RI == 1) & (tt.f == 1)].storm_id.nunique(), "RI_storms": len(ri)})
R = pd.DataFrame(rows).sort_values("val_PR_AUC", ascending=False)
R.to_csv(f"{out}/improvement_candidates.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
