"""Build the data for the website from the real model and replay (no hand-entered numbers).

usage: python3 build_site_data.py OBS.json STORMS.json COAST.geojson COUNTRIES.geojson MODELS_DIR OUT.json
"""
import sys, json
import numpy as np, pandas as pd
import prep, model, warn

obs, storms, coast_f, countries_f, mdir, out = sys.argv[1:7]
d = prep.build_rows(prep.load(obs), storms)
lab = d[d.wind_24.notna()]
sp = prep.split(lab)
m, th = model.fit(sp["train"], sp["val"])
d["p"] = m.predict_proba(d[prep.FEATURES])[:, 1]
d["flag"] = (d.p >= th).astype(int)
coast = warn.Coast(coast_f)

BBOX = (40, 110, -5, 35)


def simplify(ring, step):
    ring = np.asarray(ring)
    if len(ring) > 40:
        ring = np.vstack([ring[::step], ring[-1:]])
    return [[round(float(x), 2), round(float(y), 2)] for x, y in ring]


def geo(features, step, clip=True):
    outf = []
    for f in features:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        keep = []
        for p in polys:
            r = np.asarray(p[0])
            if clip and (r[:, 0].max() < BBOX[0] or r[:, 0].min() > BBOX[1] or r[:, 1].max() < BBOX[2] or r[:, 1].min() > BBOX[3]):
                continue
            if len(r) < 4:
                continue
            keep.append([simplify(r, step)])
        if keep:
            outf.append({"type": "Feature", "properties": {"name": f["properties"].get("name") or f["properties"].get("NAME")},
                         "geometry": {"type": "MultiPolygon", "coordinates": keep}})
    return {"type": "FeatureCollection", "features": outf}


countries = geo(json.load(open(countries_f))["features"], 3)
states = geo(json.load(open(coast_f))["features"], 6, clip=False)

S = []
for sid, g in d[(d.year >= 2019) & (d.year <= 2025)].groupby("storm_id"):
    g = g.sort_values("t")
    state, arr, sent, upd = warn.warnings_for_storm(g, g.flag.values, coast)
    name = g["name"].iloc[0] if isinstance(g["name"].iloc[0], str) else ""
    sms = []
    for k, w in enumerate(sent):
        texts = []
        for lang, text in w["sms"].items():
            n, parts, enc = warn.sms_parts(text)
            texts.append({"lang": lang, "text": text, "chars": n, "parts": parts, "enc": enc})
        sms.append({"n": k + 1, "t": w["t"].strftime("%Y-%m-%dT%H:%M:00Z"), "h": round(w["hours_before_arrival"]),
                    "kind": w["kind"], "texts": texts})
    S.append({"id": sid, "name": name.title(), "year": int(g.year.iloc[0]), "peak": int(g.wind.max()),
              "basin": str(g.storm_basin.iloc[0]), "state": state, "lang": warn.LANGUAGE.get(state, "English") if state else None,
              "arrive": arr.strftime("%Y-%m-%dT%H:%M:00Z") if state else None,
              "warned": bool(sent), "first_h": round(sent[0]["hours_before_arrival"]) if sent else None, "sms": sms,
              "pts": [[r.t.strftime("%Y-%m-%dT%H:%MZ"), round(float(r.lat), 2), round(float(r.lon), 2), int(r.wind),
                       round(float(r.p), 3), int(r.flag), int(r.RI) if not np.isnan(r.wind_24) else None]
                      for r in g.itertuples()]})

T = pd.read_csv(f"{mdir}/model_comparison_test.csv")
CV = pd.read_csv(f"{mdir}/model_cv_1982_2018.csv")
models = []
for r in T.itertuples():
    cv = CV[CV.model == r.model]
    models.append({"model": r.model, "precision": round(r.precision, 3), "recall": round(r.recall, 3),
                   "f2": round(r.F2, 3), "pr_auc": round(r.PR_AUC, 3), "roc_auc": round(r.ROC_AUC, 3),
                   "caught": r.RI_storms_caught,
                   "cv_pr_auc": round(float(cv.PR_AUC_mean.iloc[0]), 3) if len(cv) else None})

templates = {"FIRST": warn.FIRST, "REST": warn.REST, "DIGITS": warn.DIGITS, "MONTHS": warn.MONTHS,
             "MONTH_FIRST": sorted(warn.MONTH_FIRST), "LANGUAGE": warn.LANGUAGE, "REGION": warn.REGION,
             "LOCAL_NAME": warn.LOCAL_NAME}
json.dump({"threshold": th, "countries": countries, "states": states, "storms": S, "models": models,
           "templates": templates}, open(out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
cyc = [s for s in S if s["state"] and s["peak"] >= 34]
print("storms", len(S), "cyclones heading for India", len(cyc), "warned", sum(s["warned"] for s in cyc),
      "size KB", round(len(open(out, encoding="utf-8").read().encode()) / 1024))
