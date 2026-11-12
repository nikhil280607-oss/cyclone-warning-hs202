"""Replay the warning system with the LR+RF ensemble (extra inputs), threshold from 2013-2018."""
import sys, runpy, numpy as np, pandas as pd
import warn
obs, storms, coast_f, out = sys.argv[1:5]
sys.argv = ["improve_model.py", obs, storms, coast_f, out]
g = runpy.run_path("improve_model.py")
d, tr, BASE, EXTRA, cands = g["d"], g["tr"], g["BASE"], g["EXTRA"], g["cands"]
F = BASE + EXTRA
lrm, rfm = cands["LR base+extra"][0], cands["RF base+extra"][0]
th = float(g["R"].set_index("model").loc["LR+RF ensemble base+extra", "threshold"])
d["p_ens"] = (lrm.predict_proba(d)[:, 1] + rfm.predict_proba(d)[:, 1]) / 2
d["flag"] = (d.p_ens >= th).astype(int)
coast = warn.Coast(coast_f)
for y0, y1 in [(2013, 2018), (2019, 2025)]:
    rows = []
    for sid, s in d[(d.year >= y0) & (d.year <= y1)].groupby("storm_id"):
        s = s.sort_values("t")
        state, arr, sent, upd = warn.warnings_for_storm(s, s.flag.values, coast)
        rows.append({"name": s["name"].iloc[0], "year": s.year.iloc[0], "peak": s.wind.max(), "state": state,
                     "sms": len(sent), "first_h": round(sent[0]["hours_before_arrival"]) if sent else np.nan})
    R = pd.DataFrame(rows); near = R[R.state.notna()]; cyc = near[near.peak >= 34]; sev = near[near.peak >= 48]
    w = cyc[cyc.sms > 0]
    print(f"{y0}-{y1}: cyclones {len(cyc)} warned {len(w)}; severe {(sev.sms>0).sum()}/{len(sev)}; median {w.first_h.median():.0f} h; "
          f">=48h {(w.first_h>=48).sum()}; SMS {near.sms.sum()}; false warnings {((near.sms>0)&(near.peak<34)).sum()}; missed "
          + ", ".join(f"{a} {b}" for a, b in zip(cyc[cyc.sms==0].name, cyc[cyc.sms==0].year)))
d[["storm_id","name","year","t","lat","lon","wind","RI","p_ens","flag"]].to_csv(f"{out}/ensemble_probs.csv", index=False)
print("threshold", th)
