"""Run the whole warning system and replay it on every 2013-2018 and 2019-2025 storm.

usage: python3 run_replay.py OBSERVATIONS.json STORMS.json COASTAL_STATES.geojson OUTDIR
"""
import sys, os, json
import numpy as np, pandas as pd
import prep, model, warn

obs, storms, coast_file, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)

# 1. data
o = prep.load(obs)
d = prep.build_rows(o, storms)
labelled = d[d.wind_24.notna()]
sp = prep.split(labelled)
print(f"records used: {len(o)}  (clashing records removed: {o.attrs['n_clash']})")
print(f"rows with a 24 h target: {len(labelled)}, RI: {labelled.RI.sum()}")
for k, v in sp.items():
    print(f"  {k:5s}: {v.storm_id.nunique():3d} storms, {len(v):4d} times, {v.RI.sum():3d} RI")

# 2. model (logistic regression)
m, th = model.fit(sp["train"], sp["val"])
te = sp["test"]
s = model.scores(te, (m.predict_proba(te[prep.FEATURES])[:, 1] >= th).astype(int))
print(f"\nmodel threshold (chosen on 2013-2018): {th}")
print("test 2019-2025:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})

# 3. replay the warning system storm by storm, using every record (not only labelled ones)
coast = warn.Coast(coast_file)


def replay(y0, y1):
    part = d[(d.year >= y0) & (d.year <= y1)].copy()
    part["flag"] = (m.predict_proba(part[prep.FEATURES])[:, 1] >= th).astype(int)
    rows, msgs = [], []
    for sid, g in part.groupby("storm_id", sort=True):
        g = g.sort_values("t")
        state, t_arr, sent, updates = warn.warnings_for_storm(g, g.flag.values, coast)
        window = g[(g.t < t_arr) & (g.t >= t_arr - pd.Timedelta(hours=warn.WINDOW_H))] if state else g.iloc[:0]
        rows.append({"storm_id": sid, "name": g["name"].iloc[0], "year": g.year.iloc[0], "peak_kt": g.wind.max(),
                     "coast_at_risk": state or "none (not heading for an Indian coast)",
                     "language": warn.LANGUAGE.get(state, "English") if state else "",
                     "arrival_IST": t_arr.tz_convert("Asia/Kolkata").strftime("%d %b %Y %H:%M") if state else "",
                     "first_sms_h_before": round(sent[0]["hours_before_arrival"]) if sent else np.nan,
                     "sms_our_rule": len(sent), "sms_if_every_update": updates, "ever_flagged": int(g.flag.max()),
                     "flagged_in_window": int(window.flag.max()) if len(window) else 0})
        for k, w in enumerate(sent):
            for lang, text in w["sms"].items():
                n, parts, enc = warn.sms_parts(text)
                msgs.append({"storm_id": sid, "name": g["name"].iloc[0], "sms_no": k + 1, "kind": w["kind"],
                             "sent_UTC": w["t"].strftime("%Y-%m-%d %H:%M"), "hours_before_arrival": round(w["hours_before_arrival"]),
                             "language": lang, "chars": n, "sms_parts": parts, "encoding": enc, "text": text})
    return pd.DataFrame(rows), pd.DataFrame(msgs)


def summary(R, label):
    near = R[R.language != ""]
    warned = near[near.sms_our_rule > 0]
    cyc = near[near.peak_kt >= 34]
    sev = near[near.peak_kt >= 48]
    wc = warned[warned.peak_kt >= 34]
    missed = cyc[cyc.sms_our_rule == 0]
    print(f"\n== replay {label} ==")
    print(f"storms: {len(R)}, heading for an Indian coast: {len(near)}")
    print(f"cyclones heading for India: {len(cyc)}")
    print(f"  warned before arrival: {(cyc.sms_our_rule > 0).sum()}")
    print(f"  (SMS only on a fresh RI flag: {(cyc.flagged_in_window > 0).sum()})")
    print(f"severe cyclones warned: {(sev.sms_our_rule > 0).sum()} of {len(sev)}")
    print(f"first SMS: median {wc.first_sms_h_before.median():.0f} h before arrival")
    print(f"  at 48 h or earlier: {(wc.first_sms_h_before >= 48).sum()} of {len(wc)}")
    print(f"SMS sent: {near.sms_our_rule.sum()} (one per update: {near.sms_if_every_update.sum()})")
    print(f"false warnings (never 34 kt): {(warned.peak_kt < 34).sum()}")
    print("missed:", ", ".join(f"{n.title()} {y}" for n, y in zip(missed.name, missed.year)) or "none")


Rv, _ = replay(2013, 2018)
summary(Rv, "2013-2018 (check)")
R, M = replay(2019, 2025)
summary(R, "2019-2025 test storms")
R.to_csv(f"{out}/replay_storms.csv", index=False)
M.to_csv(f"{out}/messages.csv", index=False)
json.dump({"threshold": th, "test_scores": s}, open(f"{out}/model_scores.json", "w"), indent=1, default=str)
