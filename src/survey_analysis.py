"""Summarise the Google Form reading test (raw export) and draw the survey figure.
usage: python3 survey_analysis.py form_responses.csv OUTDIR"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
d = pd.read_csv(src)
d.columns = ["timestamp", "lang", "coastal", "action", "urgency", "sufficient", "effective", "act"]
d["lang"] = d.lang.str.split().str[-1]
d["t"] = pd.to_datetime(d.timestamp, dayfirst=True)
d["correct"] = d.action.str.startswith("Return to shore")
d["urg_hi"] = d.urgency >= 4; d["eff_hi"] = d.effective >= 4
d["suff"] = d.sufficient.str.startswith(("Yes", "Mostly"))
d["would_act"] = d.act == "Yes"
n = len(d)
S = {"n": n, "first": str(d.t.min()), "last": str(d.t.max()),
     "coastal_yes": int((d.coastal == "Yes").sum()),
     "correct_action": round(d.correct.mean(), 3), "urgency_mean": round(d.urgency.mean(), 2), "urgency_4_5": round(d.urg_hi.mean(), 3),
     "sufficient_yes": round((d.sufficient.str.startswith("Yes")).mean(), 3), "sufficient_yes_or_mostly": round(d.suff.mean(), 3),
     "effective_mean": round(d.effective.mean(), 2), "effective_4_5": round(d.eff_hi.mean(), 3),
     "act_yes": round(d.would_act.mean(), 3), "act_maybe": round((d.act == "Maybe").mean(), 3), "act_no": round((d.act == "No").mean(), 3)}
for c in ["lang", "coastal", "action", "sufficient", "act"]:
    S[f"counts_{c}"] = d[c].value_counts().to_dict()
for c in ["urgency", "effective"]:
    S[f"counts_{c}"] = {int(k): int(v) for k, v in d[c].value_counts().sort_index().items()}
by_lang = d.groupby("lang").agg(n=("lang", "size"), correct=("correct", "mean"), urgency=("urgency", "mean"),
                                effective=("effective", "mean"), would_act=("would_act", "mean")).sort_values("n", ascending=False)
by_coast = d.groupby("coastal").agg(n=("lang", "size"), correct=("correct", "mean"), urgency=("urgency", "mean"),
                                    effective=("effective", "mean"), would_act=("would_act", "mean"))
by_lang.round(3).to_csv(f"{out}/survey_by_language.csv"); by_coast.round(3).to_csv(f"{out}/survey_by_coastal.csv")
json.dump(S, open(f"{out}/survey_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(S, indent=1, ensure_ascii=False)); print(by_lang.round(2).to_string()); print(by_coast.round(2).to_string())
# figure
INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"; S1, S2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 300,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw={"width_ratios": [0.9, 1.25]})
bl = by_lang.sort_values("n")
a.barh(bl.index, bl.n, color=S1, height=0.62)
for y, (v, c) in enumerate(zip(bl.n, bl.correct)):
    a.text(v + 0.3, y, f"{v}  ({c:.0%} correct)", va="center", fontsize=7, color=INK2)
a.set_xlim(0, bl.n.max() * 1.9); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
a.set_title(f"(a) Respondents by language (n = {n})")
items = [("Chose the correct action\n(return to shore now)", d.correct.mean()), ("Urgency rated clear\n(4 or 5 of 5)", d.urg_hi.mean()),
         ("Information at least\nmostly sufficient", d.suff.mean()), ("Effective as an official\nalert (4 or 5 of 5)", d.eff_hi.mean()),
         ("Would act on it\nas a fisher (yes)", d.would_act.mean())][::-1]
b.barh([i[0] for i in items], [100 * i[1] for i in items], color=S2, height=0.62)
for y, (_, v) in enumerate(items):
    b.text(100 * v + 1, y, f"{100 * v:.0f}%", va="center", fontsize=7.5, color=INK2)
b.set_xlim(0, 112); b.set_xticks([0, 25, 50, 75, 100]); b.tick_params(axis="y", labelsize=7.5)
b.grid(axis="x", color=GRID, lw=0.6); b.set_axisbelow(True); b.set_title("(b) Reading test results")
fig.tight_layout(); fig.savefig(f"{out}/fig_survey.png")
