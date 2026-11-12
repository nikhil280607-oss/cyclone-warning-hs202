"""Report figures from the real outputs of run_replay.py and compare_models.py.

usage: python3 make_figures.py OBS.json STORMS.json COAST.geojson COUNTRIES.geojson RESULTS_DIR MODELS_DIR FIG_DIR
"""
import sys, os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=2.4, foreground='white')]
import prep, warn

obs, storms, coast_f, countries_f, res, mdir, fig = sys.argv[1:8]
os.makedirs(fig, exist_ok=True)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left"})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


d = prep.build_rows(prep.load(obs), storms)
R = pd.read_csv(f"{res}/replay_storms.csv")
M = pd.read_csv(f"{res}/messages.csv")

# ---------------- Fig 1: study area, tracks of test cyclones heading for India ----------------
def polys(geom):
    return geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]


fig1, ax = plt.subplots(figsize=(6.5, 5.6))
ax.set_facecolor("#eef3f8")
for f in json.load(open(countries_f))["features"]:
    for p in polys(f["geometry"]):
        ax.add_patch(Polygon(np.array(p[0]), closed=True, fc="#f4f3ef", ec="#c3c2b7", lw=0.4, zorder=1))
coast = json.load(open(coast_f))["features"]
for f in coast:
    for p in polys(f["geometry"]):
        ax.add_patch(Polygon(np.array(p[0]), closed=True, fc="#dbe6f3", ec="#9ec5f4", lw=0.5, zorder=2))
cyc = R[(R.language.notna()) & (R.peak_kt >= 34)]
for _, r in cyc.iterrows():
    g = d[d.storm_id == r.storm_id].sort_values("t")
    col = S1 if r.sms_our_rule > 0 else S2
    ax.plot(g.lon, g.lat, color=col, lw=1.1, alpha=0.9, zorder=4, solid_capstyle="round")
    ax.scatter(g.lon.iloc[-1], g.lat.iloc[-1], s=8, color=col, zorder=5)
label_at = {"FANI": (89.6, 6.0), "AMPHAN": (89.3, 11.8), "TAUKTAE": (68.6, 11.0), "BIPARJOY": (58.6, 15.0),
            "ASNA": (62.2, 23.9), "VAYU": (63.0, 22.0), "MICHAUNG": (84.6, 10.2), "NIVAR": (85.8, 8.3)}
for nm, (x, y) in label_at.items():
    rr = cyc[cyc.name == nm]
    if len(rr):
        ax.text(x, y, f"{nm.title()} {rr.year.iloc[0]}", fontsize=7, color=INK2, zorder=6, path_effects=HALO)
lang_lab = {"Gujarat": (70.0, 22.9, "Gujarat · Gujarati"), "Maharashtra": (73.6, 18.6, "Maharashtra · Marathi"),
            "Kerala": (75.2, 10.0, "Kerala · Malayalam"), "Tamil Nadu": (77.2, 11.8, "Tamil Nadu · Tamil"),
            "Andhra Pradesh": (78.2, 15.8, "Andhra Pradesh · Telugu"), "Odisha": (82.3, 20.6, "Odisha · Odia"),
            "West Bengal": (86.2, 23.6, "West Bengal · Bengali"), "Karnataka": (74.4, 14.2, "Karnataka · Kannada")}
for st, (x, y, t) in lang_lab.items():
    ax.text(x, y, t, fontsize=6.5, color="#184f95", fontweight="bold", zorder=7, path_effects=HALO)
ax.set_xlim(58, 98); ax.set_ylim(2, 27); ax.set_aspect(1 / np.cos(np.radians(14)))
ax.set_xticks(range(60, 100, 10)); ax.set_xticklabels([f"{x}°E" for x in range(60, 100, 10)])
ax.set_yticks(range(5, 30, 5)); ax.set_yticklabels([f"{y}°N" for y in range(5, 30, 5)])
for s in ax.spines.values():
    s.set_visible(True); s.set_color(AXIS)
ax.plot([], [], color=S1, lw=1.5, label=f"Warned before arrival ({(cyc.sms_our_rule > 0).sum()})")
ax.plot([], [], color=S2, lw=1.5, label=f"Not warned ({(cyc.sms_our_rule == 0).sum()})")
ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRID, fontsize=7.5)
ax.set_title("Cyclones heading for India, 2019–2025, and the language used on each coast")
fig1.tight_layout(); fig1.savefig(f"{fig}/fig_study_area.png", bbox_inches="tight"); plt.close(fig1)

# ---------------- Fig 2: Fani timeline ----------------
g = d[(d.name == "FANI") & (d.year == 2019)].sort_values("t").copy()
import model as mdl
lab = d[d.wind_24.notna()]
sp = prep.split(lab)
m, th = mdl.fit(sp["train"], sp["val"])
g["p"] = m.predict_proba(g[prep.FEATURES])[:, 1]
t_ist = g.t.dt.tz_convert("Asia/Kolkata")
fig2, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
ri = g[g.RI == 1]
for _, r in ri.iterrows():
    tt = r.t.tz_convert("Asia/Kolkata")
    for a in (a1, a2):
        a.axvspan(tt, tt + pd.Timedelta(hours=3), color="#cde2fb", lw=0, zorder=0)
a1.plot(t_ist, g.wind, color=S1, lw=2, zorder=3)
a1.set_ylabel("Max wind (kt)")
grid(a1)
fm = M[M.name == "FANI"].drop_duplicates("sms_no")
for _, r in fm.iterrows():
    tt = pd.Timestamp(r.sent_UTC, tz="UTC").tz_convert("Asia/Kolkata")
    a1.axvline(tt, color=S2, lw=1.2, ls="--", zorder=2)
    a1.text(tt, 122, f"SMS {r.sms_no}\n{r.hours_before_arrival} h before", fontsize=6.5, color=INK2, ha="center", va="bottom")
arr = pd.Timestamp("2019-05-03 11:30", tz="Asia/Kolkata")
a1.axvline(arr, color=INK, lw=1)
a1.text(arr, 25, " landfall,\n Puri", fontsize=7, color=INK, va="bottom")
a1.set_ylim(15, 140)
a1.set_title("Cyclone Fani (2019): model probability, rapid intensification and the SMS sent")
a2.plot(t_ist, g.p, color=INK2, lw=1.5)
a2.axhline(th, color=S2, lw=1, ls=":")
a2.text(0.80, th + 0.04, f"alert threshold {th}", fontsize=7, color=INK2, transform=a2.get_yaxis_transform())
a2.set_ylabel("P(RI in next 24 h)")
a2.set_ylim(0, 1)
grid(a2)
import matplotlib.dates as mdates
a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=t_ist.dt.tz))
a1.text(0.01, 0.80, "shaded: times when the storm went on to\ngain 30 kt or more in the next 24 h (observed)",
        transform=a1.transAxes, ha="left", fontsize=7, color=INK2)
fig2.tight_layout(); fig2.savefig(f"{fig}/fig_fani_timeline.png"); plt.close(fig2)

# ---------------- Fig 3: lead time per warned cyclone ----------------
w = cyc[cyc.sms_our_rule > 0].sort_values("first_sms_h_before")
fig3, ax = plt.subplots(figsize=(6.5, 4.6))
y = np.arange(len(w))
ax.barh(y, w.first_sms_h_before, color=S1, height=0.7)
ax.set_yticks(y); ax.set_yticklabels([f"{n.title()} {yr} · {lg}" for n, yr, lg in zip(w.name, w.year, w.language)], fontsize=7)
ax.axvline(48, color=S2, lw=1, ls="--"); ax.text(48.5, len(w) - 0.6, "48 h: IMD cyclone alert", fontsize=7, color=INK2)
ax.axvline(72, color=MUTED, lw=1, ls=":"); ax.text(72.5, len(w) - 2.2, "72 h:\nwindow\nopens", fontsize=7, color=INK2)
ax.set_xlabel("Hours from first SMS to arrival at the coast")
ax.set_xlim(0, 84)
grid(ax, "x")
ax.set_title(f"Warning time for the {len(w)} warned cyclones (median {w.first_sms_h_before.median():.0f} h)")
fig3.tight_layout(); fig3.savefig(f"{fig}/fig_lead_time.png"); plt.close(fig3)

# ---------------- Fig 4: precision-recall curves ----------------
C = json.load(open(f"{mdir}/curves.json"))
T = pd.read_csv(f"{mdir}/model_comparison_test.csv").set_index("model")
fig4, ax = plt.subplots(figsize=(6.5, 4.2))
cols = {"Logistic regression": S1, "Random forest": S2, "Gradient boosting": S3, "Neural network (MLP)": S4}
endpos = {}
for nm, c in cols.items():
    pr, rc = C[nm]["pr"]
    ax.plot(rc, pr, color=c, lw=1.6, label=f"{nm} (PR-AUC {T.loc[nm, 'PR_AUC']:.2f})")
    op = C[nm]["op"]
    ax.scatter(op[1], op[0], s=36, color=c, edgecolor="white", lw=1.2, zorder=5)
rule = T.loc["Rule: +10 kt in last 12 h"]
ax.scatter(rule.recall, rule.precision, marker="s", s=30, color=INK2, zorder=5)
ax.annotate("rule: +10 kt in 12 h", (rule.recall, rule.precision), (0.47, 0.08), fontsize=7, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), path_effects=HALO)
base = float(T.loc["Climatology (always 5.8%)", "PR_AUC"])
ax.axhline(base, color=MUTED, lw=1, ls=":"); ax.text(0.80, base - 0.025, f"random guess ({base:.2f})", fontsize=7, color=MUTED)
ax.set_xlabel("Recall (share of RI times flagged)"); ax.set_ylabel("Precision (share of flags that were RI)")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
grid(ax, "both")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title("Precision–recall on 2019–2025 test storms (dots: chosen alert thresholds)")
fig4.tight_layout(); fig4.savefig(f"{fig}/fig_pr_curves.png"); plt.close(fig4)

# ---------------- Fig 5: ablation + importance ----------------
A = pd.read_csv(f"{mdir}/ablation_logistic.csv")
I = pd.read_csv(f"{mdir}/permutation_importance.csv")
I = I[I.model == "Logistic regression"].sort_values("PR_AUC_drop")
nice = {"ci_no": "Satellite CI number", "wind": "Current wind", "arabian_sea": "Arabian Sea (basin)",
        "dV_past24": "Wind change, last 24 h", "lon": "Longitude", "dV_past12": "Wind change, last 12 h",
        "pressure": "Central pressure", "lat": "Latitude", "day_of_year": "Day of year", "dV_past6": "Wind change, last 6 h"}
fig5, (b1, b2) = plt.subplots(1, 2, figsize=(6.5, 3.4), gridspec_kw={"width_ratios": [1.05, 1]})
b1.barh([nice[f] for f in I.feature], I.PR_AUC_drop, xerr=I.sd, color=S1, height=0.65,
        error_kw={"ecolor": MUTED, "lw": 0.8})
b1.set_xlabel("Drop in test PR-AUC when shuffled"); grid(b1, "x")
b1.set_title("(a) Permutation importance", fontsize=9)
b1.tick_params(axis="y", labelsize=7)
A2 = A.iloc[::-1]
colors = [INK2 if r.startswith("none") else S2 for r in A2.removed]
names = {"none (full model)": "Nothing (full model)", "satellite CI number": "Satellite CI number",
         "recent intensity change": "Recent intensity change", "current intensity (wind, pressure)": "Current intensity",
         "position": "Position (lat, lon)", "season and basin": "Season and basin"}
b2.barh([names[r] for r in A2.removed], A2.PR_AUC, color=colors, height=0.65)
b2.axvline(A.PR_AUC.iloc[0], color=INK2, lw=0.8, ls=":")
for yv, v in enumerate(A2.PR_AUC):
    b2.text(v + 0.004, yv, f"{v:.3f}", va="center", fontsize=7, color=INK2, path_effects=HALO)
b2.set_xlabel("Test PR-AUC"); b2.set_xlim(0, 0.29); grid(b2, "x")
b2.set_title("(b) Inputs removed (ablation)", fontsize=9)
b2.tick_params(axis="y", labelsize=7)
fig5.tight_layout(); fig5.savefig(f"{fig}/fig_importance_ablation.png"); plt.close(fig5)

# ---------------- Fig 6: SMS length per language ----------------
first = M[M.sms_no == 1].groupby("language").agg(chars=("chars", "median"), parts=("sms_parts", "median")).reset_index()
first = first.sort_values("chars")
fig6, ax = plt.subplots(figsize=(6.5, 3.0))
yy = np.arange(len(first))
ax.barh(yy, first.chars, color=[S1 if l == "English" else S3 for l in first.language], height=0.62)
for yv, (c, p_) in enumerate(zip(first.chars, first.parts)):
    ax.text(c + 2, yv, f"{int(c)} characters · {int(p_)} SMS", va="center", fontsize=7, color=INK2, path_effects=HALO)
ax.set_yticks(yy); ax.set_yticklabels(first.language)
ax.axvline(160, color=MUTED, lw=1, ls=":"); ax.text(161, len(first) - 0.5, "160: one English SMS", fontsize=7, color=MUTED)
ax.axvline(70, color=MUTED, lw=1, ls="--"); ax.text(71, -0.9, "70: one Unicode SMS", fontsize=7, color=MUTED)
ax.set_xlim(0, 230); ax.set_ylim(-1.2, len(first) - 0.2); ax.set_xlabel("Median length of the first warning SMS (characters)")
grid(ax, "x")
ax.set_title("Same warning, different cost: Indian scripts need Unicode SMS")
fig6.tight_layout(); fig6.savefig(f"{fig}/fig_sms_length.png"); plt.close(fig6)
print("figures written to", fig)
