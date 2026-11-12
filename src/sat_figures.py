import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from PIL import Image
satf, tracks, imgdir, out = sys.argv[1:5]
INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"; S1, S2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 300,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
# montage of four Fani images
names = ["FANI_20190427_0002", "FANI_20190429_0002", "FANI_20190501_0002", "FANI_20190502_0002"]
ims = [Image.open(f"{imgdir}/{n}.png").convert("RGB") for n in names]
w, h = ims[0].size
mont = Image.new("RGB", (w * 2, h * 2), "white")
for i, im in enumerate(ims):
    mont.paste(im, ((i % 2) * w, (i // 2) * h))
mont.save(f"{out}/fig_sat_fani_montage.png")
# time series: wind and cold-cloud fraction, Fani
s = pd.read_csv(satf); s["t"] = pd.to_datetime(s.time_utc, utc=True)
tr = pd.read_csv(tracks); tr["t"] = pd.to_datetime(tr.t, utc=True)
f = s[s.storm == "FANI"].groupby(s.t.dt.round("12h")).mean(numeric_only=True).reset_index()
tf = tr[tr.storm == "FANI"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 3.8), sharex=True)
for _, r in tf[tf.RI_next24h == 1].iterrows():
    for a in (a1, a2):
        a.axvspan(r.t, r.t + pd.Timedelta(hours=3), color="#cde2fb", lw=0)
a1.plot(tf.t, tf.wind, color=S1, lw=2); a1.set_ylabel("IMD max wind (kt)"); a1.grid(axis="y", color=GRID, lw=0.6)
a1.set_title("Cyclone Fani: INSAT-3DR cold cloud and IMD intensity")
a2.plot(f.t, 100 * f.frac_lt220_200km, color=S2, lw=2, marker="o", ms=3.5, label="within 200 km")
a2.plot(f.t, 100 * f.frac_lt220_500km, color=INK2, lw=1.5, ls="--", marker="o", ms=3, label="within 500 km")
a2.set_ylabel("Cloud tops < 220 K (%)"); a2.set_ylim(0, 105); a2.grid(axis="y", color=GRID, lw=0.6)
a2.legend(frameon=False, fontsize=7.5, loc="lower right")
a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
a1.text(0.01, 0.82, "shaded: followed by +30 kt in 24 h (rapid intensification)", transform=a1.transAxes, fontsize=7, color=INK2)
fig.tight_layout(); fig.savefig(f"{out}/fig_sat_fani_timeseries.png"); print("ok")
