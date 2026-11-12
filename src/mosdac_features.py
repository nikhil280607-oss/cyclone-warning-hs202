"""
Turn INSAT-3DR IMAGER L1C (3RIMG_..._L1C_ASIA_MER_*.h5) TIR1 files into storm features.

For every file it finds the storm centre at that time (IMD best track, interpolated),
then measures the cloud-top brightness temperature (BT) around it:
  min / mean BT within 200 km and 500 km, share of pixels colder than 220 K and 200 K,
  cold-cloud area (< 220 K) within 500 km.
It also saves one cropped picture per storm per day for the report.

Run (in the folder that has this script and storm_tracks.csv):
    python3 mosdac_features.py --inspect  FOLDER_WITH_H5_FILES      # prints what is inside one file
    python3 mosdac_features.py            FOLDER_WITH_H5_FILES      # makes the CSV and pictures
Outputs: satellite_features.csv, sat_images/*.png, sat_timeseries_<storm>.png
"""
import sys, os, re, glob, math
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH = 6378137.0
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def file_time(path):
    m = re.search(r"3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})", os.path.basename(path))
    d, mon, y, hh, mm = m.groups()
    return pd.Timestamp(year=int(y), month=MONTHS[mon], day=int(d), hour=int(hh), minute=int(mm), tz="UTC")


def inspect(path):
    with h5py.File(path, "r") as f:
        print("FILE:", path)
        print("global attributes:")
        for k, v in f.attrs.items():
            print("   ", k, "=", v)
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:30s} shape={obj.shape} dtype={obj.dtype} attrs={dict(obj.attrs)}")
        f.visititems(show)


def read_bt(f):
    """Brightness temperature (K) as a 2-D array, NaN where missing."""
    names = list(f.keys())
    if "TIR1_BT" in names:                                        # MOSDAC order with "BT" chosen
        ds = f["TIR1_BT"]
        bt = np.squeeze(ds[...]).astype("float32")
        fill = float(np.ravel(ds.attrs.get("_FillValue", -999.0))[0])
        bt[bt == fill] = np.nan
        bt[(bt < 150) | (bt > 350)] = np.nan
        return bt
    temp = f["IMG_TIR1_TEMP"] if "IMG_TIR1_TEMP" in names else None
    if temp is not None and temp.ndim >= 2:                      # BT already as an image
        bt = np.squeeze(temp[...]).astype("float32")
        fill = temp.attrs.get("_FillValue")
    else:                                                         # counts + look-up table
        cnt_ds = f["IMG_TIR1"]
        cnt = np.squeeze(cnt_ds[...]).astype("int64")
        fill = cnt_ds.attrs.get("_FillValue")
        lut = np.asarray(temp[...], dtype="float32").ravel()
        bad = (cnt < 0) | (cnt >= len(lut))
        if fill is not None:
            bad |= cnt == int(np.ravel(fill)[0])
        bt = lut[np.clip(cnt, 0, len(lut) - 1)]
        bt[bad] = np.nan
        fill = None
    if fill is not None:
        bt[bt == np.ravel(fill)[0]] = np.nan
    bt[(bt < 150) | (bt > 350)] = np.nan
    return bt


def scaled(ds):
    a = ds[...].astype("float64")
    sf, off = ds.attrs.get("scale_factor"), ds.attrs.get("add_offset")
    if sf is not None:
        a = a * float(np.ravel(sf)[0])
    if off is not None:
        a = a + float(np.ravel(off)[0])
    return np.squeeze(a)


_GRID = {}


def mercator_latlon(f):
    """INSAT L1C ASIA_MER: X/Y in metres on an ellipsoidal Mercator grid (WGS84, standard parallel)."""
    pi = f["Projection_Information"].attrs
    a = float(np.ravel(pi["semi_major_axis"])[0]); b = float(np.ravel(pi["semi_minor_axis"])[0])
    lon0 = float(np.ravel(pi["longitude_of_projection_origin"])[0]); phi_ts = np.radians(float(np.ravel(pi["standard_parallel"])[0]))
    e = np.sqrt(1 - (b / a) ** 2)
    k0 = np.cos(phi_ts) / np.sqrt(1 - (e * np.sin(phi_ts)) ** 2)
    x = f["X"][...].astype("float64"); y = f["Y"][...].astype("float64")
    lon = lon0 + np.degrees(x / (a * k0))
    t = np.exp(-y / (a * k0))
    phi = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(8):
        es = e * np.sin(phi)
        phi = np.pi / 2 - 2 * np.arctan(t * ((1 - es) / (1 + es)) ** (e / 2))
    lat = np.degrees(phi)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def read_latlon(f, shape):
    """Latitude and longitude for every pixel (2-D arrays)."""
    if "Projection_Information" in f and "X" in f and "Y" in f:
        key = (f["X"].shape, f["Y"].shape, float(f["X"][0]), float(f["Y"][0]))
        if key not in _GRID:
            _GRID[key] = mercator_latlon(f)
        return _GRID[key]
    keys = {k.lower(): k for k in f.keys()}
    if "latitude" in keys and "longitude" in keys:
        lat, lon = scaled(f[keys["latitude"]]), scaled(f[keys["longitude"]])
        if np.nanmax(np.abs(lat)) > 90:          # stored as degrees x 100
            lat, lon = lat / 100.0, lon / 100.0
        if lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        return lat, lon
    if "x" in keys and "y" in keys:
        x, y = scaled(f[keys["x"]]), scaled(f[keys["y"]])
        if np.nanmax(np.abs(x)) > 360:           # Mercator metres -> degrees
            lon = np.degrees(x / R_EARTH)
            lat = np.degrees(np.arctan(np.sinh(y / R_EARTH)))
        else:
            lon, lat = x, y
        lon2, lat2 = np.meshgrid(lon, lat)
        return lat2, lon2
    a = f.attrs                                   # fall back to the corner attributes
    left, right = float(np.ravel(a["left_longitude"])[0]), float(np.ravel(a["right_longitude"])[0])
    top, bottom = float(np.ravel(a["upper_latitude"])[0]), float(np.ravel(a["lower_latitude"])[0])
    ny, nx = shape
    lon = np.linspace(left, right, nx)
    my = np.linspace(np.log(np.tan(np.pi / 4 + np.radians(top) / 2)),
                     np.log(np.tan(np.pi / 4 + np.radians(bottom) / 2)), ny)
    lat = np.degrees(2 * np.arctan(np.exp(my)) - np.pi / 2)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def km(lat1, lon1, lat2, lon2):
    r = np.radians
    a = np.sin(r(lat2 - lat1) / 2) ** 2 + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def centre_at(track, storm, t):
    g = track[track.storm == storm]
    ts = (g.t - t).dt.total_seconds().values
    if ts.min() > 0 or ts.max() < 0:
        return None
    lat = np.interp(0, ts, g.lat.values)
    lon = np.interp(0, ts, g.lon.values)
    wind = np.interp(0, ts, g.wind.values)
    return lat, lon, wind


def main(folder):
    track = pd.read_csv("storm_tracks.csv", parse_dates=["t"])
    track["t"] = pd.to_datetime(track.t, utc=True)
    files = sorted(glob.glob(os.path.join(folder, "**", "*L1C_ASIA_MER*.h5"), recursive=True))
    print(len(files), "files found")
    os.makedirs("sat_images", exist_ok=True)
    rows, saved_days = [], set()
    for i, path in enumerate(files):
        t = file_time(path)
        hit = None
        for storm in track.storm.unique():
            c = centre_at(track, storm, t)
            if c:
                hit = (storm, *c)
                break
        if hit is None:
            continue
        storm, clat, clon, cwind = hit
        try:
            with h5py.File(path, "r") as f:
                bt = read_bt(f)
                lat, lon = read_latlon(f, bt.shape)
        except Exception as e:
            print("  could not read", os.path.basename(path), "->", e)
            continue
        box = (np.abs(lat - clat) < 7) & (np.abs(lon - clon) < 7)
        if box.sum() == 0:
            continue
        rr, cc = np.where(box)
        sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
        b, la, lo = bt[sl], lat[sl], lon[sl]
        dist = km(clat, clon, la, lo)
        row = {"file": os.path.basename(path), "storm": storm, "time_utc": t.strftime("%Y-%m-%d %H:%M"),
               "centre_lat": round(clat, 2), "centre_lon": round(clon, 2), "imd_wind_kt": round(cwind, 1)}
        for rad in (200, 500):
            v = b[(dist <= rad) & np.isfinite(b)]
            row[f"n_pix_{rad}km"] = int(v.size)
            row[f"min_bt_{rad}km"] = float(np.min(v)) if v.size else np.nan
            row[f"mean_bt_{rad}km"] = float(np.mean(v)) if v.size else np.nan
            row[f"std_bt_{rad}km"] = float(np.std(v)) if v.size else np.nan
            row[f"frac_lt220_{rad}km"] = float(np.mean(v < 220)) if v.size else np.nan
            row[f"frac_lt200_{rad}km"] = float(np.mean(v < 200)) if v.size else np.nan
        row["cold_area_lt220_500km_km2"] = row["frac_lt220_500km"] * math.pi * 500 ** 2
        rows.append(row)
        day = (storm, t.strftime("%Y-%m-%d"))
        if day not in saved_days and t.hour == 0:
            saved_days.add(day)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.pcolormesh(lo, la, b, cmap="gray_r", vmin=180, vmax=300, shading="auto")
            ax.contour(lo, la, dist, levels=[200, 500], colors=["#eb6834", "#2a78d6"], linewidths=1)
            ax.plot(clon, clat, "+", color="#eb6834", ms=10)
            ax.set_aspect(1 / np.cos(np.radians(clat)))
            ax.set_title(f"{storm.title()}  {t.strftime('%d %b %Y %H:%M')} UTC  ({cwind:.0f} kt)", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8, label="TIR1 brightness temperature (K)")
            fig.tight_layout()
            fig.savefig(f"sat_images/{storm}_{t.strftime('%Y%m%d_%H%M')}.png", dpi=200)
            plt.close(fig)
        if i % 20 == 0:
            print(f"  {i + 1}/{len(files)}  {storm} {t}")
    out = pd.DataFrame(rows).sort_values(["storm", "time_utc"])
    out.to_csv("satellite_features.csv", index=False)
    print("saved satellite_features.csv with", len(out), "rows")
    for storm, g in out.groupby("storm"):
        tt = pd.to_datetime(g.time_utc)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        a1.plot(tt, g.imd_wind_kt, color="#2a78d6"); a1.set_ylabel("IMD wind (kt)")
        a2.plot(tt, 100 * g.frac_lt220_200km, color="#eb6834"); a2.set_ylabel("% < 220 K (200 km)")
        a1.set_title(storm.title()); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(f"sat_timeseries_{storm}.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "--inspect":
        first = sorted(glob.glob(os.path.join(sys.argv[2], "**", "*.h5"), recursive=True))[0]
        inspect(first)
    else:
        main(sys.argv[1])
