"""Step 1: turn IMD best-track records into model rows.

One row = one storm at one time T. Inputs use only what is known at T.
Target = did the wind rise by 30 kt or more by T+24 h (rapid intensification, RI)?
"""
import numpy as np, pandas as pd

RI_KT = 30          # RI = +30 kt in 24 h
TOL_H = 3           # target / past values matched to real timestamps within +-3 h


def load(obs_json):
    o = pd.read_json(obs_json)
    o["t"] = pd.to_datetime(o["time_resolved"].fillna(o["time"]), utc=True)
    n_clash = int(o.duplicated(["storm_id", "t"]).sum())        # records sharing a storm and time
    o = o[~o.date_suspect & ~o.pos_suspect & o.wind.notna()]
    o = o.sort_values(["storm_id", "t"]).drop_duplicates(["storm_id", "t"], keep="first")
    o.attrs["n_clash"] = n_clash
    return o.reset_index(drop=True)


def value_at(g, hours, col="wind"):
    """For every row of one storm, the value of `col` at T+hours (nearest record within TOL_H)."""
    t = g.t.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    want = t + hours
    j = np.searchsorted(t, want)
    out = np.full(len(g), np.nan)
    for i, w in enumerate(want):
        best, bd = None, TOL_H + 1e-9
        for k in (j[i] - 1, j[i]):
            if 0 <= k < len(t) and abs(t[k] - w) <= bd and k != i:
                best, bd = k, abs(t[k] - w)
        if best is not None:
            out[i] = g[col].values[best]
    return out


def km_between(lat1, lon1, lat2, lon2):
    r = np.radians
    a = np.sin(r(lat2 - lat1) / 2) ** 2 + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_rows(o, storms_json):
    st = pd.read_json(storms_json)[["storm_id", "basin"]].rename(columns={"basin": "storm_basin"})
    rows = []
    for sid, g in o.groupby("storm_id", sort=False):
        g = g.copy()
        g["wind_24"] = value_at(g, 24)
        for h in (6, 12, 24):
            g[f"dV_past{h}"] = g.wind - value_at(g, -h)
        g["age_h"] = (g.t - g.t.iloc[0]).dt.total_seconds() / 3600
        rows.append(g)
    d = pd.concat(rows).merge(st, on="storm_id", how="left")
    d["dV24"] = d.wind_24 - d.wind
    d["RI"] = (d.dV24 >= RI_KT).astype(int)
    d["day_of_year"] = d.t.dt.dayofyear                       # season
    d["arabian_sea"] = (d.storm_basin == "ARB").astype(int)
    return d


# all known at time T: current strength, satellite CI number, recent growth, position, season, sea
FEATURES = ["wind", "pressure", "ci_no", "dV_past6", "dV_past12", "dV_past24",
            "lat", "lon", "day_of_year", "arabian_sea"]


def split(d):
    yr = d.year
    return {"train": d[yr <= 2012], "val": d[(yr >= 2013) & (yr <= 2018)], "test": d[(yr >= 2019) & (yr <= 2025)]}
