# -*- coding: utf-8 -*-
"""
phone_spp.py - position from a smartphone GNSS log (Google GnssLogger) with gnss_spp.py
====================================================================================

Input : phone RINEX observation file (.26o) made by GnssLogger
        broadcast navigation files of the same day (GPS .26n.gz, GLONASS .26g.gz)
        GnssLogger text log (.txt) - only for the phone's own positions ("Fix")
Output: phone_output/ with a CSV file and two figures.

Two solutions are computed:  GPS only,  and  GPS + GLONASS.
They are compared with the phone's own position (Fix, provider "GPS").
Time: the phone Fix time is UTC (Unix milliseconds); the RINEX epochs are GPS time.
      GPS = UTC + 18 s. The RINEX epochs are at .413 s, the Fix at .000 s, so the
      SPP track is interpolated to the Fix times before comparing.
The phone's own position is NOT the truth; it is only a comparison.

Run in PyCharm (Run button). Needs gnss_spp.py in the same folder.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

import gnss_spp as spp

# ================================ SETTINGS ===================================
DATA = spp.HERE / "phone_data"
OBS_FILE = DATA / "gnss_log_2026_10_06_17_04_52.26o"
NAV_FILES = [DATA / "brdc2790.26n.gz", DATA / "brdc2790.26g.gz"]
LOG_FILE = DATA / "gnss_log_2026_10_06_17_04_51.txt"
OUT_DIR = spp.HERE / "phone_output"

spp.SIGMA_A = 3.0            # m  phone pseudoranges are noisier than a geodetic receiver
spp.SIGMA_B = 3.0            # m
spp.ELEV_MASK_DEG = 10.0
spp.CN0_MASK = 25.0          # dB-Hz, phone signals weaker than this are not used
# An epoch is "reliable" only if it can check itself: enough satellites, good geometry, good fit
MIN_SATS, MAX_HDOP, MAX_SIGMA0 = 6, 5.0, 3.0
SHOW_PLOTS = True
SAVE_FIGURES = True
# =============================================================================


def read_phone_fixes(path):
    """Phone positions from the GnssLogger 'Fix' lines -> DataFrame (time in GPS seconds)."""
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("Fix,"):
                p = line.strip().split(",")
                rows.append({"provider": p[1], "lat": float(p[2]), "lon": float(p[3]), "h": float(p[4]),
                             "accuracy_m": float(p[6]) if p[6] else np.nan, "unix_ms": int(p[8])})
    fx = pd.DataFrame(rows)
    fx["gps_time"] = fx["unix_ms"] / 1000.0 - spp.GPS_EPOCH_UNIX + spp.LEAP_SECONDS   # UTC -> GPS
    return fx


def geodetic_to_ecef(lat, lon, h):
    lat, lon = math.radians(lat), math.radians(lon)
    n = spp.WGS84_A / math.sqrt(1.0 - spp.WGS84_E2 * math.sin(lat) ** 2)
    return np.array([(n + h) * math.cos(lat) * math.cos(lon),
                     (n + h) * math.cos(lat) * math.sin(lon),
                     (n * (1.0 - spp.WGS84_E2) + h) * math.sin(lat)])


def run(epochs, nav, systems):
    """SPP for every epoch, using only the given systems."""
    rows, x_prev = [], None
    for t, meas in epochs:
        m = {s: v for s, v in meas.items() if s[0] in systems}
        sol = spp.solve_epoch(t, m, nav, x_prev)
        row = {"gps_time": t, "utc": spp.gps_to_utc_text(t), "tracked": len(m),
               "cn0_mean_tracked": float(np.nanmean([v[1] for v in m.values()])) if m else np.nan}
        if sol is not None:
            x_prev = sol["x"]
            n = Counter(s[0] for s in sol["sats"])
            row.update(X=sol["x"][0], Y=sol["x"][1], Z=sol["x"][2],
                       lat=math.degrees(sol["lat"]), lon=math.degrees(sol["lon"]), h=sol["h"],
                       n_sat=len(sol["sats"]), n_G=n.get("G", 0), n_R=n.get("R", 0),
                       rejected=" ".join(sol["rejected"]), hdop=sol["hdop"], sigma0=sol["sigma0"],
                       rms_residual_m=float(np.sqrt(np.mean(sol["v"] ** 2))))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["reliable"] = (df.get("n_sat", 0) >= MIN_SATS) & (df.get("hdop", np.inf) <= MAX_HDOP) \
        & (df.get("sigma0", np.inf).fillna(np.inf) <= MAX_SIGMA0)
    return df


def main():
    spp.SYSTEMS = "GR"
    epochs = spp.read_rinex_obs(OBS_FILE)
    nav = spp.Navigation(NAV_FILES)
    fixes = read_phone_fixes(LOG_FILE)
    print(f"Epochs: {len(epochs)}   ephemerides: {sum(len(v) for v in nav.eph.values())} "
          f"for {len(nav.eph)} satellites   phone fixes: {dict(Counter(fixes['provider']))}")

    sol = {"GPS": run(epochs, nav, "G"), "GPS+GLONASS": run(epochs, nav, "GR")}

    # local East/North/Up frame at the first phone GPS fix
    gfix = fixes[fixes["provider"] == "GPS"].reset_index(drop=True)
    x0 = geodetic_to_ecef(*gfix.loc[0, ["lat", "lon", "h"]])
    lat0, lon0, _ = spp.ecef_to_geodetic(x0)
    rot = spp.enu_rotation(lat0, lon0)
    for name, df in sol.items():
        ok = df["X"].notna()
        enu = (df.loc[ok, ["X", "Y", "Z"]].to_numpy(float) - x0) @ rot.T
        df.loc[ok, ["E", "N", "U"]] = enu
    fixes[["E", "N", "U"]] = np.array([rot @ (geodetic_to_ecef(a, b, c) - x0)
                                       for a, b, c in fixes[["lat", "lon", "h"]].to_numpy()])

    # compare with the phone GPS fix at the same moment (interpolate SPP to the fix time)
    summary = []
    gf = fixes[fixes["provider"] == "GPS"].copy()
    for name, df in sol.items():
        d = df[df["reliable"]]                            # only epochs that passed the quality check
        near = np.array([np.min(np.abs(d["gps_time"].to_numpy() - t)) <= 1.0 for t in gf["gps_time"]])
        e_i = np.interp(gf["gps_time"], d["gps_time"], d["E"], left=np.nan, right=np.nan)
        n_i = np.interp(gf["gps_time"], d["gps_time"], d["N"], left=np.nan, right=np.nan)
        diff = np.where(near, np.hypot(e_i - gf["E"], n_i - gf["N"]), np.nan)   # no bridging of gaps
        gf[f"diff_2D_{name}"] = diff
        summary.append({"solution": name, "epochs solved": int(df["X"].notna().sum()),
                        "epochs reliable": f"{len(d)} of {len(df)}",
                        "mean satellites": round(d["n_sat"].mean(), 1),
                        "median HDOP": round(d["hdop"].median(), 2),
                        "median sigma0": round(d["sigma0"].median(), 2),
                        "median RMS residual [m]": round(d["rms_residual_m"].median(), 2),
                        "epochs with rejections": int((d["rejected"] != "").sum()),
                        "median 2D diff to phone fix [m]": round(np.nanmedian(diff), 2),
                        "95% 2D diff to phone fix [m]": round(np.nanpercentile(diff, 95), 2)})
    summary = pd.DataFrame(summary)
    print("\n" + summary.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    both = sol["GPS"].merge(sol["GPS+GLONASS"], on=["gps_time", "utc"], suffixes=("_GPS", "_GPS+GLO"))
    both.to_csv(OUT_DIR / "phone_spp_results.csv", index=False, float_format="%.4f")
    gf.to_csv(OUT_DIR / "phone_fix_comparison.csv", index=False, float_format="%.4f")
    summary.to_csv(OUT_DIR / "phone_summary.csv", index=False)
    print(f"\nSaved in {OUT_DIR}")
    plots(sol, fixes, gf)


def plots(sol, fixes, gf):
    if not (SHOW_PLOTS or SAVE_FIGURES):
        return
    import matplotlib.pyplot as plt
    colors = {"GPS": "tab:orange", "GPS+GLONASS": "tab:blue"}

    fig, ax = plt.subplots(figsize=(8, 8))
    for prov, style in (("FLP", dict(color="0.6", lw=1.5)), ("GPS", dict(color="k", lw=1.5))):
        f = fixes[fixes["provider"] == prov]
        ax.plot(f["E"], f["N"], label=f"phone own position ({prov})", **style)
    for name, df in sol.items():
        good, bad = df[df["reliable"]], df[~df["reliable"] & df["E"].notna()]
        ax.plot(good["E"], good["N"], ".", ms=5, color=colors[name], label=f"SPP {name}: reliable")
        ax.plot(bad["E"], bad["N"], "x", ms=4, color=colors[name], alpha=0.5, label=f"SPP {name}: not reliable")
    first = sol["GPS+GLONASS"].dropna(subset=["E"]).iloc[0]
    ax.plot(first["E"], first["N"], "g^", ms=10, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Phone walk, 2 minutes: SPP positions and the phone's own positions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "phone_track.png", dpi=150)

    t0 = sol["GPS"]["gps_time"].iloc[0]
    fig2, ax2 = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for name, df in sol.items():
        ax2[0].plot(df["gps_time"] - t0, df["n_sat"], color=colors[name], label=f"used, {name}")
        ax2[2].semilogy(gf["gps_time"] - t0, gf[f"diff_2D_{name}"], ".-", ms=3, color=colors[name], label=name)
    df = sol["GPS+GLONASS"]
    ax2[0].plot(df["gps_time"] - t0, df["tracked"], color="0.6", label="tracked, GPS+GLONASS")
    ax2[1].plot(df["gps_time"] - t0, df["cn0_mean_tracked"], color="tab:green")
    weak = df["gps_time"][df["cn0_mean_tracked"] < 28] - t0
    for a in ax2:
        for t in weak:
            a.axvspan(t - 0.5, t + 0.5, color="tab:red", alpha=0.08, lw=0)
    ax2[0].set_ylabel("satellites")
    ax2[1].set_ylabel("mean C/N0 [dB-Hz]\n(red = weak signals)")
    ax2[2].set_ylabel("2D difference to\nphone GPS fix [m]")
    ax2[2].set_xlabel("seconds since first epoch")
    for a in ax2:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig2.suptitle("Phone GNSS: satellites, difference to the phone's own position, geometry")
    fig2.tight_layout()
    if SAVE_FIGURES:
        fig2.savefig(OUT_DIR / "phone_time_series.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
