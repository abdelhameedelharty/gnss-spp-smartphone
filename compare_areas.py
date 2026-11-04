# -*- coding: utf-8 -*-
"""
compare_areas.py - compare the GNSS observations of two phone recordings made in two areas
=========================================================================================

For each area it computes, from the RINEX file (all systems) and the GnssLogger txt file:
  - how many satellites and signals are tracked (per system, L1 and L5)
  - signal strength C/N0 (distribution, and against elevation)
  - sky geometry: elevation / azimuth of every satellite (sky plot), DOP
  - continuity: how often a signal is lost and found again
  - code noise: pseudorange change minus Doppler prediction ("code-minus-Doppler")
  - positions (single point positioning, all systems) and their difference to the
    phone's own GPS position (the phone position is not the truth)
Time: RINEX = GPS time; phone Fix / Status = UTC (+18 s -> GPS); local = UTC + 8 h.

Output: Phone_Areas_Comparison_2026-10-06.xlsx and figures in compare_output/.
Run in PyCharm (Run button). Needs gnss_spp.py and phone_spp.py in the same folder.
"""
import math
import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

import gnss_spp as spp
import phone_spp as ph          # phone settings: sigma 3 m, C/N0 mask 25 dB-Hz

# ================================ SETTINGS ===================================
DATA = spp.HERE / "phone_data"
AREAS = {
    "A": dict(label="Area A (22:04)", obs=DATA / "gnss_log_2026_10_06_22_04_41.26o",
              txt=DATA / "gnss_log_2026_10_06_22_04_41.txt"),
    "B": dict(label="Area B (22:07)", obs=DATA / "gnss_log_2026_10_06_22_07_13.26o",
              txt=DATA / "gnss_log_2026_10_06_22_07_13.txt"),
}
NAV_FILES = [DATA / "BRDM00DLR_S_20262790000_01D_MN.rnx.gz"]   # all systems, whole day
SYSTEMS = "GRECJ"
OUT_DIR = spp.HERE / "compare_output"
OUT_XLSX = spp.HERE / "Phone_Areas_Comparison_2026-10-06.xlsx"
LOCAL_OFFSET_H = 8
SHOW_PLOTS = True
SAVE_FIGURES = True
# =============================================================================

SYS_NAME = {"G": "GPS", "R": "GLONASS", "E": "Galileo", "C": "BeiDou", "J": "QZSS"}
SYS_COLOR = {"G": "tab:blue", "R": "tab:red", "E": "tab:green", "C": "tab:orange", "J": "tab:purple"}
CONST = {1: "G", 3: "R", 4: "J", 5: "C", 6: "E"}                  # Android ConstellationType
FREQ = {"1C": 1575.42e6, "5Q": 1176.45e6, "2I": 1561.098e6, "1P": 1575.42e6, "5P": 1176.45e6}
BAND = {"1C": "L1", "2I": "L1", "1P": "L1", "5Q": "L5", "5P": "L5"}
AREA_COLOR = {"A": "tab:blue", "B": "tab:orange"}


# ------------------------------------------------------------------ reading
def read_rinex_all(path):
    """Every signal of every satellite: one row per (time, satellite, signal) with code C, Doppler D, C/N0 S."""
    types, fcn, cur, rows = {}, {}, None, []
    with spp._open(path) as f:
        for line in f:
            label = line[60:].strip()
            if label == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    cur = line[0]
                    types[cur] = []
                types[cur] += line[7:60].split()
            elif label == "GLONASS SLOT / FRQ #":
                for sat, k in re.findall(r"(R\s?\d+)\s+(-?\d+)", line[4:60]):
                    fcn[sat.replace(" ", "0")] = int(k)
            elif label == "END OF HEADER":
                break
        for line in f:
            if not line.startswith(">"):
                continue
            flag, nsat = int(line[31]), int(line[32:35])
            if flag > 1:
                for _ in range(nsat):
                    next(f)
                continue
            t = spp.gps_time(int(line[2:6]), int(line[7:9]), int(line[10:12]),
                             int(line[13:15]), int(line[16:18]), float(line[18:29]))
            for _ in range(nsat):
                rec = next(f)
                sat = rec[:3].replace(" ", "0")
                s = sat[0]
                if s not in types or s not in SYSTEMS:
                    continue
                vals = {ot: spp._obs_value(rec, i) for i, ot in enumerate(types[s])}
                for sig in sorted({ot[1:] for ot in types[s]}):
                    c = vals.get("C" + sig)
                    if c is None or c < 1.0e7:
                        continue
                    freq = spp.GLO_F1 + fcn.get(sat, 0) * spp.GLO_DF1 if s == "R" else FREQ.get(sig, np.nan)
                    rows.append((t, sat, s, sig, "L1" if s == "R" else BAND.get(sig, sig), c,
                                 vals.get("D" + sig), vals.get("S" + sig), freq))
    df = pd.DataFrame(rows, columns=["t", "sat", "sys", "sig", "band", "C", "D", "S", "freq"])
    return df, fcn


def read_status(path, fcn):
    """Phone 'Status' lines: which satellites the phone used in its own position (time UTC -> GPS)."""
    slot_of = {k: s for s, k in fcn.items()}
    rows, last_t = [], None
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("Status,"):
                continue
            p = line.strip().split(",")
            if p[1]:
                last_t = int(p[1]) / 1000.0 - spp.GPS_EPOCH_UNIX + spp.LEAP_SECONDS
            s = CONST.get(int(p[4])) if p[4] else None
            if last_t is None or s is None:
                continue
            svid = int(p[5])
            sat = slot_of.get(svid - 100, f"R{svid}") if (s == "R" and svid > 90) else f"{s}{svid:02d}"
            rows.append((last_t, sat, s, float(p[6]), float(p[7]), int(p[10])))
    return pd.DataFrame(rows, columns=["t", "sat", "sys", "freq", "cn0", "used"])


def geodetic_ecef(lat, lon, h):
    return ph.geodetic_to_ecef(lat, lon, h)


# ------------------------------------------------------------------ processing
def process(key, nav, origin):
    area = AREAS[key]
    sig, fcn = read_rinex_all(area["obs"])
    spp.SYSTEMS = SYSTEMS
    epochs = spp.read_rinex_obs(area["obs"])
    fixes = ph.read_phone_fixes(area["txt"])
    status = read_status(area["txt"], fcn)
    gf = fixes[fixes["provider"] == "GPS"].reset_index(drop=True)
    x0, rot = origin
    gf_enu = np.array([rot @ (geodetic_ecef(a, b, c) - x0) for a, b, c in gf[["lat", "lon", "h"]].to_numpy()])
    xref = geodetic_ecef(gf["lat"].median(), gf["lon"].median(), gf["h"].median())   # area centre for geometry
    lat_r, lon_r, _ = spp.ecef_to_geodetic(xref)
    rot_r = spp.enu_rotation(lat_r, lon_r)

    # satellite elevation / azimuth for each (time, satellite) from the broadcast ephemeris
    geo = {}
    for t, sat, p in sig[sig["band"] == "L1"].drop_duplicates(["t", "sat"])[["t", "sat", "C"]].itertuples(index=False):
        e = nav.get(sat, t)
        if e is None:
            continue
        t_tx = t - p / spp.C
        t_tx -= spp.clock_polynomial(e, t_tx)
        rs, _ = spp.satellite_position_clock(e, t_tx)
        u = rot_r @ ((rs - xref) / np.linalg.norm(rs - xref))
        geo[(t, sat)] = (math.degrees(math.asin(u[2])), math.degrees(math.atan2(u[0], u[1])) % 360)
    sig["elevation"] = [geo.get((t, s), (np.nan, np.nan))[0] for t, s in zip(sig["t"], sig["sat"])]
    sig["azimuth"] = [geo.get((t, s), (np.nan, np.nan))[1] for t, s in zip(sig["t"], sig["sat"])]

    # code-minus-Doppler (code noise and multipath change) and C/N0 change, per signal track
    sig = sig.sort_values(["sat", "sig", "t"]).reset_index(drop=True)
    sig["cmd"] = np.nan
    sig["dS"] = np.nan
    for _, idx in sig.groupby(["sat", "sig"]).groups.items():
        d = sig.loc[idx]
        dt = d["t"].diff()
        lam = spp.C / d["freq"]
        r = d["C"].diff() + lam * (d["D"] + d["D"].shift()) / 2.0 * dt
        ok = dt.between(0.5, 1.5) & d["D"].notna() & d["D"].shift().notna()
        r = r.where(ok)
        sig.loc[idx, "cmd"] = r - r.median()
        sig.loc[idx, "dS"] = d["S"].diff().where(ok)

    # single point positioning, all systems
    ep_rows, x_prev, used_status = [], None, {}
    status_by_t = status.groupby("t")
    st_times = np.array(sorted(status["t"].unique())) if len(status) else np.array([])
    for t, meas in epochs:
        sol = spp.solve_epoch(t, meas, nav, x_prev)
        n_sys = Counter(s[0] for s in meas)
        row = {"t": t, "satellites tracked": len(meas)}
        for s in SYSTEMS:
            row[f"tracked {SYS_NAME[s]}"] = n_sys.get(s, 0)
        es = sig[(sig["t"] == t)]
        row["signals tracked"] = len(es)
        row["dual-frequency satellites"] = int((es.groupby("sat")["band"].nunique() > 1).sum())
        row["mean C/N0 L1 [dB-Hz]"] = es.loc[es["band"] == "L1", "S"].mean()
        row["mean C/N0 L5 [dB-Hz]"] = es.loc[es["band"] == "L5", "S"].mean()
        if len(st_times):
            k = int(np.argmin(np.abs(st_times - t)))
            if abs(st_times[k] - t) <= 1.5:
                row["satellites used by the phone"] = int(status_by_t.get_group(st_times[k])
                                                          .drop_duplicates("sat")["used"].sum())
        if sol is not None:
            x_prev = sol["x"]
            e, n, u = rot @ (sol["x"] - x0)
            redundancy = len(sol["sats"]) - 3 - len(sol["clock_m"])
            row.update({"SPP latitude": math.degrees(sol["lat"]), "SPP longitude": math.degrees(sol["lon"]),
                        "SPP height [m]": sol["h"], "SPP East [m]": e, "SPP North [m]": n,
                        "SPP satellites used": len(sol["sats"]), "SPP systems used": len(sol["clock_m"]),
                        "SPP rejected": " ".join(sol["rejected"]), "SPP HDOP": sol["hdop"], "SPP PDOP": sol["pdop"],
                        "SPP sigma0": sol["sigma0"], "SPP RMS residual [m]": float(np.sqrt(np.mean(sol["v"] ** 2))),
                        "SPP reliable": bool(redundancy >= 2 and sol["hdop"] <= 5 and
                                             (sol["sigma0"] if sol["sigma0"] == sol["sigma0"] else 99) <= 3)})
            for s_, v_ in zip(sol["sats"], sol["v"]):
                used_status[(t, s_)] = ("used", v_)
            for s_ in sol["rejected"]:
                used_status[(t, s_)] = ("rejected (w-test)", np.nan)
        else:
            row["SPP reliable"] = False
        row["phone GPS fix East [m]"] = np.interp(t, gf["gps_time"], gf_enu[:, 0], left=np.nan, right=np.nan)
        row["phone GPS fix North [m]"] = np.interp(t, gf["gps_time"], gf_enu[:, 1], left=np.nan, right=np.nan)
        ep_rows.append(row)
    ep = pd.DataFrame(ep_rows)
    ep["SPP - phone fix [m]"] = np.hypot(ep.get("SPP East [m]") - ep["phone GPS fix East [m]"],
                                         ep.get("SPP North [m]") - ep["phone GPS fix North [m]"])

    main_sig = {s: spp.SIGNALS[s] for s in SYSTEMS}
    status_col, resid_col = [], []
    for t, sat, sg, cn0, el in sig[["t", "sat", "sig", "S", "elevation"]].itertuples(index=False):
        if sg != main_sig[sat[0]]:
            status_col.append("second signal (not used in SPP)")
            resid_col.append(np.nan)
            continue
        st, v = used_status.get((t, sat), (None, np.nan))
        if st is None:
            if nav.get(sat, t) is None:
                st = "no ephemeris"
            elif not cn0 >= spp.CN0_MASK:
                st = f"weak signal (< {spp.CN0_MASK:.0f} dB-Hz)"
            elif el == el and el < spp.ELEV_MASK_DEG:
                st = f"below {spp.ELEV_MASK_DEG:.0f} deg"
            else:
                st = "not used (too few satellites / only one of its system)"
        status_col.append(st)
        resid_col.append(v)
    sig["SPP status"] = status_col
    sig["SPP residual [m]"] = resid_col
    return dict(sig=sig, ep=ep, fixes=fixes, gf=gf, gf_enu=gf_enu, status=status, epochs=len(epochs))


def summarise(res):
    """Side-by-side table of observation properties. 'better' = which area is better for that property."""
    def m(key, fn):
        return {k: fn(r) for k, r in res.items()}

    def track_stats(sig):
        out = []
        for (sat, sg), d in sig.groupby(["sat", "sig"]):
            t = np.sort(d["t"].to_numpy())
            runs = 1 + int((np.diff(t) > 1.5).sum())
            out.append((sat, sg, len(t), runs - 1, (t[-1] - t[0]) / 60.0 + 1 / 60.0))
        return pd.DataFrame(out, columns=["sat", "sig", "n", "losses", "minutes"])

    rows = []

    def add(name, values, higher_is_better=None, unit=""):
        a, b = values["A"], values["B"]
        better = ""
        if higher_is_better is not None and a == a and b == b and a != b:
            better = "A" if (a > b) == higher_is_better else "B"
        rows.append({"property": name + (f" [{unit}]" if unit else ""), AREAS["A"]["label"]: a,
                     AREAS["B"]["label"]: b, "better": better})

    add("recording length", m("", lambda r: round(r["ep"]["t"].iloc[-1] - r["ep"]["t"].iloc[0] + 1)), None, "s")
    add("walking distance (phone GPS track)",
        m("", lambda r: round(float(np.hypot(*np.diff(r["gf_enu"][:, :2], axis=0).T).sum()), 1)), None, "m")
    add("satellites tracked per epoch (mean)", m("", lambda r: round(r["ep"]["satellites tracked"].mean(), 1)), True)
    for s in SYSTEMS:
        add(f"   {SYS_NAME[s]}", m("", lambda r, s=s: round(r["ep"][f"tracked {SYS_NAME[s]}"].mean(), 1)), True)
    add("satellites per epoch, always-on systems GPS+Galileo+QZSS",
        m("", lambda r: round((r["ep"]["tracked GPS"] + r["ep"]["tracked Galileo"] + r["ep"]["tracked QZSS"]).mean(), 1)),
        True)
    add("chip switched GLONASS on/off (times, chip behaviour)",
        m("", lambda r: int(((r["ep"]["tracked GLONASS"] > 0).astype(int).diff().abs() > 0).sum())), False)
    add("signals tracked per epoch (mean)", m("", lambda r: round(r["ep"]["signals tracked"].mean(), 1)), True)
    add("dual-frequency satellites per epoch (mean)",
        m("", lambda r: round(r["ep"]["dual-frequency satellites"].mean(), 1)), True)
    add("C/N0 L1 median", m("", lambda r: round(r["sig"].loc[r["sig"]["band"] == "L1", "S"].median(), 1)), True, "dB-Hz")
    add("C/N0 L1 10th percentile (weak end)",
        m("", lambda r: round(r["sig"].loc[r["sig"]["band"] == "L1", "S"].quantile(0.1), 1)), True, "dB-Hz")
    add("C/N0 L5 median", m("", lambda r: round(r["sig"].loc[r["sig"]["band"] == "L5", "S"].median(), 1)), True, "dB-Hz")
    add("share of L1 signals weaker than 30 dB-Hz",
        m("", lambda r: round(100 * (r["sig"].loc[r["sig"]["band"] == "L1", "S"] < 30).mean(), 1)), False, "%")
    add("C/N0 L1, satellites above 45 deg", m("", lambda r: round(r["sig"].loc[(r["sig"]["band"] == "L1") &
                                                                               (r["sig"]["elevation"] > 45), "S"].median(), 1)), True, "dB-Hz")
    add("C/N0 L1, satellites below 30 deg", m("", lambda r: round(r["sig"].loc[(r["sig"]["band"] == "L1") &
                                                                               (r["sig"]["elevation"] < 30), "S"].median(), 1)), True, "dB-Hz")
    add("C/N0 change epoch-to-epoch (fading, median |dS|)",
        m("", lambda r: round(r["sig"]["dS"].abs().median(), 2)), False, "dB-Hz")
    add("mean elevation of tracked satellites",
        m("", lambda r: round(r["sig"].drop_duplicates(["t", "sat"])["elevation"].mean(), 1)), None, "deg")

    def az_gap(r):
        d = r["sig"][(r["sig"]["elevation"] > 10)].groupby("sat")["azimuth"].median().sort_values().to_numpy()
        if len(d) < 2:
            return np.nan
        return round(float(np.max(np.diff(np.r_[d, d[0] + 360]))), 1)
    add("largest empty azimuth sector (sky opening)", m("", az_gap), False, "deg")
    ts = {k: track_stats(r["sig"]) for k, r in res.items()}
    add("signal losses per signal per minute (all systems)",
        {k: round(t["losses"].sum() / t["minutes"].sum(), 3) for k, t in ts.items()}, False)
    always = {k: t[t["sat"].str[0].isin(list("GEJ")) & t["sig"].isin(["1C"])] for k, t in ts.items()}
    add("signal losses per signal per minute (GPS+Galileo+QZSS L1)",
        {k: round(t["losses"].sum() / t["minutes"].sum(), 3) for k, t in always.items()}, False)
    add("code noise L1 (code-minus-Doppler / sqrt 2, robust)",
        m("", lambda r: round(1.4826 * r["sig"].loc[r["sig"]["band"] == "L1", "cmd"].abs().median() / math.sqrt(2), 2)),
        False, "m")
    add("code noise L5 (code-minus-Doppler / sqrt 2, robust)",
        m("", lambda r: round(1.4826 * r["sig"].loc[r["sig"]["band"] == "L5", "cmd"].abs().median() / math.sqrt(2), 2)),
        False, "m")
    add("satellites used by the phone's own fix (mean)",
        m("", lambda r: round(r["ep"]["satellites used by the phone"].mean(), 1)
          if "satellites used by the phone" in r["ep"] else np.nan), True)
    add("SPP epochs with a position", m("", lambda r: round(100 * r["ep"]["SPP East [m]"].notna().mean(), 1)), True, "%")
    add("SPP reliable epochs", m("", lambda r: round(100 * r["ep"]["SPP reliable"].mean(), 1)), True, "%")
    add("SPP satellites used (mean)", m("", lambda r: round(r["ep"]["SPP satellites used"].mean(), 1)), True)
    add("SPP HDOP (median)", m("", lambda r: round(r["ep"]["SPP HDOP"].median(), 2)), False)
    add("SPP rejected satellites per epoch (mean)",
        m("", lambda r: round(r["ep"]["SPP rejected"].fillna("").str.split().str.len().mean(), 2)), False)
    add("SPP RMS residual (median)", m("", lambda r: round(r["ep"]["SPP RMS residual [m]"].median(), 2)), False, "m")
    add("SPP - phone fix, median (reliable epochs)",
        m("", lambda r: round(r["ep"].loc[r["ep"]["SPP reliable"], "SPP - phone fix [m]"].median(), 2)), False, "m")
    add("SPP - phone fix, 95% (reliable epochs)",
        m("", lambda r: round(r["ep"].loc[r["ep"]["SPP reliable"], "SPP - phone fix [m]"].quantile(0.95), 2)), False, "m")

    per_sys = []
    for k, r in res.items():
        s_ = r["sig"]
        for s in SYSTEMS:
            for band in ("L1", "L5"):
                d = s_[(s_["sys"] == s) & (s_["band"] == band)]
                if d.empty:
                    continue
                t = ts[k][ts[k]["sat"].str[0] == s]
                t = t[t["sig"].isin(d["sig"].unique())]
                main = d[d["sig"] == spp.SIGNALS[s]]                  # only the signal used in SPP
                used = main["SPP status"].eq("used").sum()
                per_sys.append({"area": AREAS[k]["label"], "system": SYS_NAME[s], "band": band,
                                "satellites": d["sat"].nunique(), "observations": len(d),
                                "C/N0 median [dB-Hz]": round(d["S"].median(), 1),
                                "elevation mean [deg]": round(d["elevation"].mean(), 1),
                                "code noise [m]": round(1.4826 * d["cmd"].abs().median() / math.sqrt(2), 2),
                                "losses per signal-minute": round(t["losses"].sum() / max(t["minutes"].sum(), 1e-9), 3),
                                "used in SPP [%]": round(100 * used / len(main), 1) if len(main) else np.nan,
                                "SPP residual RMS [m]": round(float(np.sqrt(np.nanmean(d["SPP residual [m]"] ** 2))), 2)
                                if d["SPP residual [m]"].notna().any() else np.nan})
    per_sat = []
    for k, r in res.items():
        for (sat, sg), d in r["sig"].groupby(["sat", "sig"]):
            t = ts[k][(ts[k]["sat"] == sat) & (ts[k]["sig"] == sg)]
            per_sat.append({"area": AREAS[k]["label"], "satellite": sat, "system": SYS_NAME[sat[0]], "signal": "C" + sg,
                            "band": d["band"].iloc[0], "epochs tracked": len(d),
                            "tracked [% of epochs]": round(100 * len(d) / r["epochs"], 1),
                            "losses of lock": int(t["losses"].iloc[0]) if len(t) else 0,
                            "C/N0 median [dB-Hz]": round(d["S"].median(), 1),
                            "elevation [deg]": round(d["elevation"].median(), 1),
                            "azimuth [deg]": round(d["azimuth"].median(), 1),
                            "code noise [m]": round(1.4826 * d["cmd"].abs().median() / math.sqrt(2), 2),
                            "used in SPP [% of its epochs]": round(100 * d["SPP status"].eq("used").mean(), 1)})
    return pd.DataFrame(rows), pd.DataFrame(per_sys), pd.DataFrame(per_sat)


# ------------------------------------------------------------------ output
def time_columns(gps_seconds, t0):
    t = pd.Series(np.asarray(gps_seconds, float))
    gps = pd.Timestamp(spp.GPS_EPOCH) + pd.to_timedelta(t, unit="s")
    utc = gps - pd.Timedelta(seconds=spp.LEAP_SECONDS)
    loc = utc + pd.Timedelta(hours=LOCAL_OFFSET_H)
    f = lambda s: s.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    return pd.DataFrame({"GPS time": f(gps), "UTC": f(utc), f"Local time (UTC+{LOCAL_OFFSET_H})": f(loc),
                         "seconds since start": (t - t0).round(3)})


README = [
    "COMPARISON OF TWO AREAS - PHONE GNSS OBSERVATIONS (6 October 2026, night, about 22:05 local time)",
    "",
    "DATA",
    "Area A: gnss_log_2026_10_06_22_04_41 (.26o and .txt). Area B: gnss_log_2026_10_06_22_07_13 (.26o and .txt).",
    "Samsung SM-S928B (Qualcomm GNSS GEN9), GnssLogger v3.1.1.3. In both areas the phone was carried while walking",
    "(about 140 m in 2 minutes, about 1.1 m/s). Broadcast orbits of all systems: BRDM00DLR_S_20262790000_01D_MN.rnx.gz (DLR).",
    "Note: the txt file of area A has no 'Raw' lines; everything here uses the RINEX files, which both areas have.",
    "Settings: area A = developer option 'Force full GNSS measurements' ON, GnssLogger raw measurements OFF;",
    "          area B = 'Force full GNSS measurements' OFF, raw measurements ON. In B the chip still reported",
    "          IsFullTracking = 1 for every measurement, so both areas were most likely recorded with full tracking.",
    "Note: the chip stopped reporting GLONASS (and BeiDou B1I; B1C/B2a continued) in blocks, in both areas and with full",
    "      tracking on. This is the chip's own behaviour, not the area. Counts and continuity of all systems therefore",
    "      also show the phone; the rows 'always-on systems GPS+Galileo+QZSS' are the fair comparison of the areas.",
    "",
    "TIME",
    "GPS time = time of the RINEX file. UTC = GPS time - 18 s. Local time = UTC + 8 h.",
    "",
    "SHEETS",
    "Summary          - the properties side by side; column 'better' shows which area is better for that property.",
    "Per_system       - the same properties for each system (GPS, GLONASS, Galileo, BeiDou, QZSS) and band (L1, L5).",
    "Per_satellite    - one row per satellite and signal: tracking, losses of lock, C/N0, elevation, azimuth, code noise.",
    "Epochs_A/B       - one row per second: satellites, signals, C/N0, SPP position and quality, phone position.",
    "Signals_A/B      - one row per signal and second: code, Doppler, C/N0, elevation, azimuth, code-minus-Doppler, SPP use.",
    "",
    "HOW THE PROPERTIES ARE COMPUTED",
    "C/N0: signal-to-noise density from the RINEX file (S). L1 = GPS/Galileo/QZSS L1, BeiDou B1I/B1C, GLONASS G1. L5 = L5/E5a/B2a.",
    "Elevation and azimuth: from the broadcast orbits and the centre of each walk.",
    "Largest empty azimuth sector: the widest direction (in degrees) with no satellite above 10 deg - large = blocked sky.",
    "Losses of lock: a signal disappears for more than 1.5 s and comes back.",
    "Code noise: change of the pseudorange from one second to the next, minus the change predicted by the Doppler;",
    "            robust spread divided by sqrt(2). It shows code noise and fast multipath (no ephemeris needed).",
    "Fading: median change of C/N0 from one second to the next.",
    "SPP: single point positioning with code, all systems, Klobuchar, Saastamoinen, mask 10 deg, C/N0 >= 25 dB-Hz,",
    "     sigma 3 m, one receiver clock per system, Baarda w-test. Reliable = redundancy >= 2, HDOP <= 5, sigma0 <= 3.",
    "The phone's own position (Fix) is NOT the truth; 'SPP - phone fix' is only a comparison.",
    "",
    "Made with compare_areas.py (folder GNSS_SPP). AI-assisted processing; check before publishing.",
]


def write_excel(summary, per_sys, per_sat, res):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    t0 = {k: r["ep"]["t"].iloc[0] for k, r in res.items()}
    sheets = [("Summary", summary), ("Per_system", per_sys), ("Per_satellite", per_sat)]
    for k, r in res.items():
        ep = r["ep"].copy()
        ep["SPP reliable"] = ep["SPP reliable"].map({True: "yes", False: "no"})
        sheets.append((f"Epochs_{k}", pd.concat([time_columns(ep["t"], t0[k]), ep.drop(columns=["t"])], axis=1)))
    for k, r in res.items():
        s = r["sig"].sort_values(["t", "sat", "sig"]).reset_index(drop=True)
        s = s.rename(columns={"sat": "satellite", "sys": "system", "sig": "signal", "C": "pseudorange [m]",
                              "D": "Doppler [Hz]", "S": "C/N0 [dB-Hz]", "freq": "frequency [Hz]",
                              "elevation": "elevation [deg]", "azimuth": "azimuth [deg]",
                              "cmd": "code-minus-Doppler [m]", "dS": "C/N0 change [dB-Hz]"})
        s["system"] = s["system"].map(SYS_NAME)
        s["signal"] = "C" + s["signal"]
        sheets.append((f"Signals_{k}", pd.concat([time_columns(s["t"], t0[k]), s.drop(columns=["t"])], axis=1)))
    target = OUT_XLSX
    try:
        writer = pd.ExcelWriter(target, engine="openpyxl")
    except PermissionError:
        target = OUT_XLSX.with_name(OUT_XLSX.stem + datetime.now().strftime("_%H%M%S") + ".xlsx")
        writer = pd.ExcelWriter(target, engine="openpyxl")
    with writer as xw:
        pd.DataFrame({"README": README}).to_excel(xw, sheet_name="README", index=False, header=False)
        for name, df in sheets:
            df = df.copy()
            for c in df.columns:
                if df[c].dtype.kind == "f":
                    df[c] = df[c].round(8 if ("latitude" in c or "longitude" in c) else 4)
            df.to_excel(xw, sheet_name=name, index=False)
        ws = xw.book["README"]
        ws.column_dimensions["A"].width = 140
        for cell in ws["A"]:
            if cell.value and str(cell.value).split(" -")[0].isupper():
                cell.font = Font(bold=True, size=14 if cell.row == 1 else 11)
        for name, df in sheets:
            ws = xw.book[name]
            ws.freeze_panes = "B2" if name in ("Summary", "Per_system", "Per_satellite") else "D2"
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col)
                text = str(cell.value)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                time_col = text in ("GPS time", "UTC", "seconds since start") or text.startswith("Local time")
                cell.fill = PatternFill("solid", fgColor="FFE699" if time_col else "BDD7EE")
                width = 46 if (name == "Summary" and col == 1) else max(10, min(24, len(text) * 0.55 + 4))
                ws.column_dimensions[get_column_letter(col)].width = width
            ws.row_dimensions[1].height = 45
            if name == "Summary":
                for row in range(2, ws.max_row + 1):
                    b = ws.cell(row=row, column=4).value
                    if b in ("A", "B"):
                        ws.cell(row=row, column=2 if b == "A" else 3).fill = PatternFill("solid", fgColor="C6E0B4")
    print(f"Saved: {target}")


def plots(res):
    if not (SHOW_PLOTS or SAVE_FIGURES):
        return
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(res)

    fig, ax = plt.subplots(3, 2, sharex="col", figsize=(14, 10))
    for j, k in enumerate(keys):
        ep = res[k]["ep"]
        t = ep["t"] - ep["t"].iloc[0]
        ys = [ep[f"tracked {SYS_NAME[s]}"] for s in SYSTEMS]
        ax[0, j].stackplot(t, ys, colors=[SYS_COLOR[s] for s in SYSTEMS], labels=[SYS_NAME[s] for s in SYSTEMS], alpha=0.8)
        ax[0, j].plot(t, ep["SPP satellites used"], "k-", lw=1.2, label="used in SPP")
        ax[0, j].set_title(AREAS[k]["label"])
        ax[0, j].set_ylabel("satellites")
        ax[1, j].plot(t, ep["mean C/N0 L1 [dB-Hz]"], color="tab:blue", label="mean C/N0 L1")
        ax[1, j].plot(t, ep["mean C/N0 L5 [dB-Hz]"], color="tab:red", label="mean C/N0 L5")
        ax[1, j].set_ylabel("C/N0 [dB-Hz]")
        ax[1, j].set_ylim(15, 50)
        ok = ep["SPP reliable"]
        ax[2, j].semilogy(t[ok], ep.loc[ok, "SPP - phone fix [m]"], ".", color="tab:green", ms=4, label="reliable")
        ax[2, j].semilogy(t[~ok], ep.loc[~ok, "SPP - phone fix [m]"], "x", color="0.5", ms=4, label="not reliable")
        ax[2, j].set_ylim(0.5, 300)
        ax[2, j].set_ylabel("SPP - phone fix [m]")
        ax[2, j].set_xlabel("seconds since start")
        for a in ax[:, j]:
            a.grid(alpha=0.3)
            a.legend(fontsize=7, loc="lower left", ncol=3)
    fig.suptitle("Two areas: satellites, signal strength and position difference")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "compare_overview.png", dpi=140)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    bins = np.arange(10, 52, 2)
    for k in keys:
        s = res[k]["sig"]
        ax[0].hist(s.loc[s["band"] == "L1", "S"], bins=bins, histtype="step", lw=2, density=True,
                   color=AREA_COLOR[k], label=AREAS[k]["label"])
        ax[1].hist(s.loc[s["band"] == "L5", "S"], bins=bins, histtype="step", lw=2, density=True,
                   color=AREA_COLOR[k], label=AREAS[k]["label"])
        d = s[s["band"] == "L1"]
        ax[2].plot(d["elevation"], d["S"], ".", ms=1.5, alpha=0.25, color=AREA_COLOR[k])
        e_bins = np.arange(0, 91, 10)
        med = d.groupby(pd.cut(d["elevation"], e_bins), observed=False)["S"].median()
        ax[2].plot(e_bins[:-1] + 5, med.to_numpy(), "o-", lw=2, color=AREA_COLOR[k], label=f"{AREAS[k]['label']} median")
    ax[0].set_title("C/N0 on L1")
    ax[1].set_title("C/N0 on L5")
    ax[2].set_title("C/N0 (L1) against elevation")
    for a in ax[:2]:
        a.set_xlabel("C/N0 [dB-Hz]")
        a.set_ylabel("share")
    ax[2].set_xlabel("elevation [deg]")
    ax[2].set_ylabel("C/N0 [dB-Hz]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "compare_cn0.png", dpi=140)

    fig = plt.figure(figsize=(13, 6.5))
    for j, k in enumerate(keys):
        a = fig.add_subplot(1, 2, j + 1, projection="polar")
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)
        s = res[k]["sig"]
        d = s[(s["band"] == "L1") & s["elevation"].notna()].drop_duplicates(["t", "sat"])
        sc = a.scatter(np.radians(d["azimuth"]), 90 - d["elevation"], c=d["S"], cmap="viridis", vmin=15, vmax=48, s=6)
        last = d.sort_values("t").groupby("sat").tail(1)
        for sat, az, el in last[["sat", "azimuth", "elevation"]].itertuples(index=False):
            a.text(np.radians(az), 90 - el, sat, fontsize=6)
        a.set_ylim(0, 90)
        a.set_yticks([0, 30, 60, 90])
        a.set_yticklabels(["90", "60", "30", "0"], fontsize=7)
        a.set_title(f"{AREAS[k]['label']}: sky plot (colour = C/N0 L1)", pad=18)
    fig.colorbar(sc, ax=fig.axes, shrink=0.7, label="C/N0 [dB-Hz]")
    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "compare_skyplots.png", dpi=140)

    fig, ax = plt.subplots(figsize=(14, 4.2))
    for k in keys:
        gf = res[k]["gf_enu"]
        ax.plot(gf[:, 0], gf[:, 1], "-", color=AREA_COLOR[k], lw=2, label=f"{AREAS[k]['label']}: phone GPS track")
        ep = res[k]["ep"]
        ok = ep["SPP reliable"]
        ax.plot(ep.loc[ok, "SPP East [m]"], ep.loc[ok, "SPP North [m]"], ".", color=AREA_COLOR[k], ms=5, alpha=0.6,
                label=f"{AREAS[k]['label']}: SPP reliable")
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set_title("Walks in the two areas: phone GPS track and SPP (all systems)")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "compare_tracks.png", dpi=140)
    if SHOW_PLOTS:
        plt.show()


def main():
    spp.SYSTEMS = SYSTEMS
    nav = spp.Navigation(NAV_FILES)
    fx_a = ph.read_phone_fixes(AREAS["A"]["txt"])
    g0 = fx_a[fx_a["provider"] == "GPS"].iloc[0]
    x0 = geodetic_ecef(g0["lat"], g0["lon"], g0["h"])
    lat0, lon0, _ = spp.ecef_to_geodetic(x0)
    origin = (x0, spp.enu_rotation(lat0, lon0))
    res = {k: process(k, nav, origin) for k in AREAS}
    summary, per_sys, per_sat = summarise(res)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print()
    print(per_sys.to_string(index=False))
    write_excel(summary, per_sys, per_sat, res)
    plots(res)


if __name__ == "__main__":
    main()
