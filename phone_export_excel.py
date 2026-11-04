# -*- coding: utf-8 -*-
"""
phone_export_excel.py - phone GNSS + IMU processed data -> one Excel file, all on one time axis
=============================================================================================

Every sheet has the same time columns:
  GPS time  (time of the satellites and of the RINEX file)
  UTC       = GPS time - 18 s
  Local     = UTC + LOCAL_OFFSET_H  (from the file name: 17_04_51 local = 09:04:51 UTC)
  seconds since start  (0 = first GNSS epoch)

Run phone_spp.py once before (it makes phone_output/phone_spp_results.csv). Needs openpyxl.
"""
import math
from datetime import datetime

import numpy as np
import pandas as pd

import gnss_spp as spp
import phone_spp as ph
import phone_imu as im

LOCAL_OFFSET_H = 8
OUT_FILE = spp.HERE / "Phone_GNSS_IMU_2026-10-06.xlsx"
SYSTEM_NAME = {"G": "GPS", "R": "GLONASS", "E": "Galileo", "C": "BeiDou", "J": "QZSS"}
SEGMENTS = [(-1, 28, "outside, between tall buildings"), (28, 36, "entering (door 1) - estimated"),
            (36, 62, "inside the 18-floor building"), (62, 67, "leaving (door 2) - estimated"),
            (67, 121, "outside again")]


def time_columns(gps_seconds, t0):
    """GPS / UTC / local time text and seconds since start for an array of GPS seconds."""
    t = pd.Series(np.asarray(gps_seconds, float))
    gps = pd.Timestamp(spp.GPS_EPOCH) + pd.to_timedelta(t, unit="s")
    utc = gps - pd.Timedelta(seconds=spp.LEAP_SECONDS)
    loc = utc + pd.Timedelta(hours=LOCAL_OFFSET_H)
    f = lambda s: s.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    return pd.DataFrame({"GPS time": f(gps), "UTC": f(utc), f"Local time (UTC+{LOCAL_OFFSET_H})": f(loc),
                         "GPS week": (t // 604800).astype(int), "GPS seconds of week": (t % 604800).round(3),
                         "seconds since start": (t - t0).round(3)})


def segment_of(sec):
    for a, b, name in SEGMENTS:
        if a <= sec < b:
            return name
    return ""


def enu_to_geodetic(e, n, u, x0, rot):
    lat, lon, h = spp.ecef_to_geodetic(x0 + rot.T @ np.array([e, n, u]))
    return math.degrees(lat), math.degrees(lon), h


def main():
    # ------------------------------------------------------------ GNSS
    spp.SYSTEMS = "GRECJ"                               # read every system (only G and R have ephemerides)
    epochs = spp.read_rinex_obs(ph.OBS_FILE)
    nav = spp.Navigation(ph.NAV_FILES)
    fixes = ph.read_phone_fixes(ph.LOG_FILE)
    t0 = epochs[0][0]
    gfix = fixes[fixes["provider"] == "GPS"].reset_index(drop=True)
    x0 = ph.geodetic_to_ecef(*gfix.loc[0, ["lat", "lon", "h"]])
    lat0, lon0, _ = spp.ecef_to_geodetic(x0)
    rot = spp.enu_rotation(lat0, lon0)
    enu = lambda xyz: rot @ (np.asarray(xyz, float) - x0)

    sol = {"GPS+GLO": ph.run(epochs, nav, "GR"), "GPS": ph.run(epochs, nav, "G")}
    for df in sol.values():
        ok = df["X"].notna()
        df.loc[ok, ["E", "N", "U"]] = np.array([enu(v) for v in df.loc[ok, ["X", "Y", "Z"]].to_numpy()])

    # per-satellite table (main solution GPS+GLONASS)
    sat_rows, x_prev = [], None
    mask = math.radians(spp.ELEV_MASK_DEG)
    for t, meas in epochs:
        m = {s: v for s, v in meas.items() if s[0] in "GR"}
        s_ep = spp.solve_epoch(t, m, nav, x_prev)
        if s_ep is not None:
            x_prev = s_ep["x"]
        sats = {d["sat"]: d for d in spp.satellites_for_epoch(t, m, nav)}
        resid = dict(zip(s_ep["sats"], zip(s_ep["v"], s_ep["w"]))) if s_ep else {}
        for sat in sorted(meas):
            p, cn0 = meas[sat]
            r = {"gps_time": t, "satellite": sat, "system": SYSTEM_NAME.get(sat[0], sat[0]),
                 "signal": "C" + spp.SIGNALS[sat[0]], "pseudorange [m]": p, "C/N0 [dB-Hz]": cn0}
            d = sats.get(sat)
            if sat[0] not in "GR":
                r["status"] = "not used: no ephemeris (nav files have only GPS + GLONASS)"
            elif not cn0 >= spp.CN0_MASK:
                r["status"] = f"not used: weak signal (< {spp.CN0_MASK:.0f} dB-Hz)"
            elif d is None:
                r["status"] = "not used: no ephemeris for this satellite"
            else:
                r.update({"transmit time (GPS seconds of week)": d["t_tx"] % 604800,
                          "sat X [m]": d["rs"][0], "sat Y [m]": d["rs"][1], "sat Z [m]": d["rs"][2],
                          "sat clock [m]": spp.C * d["dts"]})
                if s_ep is None:
                    r["status"] = "no position this epoch (too few satellites)"
                else:
                    v = d["rs"] - s_ep["x"]
                    e_, n_, u_ = spp.enu_rotation(s_ep["lat"], s_ep["lon"]) @ (v / np.linalg.norm(v))
                    el, az = math.asin(u_), math.atan2(e_, n_)
                    r.update({"elevation [deg]": math.degrees(el), "azimuth [deg]": math.degrees(az) % 360,
                              "iono delay [m]": spp.klobuchar(t % 604800, s_ep["lat"], s_ep["lon"], az, el,
                                                              nav.alpha, nav.beta) * (spp.F_L1 / d["freq"]) ** 2,
                              "tropo delay [m]": spp.saastamoinen(s_ep["lat"], s_ep["h"], el)})
                    if sat in resid:
                        r.update({"status": "used", "residual [m]": resid[sat][0], "w-test": resid[sat][1]})
                    elif sat in s_ep["rejected"]:
                        r["status"] = "rejected by w-test (outlier)"
                    elif el < mask:
                        r["status"] = f"not used: below {spp.ELEV_MASK_DEG:.0f} deg elevation"
                    else:
                        r["status"] = "not used: only satellite of its system"
            sat_rows.append(r)
    sats_df = pd.DataFrame(sat_rows)

    # ------------------------------------------------------------ IMU
    r = im.analyse()
    grid, t_imu0 = r["grid"], r["t0"]
    s = r["s"]
    mag_cal = r["mcal"]
    m = r["m"]
    p = r["p"]
    o = s["OrientationDeg"]
    step_flag = np.zeros(len(grid), dtype=int)
    step_flag[r["peaks"]] = 1
    imu = pd.DataFrame({
        "acc X [m/s2]": r["a"][:, 0], "acc Y [m/s2]": r["a"][:, 1], "acc Z [m/s2]": r["a"][:, 2],
        "|a| [m/s2]": r["amag"], "|a| smoothed [m/s2]": r["af"], "walking (1=yes)": r["walking"].astype(int),
        "step (1=step)": step_flag,
        "gyro X [deg/s]": np.degrees(r["gyr"][:, 0]), "gyro Y [deg/s]": np.degrees(r["gyr"][:, 1]),
        "gyro Z [deg/s]": np.degrees(r["gyr"][:, 2]), "turn rate about vertical [deg/s]": np.degrees(r["yaw_rate"]),
        "heading from gyro [deg]": r["head_gyro"], "heading from magnetometer [deg]": r["head_mag"],
        "pitch [deg]": np.interp(grid, o["sec"], o["pitchDeg"]), "roll [deg]": np.interp(grid, o["sec"], o["rollDeg"]),
        "mag X [uT]": np.interp(grid, m["sec"], mag_cal[:, 0]), "mag Y [uT]": np.interp(grid, m["sec"], mag_cal[:, 1]),
        "mag Z [uT]": np.interp(grid, m["sec"], mag_cal[:, 2]), "|B| [uT]": np.interp(grid, m["sec"], m["B"]),
        "pressure [hPa]": np.interp(grid, p["sec"], p["PressureHpa"]),
        "height change from pressure [m]": np.interp(grid, p["sec"], p["dh"].bfill().ffill()),
    })
    imu.insert(0, "environment (estimated)", [segment_of(x) for x in grid])
    imu = pd.concat([time_columns(grid + t_imu0, t0), imu], axis=1)

    t_step = r["t_step"]
    pdr_geo = np.array([enu_to_geodetic(e, n, 0.0, x0, rot) for e, n in zip(r["E"], r["N"])])
    steps = pd.concat([time_columns(t_step + t_imu0, t0), pd.DataFrame({
        "step no.": np.arange(1, len(t_step) + 1), "environment (estimated)": [segment_of(x) for x in t_step],
        "step length [m]": r["L"], "heading [deg]": np.degrees(r["h_step"]) % 360,
        "PDR East [m]": r["E"], "PDR North [m]": r["N"], "PDR latitude [deg]": pdr_geo[:, 0],
        "PDR longitude [deg]": pdr_geo[:, 1], "phone GNSS East [m]": r["gE"], "phone GNSS North [m]": r["gN"],
        "distance PDR - phone GNSS [m]": r["pdr_diff"]})], axis=1)

    # ------------------------------------------------------------ combined, one row per GNSS epoch
    a = sol["GPS+GLO"]
    b = sol["GPS"]
    t_ep = a["gps_time"].to_numpy()
    sec_ep = t_ep - t0
    gsec = gfix["gps_time"].to_numpy()
    genu = np.array([enu(ph.geodetic_to_ecef(la, lo, h)) for la, lo, h in gfix[["lat", "lon", "h"]].to_numpy()])
    near_step = np.array([np.sum((t_step > s1 - 1.0) & (t_step <= s1)) for s1 in sec_ep])
    combined = pd.concat([time_columns(t_ep, t0), pd.DataFrame({
        "environment (estimated)": [segment_of(x) for x in sec_ep],
        "satellites tracked (G+R)": a["tracked"], "mean C/N0 tracked [dB-Hz]": a["cn0_mean_tracked"],
        "SPP G+R reliable": a["reliable"].map({True: "yes", False: "no"}),
        "SPP G+R satellites used": a["n_sat"], "SPP G+R latitude [deg]": a["lat"], "SPP G+R longitude [deg]": a["lon"],
        "SPP G+R height [m]": a["h"], "SPP G+R East [m]": a["E"], "SPP G+R North [m]": a["N"],
        "SPP G+R HDOP": a["hdop"], "SPP G+R rejected": a["rejected"],
        "SPP GPS-only reliable": b["reliable"].map({True: "yes", False: "no"}),
        "SPP GPS-only East [m]": b["E"], "SPP GPS-only North [m]": b["N"],
        "phone own GPS fix East [m]": np.interp(t_ep, gsec, genu[:, 0], left=np.nan, right=np.nan),
        "phone own GPS fix North [m]": np.interp(t_ep, gsec, genu[:, 1], left=np.nan, right=np.nan),
        "PDR East [m]": np.interp(sec_ep, t_step, r["E"], left=np.nan, right=np.nan),
        "PDR North [m]": np.interp(sec_ep, t_step, r["N"], left=np.nan, right=np.nan),
        "steps in last second": near_step,
        "heading from gyro [deg]": im.interp_angle(sec_ep - (t_imu0 - t0), grid, r["head_gyro"]),
        "|B| [uT]": np.interp(sec_ep, m["sec"], m["B"]),
        "height change from pressure [m]": np.interp(sec_ep, p["sec"], p["dh"].bfill().ffill()),
    })], axis=1)
    combined["SPP G+R - phone fix [m]"] = np.hypot(combined["SPP G+R East [m]"] - combined["phone own GPS fix East [m]"],
                                                   combined["SPP G+R North [m]"] - combined["phone own GPS fix North [m]"])
    combined["PDR - phone fix [m]"] = np.hypot(combined["PDR East [m]"] - combined["phone own GPS fix East [m]"],
                                               combined["PDR North [m]"] - combined["phone own GPS fix North [m]"])

    gnss_ep = {}
    for name, df in sol.items():
        cols = ["tracked", "cn0_mean_tracked", "reliable", "n_sat", "n_G", "n_R", "lat", "lon", "h", "X", "Y", "Z",
                "E", "N", "U", "hdop", "sigma0", "rms_residual_m", "rejected"]
        d = df[cols].copy()
        d["reliable"] = d["reliable"].map({True: "yes", False: "no"})
        d.columns = [f"{name} {c}" for c in cols]
        gnss_ep[name] = d
    gnss = pd.concat([time_columns(t_ep, t0), pd.Series([segment_of(x) for x in sec_ep], name="environment (estimated)"),
                      gnss_ep["GPS+GLO"], gnss_ep["GPS"]], axis=1)

    fx = fixes.copy()
    fxe = np.array([enu(ph.geodetic_to_ecef(la, lo, h)) for la, lo, h in fx[["lat", "lon", "h"]].to_numpy()])
    fx_out = pd.concat([time_columns(fx["gps_time"], t0), pd.DataFrame({
        "provider": fx["provider"], "latitude [deg]": fx["lat"], "longitude [deg]": fx["lon"],
        "height [m]": fx["h"], "accuracy (phone estimate) [m]": fx["accuracy_m"],
        "East [m]": fxe[:, 0], "North [m]": fxe[:, 1]})], axis=1)

    sats_out = pd.concat([time_columns(sats_df["gps_time"], t0), sats_df.drop(columns=["gps_time"])], axis=1)
    seg_rows = []
    for a1, b1, name in SEGMENTS:
        st, en = max(a1, 0), min(b1, 120)
        tc = time_columns([t0 + st, t0 + en], t0)
        seg_rows.append({"environment (estimated)": name, "from [s]": st, "to [s]": en,
                         "from GPS": tc["GPS time"][0], "to GPS": tc["GPS time"][1],
                         "from UTC": tc["UTC"][0], "to UTC": tc["UTC"][1],
                         f"from local": tc.iloc[0, 2], "to local": tc.iloc[1, 2]})
    segments = pd.DataFrame(seg_rows)
    imu_summary = r["table"]
    write(segments, imu_summary, combined, gnss, sats_out, fx_out, steps, imu, r, t0)


README = [
    "PHONE GNSS + IMU - PROCESSED DATA (training recording, 6 October 2026)",
    "",
    "DEVICE AND FILES",
    "Samsung SM-S928B (Android 16, Qualcomm GNSS GEN9), app GnssLogger v3.1.1.3. Recording: 2 minutes.",
    "Files (folder GNSS_SPP/phone_data): gnss_log_2026_10_06_17_04_51.txt (all raw data), gnss_log_..._52.26o (RINEX 4.01),",
    "brdc2790.26n.gz and brdc2790.26g.gz (broadcast orbits: GPS and GLONASS only).",
    "The walk: outside between tall buildings -> into an 18-floor building -> out through another door.",
    "",
    "TIME (the same columns in every sheet)",
    "GPS time: time of the satellites and of the RINEX file. No leap seconds.",
    "UTC = GPS time - 18 s. The phone sensors and the phone positions are logged in UTC and were converted.",
    "Local time = UTC + 8 h (taken from the file name: 17:04:51 local = 09:04:51 UTC = 09:05:09 GPS).",
    "GPS week and seconds of week: the time used by the satellite orbits.",
    "Seconds since start: 0 = first GNSS epoch (09:05:09.413 GPS). Use this column to plot everything together.",
    "",
    "SHEETS",
    "Segments         - the parts of the walk with their start and end in GPS, UTC and local time (estimated from the sensors).",
    "Combined_1s      - one row per GNSS epoch (1 s): SPP position, phone position, PDR position, steps, heading,",
    "                   magnetic field and pressure height - everything on one time axis.",
    "GNSS_epochs      - single point positioning (this program) for each epoch: GPS+GLONASS and GPS only.",
    "GNSS_satellites  - one row per satellite and epoch: raw pseudorange and C/N0, satellite position and clock,",
    "                   elevation, azimuth, ionosphere and troposphere delays, residual, w-test, and why a satellite was not used.",
    "Phone_fixes      - the phone's own positions (GPS = GNSS chip, FLP = fused with Wi-Fi/cell/sensors, NLP = network).",
    "IMU_summary      - steps, cadence, step length, magnetic field, height change and PDR difference for each part.",
    "PDR_steps        - one row per detected step: step length, heading, PDR position (also latitude/longitude).",
    "IMU_100Hz        - motion sensors resampled to 100 Hz: accelerometer, gyroscope (bias removed), heading,",
    "                   pitch/roll, magnetometer (bias removed), pressure and height change, walking and step flags.",
    "",
    "PROCESSING",
    "GNSS: code pseudoranges GPS L1 C/A and GLONASS L1; broadcast orbits; Klobuchar and Saastamoinen; elevation mask 10 deg;",
    "C/N0 mask 25 dB-Hz; weights sigma = 3 m (elevation dependent); one receiver clock per system; Baarda w-test (3.29).",
    "Reliable epoch = at least 6 satellites, HDOP <= 5 and a-posteriori sigma0 <= 3.",
    "IMU: steps from peaks of |acceleration|; step length with the Weinberg model, K calibrated with GNSS in the open area (65-97 s);",
    "heading from the integrated gyroscope (bias removed), aligned with the GNSS walking direction in the same window.",
    "East / North / Up: local frame with origin at the first phone GPS fix.",
    "",
    "IMPORTANT",
    "This is a training data set. The phone's own position is NOT the truth; differences are only comparisons.",
    "Galileo, BeiDou and QZSS were tracked but not used, because their orbit file was not downloaded.",
    "The environment labels (entering, inside, leaving) are estimated from GNSS signal loss, magnetic field and pressure.",
    "Made with gnss_spp.py, phone_spp.py, phone_imu.py and phone_export_excel.py (folder GNSS_SPP).",
]


def write(segments, imu_summary, combined, gnss, sats, fixes, steps, imu, r, t0):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    def rounded(df):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                deg = "latitude" in c or "longitude" in c or c.endswith(" lat") or c.endswith(" lon")
                df[c] = df[c].round(8 if deg else 4)
        return df

    sheets = [("Segments", segments), ("Combined_1s", combined), ("GNSS_epochs", gnss), ("GNSS_satellites", sats),
              ("Phone_fixes", fixes), ("IMU_summary", imu_summary), ("PDR_steps", steps), ("IMU_100Hz", imu)]
    target = OUT_FILE
    try:
        writer = pd.ExcelWriter(target, engine="openpyxl")
    except PermissionError:
        target = OUT_FILE.with_name(OUT_FILE.stem + datetime.now().strftime("_%H%M%S") + ".xlsx")
        writer = pd.ExcelWriter(target, engine="openpyxl")
    time_fill = PatternFill("solid", fgColor="FFE699")
    gnss_fill = PatternFill("solid", fgColor="BDD7EE")
    imu_fill = PatternFill("solid", fgColor="C6E0B4")
    with writer as xw:
        pd.DataFrame({"README": README}).to_excel(xw, sheet_name="README", index=False, header=False)
        for name, df in sheets:
            rounded(df).to_excel(xw, sheet_name=name, index=False)
        ws = xw.book["README"]
        ws.column_dimensions["A"].width = 140
        for cell in ws["A"]:
            if cell.value and str(cell.value).split(" (")[0].isupper():
                cell.font = Font(bold=True, size=14 if cell.row == 1 else 11)
        imu_words = ("acc", "|a|", "gyro", "turn", "heading", "pitch", "roll", "mag", "|B|", "pressure", "height",
                     "step", "walking", "PDR", "cadence")
        for name, df in sheets:
            ws = xw.book[name]
            ws.freeze_panes = "B2" if name in ("Segments", "IMU_summary") else "D2"
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col)
                text = str(cell.value)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if text in ("GPS time", "UTC", "GPS week", "GPS seconds of week", "seconds since start") \
                        or text.startswith("Local time") or text.startswith("from") or text.startswith("to "):
                    cell.fill = time_fill
                elif any(w in text for w in imu_words):
                    cell.fill = imu_fill
                else:
                    cell.fill = gnss_fill
                ws.column_dimensions[get_column_letter(col)].width = max(10, min(24, len(text) * 0.55 + 4))
            ws.row_dimensions[1].height = 45
    print(f"Saved: {target}")
    for name, df in sheets:
        print(f"  {name:16s} {len(df):6d} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
