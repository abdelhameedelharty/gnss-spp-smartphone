# -*- coding: utf-8 -*-
"""
phone_export_excel.py - phone GNSS + IMU processed data -> one Excel file, all on one time axis
=============================================================================================
"""
import io
import math
from collections import Counter
from datetime import datetime
import os

import numpy as np
import pandas as pd

import gnss_spp as spp
import phone_spp as ph
import phone_imu as im

# Default fallback values
LOCAL_OFFSET_H = 8
SYSTEM_NAME = {"G": "GPS", "R": "GLONASS", "E": "Galileo", "C": "BeiDou", "J": "QZSS"}
ALL_SYSTEMS = "GRECJ"      # every system gnss_spp.py can use (NavIC and SBAS are not supported)
# ALL = new main solution, GPS+GLO = the last results (GPS + GLONASS only), GPS = GPS only
SOLUTIONS = {"ALL": ALL_SYSTEMS, "GPS+GLO": "GR", "GPS": "G"}

# Environment segments (from, to [s since start], name). They are found from the signals in main()
# (signal_segments), so they fit every recording. The old fixed list fitted only the 6 Oct walk.
SEGMENTS = []
OPEN_CN0, BLOCKED_CN0, MIN_TRACKED = 36.0, 30.0, 10    # dB-Hz, dB-Hz, satellites (5 s means)


def signal_segments(sec, cn0, tracked, no_position):
    """Environment class of every epoch from the signals only, as runs (from, to, name).
    open sky: 5 s mean C/N0 >= OPEN_CN0; blocked: C/N0 < BLOCKED_CN0, fewer than MIN_TRACKED satellites
    or no position; otherwise partly blocked."""
    cn0_5 = pd.Series(cn0).rolling(5, center=True, min_periods=1).mean().to_numpy()
    trk_5 = pd.Series(tracked).rolling(5, center=True, min_periods=1).mean().to_numpy()
    env = np.where((cn0_5 < BLOCKED_CN0) | (trk_5 < MIN_TRACKED) | no_position, "blocked (GNSS lost or weak)",
                   np.where(cn0_5 >= OPEN_CN0, "open sky", "partly blocked"))
    segs, start = [], 0
    for k in range(1, len(env) + 1):
        if k == len(env) or env[k] != env[start]:
            a = -1.0 if start == 0 else sec[start] - 0.5
            b = sec[-1] + 1.0 if k == len(env) else sec[k] - 0.5
            segs.append((float(a), float(b), str(env[start])))
            start = k
    return segs


def longest(segments, name):
    """Longest segment with this name as (from, to), or None."""
    runs = [(a, b) for a, b, n in segments if n == name]
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def imu_table(r):
    """IMU summary for each environment segment (same columns as phone_imu's own table)."""
    t_step, grid, af, walking, L = r["t_step"], r["grid"], r["af"], r["walking"], r["L"]
    m, p, mag_minus_gyro, pdr_diff = r["m"], r["p"], r["mag_minus_gyro"], r["pdr_diff"]
    rows = []
    for a, b, name in SEGMENTS:
        t1, t2 = max(a, grid[0]), min(b, grid[-1])
        st = (t_step >= t1) & (t_step < t2)
        gg = (grid >= t1) & (grid < t2)
        mm = (m["sec"] >= t1) & (m["sec"] < t2)
        pp = (p["sec"] >= t1) & (p["sec"] < t2)
        f_peak = np.nan
        if gg.sum() > im.FS:                              # cadence check with the spectrum of |a| (>= 1 s of data)
            seg_a = af[gg] - af[gg].mean()
            freq = np.fft.rfftfreq(len(seg_a), 1.0 / im.FS)
            band = (freq > 1.0) & (freq < 3.0)
            if band.any():
                f_peak = freq[band][np.argmax(np.abs(np.fft.rfft(seg_a))[band])]
        rows.append({"segment": name, "from [s]": round(max(t1, 0.0), 1), "to [s]": round(t2, 1),
                     "steps": int(st.sum()),
                     "cadence [steps/s]": round(st.sum() / (t2 - t1), 2) if t2 > t1 else np.nan,
                     "cadence FFT [steps/s]": round(f_peak, 2),
                     "mean step length [m]": round(L[st].mean(), 2) if st.any() else np.nan,
                     "walking share [%]": round(100 * walking[gg].mean()) if gg.any() else np.nan,
                     "|B| median [uT]": round(m.loc[mm, "B"].median(), 1),
                     "|B| range [uT]": round(m.loc[mm, "B"].max() - m.loc[mm, "B"].min(), 1),
                     "heading mag - gyro, median abs [deg]":
                         round(np.nanmedian(np.abs(mag_minus_gyro[gg])), 1) if gg.any() else np.nan,
                     "height change [m]": round(p.loc[pp, "dh"].mean(), 2),
                     "PDR - phone GNSS at end of segment [m]": round(pdr_diff[st][-1], 1) if st.any() else np.nan})
    return pd.DataFrame(rows)


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
    global LOCAL_OFFSET_H
    global OUT_FILE
    global SEGMENTS

    # =====================================================================
    # AUTOMATIC FOLDER READER
    # =====================================================================
    print("\\n=== SETUP: Provide the folder containing your files ===")
    folder_input = input("Enter the folder path: ").strip("'\" ")

    if not os.path.isdir(folder_input):
        print(f"Error: The folder '{folder_input}' does not exist.")
        return

    obs_file, nav_files, log_file = None, [], None

    for file in os.listdir(folder_input):
        filepath = os.path.join(folder_input, file)
        if not os.path.isfile(filepath):
            continue

        ext = file.lower().split('.')[-1]

        if ext in ['26o', '25o', '24o', '23o', '22o', '21o', '20o', '19o', '18o', 'rnx'] and 'obs' not in file.lower():
            if obs_file is None:
                obs_file = filepath
        elif ext in ['26p', '25p', '24p', '23p', '22p', '21p', '20p', '19p', '18p', 'n', 'p', 'g', 'rnx', 'gz']:
            if filepath != obs_file:
                nav_files.append(filepath)
        elif ext == 'txt':
            log_file = filepath

    if not obs_file:
        print("Error: No RINEX Observation file found.")
        return
    if not nav_files:
        print("Error: No Navigation files found.")
        return
    if not log_file:
        print("Error: No GnssLogger text file found.")
        return

    print(f"\\nFound Observation: {os.path.basename(obs_file)}")
    print(f"Found Navigation:  {[os.path.basename(n) for n in nav_files]}")
    print(f"Found Logger txt:  {os.path.basename(log_file)}")

    ph.OBS_FILE = obs_file
    ph.NAV_FILES = nav_files
    ph.LOG_FILE = log_file

    OUT_FILE = os.path.join(folder_input, "Phone_GNSS_IMU_Output.xlsx")
    print(f"Output will be saved to: {OUT_FILE}")

    tz_input = input(f"\\nEnter timezone offset in hours (default is {LOCAL_OFFSET_H}): ").strip()
    if tz_input:
        LOCAL_OFFSET_H = int(tz_input)

    print("\\nProcessing data, please wait...\\n")
    # =====================================================================

    # ------------------------------------------------------------ GNSS
    spp.SYSTEMS = ALL_SYSTEMS
    epochs = spp.read_rinex_obs(ph.OBS_FILE)
    nav = spp.Navigation(ph.NAV_FILES)
    fixes = ph.read_phone_fixes(ph.LOG_FILE)
    t0 = epochs[0][0]
    gfix = fixes[fixes["provider"] == "GPS"].reset_index(drop=True)
    x0 = ph.geodetic_to_ecef(*gfix.loc[0, ["lat", "lon", "h"]])
    lat0, lon0, _ = spp.ecef_to_geodetic(x0)
    rot = spp.enu_rotation(lat0, lon0)
    enu = lambda xyz: rot @ (np.asarray(xyz, float) - x0)

    n_eph = Counter(sat[0] for sat in nav.eph)
    print("Satellites with ephemeris: " + ", ".join(f"{SYSTEM_NAME[k]} {n_eph[k]}" for k in ALL_SYSTEMS))

    sol = {name: ph.run(epochs, nav, systems) for name, systems in SOLUTIONS.items()}
    for df in sol.values():
        ok = df["X"].notna()
        df.loc[ok, ["E", "N", "U"]] = np.array([enu(v) for v in df.loc[ok, ["X", "Y", "Z"]].to_numpy()])

    # per-satellite table (main solution: all systems)
    sat_rows, x_prev, n_used = [], None, []
    mask = math.radians(spp.ELEV_MASK_DEG)
    for t, meas in epochs:
        m = {s: v for s, v in meas.items() if s[0] in ALL_SYSTEMS}
        s_ep = spp.solve_epoch(t, m, nav, x_prev)
        if s_ep is not None:
            x_prev = s_ep["x"]
        n_used.append(Counter(x[0] for x in s_ep["sats"]) if s_ep else None)
        sats = {d["sat"]: d for d in spp.satellites_for_epoch(t, m, nav)}
        resid = dict(zip(s_ep["sats"], zip(s_ep["v"], s_ep["w"]))) if s_ep else {}
        for sat in sorted(meas):
            p, cn0 = meas[sat]
            r = {"gps_time": t, "satellite": sat, "system": SYSTEM_NAME.get(sat[0], sat[0]),
                 "signal": "C" + spp.SIGNALS[sat[0]], "pseudorange [m]": p, "C/N0 [dB-Hz]": cn0}
            d = sats.get(sat)
            if sat[0] not in ALL_SYSTEMS:
                r["status"] = "not used: system not supported by gnss_spp.py"
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
    for k in "ECJ":                                      # ph.run counts only GPS (n_G) and GLONASS (n_R)
        sol["ALL"][f"n_{k}"] = [np.nan if c is None else c[k] for c in n_used]

    # ------------------------------------------------------------ environment segments (from the signals)
    a = sol["ALL"]
    SEGMENTS = signal_segments((a["gps_time"] - t0).to_numpy(), a["cn0_mean_tracked"].to_numpy(),
                               a["tracked"].to_numpy(), a["X"].isna().to_numpy())
    print("Environment segments:", ", ".join(f"{max(s0, 0):.0f}-{s1:.0f} s {n}" for s0, s1, n in SEGMENTS))

    # ------------------------------------------------------------ IMU (same log file, same time zero)
    # phone_imu.py reads its settings when it is imported (the 6 Oct walk), so set them for this recording:
    im.LOG_FILE = ph.LOG_FILE
    im.SPP_RESULTS = io.StringIO(f"gps_time\n{t0!r}\n")          # analyse() takes time zero from here
    im.CAL_WINDOW = longest(SEGMENTS, "open sky") or longest(SEGMENTS, "partly blocked")
    im.BLOCKED = longest(SEGMENTS, "blocked (GNSS lost or weak)") or (0.0, 5.0)
    print(f"IMU: step length and heading calibrated in {im.CAL_WINDOW[0]:.0f}-{im.CAL_WINDOW[1]:.0f} s "
          f"(longest open-sky part)")
    r = im.analyse()
    acc_t = r["s"]["UncalAccel"]["sec"].to_numpy()
    sensor_gaps = [(g0, g1) for g0, g1 in zip(acc_t[:-1], acc_t[1:]) if g1 - g0 > 1.0]
    for g0, g1 in sensor_gaps:
        print(f"WARNING: no IMU data from {g0:.1f} to {g1:.1f} s - steps there are missing in the PDR")
        README.append(f"WARNING: the phone logged no IMU data from {g0:.1f} to {g1:.1f} s; "
                      f"the PDR has no steps in this time.")
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

    # ------------------------------------------------------------ combined
    a = sol["ALL"]
    last = sol["GPS+GLO"]
    b = sol["GPS"]
    t_ep = a["gps_time"].to_numpy()
    sec_ep = t_ep - t0
    gsec = gfix["gps_time"].to_numpy()
    genu = np.array([enu(ph.geodetic_to_ecef(la, lo, h)) for la, lo, h in gfix[["lat", "lon", "h"]].to_numpy()])
    near_step = np.array([np.sum((t_step > s1 - 1.0) & (t_step <= s1)) for s1 in sec_ep])

    combined = pd.concat([time_columns(t_ep, t0), pd.DataFrame({
        "environment (estimated)": [segment_of(x) for x in sec_ep],
        "satellites tracked (all systems)": a["tracked"], "satellites tracked (G+R)": last["tracked"],
        "mean C/N0 tracked, all systems [dB-Hz]": a["cn0_mean_tracked"],
        "SPP ALL reliable": a["reliable"].map({True: "yes", False: "no"}),
        "SPP ALL satellites used": a["n_sat"], "SPP ALL latitude [deg]": a["lat"], "SPP ALL longitude [deg]": a["lon"],
        "SPP ALL height [m]": a["h"], "SPP ALL East [m]": a["E"], "SPP ALL North [m]": a["N"],
        "SPP ALL HDOP": a["hdop"], "SPP ALL rejected": a["rejected"],
        "SPP G+R reliable": last["reliable"].map({True: "yes", False: "no"}),
        "SPP G+R satellites used": last["n_sat"], "SPP G+R East [m]": last["E"], "SPP G+R North [m]": last["N"],
        "SPP G+R HDOP": last["hdop"],
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

    combined["SPP ALL - phone fix [m]"] = np.hypot(
        combined["SPP ALL East [m]"] - combined["phone own GPS fix East [m]"],
        combined["SPP ALL North [m]"] - combined["phone own GPS fix North [m]"]
    )

    combined["SPP G+R - phone fix [m]"] = np.hypot(
        combined["SPP G+R East [m]"] - combined["phone own GPS fix East [m]"],
        combined["SPP G+R North [m]"] - combined["phone own GPS fix North [m]"]
    )

    combined["SPP ALL - SPP G+R [m]"] = np.hypot(
        combined["SPP ALL East [m]"] - combined["SPP G+R East [m]"],
        combined["SPP ALL North [m]"] - combined["SPP G+R North [m]"]
    )

    combined["PDR - phone fix [m]"] = np.hypot(
        combined["PDR East [m]"] - combined["phone own GPS fix East [m]"],
        combined["PDR North [m]"] - combined["phone own GPS fix North [m]"]
    )

    # =====================================================================
    # NEW: RESAMPLED 1-HZ COORDINATES TABLE
    # Calculates exact 1-second intervals for both GNSS and IMU (PDR)
    # =====================================================================
    t_max = max(sec_ep[-1], t_step[-1]) if len(t_step) > 0 else sec_ep[-1]
    t_1hz = np.arange(0, t_max, 1.0)  # Array of seconds: 0, 1, 2, 3...

    # 1. Interpolate GNSS Coordinates (all-systems solution)
    gnss_e_1hz = np.interp(t_1hz, sec_ep, a["E"], left=np.nan, right=np.nan)
    gnss_n_1hz = np.interp(t_1hz, sec_ep, a["N"], left=np.nan, right=np.nan)
    gnss_u_1hz = np.interp(t_1hz, sec_ep, a["U"], left=np.nan, right=np.nan)

    # 2. Interpolate IMU (PDR) Coordinates
    pdr_e_1hz = np.interp(t_1hz, t_step, r["E"], left=np.nan, right=np.nan)
    pdr_n_1hz = np.interp(t_1hz, t_step, r["N"], left=np.nan, right=np.nan)

    # 3. Convert interpolated ENU back to Latitude / Longitude
    gnss_llh_1hz = [enu_to_geodetic(e, n, u, x0, rot) for e, n, u in zip(gnss_e_1hz, gnss_n_1hz, gnss_u_1hz)]
    pdr_llh_1hz = [enu_to_geodetic(e, n, 0.0, x0, rot) for e, n in zip(pdr_e_1hz, pdr_n_1hz)]

    coords_1hz = pd.concat([time_columns(t_1hz + t0, t0), pd.DataFrame({
        "environment (estimated)": [segment_of(x) for x in t_1hz],
        # GNSS Coordinates
        "GNSS Latitude [deg]": [c[0] for c in gnss_llh_1hz],
        "GNSS Longitude [deg]": [c[1] for c in gnss_llh_1hz],
        "GNSS Height [m]": [c[2] for c in gnss_llh_1hz],
        "GNSS East [m]": gnss_e_1hz,
        "GNSS North [m]": gnss_n_1hz,
        "GNSS Up [m]": gnss_u_1hz,
        # IMU (PDR) Coordinates
        "IMU (PDR) Latitude [deg]": [c[0] for c in pdr_llh_1hz],
        "IMU (PDR) Longitude [deg]": [c[1] for c in pdr_llh_1hz],
        "IMU (PDR) Height [m]": 0.0,  # IMU doesn't calculate height here
        "IMU (PDR) East [m]": pdr_e_1hz,
        "IMU (PDR) North [m]": pdr_n_1hz,
        # Difference
        "2D Distance GNSS vs IMU [m]": np.hypot(gnss_e_1hz - pdr_e_1hz, gnss_n_1hz - pdr_n_1hz)
    })], axis=1)
    # =====================================================================

    gnss_ep = {}
    for name, df in sol.items():
        cols = ["tracked", "cn0_mean_tracked", "reliable", "n_sat", "n_G", "n_R", "n_E", "n_C", "n_J", "lat", "lon",
                "h", "X", "Y", "Z", "E", "N", "U", "hdop", "sigma0", "rms_residual_m", "rejected"]
        cols = [c for c in cols if c in df]
        d = df[cols].copy()
        d["reliable"] = d["reliable"].map({True: "yes", False: "no"})
        d.columns = [f"{name} {c}" for c in cols]
        gnss_ep[name] = d
    gnss = pd.concat(
        [time_columns(t_ep, t0), pd.Series([segment_of(x) for x in sec_ep], name="environment (estimated)"),
         gnss_ep["ALL"], gnss_ep["GPS+GLO"], gnss_ep["GPS"]], axis=1)

    compare_sum, compare_ep = compare(sol, t_ep, t0, gsec, genu)

    fx = fixes.copy()
    fxe = np.array([enu(ph.geodetic_to_ecef(la, lo, h)) for la, lo, h in fx[["lat", "lon", "h"]].to_numpy()])
    fx_out = pd.concat([time_columns(fx["gps_time"], t0), pd.DataFrame({
        "provider": fx["provider"], "latitude [deg]": fx["lat"], "longitude [deg]": fx["lon"],
        "height [m]": fx["h"], "accuracy (phone estimate) [m]": fx["accuracy_m"],
        "East [m]": fxe[:, 0], "North [m]": fxe[:, 1]})], axis=1)

    sats_out = pd.concat([time_columns(sats_df["gps_time"], t0), sats_df.drop(columns=["gps_time"])], axis=1)
    seg_rows = []
    for a1, b1, name in SEGMENTS:
        st, en = max(a1, 0), min(b1, sec_ep[-1] + 1)
        tc = time_columns([t0 + st, t0 + en], t0)
        seg_rows.append({"environment (estimated)": name, "from [s]": st, "to [s]": en,
                         "from GPS": tc["GPS time"][0], "to GPS": tc["GPS time"][1],
                         "from UTC": tc["UTC"][0], "to UTC": tc["UTC"][1],
                         f"from local": tc.iloc[0, 2], "to local": tc.iloc[1, 2]})
    segments = pd.DataFrame(seg_rows)
    imu_summary = imu_table(r)                    # r["table"] uses the fixed windows of the 6 Oct walk

    print("\nCOMPARISON (only reliable epochs; the phone fix is not the truth)")
    print(compare_sum.set_index("solution").T.to_string())

    # Pass the new coords_1hz table to the writer
    write(segments, imu_summary, combined, gnss, sats_out, fx_out, steps, imu, r, t0, coords_1hz,
          compare_sum, compare_ep)


def compare(sol, t_ep, t0, gsec, genu):
    """New all-systems solution vs the last results (GPS+GLONASS) and GPS only.
    Returns a summary (one row per solution, only reliable epochs) and a per-epoch table."""
    fix_e = np.interp(t_ep, gsec, genu[:, 0], left=np.nan, right=np.nan)
    fix_n = np.interp(t_ep, gsec, genu[:, 1], left=np.nan, right=np.nan)
    near = np.array([np.min(np.abs(gsec - t)) <= 1.0 for t in t_ep])     # a phone fix within 1 s (no gap bridging)
    near &= np.isfinite(fix_e)                                           # and not before the first / after the last fix
    last = sol["GPS+GLO"]
    last_ok = last["reliable"].to_numpy()
    med = lambda x: round(float(np.median(x)), 2) if len(x) else np.nan
    p95 = lambda x: round(float(np.percentile(x, 95)), 2) if len(x) else np.nan

    rows, per_epoch = [], {}
    for name, df in sol.items():
        ok = df["reliable"].to_numpy()
        d = df[ok]
        to_fix = np.hypot(df["E"] - fix_e, df["N"] - fix_n).to_numpy()
        to_last = np.hypot(df["E"] - last["E"], df["N"] - last["N"]).to_numpy()
        row = {"solution": name, "systems": " + ".join(SYSTEM_NAME[k] for k in SOLUTIONS[name]),
               "epochs": len(df), "epochs solved": int(df["X"].notna().sum()), "epochs reliable": int(ok.sum()),
               "reliable [%]": round(100.0 * ok.mean(), 1),
               "reliable here, not in G+R": int((ok & ~last_ok).sum()),
               "reliable in G+R, not here": int((~ok & last_ok).sum()),
               "mean satellites used": round(d["n_sat"].mean(), 1) if len(d) else np.nan}
        for k in ALL_SYSTEMS:
            row[f"mean {SYSTEM_NAME[k]} used"] = round(d[f"n_{k}"].mean(), 1) if f"n_{k}" in d and len(d) else 0.0
        row.update({"median HDOP": med(d["hdop"]), "median sigma0": med(d["sigma0"]),
                    "median RMS residual [m]": med(d["rms_residual_m"]),
                    "epochs with rejections": int((d["rejected"] != "").sum()),
                    "median 2D diff to phone fix [m]": med(to_fix[ok & near]),
                    "95% 2D diff to phone fix [m]": p95(to_fix[ok & near]),
                    "max 2D diff to phone fix [m]": round(float(np.max(to_fix[ok & near])), 2)
                    if (ok & near).any() else np.nan,
                    "median 2D diff to G+R (both reliable) [m]": med(to_last[ok & last_ok])})
        rows.append(row)
        if name != "GPS":
            per_epoch.update({f"{name} reliable": df["reliable"].map({True: "yes", False: "no"}),
                              f"{name} satellites used": df["n_sat"], f"{name} HDOP": df["hdop"],
                              f"{name} sigma0": df["sigma0"], f"{name} RMS residual [m]": df["rms_residual_m"],
                              f"{name} rejected": df["rejected"], f"{name} East [m]": df["E"],
                              f"{name} North [m]": df["N"], f"{name} Up [m]": df["U"],
                              f"{name} - phone fix 2D [m]": to_fix})
    a = sol["ALL"]
    per_epoch.update({"ALL - G+R 2D [m]": np.hypot(a["E"] - last["E"], a["N"] - last["N"]),
                      "ALL - G+R Up [m]": a["U"] - last["U"],
                      **{f"ALL {SYSTEM_NAME[k]} used": a[f"n_{k}"] for k in ALL_SYSTEMS}})
    sec_ep = t_ep - t0
    epochs = pd.concat([time_columns(t_ep, t0), pd.DataFrame({
        "environment (estimated)": [segment_of(x) for x in sec_ep],
        "satellites tracked (all systems)": a["tracked"], "satellites tracked (G+R)": last["tracked"],
        "phone fix within 1 s": np.where(near, "yes", "no"), **per_epoch})], axis=1)
    return pd.DataFrame(rows), epochs


README = [
    "PHONE GNSS + IMU - PROCESSED DATA (NEW RECORDING)",
    "",
    "DEVICE AND FILES",
    "Processed via dynamic folder input.",
    "",
    "TIME (the same columns in every sheet)",
    "GPS time: time of the satellites and of the RINEX file. No leap seconds.",
    "UTC = GPS time - 18 s. The phone sensors and the phone positions are logged in UTC and were converted.",
    f"Local time = UTC + {LOCAL_OFFSET_H} h",
    "GPS week and seconds of week: the time used by the satellite orbits.",
    "Seconds since start: 0 = first GNSS epoch. Use this column to plot everything together.",
    "",
    "SATELLITE SYSTEMS",
    "ALL (main solution) = GPS + GLONASS + Galileo + BeiDou + QZSS, with the merged navigation file (BRDM ... MN.rnx.gz).",
    "Code only: GPS L1 C/A, GLONASS L1, Galileo E1, BeiDou B1I, QZSS L1 C/A. One receiver clock per system.",
    "NavIC (IRNSS) and SBAS are in the navigation file but are not used (gnss_spp.py does not support them).",
    "GPS+GLO = the last results (GPS + GLONASS only). They are computed again in the same run, with the same data and settings.",
    "Combined_1s, Coordinates_1Hz and GNSS_satellites use the ALL solution.",
    "",
    "COMPARISON",
    "Compare_summary - one row per solution (ALL, GPS+GLO = last results, GPS only). Only reliable epochs are used.",
    "Compare_epochs  - every epoch side by side: ALL and GPS+GLO, their difference, and satellites used per system.",
    "Reliable epoch = at least 6 satellites, HDOP <= 5 and sigma0 <= 3.",
    "The phone's own fix is NOT the truth; the differences to it are only a comparison.",
    "",
    "ENVIRONMENT SEGMENTS AND IMU",
    "Segments are found from the signals of the ALL solution (5 s means): open sky = C/N0 >= 36 dB-Hz;",
    "blocked = C/N0 < 30 dB-Hz, fewer than 10 satellites, or no position; partly blocked = the rest.",
    "IMU (IMU_100Hz, PDR_steps, IMU_summary) is read from the same GnssLogger txt as the GNSS (this recording).",
    "Step length (Weinberg K) and heading are calibrated with the phone's own GPS track in the longest open-sky segment.",
]


def write(segments, imu_summary, combined, gnss, sats, fixes, steps, imu, r, t0, coords_1hz, compare_sum, compare_ep):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    def rounded(df):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                deg = "latitude" in c or "longitude" in c or c.endswith(" lat") or c.endswith(" lon")
                df[c] = df[c].round(8 if deg else 4)
        return df

    # Added "Coordinates_1Hz" and the two comparison sheets to the list of sheets
    sheets = [("Compare_summary", compare_sum), ("Segments", segments), ("Combined_1s", combined),
              ("Compare_epochs", compare_ep), ("Coordinates_1Hz", coords_1hz),
              ("GNSS_epochs", gnss), ("GNSS_satellites", sats), ("Phone_fixes", fixes),
              ("IMU_summary", imu_summary), ("PDR_steps", steps), ("IMU_100Hz", imu)]

    target = OUT_FILE
    try:
        writer = pd.ExcelWriter(target, engine="openpyxl")
    except PermissionError:
        target = OUT_FILE.replace(".xlsx", f"_{datetime.now().strftime('%H%M%S')}.xlsx")
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
                     "step", "walking", "PDR", "cadence", "IMU")
        for name, df in sheets:
            ws = xw.book[name]
            ws.freeze_panes = "B2" if name in ("Compare_summary", "Segments", "IMU_summary") else "D2"
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col)
                text = str(cell.value)
                cell.alignment = Alignment(wrap_text=True, vertical="top")

                is_time_col = text in ("GPS time", "UTC", "GPS week", "GPS seconds of week", "seconds since start")
                is_time_col = is_time_col or text.startswith("Local time") or text.startswith(
                    "from") or text.startswith("to ")

                if is_time_col:
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