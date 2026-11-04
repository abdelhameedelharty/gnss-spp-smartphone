# -*- coding: utf-8 -*-
"""
phone_imu.py - analysis of the phone motion sensors (GnssLogger log) + simple PDR
===============================================================================

Sensors in the log: accelerometer and gyroscope (~110 Hz), magnetometer (100 Hz),
orientation / game rotation vector (100 Hz), barometer (25 Hz).
Time: the sensor time 'utcTimeMillis' is UTC -> converted to GPS time (+18 s),
      so the x-axis is the same as in phone_spp.py (seconds since the first GNSS epoch).

What it does
 1. Walking or standing (accelerometer variation), step detection (peaks of |a|).
 2. Step length with the Weinberg model  L = K * (a_max - a_min)^(1/4);
    K is calibrated with the phone GNSS track in the open-sky part (CAL_WINDOW).
 3. Heading: gyro-based (game rotation vector, no magnetometer) and magnetic
    (orientation); both aligned to the GNSS walking direction in CAL_WINDOW.
    Their difference shows magnetic disturbances (e.g. steel in buildings).
 4. Magnetic field strength and air pressure (height change) along the walk.
 5. PDR track (steps + heading) compared with the phone GNSS track, also
    through the part where GNSS was blocked.

Run phone_spp.py first (this script reads its results). Needs numpy, pandas, scipy, matplotlib.
"""
import math

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

import gnss_spp as spp
import phone_spp as ph

# ================================ SETTINGS ===================================
LOG_FILE = ph.LOG_FILE
SPP_RESULTS = ph.OUT_DIR / "phone_spp_results.csv"
OUT_DIR = ph.OUT_DIR
CAL_WINDOW = (65.0, 97.0)        # s, open-sky part used to calibrate step length and heading
BLOCKED = (35.0, 65.0)           # s, part where GNSS was blocked (from phone_spp results)
FS = 100.0                       # Hz, sensors are resampled to this rate
SHOW_PLOTS = True
SAVE_FIGURES = True
# =============================================================================


def read_sensors(path):
    heads, data = {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("# ") and "," in line:
                p = line[2:].strip().split(",")
                heads[p[0]] = p
            elif line[:1].isalpha():
                k = line.split(",", 1)[0]
                if k in ("UncalAccel", "UncalGyro", "UncalMag", "Pressure", "OrientationDeg", "GameRotationVector"):
                    data.setdefault(k, []).append(line.strip().split(","))
    out = {}
    for k, rows in data.items():
        df = pd.DataFrame(rows, columns=heads[k][:len(rows[0])]).iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        df["gps_time"] = df["utcTimeMillis"] / 1000.0 - spp.GPS_EPOCH_UNIX + spp.LEAP_SECONDS   # UTC -> GPS
        out[k] = df.sort_values("gps_time").reset_index(drop=True)
    return out


def resample(df, cols, grid):
    return np.column_stack([np.interp(grid, df["sec"], df[c]) for c in cols])


def circ_mean_deg(a):
    a = np.radians(np.asarray(a, float))
    return math.degrees(math.atan2(np.nanmean(np.sin(a)), np.nanmean(np.cos(a))))


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def interp_angle(t, grid, deg):
    """Interpolate an angle in degrees without jumping at 0/360."""
    return np.degrees(np.interp(t, grid, np.unwrap(np.radians(deg)))) % 360.0


def analyse():
    """All IMU processing. Returns a dict with every result (used by main() and phone_export_excel.py)."""
    s = read_sensors(LOG_FILE)
    spp_res = pd.read_csv(SPP_RESULTS)
    t0 = spp_res["gps_time"].iloc[0]                      # first GNSS epoch = time zero
    for df in s.values():
        df["sec"] = df["gps_time"] - t0
    rates = {k: round(1.0 / df["sec"].diff().median(), 1) for k, df in s.items()}

    grid = np.arange(max(df["sec"].iloc[0] for df in s.values()),
                     min(df["sec"].iloc[-1] for df in s.values()), 1.0 / FS)

    # ---------------- accelerometer: walking and steps
    acc = s["UncalAccel"]
    a = resample(acc, ["UncalAccelXMps2", "UncalAccelYMps2", "UncalAccelZMps2"], grid) \
        - acc[["BiasXMps2", "BiasYMps2", "BiasZMps2"]].median().to_numpy()
    amag = np.linalg.norm(a, axis=1)
    b, c = butter(4, 3.0 / (FS / 2))
    af = filtfilt(b, c, amag)                             # smooth |a| (steps are 1.5-2.5 Hz)
    win = int(FS)
    roll_std = pd.Series(amag).rolling(win, center=True, min_periods=10).std().to_numpy()
    walking = roll_std > 0.6
    peaks, _ = find_peaks(af, distance=int(0.3 * FS), prominence=1.0)
    peaks = peaks[walking[peaks]]
    half = int(0.25 * FS)
    amin = np.array([af[max(0, p - half): p + half].min() for p in peaks])
    weinberg = (af[peaks] - amin) ** 0.25
    t_step = grid[peaks]

    # ---------------- heading: own gyro integration and magnetic (orientation)
    # (Android's game rotation vector re-sets its heading sometimes, so the gyro is integrated here.)
    gy = s["UncalGyro"]
    gyr = resample(gy, ["UncalGyroXRadPerSec", "UncalGyroYRadPerSec", "UncalGyroZRadPerSec"], grid) \
        - gy[["DriftXRadPerSec", "DriftYRadPerSec", "DriftZRadPerSec"]].median().to_numpy()    # remove gyro bias
    bl, al = butter(2, 0.3 / (FS / 2))
    up = filtfilt(bl, al, a, axis=0)                     # slow part of the accelerometer = "up" direction
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    yaw_rate = -np.sum(gyr * up, axis=1)                  # turn rate about the vertical, clockwise positive
    yaw_game = np.degrees(np.cumsum(yaw_rate) / FS) % 360
    o = s["OrientationDeg"]
    yaw_mag = np.degrees(np.unwrap(np.radians(o["yawDeg"].to_numpy())))
    yaw_mag = np.interp(grid, o["sec"], yaw_mag) % 360

    # ---------------- GNSS walking direction and distance (phone's own GPS track, 1 Hz)
    fixes = ph.read_phone_fixes(LOG_FILE)
    g = fixes[fixes["provider"] == "GPS"].reset_index(drop=True)
    x0 = ph.geodetic_to_ecef(*g.loc[0, ["lat", "lon", "h"]])
    lat0, lon0, _ = spp.ecef_to_geodetic(x0)
    rot = spp.enu_rotation(lat0, lon0)
    enu = np.array([rot @ (ph.geodetic_to_ecef(la, lo, h) - x0) for la, lo, h in g[["lat", "lon", "h"]].to_numpy()])
    g["E"], g["N"], g["sec"] = enu[:, 0], enu[:, 1], g["gps_time"] - t0
    dE, dN = np.diff(g["E"]), np.diff(g["N"])
    step_len_gnss = np.hypot(dE, dN)
    course = np.degrees(np.arctan2(dE, dN)) % 360
    t_course = (g["sec"].to_numpy()[1:] + g["sec"].to_numpy()[:-1]) / 2
    moving = step_len_gnss > 0.5
    cal = moving & (t_course >= CAL_WINDOW[0]) & (t_course <= CAL_WINDOW[1])

    off_game = circ_mean_deg(course[cal] - interp_angle(t_course[cal], grid, yaw_game))
    off_mag = circ_mean_deg(course[cal] - interp_angle(t_course[cal], grid, yaw_mag))
    head_gyro = (yaw_game + off_game) % 360
    head_mag = (yaw_mag + off_mag) % 360
    mag_minus_gyro = wrap180(head_mag - head_gyro)

    # ---------------- step length: calibrate K in the open-sky window
    in_cal = (t_step >= CAL_WINDOW[0]) & (t_step <= CAL_WINDOW[1])
    in_cal_g = (g["sec"] >= CAL_WINDOW[0]) & (g["sec"] <= CAL_WINDOW[1])
    dist_gnss_cal = np.hypot(np.diff(g.loc[in_cal_g, "E"]), np.diff(g.loc[in_cal_g, "N"])).sum()
    K = dist_gnss_cal / weinberg[in_cal].sum()
    L = K * weinberg

    # ---------------- PDR track, starting at the phone GNSS position at the first step
    h_step = np.radians(interp_angle(t_step, grid, head_gyro))
    E = np.interp(t_step[0], g["sec"], g["E"]) + np.cumsum(L * np.sin(h_step))
    N = np.interp(t_step[0], g["sec"], g["N"]) + np.cumsum(L * np.cos(h_step))
    gE, gN = np.interp(t_step, g["sec"], g["E"]), np.interp(t_step, g["sec"], g["N"])
    pdr_diff = np.hypot(E - gE, N - gN)

    # ---------------- magnetic field strength and pressure
    m = s["UncalMag"]
    mcal = m[["UncalMagXMicroT", "UncalMagYMicroT", "UncalMagZMicroT"]].to_numpy() \
        - m[["BiasXMicroT", "BiasYMicroT", "BiasZMicroT"]].to_numpy()
    m["B"] = np.linalg.norm(mcal, axis=1)
    p = s["Pressure"]
    p["p_smooth"] = p["PressureHpa"].rolling(25, center=True, min_periods=5).mean()
    p["dh"] = -8.43 * (p["p_smooth"] - p["p_smooth"].iloc[:50].mean())     # ~8.4 m per hPa near sea level

    # ---------------- gyro bias (Android estimate)
    gb = s["UncalGyro"][["DriftXRadPerSec", "DriftYRadPerSec", "DriftZRadPerSec"]].median().to_numpy()

    seg = {"0-35 s near buildings": (0, 35), "35-65 s GNSS blocked": BLOCKED,
           "65-97 s open area": CAL_WINDOW, "97-120 s end": (97, 121)}
    rows = []
    for name, (t1, t2) in seg.items():
        st = (t_step >= t1) & (t_step < t2)
        gg = (grid >= t1) & (grid < t2)
        mm = (m["sec"] >= t1) & (m["sec"] < t2)
        pp = (p["sec"] >= t1) & (p["sec"] < t2)
        seg_a = af[gg] - af[gg].mean()                     # cadence check with the spectrum of |a|
        freq = np.fft.rfftfreq(len(seg_a), 1.0 / FS)
        band = (freq > 1.0) & (freq < 3.0)
        f_peak = freq[band][np.argmax(np.abs(np.fft.rfft(seg_a))[band])]
        rows.append({"segment": name, "steps": int(st.sum()),
                     "cadence [steps/s]": round(st.sum() / (t2 - t1), 2),
                     "cadence FFT [steps/s]": round(f_peak, 2),
                     "mean step length [m]": round(L[st].mean(), 2) if st.any() else np.nan,
                     "walking share [%]": round(100 * walking[gg].mean()),
                     "|B| median [uT]": round(m.loc[mm, "B"].median(), 1),
                     "|B| range [uT]": round(m.loc[mm, "B"].max() - m.loc[mm, "B"].min(), 1),
                     "heading mag - gyro, median abs [deg]": round(np.nanmedian(np.abs(mag_minus_gyro[gg])), 1),
                     "height change [m]": round(p.loc[pp, "dh"].mean(), 2),
                     "PDR - phone GNSS at end of segment [m]": round(pdr_diff[st][-1], 1) if st.any() else np.nan})
    table = pd.DataFrame(rows)
    return dict(locals())


def main():
    r = analyse()
    rates, peaks, K, L, step_len_gnss = r["rates"], r["peaks"], r["K"], r["L"], r["step_len_gnss"]
    off_game, off_mag, gb, table, t0 = r["off_game"], r["off_mag"], r["gb"], r["table"], r["t0"]
    t_step, h_step, E, N, gE, gN, pdr_diff = (r[k] for k in ("t_step", "h_step", "E", "N", "gE", "gN", "pdr_diff"))
    print("Sensor rates [Hz]:", rates)
    print(f"Steps detected: {len(peaks)}   Weinberg K = {K:.3f}   PDR distance = {L.sum():.1f} m   "
          f"phone GNSS track length = {step_len_gnss.sum():.1f} m")
    print(f"Heading offsets (GNSS course - sensor yaw): gyro/game {off_game:.1f} deg, magnetic {off_mag:.1f} deg")
    print("Android gyro bias estimate [deg/s]:", np.round(np.degrees(gb), 3),
          f"-> if not removed, heading drift up to {abs(np.degrees(gb[2])) * 60:.1f} deg per minute")
    print("\n" + table.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sec": t_step, "gps_time": t_step + t0, "step_length_m": L, "heading_deg": np.degrees(h_step) % 360,
                  "E_pdr": E, "N_pdr": N, "E_phone": gE, "N_phone": gN, "diff_m": pdr_diff}) \
        .to_csv(OUT_DIR / "imu_steps_pdr.csv", index=False, float_format="%.3f")
    table.to_csv(OUT_DIR / "imu_summary.csv", index=False)
    plots(*(r[k] for k in ("grid", "amag", "af", "peaks", "walking", "s", "head_gyro", "head_mag", "t_course",
                            "course", "moving", "m", "p", "g", "E", "N", "t_step", "spp_res", "pdr_diff")))


def plots(grid, amag, af, peaks, walking, s, head_gyro, head_mag, t_course, course, moving, m, p, g, E, N, t_step,
          spp_res, pdr_diff):
    if not (SHOW_PLOTS or SAVE_FIGURES):
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(11, 12))
    ax[0].plot(grid, amag, color="0.75", lw=0.5, label="|a| raw")
    ax[0].plot(grid, af, color="tab:blue", lw=1, label="|a| smoothed")
    ax[0].plot(grid[peaks], af[peaks], "rv", ms=4, label=f"steps ({len(peaks)})")
    ax[0].set_ylabel("acceleration\n[m/s²]")
    gy = s["UncalGyro"]
    for c, col in zip("XYZ", ("tab:red", "tab:green", "tab:blue")):
        ax[1].plot(gy["sec"], np.degrees(gy[f"UncalGyro{c}RadPerSec"]), lw=0.5, color=col, label=f"gyro {c}")
    ax[1].set_ylabel("turn rate\n[deg/s]")
    ax[2].plot(grid, head_gyro, ".", ms=1, color="tab:blue", label="heading from gyroscope (integrated)")
    ax[2].plot(grid, head_mag, ".", ms=1, color="tab:orange", label="heading from magnetometer (orientation)")
    ax[2].plot(t_course[moving], course[moving], "k+", ms=5, label="GNSS walking direction")
    ax[2].set_ylabel("heading\n[deg from north]")
    ax[3].plot(m["sec"], m["B"], lw=0.7, color="tab:purple")
    ax[3].set_ylabel("magnetic field\n|B| [µT]")
    ax[4].plot(p["sec"], p["dh"], color="tab:brown")
    ax[4].set_ylabel("height change from\npressure [m]")
    ax[4].set_xlabel("seconds since the first GNSS epoch (GPS time)")
    for a in ax:
        a.axvspan(*BLOCKED, color="tab:red", alpha=0.08, lw=0)
        a.axvspan(*CAL_WINDOW, color="tab:green", alpha=0.06, lw=0)
        a.grid(alpha=0.3)
    for a in ax[:3]:
        a.legend(fontsize=7, loc="upper right", markerscale=3)
    ax[0].set_title("Phone IMU along the walk  (red = GNSS blocked, green = calibration window, open sky)")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "imu_overview.png", dpi=150)

    fig2, (axm, axd) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.1]})
    axm.plot(g["E"], g["N"], "k-", lw=1.5, label="phone's own GNSS track")
    ok = spp_res["reliable_GPS+GLO"].astype(bool)
    axm.plot(spp_res.loc[ok, "E_GPS+GLO"], spp_res.loc[ok, "N_GPS+GLO"], ".", color="tab:blue", ms=4,
             alpha=0.6, label="SPP GPS+GLONASS (reliable)")
    blk = (t_step >= BLOCKED[0]) & (t_step <= BLOCKED[1])
    axm.plot(E, N, "-", color="tab:orange", lw=2, label="PDR (steps + gyro heading)")
    axm.plot(E[blk], N[blk], "-", color="tab:red", lw=3, label="PDR while GNSS was blocked")
    axm.plot(E[0], N[0], "g^", ms=10, label="start")
    axm.set_aspect("equal")
    axm.set_xlabel("East [m]")
    axm.set_ylabel("North [m]")
    axm.grid(alpha=0.3)
    axm.legend(fontsize=8)
    axm.set_title("Track: PDR vs GNSS")
    axd.plot(t_step, pdr_diff, ".-", ms=3, color="tab:orange")
    axd.axvspan(*BLOCKED, color="tab:red", alpha=0.08, lw=0)
    axd.axvspan(*CAL_WINDOW, color="tab:green", alpha=0.06, lw=0)
    axd.set_xlabel("seconds since the first GNSS epoch")
    axd.set_ylabel("distance PDR - phone GNSS track [m]")
    axd.set_title("How far PDR is from the phone GNSS track")
    axd.grid(alpha=0.3)
    fig2.tight_layout()
    if SAVE_FIGURES:
        fig2.savefig(OUT_DIR / "imu_pdr_track.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
