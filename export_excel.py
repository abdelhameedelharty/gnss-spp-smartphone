# -*- coding: utf-8 -*-
"""
export_excel.py - HYMN GNSS coordinates BEFORE and AFTER the time calibration -> Excel
===================================================================================

BEFORE = satellite coordinates and clocks as given in HYMN's raw file all_raw.csv.
         HYMN computed them for (RINEX epoch + 18 s): the RINEX epochs are already
         GPS time, but they were treated as UTC and 18 leap seconds were added.
AFTER  = satellite coordinates and clocks computed here from the broadcast
         ephemeris at the correct transmission time (RINEX epoch - P/c).

The receiver position is computed in the same way for both (weighted least
squares from gnss_spp.py, same pseudoranges, same corrections). Only AFTER uses
data snooping, because with 51 km satellite errors almost every satellite would
be rejected.

Run in PyCharm (Run button). Needs gnss_spp.py in the same folder and openpyxl.
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import gnss_spp as spp

ALL_RAW = spp.HYMN / "data" / "raw" / "gnss" / "all_raw.csv"
OUT_FILE = spp.HERE / "HYMN_GNSS_Coordinates_Before_After.xlsx"
TIME_ERROR_S = 18.0          # HYMN gps_millis = RINEX epoch + 18 s


def gps_text(t):
    return (spp.GPS_EPOCH + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")


def load_hymn_satellites():
    """(RINEX epoch, satellite) -> (HYMN time used [s], satellite XYZ [m], clock [m])."""
    df = pd.read_csv(ALL_RAW, usecols=["gps_millis", "gnss_sv_id", "observation_code",
                                       "x_sv_m", "y_sv_m", "z_sv_m", "b_sv_m"])
    keep = np.zeros(len(df), dtype=bool)
    for s, sig in spp.SIGNALS.items():
        keep |= ((df["gnss_sv_id"].str[0] == s) & (df["observation_code"] == sig)).to_numpy()
    df = df[keep & df["x_sv_m"].notna().to_numpy()]
    table = {}
    for r in df.itertuples(index=False):
        t_used = r.gps_millis / 1000.0
        table[(int(round(t_used - TIME_ERROR_S)), r.gnss_sv_id)] = (
            t_used, np.array([r.x_sv_m, r.y_sv_m, r.z_sv_m]), r.b_sv_m)
    return table


def point_of_epoch(t, ref):
    """HYMN point of a GPS epoch. ref t0/t1 are the UTC windows already converted to GPS (+18 s)."""
    if ref is None:
        return "", None
    hit = ref[(ref["t0"] <= t) & (t <= ref["t1"])]
    if hit.empty:
        return "", None
    p = hit.iloc[0]
    return p["point_id"], np.array([p["X"], p["Y"], p["Z"]])


def describe(prefix, sol, xyz_true):
    """Receiver coordinates and errors of one solution as a dict."""
    keys = ["n_sat", "X [m]", "Y [m]", "Z [m]", "lat [deg]", "lon [deg]", "h [m]",
            "dE [m]", "dN [m]", "dU [m]", "2D error [m]"]
    out = {f"{prefix} {k}": np.nan for k in keys}
    if sol is None:
        return out
    x = sol["x"]
    out.update({f"{prefix} n_sat": len(sol["sats"]),
                f"{prefix} X [m]": x[0], f"{prefix} Y [m]": x[1], f"{prefix} Z [m]": x[2],
                f"{prefix} lat [deg]": math.degrees(sol["lat"]),
                f"{prefix} lon [deg]": math.degrees(sol["lon"]),
                f"{prefix} h [m]": sol["h"]})
    if xyz_true is not None:
        lat, lon, _ = spp.ecef_to_geodetic(xyz_true)
        e, n, u = spp.enu_rotation(lat, lon) @ (x - xyz_true)
        out.update({f"{prefix} dE [m]": e, f"{prefix} dN [m]": n, f"{prefix} dU [m]": u,
                    f"{prefix} 2D error [m]": math.hypot(e, n)})
    return out


def main():
    print("Reading files ...")
    epochs = spp.read_rinex_obs(spp.OBS_FILE)
    nav = spp.Navigation(spp.NAV_FILE)
    ref = spp.load_reference(epochs[0][0], epochs[-1][0])
    hymn = load_hymn_satellites()
    mask = math.radians(spp.ELEV_MASK_DEG)
    use_iono = spp.USE_IONO and nav.alpha is not None and nav.beta is not None

    rx_rows, sv_rows = [], []
    x_after = x_before = None
    print(f"Processing {len(epochs)} epochs ...")
    for t, meas in epochs:
        key, tow = int(round(t)), t % 604800.0
        point, xyz_true = point_of_epoch(t, ref)

        # ---------------- AFTER: broadcast ephemeris at the correct time
        sol_a = spp.solve_epoch(t, meas, nav, x_after)
        if sol_a is not None:
            x_after = sol_a["x"]
        sats_a = {s["sat"]: s for s in spp.satellites_for_epoch(t, meas, nav)}

        # ---------------- BEFORE: HYMN satellite coordinates (time + 18 s)
        sats_b = []
        for sat, (p, cn0) in meas.items():
            h = hymn.get((key, sat))
            if h is not None:
                sats_b.append({"sat": sat, "sys": sat[0], "P": p, "cn0": cn0,
                               "rs": h[1], "dts": h[2] / spp.C})
        sol_b = spp.least_squares(sats_b, np.zeros(3) if x_before is None else x_before, set(), nav, tow)
        if sol_b is None and x_before is not None:
            sol_b = spp.least_squares(sats_b, np.zeros(3), set(), nav, tow)
        if sol_b is not None:
            x_before = sol_b["x"]

        row = {"GPS time (RINEX epoch)": gps_text(t),
               "UTC": spp.gps_to_utc_text(t),
               "point": point,
               "satellites tracked": len(meas),
               "BEFORE satellite time used (GPS)": gps_text(t + TIME_ERROR_S)}
        row.update(describe("BEFORE", sol_b, xyz_true))
        row["AFTER satellite time used (GPS)"] = gps_text(t) + " - P/c"
        row.update(describe("AFTER", sol_a, xyz_true))
        row.update({"AFTER rejected by w-test": " ".join(sol_a["rejected"]) if sol_a else "",
                    "AFTER HDOP": sol_a["hdop"] if sol_a else np.nan,
                    "AFTER PDOP": sol_a["pdop"] if sol_a else np.nan,
                    "AFTER sigma0": sol_a["sigma0"] if sol_a else np.nan})
        rx_rows.append(row)

        # ---------------- one row per satellite
        resid = dict(zip(sol_a["sats"], zip(sol_a["v"], sol_a["w"]))) if sol_a else {}
        for sat in sorted(meas):
            p, cn0 = meas[sat]
            r = {"GPS time (RINEX epoch)": gps_text(t), "point": point, "satellite": sat,
                 "signal": "C" + spp.SIGNALS[sat[0]],
                 "RAW pseudorange [m]": p, "RAW C/N0 [dB-Hz]": cn0}
            h = hymn.get((key, sat))
            if h is not None:
                r.update({"BEFORE satellite time (GPS seconds of week)": h[0] % 604800.0,
                          "BEFORE Xsat [m]": h[1][0], "BEFORE Ysat [m]": h[1][1], "BEFORE Zsat [m]": h[1][2],
                          "BEFORE satellite clock [m]": h[2]})
            a = sats_a.get(sat)
            if a is None:
                r["AFTER status"] = "no ephemeris"
            else:
                r.update({"AFTER satellite time = transmit (GPS seconds of week)": a["t_tx"] % 604800.0,
                          "AFTER Xsat [m]": a["rs"][0], "AFTER Ysat [m]": a["rs"][1], "AFTER Zsat [m]": a["rs"][2],
                          "AFTER satellite clock [m]": spp.C * a["dts"]})
                if h is not None:
                    r["satellite moved: AFTER - BEFORE [m]"] = float(np.linalg.norm(a["rs"] - h[1]))
                if sol_a is not None:
                    lat, lon, hgt = sol_a["lat"], sol_a["lon"], sol_a["h"]
                    d = a["rs"] - sol_a["x"]
                    e, n, u = spp.enu_rotation(lat, lon) @ (d / np.linalg.norm(d))
                    el, az = math.asin(u), math.atan2(e, n)
                    r.update({"elevation [deg]": math.degrees(el), "azimuth [deg]": math.degrees(az) % 360.0,
                              "iono delay [m]": (spp.klobuchar(tow, lat, lon, az, el, nav.alpha, nav.beta)
                                                 * (spp.F_L1 / spp.FREQ[sat[0]]) ** 2) if use_iono and el > 0 else np.nan,
                              "tropo delay [m]": spp.saastamoinen(lat, hgt, el) if spp.USE_TROPO else np.nan})
                    if sat in resid:
                        r.update({"AFTER status": "used", "AFTER residual [m]": resid[sat][0],
                                  "AFTER w-test": resid[sat][1]})
                    elif sat in sol_a["rejected"]:
                        r["AFTER status"] = "rejected (w-test)"
                    elif el < mask:
                        r["AFTER status"] = f"below {spp.ELEV_MASK_DEG:.0f} deg mask"
                    else:
                        r["AFTER status"] = "not used (only satellite of its system)"
                else:
                    r["AFTER status"] = "no position (too few satellites)"
            sv_rows.append(r)

    rx = pd.DataFrame(rx_rows)
    sv = pd.DataFrame(sv_rows)
    points = summarise(rx, ref)
    write_excel(points, rx, sv)


def summarise(rx, ref):
    out = []
    for pid in ref["point_id"]:
        g = rx[rx["point"] == pid]
        if g.empty:
            continue
        xyz = ref.loc[ref["point_id"] == pid, ["X", "Y", "Z"]].to_numpy(float)[0]
        lat, lon, h = spp.ecef_to_geodetic(xyz)
        b = g.dropna(subset=["BEFORE X [m]"])
        a = g.dropna(subset=["AFTER X [m]"])
        out.append({
            "point": pid, "epochs": len(g),
            "TRUE X [m]": xyz[0], "TRUE Y [m]": xyz[1], "TRUE Z [m]": xyz[2],
            "TRUE lat [deg]": math.degrees(lat), "TRUE lon [deg]": math.degrees(lon), "TRUE h [m] (see README)": h,
            "BEFORE solved epochs": len(b),
            "BEFORE median X [m]": b["BEFORE X [m]"].median(), "BEFORE median Y [m]": b["BEFORE Y [m]"].median(),
            "BEFORE median Z [m]": b["BEFORE Z [m]"].median(),
            "BEFORE median 2D error [m]": b["BEFORE 2D error [m]"].median(),
            "BEFORE median dU [m]": b["BEFORE dU [m]"].median(),
            "AFTER solved epochs": len(a), "AFTER mean satellites used": a["AFTER n_sat"].mean(),
            "AFTER median X [m]": a["AFTER X [m]"].median(), "AFTER median Y [m]": a["AFTER Y [m]"].median(),
            "AFTER median Z [m]": a["AFTER Z [m]"].median(),
            "AFTER median lat [deg]": a["AFTER lat [deg]"].median(),
            "AFTER median lon [deg]": a["AFTER lon [deg]"].median(),
            "AFTER median h [m]": a["AFTER h [m]"].median(),
            "AFTER median 2D error [m]": a["AFTER 2D error [m]"].median(),
            "AFTER 95% 2D error [m]": a["AFTER 2D error [m]"].quantile(0.95) if len(a) else np.nan,
            "AFTER median dU [m] (see README)": a["AFTER dU [m]"].median()})
    return pd.DataFrame(out)


README = [
    "HYMN GNSS - receiver and satellite coordinates BEFORE and AFTER the time calibration",
    "",
    "DATA",
    "Observations: HYMN v1.3, data/raw/gnss/all.24O (NovAtel, RINEX 3.03, 1 Hz, 23 Oct 2024, 15:39:41-16:36:28 GPS time).",
    "Broadcast ephemeris: data/raw/gnss/LEIJ00DEU_R_20242970000_01D_MN.rnx (IGS station Leipzig, mixed navigation file).",
    "HYMN satellite coordinates: data/raw/gnss/all_raw.csv (made by the HYMN authors with gnss_lib_py).",
    "True coordinates: HYMN tachymeter coordinates of the GNSS antenna (data/reference/csv/point_coordinates.csv).",
    "Time windows of the points: data/reference/csv/time_reference.csv (given in local time and in UTC).",
    "",
    "TIME SCALES",
    "GPS time: the time of the satellites and of the RINEX file. It has no leap seconds.",
    "UTC: civil world time. It has leap seconds. Since 1 January 2017: UTC = GPS time - 18 s, and GPS time = UTC + 18 s.",
    "Local time in Germany on 23 Oct 2024 (summer time, CEST): local = UTC + 2 h.",
    "Galileo time = GPS time. BeiDou time = GPS time - 14 s.",
    "Example: RINEX epoch 15:42:18 GPS time = 15:42:00 UTC = 17:42:00 local time.",
    "",
    "THE TIME PROBLEMS IN HYMN",
    "1) Satellites: in all_raw.csv the time 'gps_millis' is always the RINEX epoch + 18 s (checked for all 146,985 rows).",
    "   The RINEX epochs are already GPS time, but they were treated as UTC and 18 s were added (UTC -> GPS) a second time.",
    "   So HYMN computed every satellite position for a time 18 s too late, and also at the reception time instead of the transmission time.",
    "   A GNSS satellite moves about 3 km per second, so the HYMN satellite coordinates are about 51 km wrong (see sheet Satellites).",
    "2) Point windows: the windows are in UTC, but HYMN compared them directly with the GPS-time epochs (no -18 s).",
    "   So the HYMN GNSS data of each point starts 18 s too early. Here the epochs are converted first: UTC = GPS time - 18 s.",
    "",
    "BEFORE AND AFTER",
    "BEFORE = receiver position computed with the HYMN satellite coordinates and clocks (time = RINEX epoch + 18 s).",
    "AFTER  = receiver position computed with satellite coordinates and clocks from the broadcast ephemeris at the correct",
    "         transmission time (time = RINEX epoch - pseudorange / speed of light). This is the time calibration.",
    "Everything else is the same: pseudoranges GPS L1 (C1C), Galileo E1 (C1C), BeiDou B1I (C2I); elevation mask 10 deg;",
    "Klobuchar ionosphere; Saastamoinen troposphere; weights sigma^2 = 0.5^2 + 0.5^2/sin^2(elevation); one receiver clock per system.",
    "Only AFTER uses data snooping (w-test, critical value 3.29). With 51 km satellite errors almost all satellites would be rejected.",
    "",
    "BEFORE THE EPHEMERIS PROCESSING",
    "Raw GNSS data has no coordinates. It has only distances to the satellites (pseudoranges) and signal strength (C/N0).",
    "See the RAW columns in sheet Satellites. The satellite coordinates come from the ephemeris; only then the receiver position can be computed.",
    "",
    "THE TIME CALIBRATION DONE HERE",
    "a) Satellite time: the RINEX epoch is used as GPS time (no conversion), and each satellite is computed at its",
    "   transmission time: t_transmit = t_RINEX - pseudorange / c - satellite clock error. BeiDou: GPS time - 14 s.",
    "b) Matching with the points: GPS time - 18 s = UTC, then compared with the UTC windows of time_reference.csv.",
    "",
    "COORDINATES AND ERRORS",
    "X, Y, Z: Earth-centred coordinates (WGS84). lat, lon, h: WGS84 latitude, longitude, ellipsoidal height.",
    "dE, dN, dU = computed - true, in the local East / North / Up directions at the true point. 2D error = sqrt(dE^2 + dN^2).",
    "HEIGHT WARNING: the HYMN true heights (about 85 m) look like heights above sea level, not ellipsoidal heights.",
    "The GNSS height is about 125 m. So dU contains about +40 m (the geoid height in this area). Use the 2D error.",
    "",
    "SHEETS",
    "Points_summary  - one row per HYMN point: true coordinates, BEFORE and AFTER coordinates (median) and errors.",
    "Receiver_epochs - one row per second (3,406 epochs): BEFORE and AFTER receiver coordinates and errors.",
    "Satellites      - one row per satellite and second: raw data, BEFORE and AFTER satellite coordinates, elevation,",
    "                  atmosphere delays, and whether the satellite was used in the AFTER solution.",
    "Empty cells: no value (for example, no position because fewer than 4 satellites were visible).",
    "Epochs without a point name were recorded while the receiver was carried between two points.",
    "",
    "Made with gnss_spp.py and export_excel.py (folder GNSS_SPP). AI-assisted processing; check before publishing.",
]


def write_excel(points, rx, sv):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    def rounded(df):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(8 if "[deg]" in c and ("lat" in c or "lon" in c) else 6 if "seconds" in c else 3)
        return df

    fills = {"BEFORE": "F8CBAD", "AFTER": "C6E0B4", "RAW": "BDD7EE", "TRUE": "FFE699"}
    target = OUT_FILE
    try:
        writer = pd.ExcelWriter(target, engine="openpyxl")
    except PermissionError:
        target = OUT_FILE.with_name(OUT_FILE.stem + datetime.now().strftime("_%H%M%S") + ".xlsx")
        writer = pd.ExcelWriter(target, engine="openpyxl")
    with writer as xw:
        pd.DataFrame({"README": README}).to_excel(xw, sheet_name="README", index=False, header=False)
        rounded(points).to_excel(xw, sheet_name="Points_summary", index=False)
        rounded(rx).to_excel(xw, sheet_name="Receiver_epochs", index=False)
        rounded(sv).to_excel(xw, sheet_name="Satellites", index=False)

        ws = xw.book["README"]
        ws.column_dimensions["A"].width = 150
        for cell in ws["A"]:
            if cell.value and (cell.row == 1 or str(cell.value).isupper()):
                cell.font = Font(bold=True, size=14 if cell.row == 1 else 11)
        for name in ("Points_summary", "Receiver_epochs", "Satellites"):
            ws = xw.book[name]
            ws.freeze_panes = "B2" if name == "Points_summary" else "D2"
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col)
                text = str(cell.value)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                for word, color in fills.items():
                    if text.startswith(word):
                        cell.fill = PatternFill("solid", fgColor=color)
                ws.column_dimensions[get_column_letter(col)].width = max(11, min(22, len(text) * 0.6 + 4))
            ws.row_dimensions[1].height = 48
    print(f"Saved: {target}")
    print("\nPoints summary (2D errors in metres):")
    print(points[["point", "BEFORE median 2D error [m]", "AFTER median 2D error [m]",
                  "AFTER solved epochs"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
