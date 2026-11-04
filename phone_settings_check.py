# -*- coding: utf-8 -*-
"""
phone_settings_check.py - check what the phone settings did to the GNSS measurements
===================================================================================

For each GnssLogger recording (RINEX .26o + text log .txt) it checks:
  1. Is there 'Raw' data in the txt file? (GnssLogger setting "measurements" on/off)
  2. Full tracking: the chip writes IsFullTracking (1 = full tracking) in every Raw line
  3. Carrier phase: AccumulatedDeltaRangeState (bit 1 = valid; 16 alone = not valid)
  4. Epoch timing: interval between epochs, gaps, changes of the fraction of the second
  5. Satellites per epoch for each system and signal (RINEX), and in the Raw lines,
     to see when the chip switches GLONASS or BeiDou B1I on and off
Output: Phone_Settings_Check.xlsx (+ a figure in compare_output/).

Run in PyCharm (Run button). Change only the SETTINGS block.
Needs numpy, pandas, matplotlib, openpyxl, and compare_areas.py / gnss_spp.py in the same folder.
"""
from datetime import datetime

import numpy as np
import pandas as pd

import gnss_spp as spp
import compare_areas as ca

# ================================ SETTINGS ===================================
DATA = spp.HERE / "phone_data"
# name: (RINEX file, txt file, what you set on the phone)
LOGS = {
    "Test 1 (17:04)": ("gnss_log_2026_10_06_17_04_52.26o", "gnss_log_2026_10_06_17_04_51.txt", "not noted"),
    "Area A (22:04)": ("gnss_log_2026_10_06_22_04_41.26o", "gnss_log_2026_10_06_22_04_41.txt",
                       "Force full GNSS measurements ON, raw measurements OFF"),
    "Area B (22:07)": ("gnss_log_2026_10_06_22_07_13.26o", "gnss_log_2026_10_06_22_07_13.txt",
                       "Force full GNSS measurements OFF, raw measurements ON"),
}
OUT_XLSX = spp.HERE / "Phone_Settings_Check.xlsx"
OUT_FIG = spp.HERE / "compare_output" / "phone_settings_check.png"
LOCAL_OFFSET_H = 8
SHOW_PLOTS = True
# =============================================================================

CONST = {1: "G", 3: "R", 4: "J", 5: "C", 6: "E"}
SIGNAL_NAME = {("G", "1C"): "GPS L1", ("G", "5Q"): "GPS L5", ("R", "1C"): "GLONASS G1",
               ("E", "1C"): "Galileo E1", ("E", "5Q"): "Galileo E5a", ("C", "2I"): "BeiDou B1I",
               ("C", "1P"): "BeiDou B1C", ("C", "5P"): "BeiDou B2a", ("J", "1C"): "QZSS L1", ("J", "5Q"): "QZSS L5"}


def read_raw(path):
    """'Raw' lines of the GnssLogger txt file (empty DataFrame if there are none)."""
    head, rows, kinds = None, [], {}
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("# Raw,"):
                head = line[2:].strip().split(",")
            elif line[:1].isalpha():
                k = line.split(",", 1)[0]
                kinds[k] = kinds.get(k, 0) + 1
                if k == "Raw":
                    rows.append(line.strip().split(",")[:len(head)])
    if not rows:
        return pd.DataFrame(), kinds
    raw = pd.DataFrame(rows, columns=head[:len(rows[0])])
    for c in ["utcTimeMillis", "ConstellationType", "Svid", "CarrierFrequencyHz", "IsFullTracking",
              "AccumulatedDeltaRangeState", "HardwareClockDiscontinuityCount", "State"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["gps_time"] = raw["utcTimeMillis"] / 1000.0 - spp.GPS_EPOCH_UNIX + spp.LEAP_SECONDS   # UTC -> GPS
    return raw, kinds


def check(label, obs_name, txt_name, settings):
    sig, _ = ca.read_rinex_all(DATA / obs_name)
    raw, kinds = read_raw(DATA / txt_name)

    t = np.sort(sig["t"].unique())
    dt = np.diff(t)
    summary = {"recording": label, "what you set": settings,
               "Raw lines in txt": kinds.get("Raw", 0), "epochs (RINEX)": len(t),
               "epoch interval median [s]": round(float(np.median(dt)), 3),
               "gaps > 1.5 s": int((dt > 1.5).sum()), "largest gap [s]": round(float(dt.max()), 3),
               "different fractions of the second": len(set(np.round(t % 1, 4)))}
    if len(raw):
        summary.update({
            "IsFullTracking = 1 [% of Raw lines]": round(100 * (raw["IsFullTracking"] == 1).mean(), 1),
            "carrier phase valid [% of Raw lines]": round(100 * ((raw["AccumulatedDeltaRangeState"].astype(int) & 1) > 0).mean(), 1),
            "clock discontinuities": int(raw["HardwareClockDiscontinuityCount"].nunique() - 1)})
    else:
        summary.update({"IsFullTracking = 1 [% of Raw lines]": "no Raw lines",
                        "carrier phase valid [% of Raw lines]": "no Raw lines", "clock discontinuities": "no Raw lines"})

    # satellites per epoch for each signal (RINEX) and GLONASS / BeiDou B1I in the Raw lines
    counts = sig.groupby(["t", "sys", "sig"]).size().unstack(["sys", "sig"], fill_value=0)
    counts.columns = [SIGNAL_NAME.get(c, f"{c[0]} {c[1]}") for c in counts.columns]
    counts = counts.reindex(t, fill_value=0)
    if len(raw):
        rt = raw.groupby("utcTimeMillis")["gps_time"].first()
        glo = raw[raw["ConstellationType"] == 3].groupby("utcTimeMillis").size().reindex(rt.index, fill_value=0)
        b1i = raw[(raw["ConstellationType"] == 5) & raw["CarrierFrequencyHz"].between(1.560e9, 1.562e9)] \
            .groupby("utcTimeMillis").size().reindex(rt.index, fill_value=0)
        counts["Raw: GLONASS"] = np.interp(t, rt.to_numpy(), glo.to_numpy()).round()
        counts["Raw: BeiDou B1I"] = np.interp(t, rt.to_numpy(), b1i.to_numpy()).round()
    glo_on = counts.get("GLONASS G1", pd.Series(0, index=t)) > 0
    switches = np.flatnonzero(np.diff(glo_on.astype(int).to_numpy()) != 0) + 1
    summary["GLONASS switched on/off (times)"] = len(switches)
    summary["GLONASS switch at [s since start]"] = ", ".join(f"{t[k] - t[0]:.0f}" for k in switches) or "-"
    counts = counts.reset_index(drop=True)
    table = pd.concat([ca.time_columns(t, t[0]), counts], axis=1)
    return summary, table


def main():
    ca.LOCAL_OFFSET_H = LOCAL_OFFSET_H
    summaries, tables = [], {}
    for label, (obs, txt, settings) in LOGS.items():
        s, tab = check(label, obs, txt, settings)
        summaries.append(s)
        tables[label] = tab
        print(f"{label}: done")
    summary = pd.DataFrame(summaries).set_index("recording").T.reset_index().rename(columns={"index": "check"})
    print("\n" + summary.to_string(index=False))

    readme = pd.DataFrame({"README": [
        "PHONE SETTINGS CHECK - what the settings did to the GNSS measurements",
        "",
        "Summary: one column per recording. 'IsFullTracking = 1' is written by the chip in every Raw line.",
        "Carrier phase valid: AccumulatedDeltaRangeState bit 1. The value 16 alone means 'not valid'.",
        "Different fractions of the second: 1 = regular epochs; 2 or more = the measurement timing was re-aligned.",
        "Counts_...: satellites per epoch for each system and signal (RINEX); 'Raw:' columns come from the txt file.",
        "If GLONASS goes to 0 in both the RINEX and the Raw columns, the chip itself stopped reporting it.",
        "Time: GPS time (RINEX), UTC = GPS - 18 s, local = UTC + 8 h.",
        "Made with phone_settings_check.py (folder GNSS_SPP)."]})
    target = OUT_XLSX
    try:
        writer = pd.ExcelWriter(target, engine="openpyxl")
    except PermissionError:
        target = OUT_XLSX.with_name(OUT_XLSX.stem + datetime.now().strftime("_%H%M%S") + ".xlsx")
        writer = pd.ExcelWriter(target, engine="openpyxl")
    with writer as xw:
        readme.to_excel(xw, sheet_name="README", index=False, header=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
        for label, tab in tables.items():
            tab.to_excel(xw, sheet_name=("Counts_" + label.split(" (")[0].replace(" ", "_"))[:31], index=False)
        xw.book["README"].column_dimensions["A"].width = 120
        ws = xw.book["Summary"]
        ws.column_dimensions["A"].width = 40
        for col in "BCD":
            ws.column_dimensions[col].width = 34
    print(f"\nSaved: {target}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(tables), 1, sharex=True, figsize=(11, 3 * len(tables)))
    for a, (label, tab) in zip(np.atleast_1d(ax), tables.items()):
        x = tab["seconds since start"]
        for col, style in (("GLONASS G1", "r-"), ("BeiDou B1I", "-"), ("BeiDou B1C", "--"), ("BeiDou B2a", ":"),
                           ("GPS L1", "b-"), ("Galileo E1", "g-")):
            if col in tab:
                a.plot(x, tab[col], style, label=col, color=None if style[0] in "-:" else None)
        a.set_title(f"{label}: {LOGS[label][2]}", fontsize=9)
        a.set_ylabel("satellites")
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=6, loc="lower left")
    np.atleast_1d(ax)[-1].set_xlabel("seconds since start")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    print(f"Figure: {OUT_FIG}")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
