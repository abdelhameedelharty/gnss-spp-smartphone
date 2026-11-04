# GNSS single point positioning and smartphone GNSS/IMU analysis

Python tools that compute GNSS single point positions (SPP) from RINEX files. They also analyse raw GNSS and motion-sensor logs from an Android phone (Google GnssLogger) for indoor–outdoor walks. The repository contains the code, the phone logs it was run on (October 2026) and the results.

## Scripts

| Script | What it does |
|---|---|
| `gnss_spp.py` | SPP from a RINEX 3/4 observation file and broadcast ephemeris. Supports GPS, Galileo, BeiDou (Kepler orbits) and GLONASS (Runge-Kutta orbit integration). Uses weighted least squares with Klobuchar ionosphere and a troposphere model, and Baarda data snooping (w-test). Outputs CSV files and figures. |
| `phone_spp.py` | Runs `gnss_spp.py` on a GnssLogger RINEX file (GPS only, and GPS + GLONASS). Compares the result with the phone's own fix. |
| `phone_imu.py` | Analyses the phone sensors: walking/standing detection, step detection, Weinberg step length (calibrated with GNSS in open sky), gyro heading, simple PDR, barometer height. |
| `phone_export_excel.py` | Puts the phone GNSS and IMU results in one Excel file, all on one time axis (GPS time, UTC, local time). |
| `SensorLog/phone_export_excel.py` | The newer version for the 16 Oct 2026 log. It uses all systems (GPS, GLONASS, Galileo, BeiDou, QZSS) and compares them with GPS + GLONASS and GPS only. It also finds open-sky / partly blocked / blocked segments from the signal strength, and summarises the IMU per segment. |
| `compare_areas.py` | Compares two recordings made in two areas. Looks at tracked signals, C/N0 and elevation, sky plot and DOP, loss of lock, code-minus-Doppler noise, and positions. |
| `phone_settings_check.py` | Checks what the GnssLogger settings did to the measurements: raw data, full tracking, carrier-phase validity, epoch timing, satellites per system. |
| `export_excel.py` | Exports HYMN dataset GNSS coordinates before and after a time-tag correction (the 18 s GPS/UTC offset) to Excel. |

Time conventions used everywhere: the RINEX epochs are GPS time; GPS = UTC + 18 s; local time = UTC + 8 h.

## Data and results

- `phone_data/`: GnssLogger logs (`.txt`) and RINEX observation files (`.26o`) from 6 Oct 2026. Also the broadcast navigation files of that day (`brdc*.26n/g.gz`, `BRDM…rnx.gz`).
- `SensorLog/`: the 16 Oct 2026 walk (log, RINEX, merged multi-GNSS broadcast file, IMU log) with its Excel results and graph reports (PDF).
- `output/`, `phone_output/`, `compare_output/`, `analysis_2026-10-06_22-09-55/`: CSV results and figures.
- `*.xlsx` in the top folder: Excel results of the scripts above.

The phone was a Samsung Galaxy S24 Ultra (Qualcomm GNSS), logged with Google GnssLogger v3.1.

`HYMN_GNSS_Coordinates_Before_After.xlsx` is derived from the HYMN dataset: Ammad et al., *Descriptor: A Hybrid Indoor and Indoor–Outdoor Positioning Multitechnology Dataset (HYMN)*, IEEE Data Descriptions 3 (2026) 379–387, doi:10.1109/IEEEDATA.2026.3691044. The dataset itself is not included.

## Requirements

Python 3 with `numpy`, `pandas`, `scipy`, `matplotlib` and `openpyxl`:

```bash
pip install numpy pandas scipy matplotlib openpyxl
```
