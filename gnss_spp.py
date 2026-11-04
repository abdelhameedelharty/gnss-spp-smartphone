# -*- coding: utf-8 -*-
"""
gnss_spp.py - GNSS Single Point Positioning (SPP) from a RINEX observation file
==============================================================================

Input : RINEX 3 or 4 observation file (pseudoranges)
        RINEX 2 or 3 navigation file(s) (broadcast ephemeris + Klobuchar parameters);
        gzip files (.gz) can be used directly
Output: one position per epoch (X, Y, Z and latitude, longitude, height),
        quality numbers (satellites, DOP, residuals), a CSV file and figures.

Systems: GPS (G), Galileo (E), BeiDou (C) with Kepler orbits, and
GLONASS (R) with numerical orbit integration (Runge-Kutta). Choose with SYSTEMS.

ALGORITHM (for every epoch)
---------------------------
 1. Take the code pseudorange P of each satellite
    (GPS C1C, Galileo C1C, BeiDou C2I).
 2. Signal transmission time:   t_tx = t_rx - P/c - dt_sat
 3. Satellite position at t_tx from the broadcast ephemeris
    (Kepler orbit + harmonic corrections; special rotation for BeiDou GEO).
 4. Satellite clock error:  dt_sat = a0 + a1*dt + a2*dt^2 + relativity - group delay
 5. Start at the last position (or the Earth centre) and iterate:
      rho = |X_sat - X_rx| + Earth-rotation (Sagnac) term
      elevation, azimuth  -> elevation mask, troposphere (Saastamoinen),
                             ionosphere (Klobuchar)
      l   = P - (rho - c*dt_sat + T + I)              (observed - computed)
      a   = [-(Xs-X)/rho, -(Ys-Y)/rho, -(Zs-Z)/rho, 1 for the clock of this system]
      W   = 1/sigma^2   with  sigma^2 = SIGMA_A^2 + SIGMA_B^2 / sin^2(elevation)
      dx  = (A^T W A)^-1 A^T W l ;  X = X + dx ;  stop when |dx| < 0.1 mm
 6. Quality: DOP, a-posteriori variance factor, residuals.
 7. Optional Baarda data snooping: w_i = v_i / sigma(v_i).
    If max|w| > W_CRIT, remove that satellite and solve again (one at a time).
 8. XYZ -> latitude, longitude, height. If a reference point is known:
    error in East / North / Up.

Run it in PyCharm (green Run button). Change only the SETTINGS block.
Needs: numpy, pandas, matplotlib.
"""
import gzip
import math
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ================================ SETTINGS ===================================
HERE = Path(__file__).resolve().parent
HYMN = HERE.parent / "HYMN-dataset-1.3" / "TUD-ITVS-HYMN-dataset-8ae0f89"

OBS_FILE = HYMN / "data" / "raw" / "gnss" / "all.24O"
NAV_FILE = HYMN / "data" / "raw" / "gnss" / "LEIJ00DEU_R_20242970000_01D_MN.rnx"

# Reference (true) coordinates, used only to compute errors:
#   REF_POINTS + REF_TIMES = HYMN true antenna position of each point and the
#                            UTC time window when the antenna stood on it, or
#   REF_XYZ = (X, Y, Z) in metres for one static point of your own, or
#   all None = no errors are computed.
# Time: the RINEX epochs are GPS time, the HYMN windows are UTC.
#       UTC = GPS - LEAP_SECONDS   (GPS = UTC + LEAP_SECONDS)
# HYMN note: its true heights look like heights above sea level, not
#   ellipsoidal heights, so the Up error contains about +40 m (geoid height).
#   Use the horizontal error.
REF_POINTS = HYMN / "data" / "reference" / "csv" / "point_coordinates.csv"
REF_TIMES = HYMN / "data" / "reference" / "csv" / "time_reference.csv"
REF_XYZ = None

OUT_DIR = HERE / "output"

SYSTEMS = "GEC"                              # G = GPS, E = Galileo, C = BeiDou, R = GLONASS
SIGNALS = {"G": "1C", "E": "1C", "C": "2I", "R": "1C", "J": "1C"}  # code obs = "C"+signal, C/N0 = "S"+signal
ELEV_MASK_DEG = 10.0
CN0_MASK = 0.0               # dB-Hz, do not use weaker signals (0 = off; about 25 for phones)
USE_IONO = True             # Klobuchar model (GPSA/GPSB from the nav-file header)
USE_TROPO = True             # Saastamoinen model, standard atmosphere
SIGMA_A = 0.5                # m   pseudorange sigma^2 = SIGMA_A^2 + SIGMA_B^2/sin^2(elev)
SIGMA_B = 0.5                # m
USE_DATA_SNOOPING = True     # Baarda w-test, removes one outlier at a time
W_CRIT = 3.29                # critical value of the w-test (alpha = 0.001)
MAX_EPH_AGE = 7200.0         # s, use an ephemeris only if |t - toe| < this
MAX_EPH_AGE_GLO = 1800.0     # s, the same for GLONASS (new ephemeris every 30 min)
LEAP_SECONDS = 18            # GPS - UTC (valid since 1 Jan 2017)
SHOW_PLOTS = True
SAVE_FIGURES = True
# =============================================================================

C = 299792458.0                                       # speed of light [m/s]
OMEGA_E = 7.2921151467e-5                             # Earth rotation [rad/s] (Sagnac)
GM = {"G": 3.986005e14, "E": 3.986004418e14, "C": 3.986004418e14, "J": 3.986005e14}
OMEGA_E_SYS = {"G": 7.2921151467e-5, "E": 7.2921151467e-5, "C": 7.2921150e-5, "J": 7.2921151467e-5}
FREQ = {"G": 1575.42e6, "E": 1575.42e6, "C": 1561.098e6, "J": 1575.42e6}   # L1, E1, B1I, QZSS L1 [Hz]
F_L1 = 1575.42e6
BDS_GEO = {1, 2, 3, 4, 5, 59, 60, 61, 62, 63}         # BeiDou GEO satellites
BDT_TO_GPST = 14.0                                    # GPST = BDT + 14 s
BDS_WEEK_0 = 1356                                     # BDS week 0 = GPS week 1356
GPS_EPOCH = datetime(1980, 1, 6)
GPS_EPOCH_UNIX = 315964800                            # 6 Jan 1980 in Unix seconds
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
GLO_MU, GLO_J2, GLO_RE, GLO_OMEGA = 3.9860044e14, 1.0826257e-3, 6378136.0, 7.292115e-5   # PZ-90
GLO_F1, GLO_DF1 = 1602.0e6, 0.5625e6                  # GLONASS L1 = 1602 MHz + k * 0.5625 MHz
GLO_STEP = 60.0                                       # s, Runge-Kutta step
MAX_ITER = 10


# ------------------------------------------------------------------ time ----
def gps_time(year, month, day, hour, minute, second):
    """Calendar date (GPS time scale) -> seconds since 6 Jan 1980."""
    return (datetime(year, month, day, hour, minute) - GPS_EPOCH).total_seconds() + second


def gps_to_utc_text(t):
    return (GPS_EPOCH + timedelta(seconds=t - LEAP_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------- RINEX obs file -----
def read_rinex_obs(path):
    """Read a RINEX 3 observation file.
    Returns a list of (t_gps, {sat: (pseudorange_m, cn0_dbhz)})."""
    obs_types, current = {}, None
    epochs = []
    with _open(path) as f:
        for line in f:                                   # ---- header
            label = line[60:].strip()
            if label == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    current = line[0]
                    obs_types[current] = []
                obs_types[current] += line[7:60].split()
            elif label == "END OF HEADER":
                break
        index = {}                                       # column of code and C/N0
        for s in SYSTEMS:
            types = obs_types.get(s, [])
            code, snr = "C" + SIGNALS[s], "S" + SIGNALS[s]
            if code in types:
                index[s] = (types.index(code), types.index(snr) if snr in types else None)
        if not index:
            raise ValueError("None of the requested signals is in the observation file.")

        for line in f:                                   # ---- data
            if not line.startswith(">"):
                continue
            flag, nsat = int(line[31]), int(line[32:35])
            if flag > 1:                                 # event record: skip its lines
                for _ in range(nsat):
                    next(f)
                continue
            t = gps_time(int(line[2:6]), int(line[7:9]), int(line[10:12]),
                         int(line[13:15]), int(line[16:18]), float(line[18:29]))
            meas = {}
            for _ in range(nsat):
                rec = next(f)
                sat = rec[:3].replace(" ", "0")
                if sat[0] not in index:
                    continue
                i_code, i_snr = index[sat[0]]
                p = _obs_value(rec, i_code)
                if p is None or p < 1.0e7:
                    continue
                cn0 = _obs_value(rec, i_snr) if i_snr is not None else None
                meas[sat] = (p, np.nan if cn0 is None else cn0)
            epochs.append((t, meas))
    return epochs


def _obs_value(rec, i):
    text = rec[3 + 16 * i: 17 + 16 * i]
    try:
        return float(text)
    except ValueError:
        return None


# ------------------------------------------------------- RINEX nav file -----
def _num(text):
    text = text.strip().replace("D", "E").replace("d", "e")
    return float(text) if text else 0.0


def _open(path):
    """Open a text file for reading; gzip-compressed files (.gz) are opened directly."""
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def _kepler_ephemeris(sat, date, v):
    """GPS / Galileo / BeiDou record -> dict (None if unhealthy)."""
    s = sat[0]
    if len(v) < 27:
        return None
    health = int(v[24])
    if s == "J":
        health &= 0xFE                                    # QZSS: bit 0 is not about L1 C/A (as in RTKLIB)
    if health != 0:                                       # unhealthy satellite
        return None
    toc = gps_time(*date)
    e = dict(sat=sat, sys=s, prn=int(sat[1:]),
             af0=v[0], af1=v[1], af2=v[2],
             crs=v[4], dn=v[5], m0=v[6],
             cuc=v[7], ecc=v[8], cus=v[9], sqrta=v[10],
             toe=v[11], cic=v[12], omg0=v[13], cis=v[14],
             i0=v[15], crc=v[16], omg=v[17], omgd=v[18],
             idot=v[19])
    if s == "C":                                          # BeiDou time -> GPS time
        e["toc_abs"] = toc + BDT_TO_GPST
        e["toe_abs"] = (v[21] + BDS_WEEK_0) * 604800.0 + v[11] + BDT_TO_GPST
        e["tgd"] = v[25]                                  # TGD1 (B1I)
    else:
        e["toc_abs"] = toc
        e["toe_abs"] = v[21] * 604800.0 + v[11]
        if s == "E":
            if not int(v[20]) & 512:                      # keep I/NAV (E1,E5b clock) only
                return None
            e["tgd"] = v[26]                              # BGD(E1,E5b)
        else:
            e["tgd"] = v[25]                              # TGD (L1 C/A)
    return e


def _glonass_ephemeris(sat, date, v):
    """GLONASS record -> dict. Its epoch is UTC; position, velocity, acceleration in km."""
    if len(v) < 15 or v[6] != 0:                          # health flag (0 = OK)
        return None
    t = gps_time(*date) + LEAP_SECONDS                    # UTC -> GPS time
    return dict(sat=sat, sys="R", prn=int(sat[1:]),
                toe_abs=t, toc_abs=t, af0=v[0], af1=v[1], af2=0.0, tgd=0.0,   # dt = -TauN + GammaN*(t-tb)
                pos=np.array([v[3], v[7], v[11]]) * 1e3,
                vel=np.array([v[4], v[8], v[12]]) * 1e3,
                acc=np.array([v[5], v[9], v[13]]) * 1e3,
                freq=GLO_F1 + v[10] * GLO_DF1)


class Navigation:
    """Broadcast ephemerides from RINEX 2 or 3 navigation files (also .gz).
    One file or a list of files, e.g. [GPS file, GLONASS file]."""

    def __init__(self, paths):
        self.eph = {}
        self.alpha = self.beta = None
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for path in paths:
            with _open(path) as f:
                lines = f.readlines()
            if float(lines[0][:9]) < 3.0:
                self._read_v2(lines, "R" if lines[0][20] == "G" else "G")   # RINEX 2: N = GPS, G = GLONASS
            else:
                self._read_v3(lines)

    def _add(self, e):
        if e is not None and e["sys"] in SYSTEMS:
            self.eph.setdefault(e["sat"], []).append(e)

    def _read_v3(self, lines):
        records, current, body = [], None, False
        for line in lines:
            if not body:
                label = line[60:].strip()
                if label == "IONOSPHERIC CORR":
                    values = [_num(line[5 + 12 * k: 17 + 12 * k]) for k in range(4)]
                    if line[:4] == "GPSA":
                        self.alpha = values
                    elif line[:4] == "GPSB":
                        self.beta = values
                body = label == "END OF HEADER"
                continue
            if line[:1].strip():                          # a record starts with a letter (G05, R11 ...)
                if current:
                    records.append(current)
                current = [line]
            elif current is not None:
                current.append(line)
        if current:
            records.append(current)
        for rec in records:
            head = rec[0]
            sat = head[:3].replace(" ", "0")
            date = (int(head[4:8]), int(head[9:11]), int(head[12:14]),
                    int(head[15:17]), int(head[18:20]), int(head[21:23]))
            v = [_num(head[23 + 19 * k: 42 + 19 * k]) for k in range(3)]
            for line in rec[1:]:
                v += [_num(line[4 + 19 * k: 23 + 19 * k]) for k in range(4)]
            if sat[0] == "R" and len(rec) >= 4:
                self._add(_glonass_ephemeris(sat, date, v))
            elif sat[0] in "GECJ" and len(rec) >= 8:
                self._add(_kepler_ephemeris(sat, date, v))

    def _read_v2(self, lines, system):
        i = 0
        while i < len(lines):                             # header
            label = lines[i][60:].strip()
            if label in ("ION ALPHA", "ION BETA"):
                values = [_num(lines[i][2 + 12 * k: 14 + 12 * k]) for k in range(4)]
                if label == "ION ALPHA":
                    self.alpha = values
                else:
                    self.beta = values
            i += 1
            if label == "END OF HEADER":
                break
        n_lines = 4 if system == "R" else 8               # lines per record
        while i + n_lines <= len(lines):
            rec, i = lines[i:i + n_lines], i + n_lines
            head = rec[0]
            if not head.strip():
                continue
            yy = int(head[3:5])
            date = (yy + (2000 if yy < 80 else 1900), int(head[6:8]), int(head[9:11]),
                    int(head[12:14]), int(head[15:17]), float(head[17:22]))
            v = [_num(head[22 + 19 * k: 41 + 19 * k]) for k in range(3)]
            for line in rec[1:]:
                v += [_num(line[3 + 19 * k: 22 + 19 * k]) for k in range(4)]
            sat = f"{system}{int(head[0:2]):02d}"
            self._add(_glonass_ephemeris(sat, date, v) if system == "R" else _kepler_ephemeris(sat, date, v))

    def get(self, sat, t):
        """Ephemeris with toe closest to time t (None if too old)."""
        best, best_dt = None, MAX_EPH_AGE_GLO if sat[0] == "R" else MAX_EPH_AGE
        for e in self.eph.get(sat, ()):
            dt = abs(t - e["toe_abs"])
            if dt <= best_dt:
                best, best_dt = e, dt
        return best


# ----------------------------------------------- satellite orbit & clock ----
def clock_polynomial(e, t):
    dt = t - e["toc_abs"]
    return e["af0"] + e["af1"] * dt + e["af2"] * dt * dt


def _glonass_derivative(x, acc):
    """Equations of motion in PZ-90 (Earth-fixed): gravity with J2, Earth rotation, lunisolar acc."""
    r2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    r3 = r2 * math.sqrt(r2)
    a = 1.5 * GLO_J2 * GLO_MU * GLO_RE ** 2 / r2 / r3
    b = 5.0 * x[2] ** 2 / r2
    c = -GLO_MU / r3 - a * (1.0 - b)
    w2 = GLO_OMEGA ** 2
    return np.array([x[3], x[4], x[5],
                     (c + w2) * x[0] + 2.0 * GLO_OMEGA * x[4] + acc[0],
                     (c + w2) * x[1] - 2.0 * GLO_OMEGA * x[3] + acc[1],
                     (c - 2.0 * a) * x[2] + acc[2]])


def glonass_position(e, t):
    """GLONASS satellite position: Runge-Kutta integration from the ephemeris time to t."""
    x = np.r_[e["pos"], e["vel"]]
    dt = t - e["toe_abs"]
    while abs(dt) > 1e-9:
        h = math.copysign(min(GLO_STEP, abs(dt)), dt)
        k1 = _glonass_derivative(x, e["acc"])
        k2 = _glonass_derivative(x + k1 * h / 2, e["acc"])
        k3 = _glonass_derivative(x + k2 * h / 2, e["acc"])
        k4 = _glonass_derivative(x + k3 * h, e["acc"])
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
        dt -= h
    return x[:3]


def satellite_position_clock(e, t):
    """Satellite ECEF position [m] and clock error [s] at GPS time t."""
    s = e["sys"]
    if s == "R":
        return glonass_position(e, t), clock_polynomial(e, t)
    gm, we = GM[s], OMEGA_E_SYS[s]
    tk = t - e["toe_abs"]
    a = e["sqrta"] ** 2
    n = math.sqrt(gm / a ** 3) + e["dn"]
    m = e["m0"] + n * tk
    ecc = e["ecc"]
    E = m
    for _ in range(30):                                   # Kepler: M = E - e sinE
        dE = (m - E + ecc * math.sin(E)) / (1.0 - ecc * math.cos(E))
        E += dE
        if abs(dE) < 1e-14:
            break
    sin_e, cos_e = math.sin(E), math.cos(E)
    nu = math.atan2(math.sqrt(1.0 - ecc * ecc) * sin_e, cos_e - ecc)
    phi = nu + e["omg"]
    s2, c2 = math.sin(2 * phi), math.cos(2 * phi)
    u = phi + e["cus"] * s2 + e["cuc"] * c2
    r = a * (1.0 - ecc * cos_e) + e["crs"] * s2 + e["crc"] * c2
    i = e["i0"] + e["idot"] * tk + e["cis"] * s2 + e["cic"] * c2
    xp, yp = r * math.cos(u), r * math.sin(u)
    cos_i, sin_i = math.cos(i), math.sin(i)

    if s == "C" and e["prn"] in BDS_GEO:                  # BeiDou GEO (ICD formula)
        om = e["omg0"] + e["omgd"] * tk - we * e["toe"]
        xg = xp * math.cos(om) - yp * cos_i * math.sin(om)
        yg = xp * math.sin(om) + yp * cos_i * math.cos(om)
        zg = yp * sin_i
        c5, s5 = math.cos(math.radians(-5.0)), math.sin(math.radians(-5.0))
        y1, z1 = c5 * yg + s5 * zg, -s5 * yg + c5 * zg    # R_X(-5 deg)
        co, so = math.cos(we * tk), math.sin(we * tk)     # R_Z(we * tk)
        pos = np.array([co * xg + so * y1, -so * xg + co * y1, z1])
    else:
        om = e["omg0"] + (e["omgd"] - we) * tk - we * e["toe"]
        pos = np.array([xp * math.cos(om) - yp * cos_i * math.sin(om),
                        xp * math.sin(om) + yp * cos_i * math.cos(om),
                        yp * sin_i])

    relativity = -2.0 * math.sqrt(gm) / C ** 2 * ecc * e["sqrta"] * sin_e
    dts = clock_polynomial(e, t) + relativity - e["tgd"]
    return pos, dts


# ------------------------------------------------------------ geodesy ------
def ecef_to_geodetic(x):
    """ECEF [m] -> latitude, longitude [rad], ellipsoidal height [m] (WGS84)."""
    p = math.hypot(x[0], x[1])
    lon = math.atan2(x[1], x[0])
    lat = math.atan2(x[2], p * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(10):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        new = math.atan2(x[2], p * (1.0 - WGS84_E2 * n / (n + h)))
        if abs(new - lat) < 1e-12:
            lat = new
            break
        lat = new
    return lat, lon, h


def enu_rotation(lat, lon):
    """Rotation matrix ECEF -> East, North, Up."""
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


# ------------------------------------------------- atmosphere models -------
def klobuchar(tow, lat, lon, az, el, alpha, beta):
    """GPS broadcast ionosphere model. Returns the L1 delay [m]."""
    psi = 0.0137 / (el / math.pi + 0.11) - 0.022
    phi = lat / math.pi + psi * math.cos(az)
    phi = max(-0.416, min(0.416, phi))
    lam = lon / math.pi + psi * math.sin(az) / math.cos(phi * math.pi)
    phi += 0.064 * math.cos((lam - 1.617) * math.pi)      # geomagnetic latitude
    tt = (43200.0 * lam + tow) % 86400.0                  # local time
    f = 1.0 + 16.0 * (0.53 - el / math.pi) ** 3           # slant factor
    amp = max(0.0, alpha[0] + phi * (alpha[1] + phi * (alpha[2] + phi * alpha[3])))
    per = max(72000.0, beta[0] + phi * (beta[1] + phi * (beta[2] + phi * beta[3])))
    x = 2.0 * math.pi * (tt - 50400.0) / per
    if abs(x) < 1.57:
        return C * f * (5e-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0)))
    return C * f * 5e-9


def saastamoinen(lat, h, el, humidity=0.7):
    """Tropospheric delay [m] with a standard atmosphere."""
    if h < -100.0 or h > 1.0e4 or el <= 0.0:
        return 0.0
    hgt = max(h, 0.0)
    pres = 1013.25 * (1.0 - 2.2557e-5 * hgt) ** 5.2568
    temp = 15.0 - 6.5e-3 * hgt + 273.16
    e = 6.108 * humidity * math.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    z = math.pi / 2.0 - el
    dry = 0.0022768 * pres / (1.0 - 0.00266 * math.cos(2.0 * lat) - 0.00028 * hgt / 1e3) / math.cos(z)
    wet = 0.002277 * (1255.0 / temp + 0.05) * e / math.cos(z)
    return dry + wet


# ------------------------------------------------------- positioning -------
def satellites_for_epoch(t_rx, meas, nav):
    """Satellite position and clock (at transmission time) for each pseudorange."""
    sats = []
    for sat, (p, cn0) in meas.items():
        if CN0_MASK > 0 and not cn0 >= CN0_MASK:          # weak (or unknown) signal strength
            continue
        e = nav.get(sat, t_rx)
        if e is None:
            continue
        t_tx = t_rx - p / C
        t_tx -= clock_polynomial(e, t_tx)
        rs, dts = satellite_position_clock(e, t_tx)
        sats.append({"sat": sat, "sys": sat[0], "P": p, "cn0": cn0, "rs": rs, "dts": dts, "t_tx": t_tx,
                     "freq": e.get("freq", FREQ.get(sat[0], F_L1))})
    return sats


def least_squares(sats, x0, excluded, nav, tow):
    """Iterative weighted least squares. Returns a solution dict or None."""
    x = np.array(x0, dtype=float)
    mask = math.radians(ELEV_MASK_DEG)
    use_iono = USE_IONO and nav.alpha is not None and nav.beta is not None
    for _ in range(MAX_ITER):
        have_pos = np.linalg.norm(x) > 6.0e6
        if have_pos:
            lat, lon, h = ecef_to_geodetic(x)
            rot = enu_rotation(lat, lon)
        rows = []
        for s in sats:
            if s["sat"] in excluded:
                continue
            d = s["rs"] - x
            r = np.linalg.norm(d)
            unit = d / r
            el = az = np.nan
            iono = tropo = 0.0
            var = SIGMA_A ** 2 + SIGMA_B ** 2
            if have_pos:
                east, north, up = rot @ unit
                el, az = math.asin(up), math.atan2(east, north)
                if el < mask:
                    continue
                if use_iono:
                    iono = klobuchar(tow, lat, lon, az, el, nav.alpha, nav.beta) * (F_L1 / s.get("freq", FREQ.get(s["sys"], F_L1))) ** 2
                if USE_TROPO:
                    tropo = saastamoinen(lat, h, el)
                var = SIGMA_A ** 2 + (SIGMA_B / math.sin(el)) ** 2
            rho = r + OMEGA_E * (s["rs"][0] * x[1] - s["rs"][1] * x[0]) / C
            rows.append((s, -unit, s["P"] - (rho - C * s["dts"] + iono + tropo), var, el, az))

        count = Counter(row[0]["sys"] for row in rows)
        if len(count) > 1:                    # a system with 1 satellite adds no information
            rows = [row for row in rows if count[row[0]["sys"]] >= 2]
        systems = sorted({row[0]["sys"] for row in rows})
        n, m = len(rows), 3 + len(systems)
        if n < m:
            return None
        A, l, w = np.zeros((n, m)), np.zeros(n), np.zeros(n)
        for k, (s, a, lk, var, _, _) in enumerate(rows):
            A[k, :3] = a
            A[k, 3 + systems.index(s["sys"])] = 1.0
            l[k], w[k] = lk, 1.0 / var
        try:
            qx = np.linalg.inv(A.T @ (A * w[:, None]))
        except np.linalg.LinAlgError:
            return None
        est = qx @ (A.T @ (w * l))
        x = x + est[:3]
        if np.linalg.norm(est[:3]) < 1e-4 and have_pos:
            break
    else:
        return None                                       # did not converge

    v = l - A @ est                                       # residuals
    qv = 1.0 / w - np.einsum("ij,jk,ik->i", A, qx, A)     # cofactor of residuals
    wtest = np.where(qv > 1e-10, v / np.sqrt(np.clip(qv, 1e-10, None)), 0.0)
    redundancy = n - m
    lat, lon, h = ecef_to_geodetic(x)
    rot = enu_rotation(lat, lon)
    try:
        q_enu = rot @ np.linalg.inv(A.T @ A)[:3, :3] @ rot.T
        pdop = math.sqrt(np.trace(q_enu))
        hdop, vdop = math.sqrt(q_enu[0, 0] + q_enu[1, 1]), math.sqrt(q_enu[2, 2])
    except (np.linalg.LinAlgError, ValueError):
        pdop = hdop = vdop = np.nan
    return {"x": x, "lat": lat, "lon": lon, "h": h,
            "sats": [row[0]["sat"] for row in rows],
            "cn0": [row[0]["cn0"] for row in rows],
            "el": [row[4] for row in rows],
            "v": v, "w": wtest, "redundancy": redundancy,
            "sigma0": math.sqrt(float(w @ (v * v)) / redundancy) if redundancy > 0 else np.nan,
            "clock_m": dict(zip(systems, est[3:])),
            "pdop": pdop, "hdop": hdop, "vdop": vdop}


def solve_epoch(t_rx, meas, nav, x0):
    sats = satellites_for_epoch(t_rx, meas, nav)
    tow = t_rx % 604800.0
    start = np.zeros(3) if x0 is None else x0
    sol = least_squares(sats, start, set(), nav, tow)
    if sol is None and x0 is not None:                    # retry from the Earth centre
        sol = least_squares(sats, np.zeros(3), set(), nav, tow)
    rejected = []
    while USE_DATA_SNOOPING and sol is not None and sol["redundancy"] >= 2:
        k = int(np.argmax(np.abs(sol["w"])))
        if abs(sol["w"][k]) <= W_CRIT:
            break
        rejected.append(sol["sats"][k])
        new = least_squares(sats, sol["x"], set(rejected), nav, tow)
        if new is None:
            rejected.pop()
            break
        sol = new
    if sol is not None:
        sol["rejected"] = rejected
        if not -1000.0 < sol["h"] < 20000.0:              # clearly wrong solution
            return None
    return sol


# ---------------------------------------------------------- reference ------
def utc_text_to_gps(text):
    """'2024-10-23 15:42:00' (UTC) -> GPS seconds since 6 Jan 1980."""
    t = datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S")
    return (t - GPS_EPOCH).total_seconds() + LEAP_SECONDS


def load_reference(t_first=None, t_last=None):
    """True ECEF position of each HYMN point and its time window in GPS seconds (t0, t1)."""
    if REF_POINTS is None or REF_TIMES is None or not Path(REF_POINTS).exists() or not Path(REF_TIMES).exists():
        return None
    pts = pd.read_csv(REF_POINTS, usecols=["point_id", "X_ECEF_GNSS", "Y_ECEF_GNSS", "Z_ECEF_GNSS"])
    times = pd.read_csv(REF_TIMES).rename(columns={"point": "point_id"})
    times["t0"] = times["start_time_UTC"].map(utc_text_to_gps)     # UTC -> GPS: + leap seconds
    times["t1"] = times["end_time_UTC"].map(utc_text_to_gps)
    ref = times[["point_id", "t0", "t1"]].merge(pts, on="point_id")
    ref = ref.rename(columns={"X_ECEF_GNSS": "X", "Y_ECEF_GNSS": "Y", "Z_ECEF_GNSS": "Z"})
    if t_first is not None:
        ref = ref[(ref["t1"] >= t_first) & (ref["t0"] <= t_last)]
    return ref.sort_values("t0").reset_index(drop=True)


# --------------------------------------------------------------- main ------
def main():
    obs_file = Path(sys.argv[1]) if len(sys.argv) > 1 else OBS_FILE
    nav_file = Path(sys.argv[2]) if len(sys.argv) > 2 else NAV_FILE
    print(f"Observation file : {obs_file}")
    print(f"Navigation file  : {nav_file}")
    epochs = read_rinex_obs(obs_file)
    nav = Navigation(nav_file)
    print(f"Epochs read      : {len(epochs)}")
    print(f"Ephemerides      : {sum(len(v) for v in nav.eph.values())} for {len(nav.eph)} satellites")
    print(f"Klobuchar        : {'yes' if nav.alpha and nav.beta else 'NOT in header - no iono correction'}")

    rows, x_prev = [], None
    for t, meas in epochs:
        sol = solve_epoch(t, meas, nav, x_prev)
        row = {"utc": gps_to_utc_text(t), "gps_week": int(t // 604800), "tow": t % 604800.0,
               "n_tracked": len(meas)}
        if sol is None:
            row["status"] = "no solution"
        else:
            x_prev = sol["x"]
            n_sys = Counter(s[0] for s in sol["sats"])
            row.update(status="ok", X=sol["x"][0], Y=sol["x"][1], Z=sol["x"][2],
                       lat_deg=math.degrees(sol["lat"]), lon_deg=math.degrees(sol["lon"]), h=sol["h"],
                       n_sat=len(sol["sats"]), n_G=n_sys.get("G", 0), n_E=n_sys.get("E", 0),
                       n_C=n_sys.get("C", 0), n_R=n_sys.get("R", 0), n_rejected=len(sol["rejected"]),
                       rejected=" ".join(sol["rejected"]),
                       pdop=sol["pdop"], hdop=sol["hdop"], vdop=sol["vdop"], sigma0=sol["sigma0"],
                       cn0_mean=float(np.nanmean(sol["cn0"])) if len(sol["cn0"]) else np.nan,
                       max_abs_w=float(np.max(np.abs(sol["w"]))) if len(sol["w"]) else np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["gps_time"] = df["gps_week"] * 604800.0 + df["tow"]

    # ---------------- errors with respect to the reference
    ref = load_reference(df["gps_time"].iloc[0], df["gps_time"].iloc[-1])
    df["point_id"] = ""
    if ref is not None or REF_XYZ is not None:
        ref_xyz = np.full((len(df), 3), np.nan)
        if ref is not None:
            for _, p in ref.iterrows():
                inside = ((df["gps_time"] >= p["t0"]) & (df["gps_time"] <= p["t1"])).to_numpy()
                df.loc[inside, "point_id"] = p["point_id"]
                ref_xyz[inside] = [p["X"], p["Y"], p["Z"]]
        else:
            df["point_id"] = "REF"
            ref_xyz[:] = REF_XYZ
        ok = (df["status"] == "ok").to_numpy() & ~np.isnan(ref_xyz[:, 0])
        enu = np.full((len(df), 3), np.nan)
        for k in np.flatnonzero(ok):
            lat, lon, _ = ecef_to_geodetic(ref_xyz[k])
            enu[k] = enu_rotation(lat, lon) @ (df.loc[k, ["X", "Y", "Z"]].to_numpy(float) - ref_xyz[k])
        df["dE"], df["dN"], df["dU"] = enu[:, 0], enu[:, 1], enu[:, 2]
        df["err_2d"] = np.hypot(df["dE"], df["dN"])
        df["err_3d"] = np.sqrt(df["err_2d"] ** 2 + df["dU"] ** 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "spp_results.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")

    solved = df["status"] == "ok"
    print(f"Solutions: {solved.sum()} of {len(df)} epochs ({100 * solved.mean():.1f} %)")
    if not solved.any():
        print("No epoch could be solved - check the files and the SIGNALS setting.")
        return
    if "err_2d" in df:
        summary = summarise(df)
        print("\nResult per point (time order)  -  errors in metres")
        print(summary.to_string(index=False))
        summary.to_csv(OUT_DIR / "spp_summary_per_point.csv", index=False, float_format="%.3f")
    make_plots(df, ref)


def summarise(df):
    out = []
    for pid, g in df[df["point_id"] != ""].groupby("point_id", sort=False):
        s = g[g["status"] == "ok"]
        out.append({"point": pid, "epochs": len(g),
                    "solved_%": 100.0 * len(s) / len(g),
                    "sats_tracked": g["n_tracked"].median(),
                    "sats_used": s["n_sat"].median() if len(s) else np.nan,
                    "C/N0_mean": s["cn0_mean"].mean() if len(s) else np.nan,
                    "rejected_mean": s["n_rejected"].mean() if len(s) else np.nan,
                    "HDOP_med": s["hdop"].median() if len(s) else np.nan,
                    "sigma0_med": s["sigma0"].median() if len(s) else np.nan,
                    "2D_median": s["err_2d"].median() if len(s) else np.nan,
                    "2D_95%": s["err_2d"].quantile(0.95) if len(s) else np.nan,
                    "Up_median": s["dU"].median() if len(s) else np.nan})
    return pd.DataFrame(out)


# ------------------------------------------------------------- plots -------
def make_plots(df, ref):
    if not (SHOW_PLOTS or SAVE_FIGURES):
        return
    import matplotlib.pyplot as plt

    t_min = (df["gps_time"] - df["gps_time"].iloc[0]) / 60.0
    has_err = "err_2d" in df

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8.5))
    ax[0].plot(t_min, df["n_tracked"], color="0.65", lw=1, label="tracked (G+E+C)")
    ax[0].plot(t_min, df["n_sat"], color="tab:blue", lw=1, label="used in solution")
    ax[0].set_ylabel("satellites")
    ax[0].legend(loc="upper right", fontsize=8)
    if has_err:
        ax[1].semilogy(t_min, df["err_2d"], ".", ms=2, color="tab:red")
        ax[1].set_ylabel("horizontal error [m]")
    else:
        ax[1].plot(t_min, df["h"], ".", ms=2)
        ax[1].set_ylabel("height [m]")
    ax[2].semilogy(t_min, df["hdop"], ".", ms=2, color="tab:green", label="HDOP")
    ax[2].semilogy(t_min, df["sigma0"], ".", ms=2, color="tab:purple", alpha=0.5, label="a-posteriori sigma0")
    ax[2].set_ylabel("HDOP / sigma0")
    ax[2].set_xlabel("minutes since first epoch")
    ax[2].legend(loc="upper right", fontsize=8)
    if has_err:                                           # mark the HYMN points
        for k, (pid, g) in enumerate(df[df["point_id"] != ""].groupby("point_id", sort=False)):
            a, b = t_min[g.index].min(), t_min[g.index].max()
            for axis in ax:
                axis.axvspan(a, b, color="0.5" if k % 2 else "0.8", alpha=0.15, lw=0)
            ax[0].text((a + b) / 2, 1.02, pid, transform=ax[0].get_xaxis_transform(),
                       ha="center", va="bottom", fontsize=7, rotation=90)
    for axis in ax:
        axis.grid(alpha=0.3)
    fig.suptitle("GNSS single point positioning (GPS + Galileo + BeiDou, code only)", y=0.995)
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "spp_time_series.png", dpi=150)

    if has_err and ref is not None:
        order = [p for p in ref["point_id"] if (df["point_id"] == p).any()]
        lat0, lon0, _ = ecef_to_geodetic(ref.loc[0, ["X", "Y", "Z"]].to_numpy(float))
        rot = enu_rotation(lat0, lon0)
        x0 = ref.loc[0, ["X", "Y", "Z"]].to_numpy(float)
        fig2, (axm, axb) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.3]})
        cmap = plt.get_cmap("tab20")
        for k, pid in enumerate(order):
            g = df[(df["point_id"] == pid) & (df["status"] == "ok")]
            if len(g):
                enu = (g[["X", "Y", "Z"]].to_numpy(float) - x0) @ rot.T
                axm.plot(enu[:, 0], enu[:, 1], ".", ms=2, color=cmap(k % 20), label=pid)
            p = ref[ref["point_id"] == pid][["X", "Y", "Z"]].to_numpy(float)[0]
            pe = rot @ (p - x0)
            axm.plot(pe[0], pe[1], "k+", ms=9, mew=1.5)
        truth = (ref[ref["point_id"].isin(order)][["X", "Y", "Z"]].to_numpy(float) - x0) @ rot.T
        pad = 25.0
        axm.set_xlim(truth[:, 0].min() - pad, truth[:, 0].max() + pad)
        axm.set_ylim(truth[:, 1].min() - pad, truth[:, 1].max() + pad)
        axm.set_aspect("equal")
        axm.set_xlabel(f"East [m] from {order[0]}")
        axm.set_ylabel(f"North [m] from {order[0]}")
        axm.set_title("SPP positions (dots) and true points (+)\npoints far away are outside the plot")
        axm.legend(fontsize=7, markerscale=4, ncol=2, loc="best")
        axm.grid(alpha=0.3)

        data = [df[(df["point_id"] == p) & (df["status"] == "ok")]["err_2d"].dropna().to_numpy() for p in order]
        keep = [k for k, d in enumerate(data) if len(d)]
        axb.boxplot([data[k] for k in keep], whis=(5, 95), showfliers=False)
        axb.set_xticks(range(1, len(keep) + 1), [order[k] for k in keep], rotation=45)
        axb.set_yscale("log")
        axb.set_ylabel("horizontal error [m]  (box 25-75 %, whiskers 5-95 %)")
        axb.set_title("Horizontal error per point (time order)")
        for j, k in enumerate(keep, start=1):
            used = df[(df["point_id"] == order[k]) & (df["status"] == "ok")]["n_sat"].median()
            axb.text(j, 0.98, f"{used:.0f} sv", transform=axb.get_xaxis_transform(),
                     ha="center", va="top", fontsize=7)
        axb.grid(alpha=0.3, which="both")
        fig2.tight_layout()
        if SAVE_FIGURES:
            fig2.savefig(OUT_DIR / "spp_points.png", dpi=150)
    if SAVE_FIGURES:
        print(f"Figures saved in: {OUT_DIR}")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
