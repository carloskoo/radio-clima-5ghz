#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import shutil
import zipfile
import gzip
from datetime import datetime, timedelta, timezone, time as dtime

import pandas as pd
import xarray as xr
import cdsapi

OUT_ROOT = "eber_burga"
ERA_CACHE_DIR = os.path.join(OUT_ROOT, "_era5_cache")

START_DATE = "2025-09-20"
END_DATE   = "2026-04-05"

LIMA_TZ = timezone(timedelta(hours=-5))

AP_LAT = -6.69240
AP_LON = -78.51418

SM_LAT = -6.76387
SM_LON = -78.60154

MID_LAT = (AP_LAT + SM_LAT) / 2.0
MID_LON = (AP_LON + SM_LON) / 2.0

BOX_DEG = 0.20
DOWNLOAD_ATTEMPTS = 2
FORCE_REDOWNLOAD_NC = False
FORCE_REBUILD_CSV = False

EXPECTED_COLUMNS = [
    "time_utc", "time_lima", "temp_c", "dewpoint_c",
    "precip_mm", "press_hpa", "wind_ms"
]

def ensure_dirs():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(ERA_CACHE_DIR, exist_ok=True)

def get_day_dir(date_obj):
    base_dir = os.path.join(OUT_ROOT, date_obj.isoformat())
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def day_csv_path(date_obj, tag):
    return os.path.join(get_day_dir(date_obj), f"era5_{tag.lower()}_hourly_{date_obj}.csv")

def month_cache_path(year, month, lat, lon, box_deg=0.20):
    return os.path.join(
        ERA_CACHE_DIR,
        f"era5land_{year:04d}-{month:02d}_lat{lat:.4f}_lon{lon:.4f}_box{box_deg:.2f}.nc"
    )

def peek_magic(path, n=16):
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except Exception:
        return b""

def is_netcdf(path):
    magic = peek_magic(path, 8)
    return magic.startswith(b"CDF") or magic.startswith(b"\x89HDF\r\n\x1a\n")

def list_months_between(start_date, end_date):
    months = []
    y, m = start_date.year, start_date.month
    end_y, end_m = end_date.year, end_date.month
    while (y < end_y) or (y == end_y and m <= end_m):
        months.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months

def detect_time_coord(ds):
    if "valid_time" in ds.coords:
        return "valid_time"
    if "time" in ds.coords:
        return "time"
    raise KeyError("No se encontró coordenada temporal")

def open_dataset_robust(nc_files):
    engines = ["netcdf4", "h5netcdf", "scipy"]
    for eng in engines:
        try:
            return xr.open_mfdataset(nc_files, combine="by_coords", engine=eng)
        except Exception:
            pass
    raise RuntimeError("No se pudo abrir el NetCDF")

def ensure_month_downloaded(year, month, lat, lon, box_deg=0.20):
    path = month_cache_path(year, month, lat, lon, box_deg=box_deg)

    if os.path.exists(path) and os.path.getsize(path) > 0 and is_netcdf(path):
        print(f"[CACHE OK] {os.path.basename(path)}")
        return path

    north = lat + box_deg / 2
    south = lat - box_deg / 2
    west = lon - box_deg / 2
    east = lon + box_deg / 2

    req = {
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "total_precipitation",
            "surface_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [north, west, south, east],
        "format": "netcdf",
    }

    cdsapi.Client().retrieve("reanalysis-era5-land", req, path)
    return path

def extract_hourly_for_day(date_obj, site_lat, site_lon):
    start_lima = datetime.combine(date_obj, dtime(0, 0), tzinfo=LIMA_TZ)
    stop_lima = start_lima + timedelta(days=1)

    start_utc = start_lima.astimezone(timezone.utc)
    stop_utc = stop_lima.astimezone(timezone.utc)

    months = list_months_between(start_utc.date(), (stop_utc - timedelta(seconds=1)).date())
    nc_files = [ensure_month_downloaded(y, m, site_lat, site_lon, BOX_DEG) for (y, m) in months]

    ds = open_dataset_robust(nc_files)
    time_name = detect_time_coord(ds)

    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"

    lon_val = site_lon
    if float(ds[lon_name].max()) > 180 and lon_val < 0:
        lon_val = 360 + lon_val

    point = ds.sel({lat_name: site_lat, lon_name: lon_val}, method="nearest")

    start_utc_naive = pd.Timestamp(start_utc).tz_convert("UTC").tz_localize(None)
    stop_utc_naive = pd.Timestamp(stop_utc).tz_convert("UTC").tz_localize(None)

    point = point.sel({time_name: slice(start_utc_naive, stop_utc_naive - pd.Timedelta(seconds=1))})
    df = point.to_dataframe().reset_index()

    if time_name in df.columns and time_name != "time":
        df = df.rename(columns={time_name: "time"})

    df["temp_c"] = df["t2m"] - 273.15
    df["dewpoint_c"] = df["d2m"] - 273.15
    df["precip_mm"] = df["tp"] * 1000.0
    df["press_hpa"] = df["sp"] / 100.0
    df["wind_ms"] = (df["u10"] ** 2 + df["v10"] ** 2) ** 0.5

    df["time_utc"] = pd.to_datetime(df["time"], utc=True)
    df["time_lima"] = df["time_utc"].dt.tz_convert("America/Lima")

    out = df[["time_utc", "time_lima", "temp_c", "dewpoint_c", "precip_mm", "press_hpa", "wind_ms"]].copy()
    return out.sort_values("time_utc").drop_duplicates(subset=["time_utc"])

def main():
    ensure_dirs()
    all_days = pd.date_range(START_DATE, END_DATE, freq="D")

    for d in all_days:
        date_obj = d.date()
        try:
            df_mid = extract_hourly_for_day(date_obj, MID_LAT, MID_LON)
            df_mid.to_csv(day_csv_path(date_obj, "mid"), index=False, sep=";")
        except Exception as e:
            print(f"[MID ERROR] {date_obj}: {e}")

        try:
            df_sm = extract_hourly_for_day(date_obj, SM_LAT, SM_LON)
            df_sm.to_csv(day_csv_path(date_obj, "sm"), index=False, sep=";")
        except Exception as e:
            print(f"[SM ERROR] {date_obj}: {e}")

if __name__ == "__main__":
    main()
