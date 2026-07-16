import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import ocha_stratus as stratus

# ── Blob locations ────────────────────────────────────────────────────────────
CONTAINER  = "projects"
STAGE      = "dev"
XML_PREFIX = "pa-aa-phl-storms/raw/ecmwf_hindcast/xml/"
CSV_PREFIX = "pa-aa-phl-storms/raw/ecmwf_hindcast/csv/"

# ── Geography ─────────────────────────────────────────────────────────────────
# Keep disturbances whose forecast track passes through the Philippines
# extended bounding box (buffer around land, lon 117–127, lat 5–21)
PHL_LON_MIN, PHL_LON_MAX = 112.0, 130.0
PHL_LAT_MIN, PHL_LAT_MAX =   4.0,  23.0
WNP_LON_MIN, WNP_LON_MAX = 100.0, 180.0
WNP_LAT_MIN, WNP_LAT_MAX =   0.0,  45.0


def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _fmt_time(iso_str):
    """Convert ISO 8601 to original YYYY/MM/DD, HH:MM:SS format."""
    if not iso_str:
        return ""
    try:
        date_part, time_part = iso_str.split("T")
        return "/".join(date_part.split("-")) + ", " + time_part.rstrip("Z")
    except Exception:
        return iso_str


def in_phl(lat, lon):
    return PHL_LAT_MIN <= lat <= PHL_LAT_MAX and PHL_LON_MIN <= lon <= PHL_LON_MAX


def in_wnp(lat, lon):
    return WNP_LAT_MIN <= lat <= WNP_LAT_MAX and WNP_LON_MIN <= lon <= WNP_LON_MAX


def xml2records(raw_bytes, blob_name):
    """Parse one TIGGE XML blob; return rows for PHL-heading WNP disturbances."""
    try:
        root = ET.fromstring(raw_bytes.decode("utf-8", errors="replace"))
    except ET.ParseError:
        print(f"  [skip] parse error: {blob_name.split('/')[-1]}")
        return []

    prod_el     = root.find("header/productionCenter")
    basetime_el = root.find("header/baseTime")
    prod_center = re.sub(r"\s+", " ", prod_el.text).strip().lower() if prod_el is not None else ""
    base_time   = basetime_el.text if basetime_el is not None else ""

    records = []
    for data_el in root.findall("data"):
        mtype = data_el.get("type")
        if mtype not in ("forecast", "ensembleForecast"):
            continue
        ensemble = data_el.get("member")

        for dist in data_el.findall("disturbance"):
            name_el = dist.find("cycloneName")
            num_el  = dist.find("cycloneNumber")
            cyc_name = name_el.text.lower().strip() if name_el is not None else "unnamed"
            cyc_num  = num_el.text.strip()           if num_el  is not None else ""

            fixes = []
            for fix in dist.findall("fix"):
                lat = _float(fix.findtext("latitude"))
                lon = _float(fix.findtext("longitude"))
                if lat is None or lon is None:
                    continue
                fixes.append((fix, lat, lon))

            if not fixes:
                continue
            if not any(in_wnp(lat, lon) for _, lat, lon in fixes):
                continue
            if not any(in_phl(lat, lon) for _, lat, lon in fixes):
                continue

            for fix, lat, lon in fixes:
                speed_el    = fix.find("cycloneData/maximumWind/speed")
                pressure_el = fix.find("cycloneData/minimumPressure/pressure")
                vtime_el    = fix.find("validTime")
                records.append({
                    "mtype":         mtype,
                    "product":       prod_center,
                    "cyc_number":    cyc_num,
                    "cyc_name":      cyc_name,
                    "ensemble":      ensemble,
                    "speed":         speed_el.text.lower().strip()    if speed_el    is not None else "",
                    "pressure":      pressure_el.text.lower().strip() if pressure_el is not None else "",
                    "time":          _fmt_time(vtime_el.text if vtime_el is not None else ""),
                    "lat":           lat,
                    "lon":           lon,
                    "lead_time":     fix.get("hour", ""),
                    "forecast_time": _fmt_time(base_time),
                })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────
xml_blobs = stratus.list_container_blobs(
    name_starts_with=XML_PREFIX,
    stage=STAGE,
    container_name=CONTAINER,
)
xml_blobs = [b for b in xml_blobs if b.endswith(".xml")]
print(f"Found {len(xml_blobs):,} XML blobs")

TYPHOON_MONTHS = {9, 10, 11, 12, 1}

def _blob_month(blob_name):
    """Extract init month from blob filename (YYYYMMDDHHMMSS)."""
    fname = blob_name.split("/")[-1]   # z_tigge_c_ecmf_YYYYMMDDHHMMSS_...xml
    return int(fname.split("_")[4][4:6])

xml_blobs = [b for b in xml_blobs if _blob_month(b) in TYPHOON_MONTHS]
print(f"After typhoon-month filter (Sep–Jan): {len(xml_blobs):,} XML blobs")

all_records = []
for i, blob_name in enumerate(xml_blobs, 1):
    for attempt in range(1, 4):
        try:
            raw = stratus.load_blob_data(blob_name, stage=STAGE, container_name=CONTAINER)
            break
        except Exception as exc:
            print(f"  [retry {attempt}/3] {blob_name.split('/')[-1]}: {exc.__class__.__name__}")
            if attempt < 3:
                time.sleep(10)
    else:
        print(f"  [skip] failed after 3 attempts: {blob_name.split('/')[-1]}")
        continue
    recs = xml2records(raw, blob_name)
    if recs:
        n_storms = len({r["cyc_number"] for r in recs})
        print(f"  [{i:>5}/{len(xml_blobs)}] {blob_name.split('/')[-1]}: "
              f"{len(recs):>5} rows, {n_storms} storm(s)")
    all_records.extend(recs)

if not all_records:
    print("No PHL-heading storms found.")
else:
    df = pd.DataFrame(all_records)
    print(f"\nTotal rows: {len(df):,}  |  Unique storms: {df['cyc_number'].nunique()}")

    out_cols = ["mtype", "product", "cyc_number", "ensemble", "speed",
                "pressure", "time", "lat", "lon", "lead_time", "forecast_time"]

    # One CSV per storm
    for cyc_name, grp in df.groupby("cyc_name"):
        out_blob = CSV_PREFIX + f"{cyc_name}_all.csv"
        stratus.upload_csv_to_blob(grp[out_cols], out_blob, stage=STAGE, container_name=CONTAINER)
        print(f"  {out_blob.split('/')[-1]}  ({len(grp):,} rows)")
