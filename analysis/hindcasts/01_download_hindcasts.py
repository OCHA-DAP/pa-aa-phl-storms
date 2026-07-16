"""
Download ECMWF TIGGE TC track XMLs for storms that made landfall in PHL target regions.

Instead of iterating every 12 hours across 18 years, this script:
  1. Detects all IBTrACS WP storms that made landfall in PH02/PH03/PH05/PH08
  2. For each storm, generates 12-hourly init times from first detection to landfall
  3. Filters those times to typhoon season months (Sep–Jan) only
  4. Downloads only those specific TIGGE XMLs
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import ocha_stratus as stratus
from shapely.geometry import LineString
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Configuration ─────────────────────────────────────────────────────────────
CONTAINER   = "projects"
STAGE       = "dev"
BLOB_PREFIX = "pa-aa-phl-storms/raw/ecmwf_hindcast/xml/"

OSDF_BASE = "https://osdf-director.osg-htc.org/ncar/gdex/d330003/"

TARGET_REGIONS = ["PH02", "PH03", "PH05", "PH08"]
MIN_WIND_KT    = 34.0
START_YEAR     = 2006   # TIGGE available from Oct 2006
END_YEAR       = 2024
TYPHOON_MONTHS = {9, 10, 11, 12, 1}   # Sep–Jan inclusive

# ── Admin boundaries ──────────────────────────────────────────────────────────
print("Loading admin boundaries...")
phl_admin_1  = stratus.codab.load_codab_from_fieldmaps(iso3="PHL", admin_level=1)
pcode_col    = [c for c in phl_admin_1.columns if "src" in c.lower()][0]
target_union = unary_union(
    phl_admin_1[phl_admin_1[pcode_col].isin(TARGET_REGIONS)].geometry
)

# ── IBTrACS: get all WP tracks in range ───────────────────────────────────────
print("Fetching IBTrACS tracks...")
tracks = gpd.read_postgis(
    f"""
    SELECT storm_id, valid_time, wind_speed, geometry
    FROM storms.ibtracs_tracks_geo
    WHERE basin = 'WP'
      AND valid_time BETWEEN '{START_YEAR}-01-01' AND '{END_YEAR}-12-31'
    """,
    stratus.get_engine(stage="prod"),
    geom_col="geometry",
)
tracks["valid_time"] = pd.to_datetime(tracks["valid_time"])

# ── Detect landfalling storms and collect their date ranges ───────────────────
print("Detecting landfalls in target regions...")
storm_ranges = []

for storm_id, grp in tracks.sort_values("valid_time").groupby("storm_id"):
    grp = grp.reset_index(drop=True)
    if grp["wind_speed"].max() < MIN_WIND_KT:
        continue
    for i in range(len(grp) - 1):
        p0, p1 = grp.iloc[i], grp.iloc[i + 1]
        seg = LineString([
            (p0.geometry.x, p0.geometry.y),
            (p1.geometry.x, p1.geometry.y),
        ])
        if seg.intersects(target_union):
            storm_ranges.append({
                "storm_id":     storm_id,
                "first_time":   grp["valid_time"].iloc[0],
                "landfall_time": p1["valid_time"],
            })
            break

print(f"Landfalling storms: {len(storm_ranges)}")
for s in storm_ranges:
    print(f"  {s['storm_id']}  {s['first_time'].date()} -> {s['landfall_time'].date()}")

# ── Build set of 12-hourly init times needed ──────────────────────────────────
needed_times = set()
for s in storm_ranges:
    current = s["first_time"].floor("12h")
    end     = s["landfall_time"].ceil("12h")
    while current <= end:
        if current.month in TYPHOON_MONTHS:
            needed_times.add(current)
        current += pd.Timedelta(hours=12)

needed_times = sorted(needed_times)
print(f"\nUnique 12-hourly init times to download: {len(needed_times)}")

# ── Pre-fetch existing blobs ──────────────────────────────────────────────────
print("Listing existing blobs...")
existing_blobs = set(
    stratus.list_container_blobs(
        name_starts_with=BLOB_PREFIX,
        stage=STAGE,
        container_name=CONTAINER,
    )
)
print(f"  {len(existing_blobs):,} files already uploaded")

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
})

n_skipped = n_downloaded = n_missing = 0

for dt in needed_times:
    ymd    = dt.strftime("%Y%m%d")
    ymdhms = dt.strftime("%Y%m%d%H%M%S")
    server = "test" if dt < pd.Timestamp("2008-08-01") else "prod"
    fname  = f"z_tigge_c_ecmf_{ymdhms}_ifs_glob_{server}_all_glo.xml"
    rel    = f"ecmf/{dt.year}/{ymd}/{fname}"
    blob_name = BLOB_PREFIX + fname

    if blob_name in existing_blobs:
        n_skipped += 1
        continue

    req = None
    for attempt in range(1, 4):
        try:
            req = session.get(OSDF_BASE + rel, allow_redirects=True, timeout=60)
            break
        except Exception as exc:
            print(f"  attempt {attempt}/3 failed ({exc.__class__.__name__}): {rel}")
            if attempt < 3:
                time.sleep(10)
    if req is None:
        print(f"  giving up after 3 attempts: {rel}")
        n_missing += 1
        continue

    if req.status_code != 200:
        print(f"  missing  {rel}")
        n_missing += 1
        continue

    for upload_attempt in range(1, 4):
        try:
            stratus.upload_blob_data(
                req.content,
                blob_name,
                stage=STAGE,
                container_name=CONTAINER,
                content_type="application/xml",
            )
            print(f"  uploaded {rel}  ({len(req.content) / 1e6:.1f} MB)")
            n_downloaded += 1
            break
        except Exception as exc:
            print(f"  upload attempt {upload_attempt}/3 failed ({exc.__class__.__name__}): {rel}")
            if upload_attempt < 3:
                time.sleep(10)
    else:
        print(f"  giving up upload after 3 attempts: {rel}")

print(
    f"\nDone — uploaded: {n_downloaded}, skipped: {n_skipped}, "
    f"missing on server: {n_missing}"
)
