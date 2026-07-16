"""
Download CHIRPS-GEFS forecasts for storms that made landfall in the target regions.

For each storm detected in IBTrACS (1998–2024), and for each issue date
1–5 days before landfall, downloads the three forecast TIFs corresponding to
valid dates: landfall-1, landfall, landfall+1.

Files are clipped to the Philippines bounding box before upload to save
storage and speed up downstream zonal-stats reads.

Blob path:
  pa-aa-phl-storms/raw/chirps-gefs/{issue_YYYYMMDD}/data.{valid_YYYY}.{valid_MMDD}.tif
"""

import io
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import ocha_stratus as stratus
import pandas as pd
import requests
import rioxarray as rxr
from shapely.geometry import LineString
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.constants import CHD_GREEN  # noqa: F401 (confirms src is importable)

# ── Configuration ─────────────────────────────────────────────────────────────
CONTAINER    = "projects"
STAGE        = "dev"
BLOB_PREFIX  = "pa-aa-phl-storms/raw/chirps-gefs/"

CHIRPS_BASE  = "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/daily/global/"

TARGET_REGIONS = ["PH02", "PH03", "PH05", "PH08"]
MAX_LEAD_DAYS  = 5     # issue dates from L-1 to L-MAX_LEAD_DAYS
DAYS_BEFORE    = 1     # valid dates: L-1, L, L+1
DAYS_AFTER     = 1
MIN_WIND_KT    = 34.0

# Clip extent — wide enough to capture approaching storm rainfall
CLIP_MINX, CLIP_MAXX = 110.0, 132.0
CLIP_MINY, CLIP_MAXY =   4.0,  22.0

# ── Landfall detection ────────────────────────────────────────────────────────
print("Loading admin boundaries...")
phl_admin_1 = stratus.codab.load_codab_from_fieldmaps(iso3="PHL", admin_level=1)
pcode_col   = [c for c in phl_admin_1.columns if "src" in c.lower()][0]
target_union = unary_union(
    phl_admin_1[phl_admin_1[pcode_col].isin(TARGET_REGIONS)].geometry
)

print("Fetching IBTrACS tracks 2001–2024...")
tracks = gpd.read_postgis(
    """
    SELECT storm_id, valid_time, wind_speed, geometry
    FROM storms.ibtracs_tracks_geo
    WHERE basin = 'WP'
      AND valid_time BETWEEN '2001-01-01' AND '2024-12-31'
    """,
    stratus.get_engine(stage="prod"),
    geom_col="geometry",
)
tracks["valid_time"] = pd.to_datetime(tracks["valid_time"])

landfalls = []
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
            landfalls.append({
                "storm_id":     storm_id,
                "landfall_date": p1["valid_time"].date(),
            })
            break

landfall_df = pd.DataFrame(landfalls)
print(f"Storms with landfall in target regions: {len(landfall_df)}")

# ── Build list of (issue_date, valid_date) pairs needed ───────────────────────
needed = set()
for _, row in landfall_df.iterrows():
    lf = pd.Timestamp(row["landfall_date"])
    for lead in range(1, MAX_LEAD_DAYS + 1):
        issue_date = lf - pd.Timedelta(days=lead)
        for d in range(-DAYS_BEFORE, DAYS_AFTER + 1):
            valid_date = lf + pd.Timedelta(days=d)
            needed.add((issue_date.date(), valid_date.date()))

print(f"Unique (issue, valid) pairs needed: {len(needed)}")

# ── Pre-fetch existing blobs ──────────────────────────────────────────────────
print("Checking existing blobs...")
existing = set(
    stratus.list_container_blobs(
        name_starts_with=BLOB_PREFIX, stage=STAGE, container_name=CONTAINER
    )
)
print(f"  {len(existing)} files already in blob")

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
})
n_uploaded = n_skipped = n_error = 0
checked_dirs = {}   # cache: issue_date -> bool (folder has forecasts)

# Files confirmed missing on the server (GET returned 404) — skip without requesting.
# These are CHIRPS-GEFS v3 data gaps; add all (issue, valid) combinations per gap.
def _gap(issue_dates, valid_fnames):
    return {f"{BLOB_PREFIX}{i}/{v}" for i in issue_dates for v in valid_fnames}

# Gap 1: valid dates 2020-05-12 to 2020-05-15
_g1_issues = [f"2020050{d}" for d in range(6, 10)] + [f"202005{d}" for d in range(10, 16)]
_g1_valids = ["c3g_2020.05.12.tif", "c3g_2020.05.13.tif",
              "c3g_2020.05.14.tif", "c3g_2020.05.15.tif"]

# Gap 2: valid dates 2020-06-10 to 2020-06-16
_g2_issues = [f"20200601", "20200602", "20200603", "20200604", "20200605",
              "20200606", "20200607", "20200608", "20200609", "20200610",
              "20200611", "20200612", "20200613", "20200614", "20200615"]
_g2_valids = ["c3g_2020.06.10.tif", "c3g_2020.06.11.tif", "c3g_2020.06.12.tif",
              "c3g_2020.06.13.tif", "c3g_2020.06.14.tif", "c3g_2020.06.15.tif",
              "c3g_2020.06.16.tif"]

KNOWN_MISSING = _gap(_g1_issues, _g1_valids) | _gap(_g2_issues, _g2_valids)

# CHIRPS-GEFS v3 has no archival data for 2020: every issue date in 2020 returns
# 404 on file GETs despite directories returning 200.  Skip the entire year.
SKIP_ISSUE_YEARS = {2020}

for issue_date, valid_date in sorted(needed):
    issue_str  = issue_date.strftime("%Y%m%d")
    fname      = f"c3g_{valid_date.strftime('%Y.%m.%d')}.tif"
    blob_name  = f"{BLOB_PREFIX}{issue_str}/{fname}"

    if blob_name in existing or blob_name in KNOWN_MISSING:
        n_skipped += 1
        continue

    if issue_date.year in SKIP_ISSUE_YEARS:
        n_skipped += 1
        continue

    # Check the issue-date folder exists and has forecasts (once per issue date)
    if issue_date not in checked_dirs:
        dir_url = (
            f"{CHIRPS_BASE}"
            f"{issue_date.year}/{issue_date.month:02d}/{issue_date.day:02d}/"
        )
        for attempt in range(1, 4):
            try:
                dir_resp = session.head(dir_url, timeout=30)
                time.sleep(10)
                break
            except Exception as exc:
                print(f"  [retry {attempt}/3] HEAD error for {issue_str}: {exc.__class__.__name__}")
                if attempt < 3:
                    time.sleep(15)
        else:
            print(f"  [skip] HEAD failed after 3 attempts for {issue_str}")
            checked_dirs[issue_date] = False
            n_skipped += 1
            continue
        if dir_resp.status_code == 404:
            print(f"  [skip] no forecast folder for {issue_str}: {dir_url}")
            checked_dirs[issue_date] = False
        else:
            checked_dirs[issue_date] = True

    if not checked_dirs[issue_date]:
        n_skipped += 1
        continue

    url = (
        f"{CHIRPS_BASE}"
        f"{issue_date.year}/{issue_date.month:02d}/{issue_date.day:02d}/{fname}"
    )
    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=60)
            break
        except Exception as exc:
            print(f"  [retry {attempt}/3] network error on {fname}: {exc.__class__.__name__}")
            if attempt < 3:
                time.sleep(15)
    else:
        print(f"  ERROR: failed after 3 attempts: {url}")
        n_error += 1
        continue
    time.sleep(15)
    if resp.status_code in (403, 404):
        print(f"  ERROR ({resp.status_code}): {url}")
        print("Stopping — server rejected request. Wait before retrying.")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"  ERROR ({resp.status_code}): {url}")
        print("Stopping — unexpected HTTP error.")
        sys.exit(1)

    try:
        da = rxr.open_rasterio(io.BytesIO(resp.content), masked=True)
        da_clip = da.rio.clip_box(
            minx=CLIP_MINX, miny=CLIP_MINY,
            maxx=CLIP_MAXX, maxy=CLIP_MAXY,
        )
        buf = io.BytesIO()
        da_clip.rio.to_raster(buf, driver="GTiff")
        buf.seek(0)
        stratus.upload_blob_data(
            buf.read(), blob_name,
            stage=STAGE, container_name=CONTAINER,
            content_type="image/tiff",
        )
        print(f"  uploaded {blob_name}")
        n_uploaded += 1
    except Exception as exc:
        print(f"  ERROR processing {url}: {exc}")
        n_error += 1

print(f"\nDone — uploaded: {n_uploaded}, skipped: {n_skipped}, errors: {n_error}")
