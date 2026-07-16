"""
Compute mean CHIRPS-GEFS rainfall per admin1 region and save to parquet.

Reads TIFs from:
  pa-aa-phl-storms/raw/chirps-gefs/{issue_YYYYMMDD}/c3g_{valid_YYYY.MM.DD}.tif

Writes parquet to:
  pa-aa-phl-storms/processed/chirps-gefs/chirps_gefs_adm1.parquet

Schema: issue_date, valid_date, lead_days, pcode, adm1_name, mean_mm

Run incrementally - existing (issue_date, valid_date) pairs are skipped.
"""

import io
import sys
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
import rioxarray as rxr
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Configuration ──────────────────────────────────────────────────────────────
CONTAINER  = "projects"
STAGE      = "dev"
RAW_PREFIX = "pa-aa-phl-storms/raw/chirps-gefs/"
OUT_BLOB   = "pa-aa-phl-storms/processed/chirps-gefs/chirps_gefs_adm1.parquet"

# ── Admin boundaries ───────────────────────────────────────────────────────────
print("Loading admin boundaries...")
phl_admin_1 = stratus.codab.load_codab_from_fieldmaps(iso3="PHL", admin_level=1)
pcode_col   = [c for c in phl_admin_1.columns if "src" in c.lower()][0]
name_col    = next((c for c in phl_admin_1.columns if "adm1_name" in c.lower()), None)
print(f"  {len(phl_admin_1)} admin1 regions  |  pcode col: {pcode_col}")

# ── Load existing parquet for incremental updates ──────────────────────────────
existing_pairs = set()
try:
    existing_df = stratus.load_parquet_from_blob(
        OUT_BLOB, stage=STAGE, container_name=CONTAINER
    )
    existing_pairs = set(
        zip(existing_df["issue_date"].astype(str),
            existing_df["valid_date"].astype(str))
    )
    n_existing = existing_df[["issue_date", "valid_date"]].drop_duplicates().shape[0]
    print(f"Existing parquet: {len(existing_df):,} rows, {n_existing} (issue,valid) pairs")
except Exception:
    existing_df = pd.DataFrame()
    print("No existing parquet — creating from scratch")

# ── List and filter TIF blobs ──────────────────────────────────────────────────
print("Listing TIF blobs...")
all_blobs = [
    b for b in stratus.list_container_blobs(
        name_starts_with=RAW_PREFIX, stage=STAGE, container_name=CONTAINER
    )
    if b.endswith(".tif")
]
print(f"Found {len(all_blobs)} TIF blobs")

blob_meta = []
for blob in all_blobs:
    # blob: .../chirps-gefs/{YYYYMMDD}/c3g_YYYY.MM.DD.tif
    parts      = blob.replace(RAW_PREFIX, "").split("/")
    issue_str  = parts[0]
    valid_str  = parts[1].replace("c3g_", "").replace(".tif", "")  # YYYY.MM.DD
    issue_date = f"{issue_str[:4]}-{issue_str[4:6]}-{issue_str[6:8]}"
    valid_date = valid_str.replace(".", "-")
    blob_meta.append({"blob": blob, "issue_date": issue_date, "valid_date": valid_date})

blob_df    = pd.DataFrame(blob_meta)
to_process = blob_df[
    ~blob_df.apply(
        lambda r: (r["issue_date"], r["valid_date"]) in existing_pairs, axis=1
    )
].reset_index(drop=True)

print(f"{len(to_process)} new (issue, valid) pairs to process")
if to_process.empty:
    print("Nothing to do.")
    sys.exit(0)

# ── Process TIFs ───────────────────────────────────────────────────────────────
new_records = []
n_ok = n_err = 0

for i, row in enumerate(to_process.itertuples(index=False), 1):
    try:
        raw = stratus.load_blob_data(row.blob, stage=STAGE, container_name=CONTAINER)
        da  = rxr.open_rasterio(io.BytesIO(raw), masked=True).squeeze(drop=True)
        da  = da.where(da >= 0)   # mask fill values / negatives
        arr       = da.values.astype(float)
        transform = da.rio.transform()

        regions_reproj = phl_admin_1.to_crs(da.rio.crs or "EPSG:4326")
        stats = zonal_stats(
            regions_reproj, arr,
            affine=transform, stats=["mean"],
            nodata=np.nan, all_touched=True,
        )

        issue_dt  = pd.Timestamp(row.issue_date).date()
        valid_dt  = pd.Timestamp(row.valid_date).date()
        lead_days = (valid_dt - issue_dt).days

        for s, (_, reg) in zip(stats, phl_admin_1.iterrows()):
            new_records.append({
                "issue_date": issue_dt,
                "valid_date": valid_dt,
                "lead_days":  lead_days,
                "pcode":      reg[pcode_col],
                "adm1_name":  reg[name_col] if name_col else "",
                "mean_mm":    float(s["mean"]) if s["mean"] is not None else np.nan,
            })
        n_ok += 1

    except Exception as exc:
        print(f"  ERROR [{i}] {row.blob}: {exc}")
        n_err += 1

    if i % 100 == 0 or i == len(to_process):
        print(f"  [{i}/{len(to_process)}] ok={n_ok}  err={n_err}")

# ── Save ───────────────────────────────────────────────────────────────────────
if not new_records:
    print("No new records to save.")
    sys.exit(0)

new_df = pd.DataFrame(new_records)
new_df["issue_date"] = pd.to_datetime(new_df["issue_date"])
new_df["valid_date"] = pd.to_datetime(new_df["valid_date"])

combined = (
    pd.concat([existing_df, new_df], ignore_index=True)
    if not existing_df.empty
    else new_df
)
combined = combined.sort_values(["issue_date", "valid_date", "pcode"]).reset_index(drop=True)

buf = io.BytesIO()
combined.to_parquet(buf, index=False)
buf.seek(0)
stratus.upload_blob_data(
    buf.read(), OUT_BLOB,
    stage=STAGE, container_name=CONTAINER,
    content_type="application/octet-stream",
)
print(f"\nSaved {len(combined):,} rows to {OUT_BLOB}")
print(f"  {combined[['issue_date','valid_date']].drop_duplicates().shape[0]} (issue,valid) pairs")
print(f"  Issue dates: {combined['issue_date'].min().date()} to {combined['issue_date'].max().date()}")
print(f"Done — ok={n_ok}  err={n_err}")
