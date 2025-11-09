---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pa-aa-phl-storms
    language: python
    name: pa-aa-phl-storms
---

# EM-DAT

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd
import geopandas as gpd
import xarray as xr
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar
from rasterio.errors import RasterioIOError

from src.datasources import codab, imerg, emdat
from src.constants import *
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
df_emdat_new = emdat.load_emat()
```

```python
query = """
SELECT *
FROM storms.ibtracs_storms
"""
with stratus.get_engine(stage="prod").connect() as con:
    df_storms = pd.read_sql(query, con)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/emdat-tropicalcyclone-2000-2022-processed-sids.csv"
df_emdat_old_raw = stratus.load_csv_from_blob(blob_name)
```

```python
df_emdat_old = df_emdat_old_raw[
    df_emdat_old_raw["iso3"] == ISO3.upper()
].copy()
```

```python
df_emdat_old.columns
```

```python
# see which ones never had SIDs filled in
df_emdat_old[df_emdat_old["sid"].isnull()][
    [
        "DisNo.",
        "Event Name",
        "Start Year",
        "Start Month",
        "Start Day",
        "Total Affected",
    ]
]
```

```python
# get values for storms not in IBTrACS (quick Google)
df_missing_ibtracs = pd.DataFrame(
    columns=[
        "name",
        "season",
        "DisNo.",
        "start_date",
        "end_date",
        "wind_speed_max_kmh",
    ],
    data=[
        ("INENG", 2003, "2003-0823-PHL", "2003-07-30", "2003-07-31", 45),
        ("WINNIE", 2004, "2004-0609-PHL", "2004-11-29", "2004-11-30", 55),
        ("MAYMAY", 2022, "2022-0665-PHL", "2022-10-11", "2022-10-12", 55),
    ],
)
```

```python
# get matches from IBTrACS for missing rows
disno2sid_old = {
    "2022-0204-PHL": "2022099N11128",
    "2022-0547-PHL": "2022232N18131",
    "2022-0680-PHL": "2022285N17140",
    "2022-0707-PHL": "2022299N11134",
}
```

```python
df_emdat_old["sid"] = df_emdat_old["sid"].fillna(
    df_emdat_old["DisNo."].apply(lambda x: disno2sid_old.get(x))
)
```

```python
df_emdat_old[df_emdat_old["sid"].isnull()][
    ["DisNo.", "Event Name", "Start Year", "Start Month", "Start Day"]
]
```

```python
# see which new EM-DAT events aren't in the old matched up CSV file
df_emdat_new[~df_emdat_new["DisNo."].isin(df_emdat_old["DisNo."].values)][
    ["DisNo.", "Event Name", "Start Year", "Start Month", "Start Day"]
]
```

```python
# limit to only before this year
df_emdat_new_recent = df_emdat_new[
    (~df_emdat_new["DisNo."].isin(df_emdat_old["DisNo."].values))
    & (df_emdat_new["Start Year"] < 2025)
].copy()
df_emdat_new_recent[
    ["DisNo.", "Event Name", "Start Year", "Start Month", "Start Day"]
]
```

```python
# match to IBTrACS
disno2sid_new = {
    "2023-0246-PHL": "2023101N14127",
    "2023-0300-PHL": "2023138N05151",
    "2023-0450-PHL": "2023194N16123",
    "2023-0464-PHL": "2023201N13134",
    "2023-0568-PHL": "2023234N18128",
    "2023-0623-PHL": "2023271N14144",
    "2023-0842-PHL": "2023349N04141",
    "2024-0337-PHL": "2024141N03142",
    "2024-0522-PHL": "2024201N12133",
    "2024-0654-PHL": "2024244N09137",
    "2024-0666-PHL": "2024253N11148",
    "2024-0695-PHL": "2024259N17126",
    "2024-0711-PHL": "2024270N24128",
    "2024-0777-PHL": "2024293N13141",
    "2024-0801-PHL": "2024298N13150",
    "2024-0825-PHL": "2024307N06143",
    "2024-0829-PHL": "2024312N14145",
    "2024-0855-PHL": "2024313N10169",
}
```

```python
df_emdat_new_recent["sid"] = df_emdat_new_recent["DisNo."].replace(
    disno2sid_new
)
```

```python
df_emdat_new_recent[df_emdat_new_recent["sid"].isnull()]
```

```python
# aggregate rainfall for storms missing from IBTrACS
quantiles = [0.8, 0.9, 0.95]


def get_storm_rainfall_aggregations(row):
    row = row.copy()
    min_date = row["valid_time_min"].date() - pd.DateOffset(days=1)
    max_date = row["valid_time_max"].date() + pd.DateOffset(days=1)
    dates = pd.date_range(min_date, max_date)
    da = imerg.open_imerg_raster_dates(dates)
    da_clip = da.rio.clip(adm0.geometry)

    # 2-day rolling sum
    da_rolling2 = da_clip.rolling(date=2).sum()
    # 3-day rolling sum
    da_rolling3 = da_clip.rolling(date=3).sum()

    # take quantiles
    for quantile in quantiles:
        for da_agg, agg_str in [
            (da_rolling2, "roll2"),
            (da_rolling3, "roll3"),
        ]:
            # get quantile threshs
            quantile_threshs = da_agg.quantile(quantile, dim=["x", "y"])
            # get max value
            row[f"q{quantile*100:.0f}_{agg_str}"] = float(
                quantile_threshs.max()
            )

    return row


import warnings

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
```

```python
df_missing_ibtracs["valid_time_min"] = pd.to_datetime(
    df_missing_ibtracs["start_date"]
)
df_missing_ibtracs["valid_time_max"] = pd.to_datetime(
    df_missing_ibtracs["end_date"]
)
df_missing_ibtracs["wind_speed_max"] = (
    df_missing_ibtracs["wind_speed_max_kmh"] / KNOTS_TO_KMH
).astype(int)
```

```python
df_missing_ibtracs
```

```python
tqdm.pandas()
```

```python
df_missing_ibtracs_with_stats = df_missing_ibtracs.progress_apply(
    get_storm_rainfall_aggregations, axis=1
)
```

```python
# create madeup SID to make merging easier
df_missing_ibtracs_with_stats["sid"] = (
    df_missing_ibtracs_with_stats["name"].str.lower()
    + "_"
    + df_missing_ibtracs_with_stats["season"].astype(str)
)
```

```python
df_missing_ibtracs_with_stats
```

```python
df_missing_ibtracs_with_stats_with_emdat = df_missing_ibtracs_with_stats.merge(
    df_emdat_old.drop(columns="sid")
)
```

```python
df_missing_ibtracs_with_stats_with_emdat
```

```python
df_emdat_combined = pd.concat(
    [df_emdat_old, df_emdat_new_recent], ignore_index=True
)
```

```python
# df_emdat_combined["sid"] = df_emdat_combined.apply(
#     lambda row: (
#         row["sid"]
#         if row["sid"]
#         else df_missing_ibtracs_with_stats.set_index("DisNo.").loc[
#             row["DisNo."], "sid"
#         ]
#     ),
#     axis=1,
# )
```

```python
df_emdat_combined[df_emdat_combined["sid"].isnull()]
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs_imerg_stats.parquet"
df_stats_raw = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats_raw
```

```python
df_stats_with_ibtracs = df_stats_raw.merge(
    df_emdat_combined, how="left"
).merge(df_storms)
```

```python
df_stats = pd.concat(
    [df_stats_with_ibtracs, df_missing_ibtracs_with_stats_with_emdat],
    ignore_index=True,
)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs_imerg_emdat_stats.parquet"
stratus.upload_parquet_to_blob(df_stats, blob_name)
```
