---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# IMERG aggregation

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

from src.datasources import codab, imerg
from src.constants import *
```

## Load data

### CODAB

```python
# codab.download_codab_to_blob()
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
adm0
```

```python
adm0.plot()
```

### IBTrACS

```python
query = """
SELECT * 
FROM storms.ibtracs_tracks_geo
WHERE basin = 'WP'
"""
with stratus.get_engine(stage="prod").connect() as con:
    gdf_tracks = gpd.read_postgis(query, con, geom_col="geometry")
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
gdf_tracks = gdf_tracks.merge(df_storms)
```

```python
gdf_tracks_recent = gdf_tracks[gdf_tracks["season"] >= 2000].copy()
```

## Processing

### Filter by distance

```python
adm0_3857 = adm0.to_crs(3857)
target_geom = adm0_3857.geometry.iloc[0]
gdf_tracks_recent_3857 = gdf_tracks_recent.to_crs(3857)
```

```python
distances = []

for geom in tqdm(gdf_tracks_recent_3857.geometry):
    distances.append(geom.distance(target_geom))

gdf_tracks_recent["distance_m"] = distances
```

```python
gdf_tracks_recent["distance_m"].hist()
```

```python
d_thresh = 50
```

```python
adm0_3857_buffer230 = adm0_3857.buffer(d_thresh * 1000)
```

```python
gdf_tracks_close = gdf_tracks_recent[
    gdf_tracks_recent["distance_m"] <= d_thresh * 1000
].copy()
```

```python
df_tracks_agg = (
    gdf_tracks_close.groupby("sid")
    .agg(
        valid_time_min=("valid_time", "min"),
        valid_time_max=("valid_time", "max"),
        wind_speed_max=("wind_speed", "max"),
    )
    .reset_index()
).dropna()
```

```python
df_tracks_agg
```

```python
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
```

```python
import warnings

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
```

```python
get_storm_rainfall_aggregations(df_tracks_agg.iloc[0])
```

```python
tqdm.pandas()
```

```python
df_tracks_agg = df_tracks_agg.progress_apply(
    get_storm_rainfall_aggregations, axis=1
)
```

```python
df_tracks_agg
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs_imerg_stats_50km.parquet"
stratus.upload_parquet_to_blob(df_tracks_agg, blob_name)
```
