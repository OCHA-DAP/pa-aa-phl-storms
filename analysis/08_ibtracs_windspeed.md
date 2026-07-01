---
jupyter:
  jupytext:
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

# Notebook to review wind speed thresholds in the trigger for the typhoon framework

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from pathlib import Path
import os
import warnings
from datetime import datetime
import zipfile
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import rasterstats as rs
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, mapping
from rasterio.features import geometry_mask
from rasterio.mask import mask
from src.datasources import imerg
from src.constants import *
import math
```

```python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
```

```python
zip_path = (
    Path()
    / "public"
    / "raw"
    / "phl"
    / "cod_ab"
    / "phl_adm_psa_namria_20231106_shp.zip"
)
phl_adm1_gdf = gpd.read_file(
    f"zip://{zip_path}!phl_admbnda_adm1_psa_namria_20231106.shp"
)
phl_adm2_gdf = gpd.read_file(
    f"zip://{zip_path}!phl_admbnda_adm2_psa_namria_20231106.shp"
)
```

```python
phl_sel_adm1 = phl_adm1_gdf[
    phl_adm1_gdf["ADM1_PCODE"].isin(["PH02", "PH05", "PH08", "PH16"])
]
fig, ax = plt.subplots(figsize=(10, 10))
phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)
phl_sel_adm1.plot(ax=ax, color="teal", edgecolor="black", alpha=0.5)

for idx, row in phl_sel_adm1.iterrows():
    ax.annotate(
        row["ADM1_EN"],
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        ha="center",
        fontsize=9,
        color="black",
    )

plt.title("ADM1 Regions in Current Framework: V, VIII, XIII")
plt.axis("off")
plt.show()
```

```python
ds = xr.open_dataset(
    Path(AA_DATA_DIR)
    / "public"
    / "raw"
    / "glb"
    / "ibtracs"
    / "IBTrACS.WP.v04r01.nc"
)
# [var for var in ds.data_vars if var.startswith("us")]
ds[
    [
        "sid",
        "name",
        "lat",
        "lon",
        "wmo_wind",
        "time",
        "usa_r50",
        "usa_r64",
        "tokyo_r50_long",
        "tokyo_r50_short",
    ]
]
```

```python
df_init = (
    ds[
        [
            "sid",
            "name",
            "lat",
            "lon",
            "wmo_wind",
            "time",
            "usa_r50",
            "usa_r64",
            "tokyo_r50_long",
            "tokyo_r50_short",
        ]
    ]
    .to_dataframe()
    .reset_index()
)
df = (
    df_init.groupby(["sid", "time", "lat", "lon", "name", "wmo_wind"])
    .agg(
        {
            "usa_r50": "max",
            "usa_r64": "max",
            "tokyo_r50_long": "max",
            "tokyo_r50_short": "max",
        }
    )
    .reset_index()
)
df = df.dropna(subset=["lat", "lon"])
df["name"] = df["name"].apply(
    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
)
df["sid"] = df["sid"].apply(
    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
)
df.head(5)
```

```python
# using the 2025 file
worldpop_data_url = (
    Path(AA_DATA_DIR)
    / "public"
    / "raw"
    / "phl"
    / "worldpop"
    / "phl_pop_2025_CN_100m_R2024B_v1.tif"
)
# read in the file
worldpop_file = rasterio.open(worldpop_data_url)
worldpop_data = worldpop_file.read(1)
worldpop_data_masked = np.ma.masked_equal(worldpop_data, -99999)
mean_value = np.mean(worldpop_data_masked)
min_value = np.min(worldpop_data_masked)
max_value = np.max(worldpop_data_masked)

print(f"Min value: {min_value}")
print(f"Max value: {max_value}")
print(f"Mean: {mean_value}")
```

Scenario 1 covers pilot areas in Region 5 and Samar Island Provinces in Region 8. Scenario 1 is based on a typhoon forecast track with a general direction and cone of uncertainty towards region 5 and Samar Island Provinces: Northern Samar/Eastern Samar/Western Samar. Scenario 2 covers Region 13 (Caraga) and Region 8, Leyte Island Provinces, namely Leyte and Southern Leyte.

```python
# leaving our Biliran data out for now
phl_adm2_gdf[(phl_adm2_gdf["ADM1_PCODE"] == "PH08")]
```

```python
gdf_adm2_scenario1 = phl_adm2_gdf[
    (phl_adm2_gdf["ADM1_PCODE"] == "PH05")
    | (phl_adm2_gdf["ADM2_PCODE"].isin(["PH08026", "PH08048", "PH08060"]))
]
gdf_adm2_scenario2 = phl_adm2_gdf[
    (phl_adm2_gdf["ADM1_PCODE"] == "PH16")
    | (phl_adm2_gdf["ADM2_PCODE"].isin(["PH08037", "PH08064"]))
]
scenario1_poly = gdf_adm2_scenario1.dissolve()
scenario2_poly = gdf_adm2_scenario2.dissolve()
```

```python
fig, ax = plt.subplots(figsize=(10, 10))
phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)
scenario1_poly.plot(ax=ax, color="teal", edgecolor="black", alpha=0.4)
scenario2_poly.plot(ax=ax, color="red", edgecolor="black", alpha=0.4)

plt.title(
    "PHL Scenario 1: V and Selected Regions in VIII \n Scenario 2: XIII and Selected Regions in VIII"
)
ax.annotate(
    "Scenario 1",
    xy=(
        scenario1_poly.geometry.centroid.x,
        scenario1_poly.geometry.centroid.y,
    ),
    ha="center",
    fontsize=10,
    fontweight="bold",
    color="black",
)
ax.annotate(
    "Scenario 2",
    xy=(
        scenario2_poly.geometry.centroid.x,
        scenario2_poly.geometry.centroid.y,
    ),
    ha="center",
    fontsize=10,
    fontweight="bold",
    color="black",
)
plt.axis("off")
plt.show()
```

```python
# get subsection of the worldpop data for the selected areas
worldpop_data_agg_1 = rs.zonal_stats(
    vectors=scenario1_poly,
    raster=worldpop_data_masked,
    stats=["sum"],
    nodata=-99999,
    affine=worldpop_file.transform,
)
worldpop_data_agg_2 = rs.zonal_stats(
    vectors=scenario2_poly,
    raster=worldpop_data_masked,
    stats=["sum"],
    nodata=-99999,
    affine=worldpop_file.transform,
)
print(worldpop_data_agg_1)
print(worldpop_data_agg_2)
# cropping the raster to the selected areas
scenario1_geom = [mapping(geom) for geom in scenario1_poly.geometry]
scenario2_geom = [mapping(geom) for geom in scenario2_poly.geometry]

worldpop_mask_1 = geometry_mask(
    scenario1_geom,
    transform=worldpop_file.transform,
    invert=True,
    out_shape=worldpop_data_masked.shape,
)
worldpop_mask_2 = geometry_mask(
    scenario2_geom,
    transform=worldpop_file.transform,
    invert=True,
    out_shape=worldpop_data_masked.shape,
)
```

```python
out_image_1, out_transform_1 = mask(worldpop_file, scenario1_geom, crop=True)
out_image_2, out_transform_2 = mask(worldpop_file, scenario2_geom, crop=True)
```

```python
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

axs[0].imshow(out_image_1[0], cmap="viridis")
axs[0].set_title("Population Raster - Scenario 1")
axs[0].axis("off")

axs[1].imshow(out_image_2[0], cmap="viridis")
axs[1].set_title("Population Raster - Scenario 2")
axs[1].axis("off")

plt.tight_layout()
plt.show()
```

```python
# starting the 15 min interpolation
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(["sid", "time"])

interpolated_groups = []

for sid, group in df.groupby("sid"):
    group = group.set_index("time")

    # Create time range only between the first and last timestamp
    time_index = pd.date_range(
        start=group.index.min(), end=group.index.max(), freq="15min"
    )

    # Reindex to that range
    group = group.reindex(time_index)

    # Interpolate numeric columns
    interpolated = group[
        [
            "lat",
            "lon",
            "wmo_wind",
            "usa_r50",
            "usa_r64",
            "tokyo_r50_long",
            "tokyo_r50_short",
        ]
    ].interpolate()

    # Forward-fill non-numeric columns
    non_numeric = group[["sid", "name"]].ffill()

    # Combine and reset
    combined = pd.concat([interpolated, non_numeric], axis=1).reset_index()
    combined = combined.rename(columns={"index": "time"})
    interpolated_groups.append(combined)

interp_df = pd.concat(interpolated_groups, ignore_index=True)

# Geometry creation
geometry = [Point(xy) for xy in zip(interp_df["lon"], interp_df["lat"])]
ibtracs_gdf = gpd.GeoDataFrame(interp_df, geometry=geometry, crs="EPSG:4326")
```

```python
# we have two conditions to test for readiness
# Readiness 1: the storm has a wind speed of at least 136 km/h (75 knots)
wind_threshold_kph_1 = 136
wind_threshold_kts_1 = wind_threshold_kph_1 * 0.539957
wind_threshold_kts_1
```

```python
# Readiness 2: the storm has a wind speed of at least 118 km/h (64 knots)
wind_threshold_kph_2 = 118
wind_threshold_kts_2 = wind_threshold_kph_2 * 0.539957
wind_threshold_kts_2
```

```python
# Readiness 3: the storm has a wind speed of at least 177 km/h (96 knots)
wind_threshold_kph_3 = 177
wind_threshold_kts_3 = wind_threshold_kph_3 * 0.539957
wind_threshold_kts_3
```

```python
# Activation/Observational: the storm has a wind speed of at least 185 km/h (100 knots)
wind_threshold_kph_4 = 185
wind_threshold_kts_4 = wind_threshold_kph_4 * 0.539957
wind_threshold_kts_4
```

```python
# checking which storm points are intersecting with the country
ibtracs_gdf.crs == phl_adm1_gdf.crs
intersecting_storms = gpd.sjoin(
    ibtracs_gdf, phl_adm1_gdf, how="inner", predicate="intersects"
)
```

```python
(
    intersecting_storms["tokyo_r50_long"]
    - intersecting_storms["tokyo_r50_short"]
).describe()
```

```python
# using tokyo data for the intersection but filling in the missing values

intersecting_storms["tok_usa_r50"] = (
    intersecting_storms["tokyo_r50_long"]
    .fillna(intersecting_storms["tokyo_r50_short"])
    .fillna(intersecting_storms["usa_r50"])
)
```

```python
intersecting_storm_ids = intersecting_storms["sid"].unique()
# Filter the storms that intersect with the Philippines
filtered_storms = ibtracs_gdf[
    ibtracs_gdf["sid"].isin(intersecting_storm_ids)
    & (ibtracs_gdf["time"] >= "1980-01-01")
]
```

```python
landfall_df = intersecting_storms.sort_values("time").drop_duplicates(
    subset="sid", keep="first"
)
landfall_df["landfall_date"] = landfall_df["time"].dt.date
landfall_df
```

```python
landfall_df.to_csv(
    Path(AA_DATA_DIR)
    / "public"
    / "raw"
    / "phl"
    / "landfall_storms.csv",
    index=False,
)
```

```python
# checking if landfall df has non na in the usa_r50
landfall_df["tok_usa_r50"].notna().any()
```

```python
# checking the number of storms that have a wind radius
landfall_df["tok_usa_r50"].notna().sum()
```

```python
# We get the time stamp from the same day as landfall but has a wind radius
df_with_landfall = intersecting_storms.merge(
    landfall_df[["sid", "landfall_date"]], on="sid", how="left"
)

# Keep only rows matching the landfall date and with non-NaN usa_r50
filtered = df_with_landfall[
    (df_with_landfall["time"].dt.date <= df_with_landfall["landfall_date"])
    & (~df_with_landfall["tok_usa_r50"].isna())
]
last_valid_r50 = (
    filtered.sort_values(["sid", "time"]).groupby("sid").last().reset_index()
)

# Convert to km
last_valid_r50["tok_usa_r50_km"] = last_valid_r50["tok_usa_r50"] * 1.852
last_valid_r50["tok_usa_r50_km"].describe()
```

Using the 50 knots radius

```python
# Not needed!!!!
filtered = df_with_landfall[
    (df_with_landfall["time"].dt.date <= df_with_landfall["landfall_date"])
    & (~df_with_landfall["usa_r64"].isna())
]
last_valid_r64 = (
    filtered.sort_values(["sid", "time"]).groupby("sid").last().reset_index()
)

# Convert to km
last_valid_r64["usa_r64_km"] = last_valid_r64["usa_r64"] * 1.852
last_valid_r64["usa_r64_km"].describe()
# even fewer storms have a wind radius of 64 knots, so we will use the 50 knot radius for our analysis
```

```python
landfall_df["tok_usa_r50_km"] = landfall_df["tok_usa_r50"] * 1.852
landfall_df["tok_usa_r50_km"].describe()
```

Using r 50 since this is suggested as a threshold for damaging winds

```python
gdf_landfall = gpd.GeoDataFrame(
    landfall_df, geometry="geometry", crs="EPSG:4326"
)
gdf_landfall = gdf_landfall[~gdf_landfall["tok_usa_r50_km"].isna()].copy()
gdf_landfall = gdf_landfall.to_crs(epsg=32651)
gdf_landfall["radius_m"] = gdf_landfall["tok_usa_r50_km"] * 1000
gdf_landfall["circle"] = gdf_landfall.buffer(gdf_landfall["radius_m"])
gdf_landfall["area_km2"] = gdf_landfall["circle"].area / 1e6
gdf_landfall = gdf_landfall.to_crs("EPSG:4326")
gdf_landfall["circle"] = gdf_landfall["circle"].to_crs(epsg=4326)
print(gdf_landfall["circle"].total_bounds)
scenario1_geom = scenario1_poly.geometry.iloc[0]
scenario2_geom = scenario2_poly.geometry.iloc[0]
gdf_landfall["circle_scenario1"] = gdf_landfall["circle"].intersection(
    scenario1_geom
)
gdf_landfall["circle_scenario2"] = gdf_landfall["circle"].intersection(
    scenario2_geom
)
gdf_landfall["scenario1_pop"] = worldpop_data_agg_1[0]["sum"]
gdf_landfall["scenario2_pop"] = worldpop_data_agg_2[0]["sum"]
gdf_landfall.head(6)
```

```python
gdf_landfall_valid_1 = gdf_landfall[
    gdf_landfall["circle_scenario1"].notnull()
    & gdf_landfall["circle_scenario1"].apply(lambda x: not x.is_empty)
]
gdf_landfall_valid_2 = gdf_landfall[
    gdf_landfall["circle_scenario2"].notnull()
    & gdf_landfall["circle_scenario2"].apply(lambda x: not x.is_empty)
]
gdf_landfall_valid_1.shape, gdf_landfall_valid_2.shape
```

```python
pop_stats_scenario1 = rs.zonal_stats(
    vectors=[
        mapping(geom) for geom in gdf_landfall_valid_1["circle_scenario1"]
    ],
    raster=worldpop_data,
    stats=["sum"],
    nodata=-99999,
    affine=worldpop_file.transform,
)
pop_stats_df_1 = pd.DataFrame(pop_stats_scenario1)
pop_stats_scenario2 = rs.zonal_stats(
    vectors=[
        mapping(geom) for geom in gdf_landfall_valid_2["circle_scenario2"]
    ],
    raster=worldpop_data,
    stats=["sum"],
    nodata=-99999,
    affine=worldpop_file.transform,
)
pop_stats_df_2 = pd.DataFrame(pop_stats_scenario2)
pop_stats_df_1.shape, pop_stats_df_2.shape
```

```python
gdf_landfall_valid_1 = gdf_landfall_valid_1.copy()
gdf_landfall_valid_1.loc[:, "sum_scenario1"] = pop_stats_df_1["sum"].values
gdf_landfall_valid_1.loc[:, "pop_ratio_scenario1"] = (
    gdf_landfall_valid_1["sum_scenario1"]
    / gdf_landfall_valid_1["scenario1_pop"]
)

gdf_landfall_valid_2 = gdf_landfall_valid_2.copy()
gdf_landfall_valid_2.loc[:, "sum_scenario2"] = pop_stats_df_2["sum"].values
gdf_landfall_valid_2.loc[:, "pop_ratio_scenario2"] = (
    gdf_landfall_valid_2["sum_scenario2"]
    / gdf_landfall_valid_2["scenario2_pop"]
)

gdf_landfall_valid_1.sort_values(by="sum_scenario1", ascending=False).head(6)
```

```python
fig, ax = plt.subplots(figsize=(5, 10), dpi=300)
phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)
scenario1_poly.plot(ax=ax, color="teal", edgecolor="black", alpha=0.4)

plt.title(
    "PHL Scenario 1: V and Selected Regions in VIII \n Typhoon Goni 2020 Landfall"
)
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2020299N11144"][
        "geometry"
    ]
).plot(
    ax=ax,
    facecolor="red",
    edgecolor="red",
    linewidth=1,
    label="Landfall Point",
)
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2020299N11144"][
        "circle"
    ]
).plot(
    ax=ax, facecolor="none", edgecolor="orange", linewidth=1, label="circle"
)
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2020299N11144"][
        "circle_scenario1"
    ]
).plot(
    ax=ax,
    facecolor="none",
    edgecolor="maroon",
    linewidth=1,
    label="circle_scenario1",
)
plt.axis("off")
plt.show()
```

```python
fig, ax = plt.subplots(figsize=(5, 10), dpi=300)
phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)
scenario1_poly.plot(ax=ax, color="teal", edgecolor="black", alpha=0.4)
scenario2_poly.plot(ax=ax, color="maroon", edgecolor="black", alpha=0.4)

plt.title("PHL Scenario 1 and 2 \n Typhoon Kammuri 2019 Landfall")
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2019329N09160"][
        "geometry"
    ]
).plot(
    ax=ax,
    facecolor="red",
    edgecolor="red",
    linewidth=1,
    label="Landfall Point",
)
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2019329N09160"][
        "circle"
    ]
).plot(
    ax=ax, facecolor="none", edgecolor="orange", linewidth=1, label="circle"
)
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2019329N09160"][
        "circle_scenario1"
    ]
).plot(
    ax=ax,
    facecolor="none",
    edgecolor="maroon",
    linewidth=1,
    label="circle_scenario1",
)
plt.axis("off")
plt.show()
```

```python
fig, ax = plt.subplots(figsize=(5, 10), dpi=300)
phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)
scenario1_poly.plot(ax=ax, color="maroon", edgecolor="black", alpha=0.4)
scenario2_poly.plot(ax=ax, color="teal", edgecolor="black", alpha=0.4)

plt.title("PHL Scenario 1 and 2 \n Typhoon Rai 2021 Landfall")
gpd.GeoSeries(
    gdf_landfall_valid_2[gdf_landfall_valid_2["sid"] == "2021346N05145"][
        "geometry"
    ]
).plot(
    ax=ax,
    facecolor="red",
    edgecolor="red",
    linewidth=1,
    label="Landfall Point",
)
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2021346N05145"][
        "circle_scenario1"
    ]
).plot(
    ax=ax,
    facecolor="none",
    edgecolor="yellow",
    linewidth=1,
    label="circle_scenario1",
)
gpd.GeoSeries(
    gdf_landfall_valid_2[gdf_landfall_valid_2["sid"] == "2021346N05145"][
        "circle"
    ]
).plot(
    ax=ax, facecolor="none", edgecolor="orange", linewidth=1, label="circle"
)
gpd.GeoSeries(
    gdf_landfall_valid_2[gdf_landfall_valid_2["sid"] == "2021346N05145"][
        "circle_scenario2"
    ]
).plot(
    ax=ax,
    facecolor="none",
    edgecolor="maroon",
    linewidth=1,
    label="circle_scenario2",
)
plt.axis("off")
plt.show()
```

```python
fig, ax = plt.subplots(figsize=(5, 10))
phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)
scenario1_poly.plot(ax=ax, color="maroon", edgecolor="black", alpha=0.4)
scenario2_poly.plot(ax=ax, color="teal", edgecolor="black", alpha=0.4)

plt.title("PHL Scenario 1 and 2 \n Typhoon Haiyan 2013 Landfall")
gpd.GeoSeries(
    gdf_landfall_valid_2[gdf_landfall_valid_2["sid"] == "2013306N07162"][
        "geometry"
    ]
).plot(
    ax=ax,
    facecolor="red",
    edgecolor="red",
    linewidth=1,
    label="Landfall Point",
)
gpd.GeoSeries(
    gdf_landfall_valid_1[gdf_landfall_valid_1["sid"] == "2013306N07162"][
        "circle_scenario1"
    ]
).plot(
    ax=ax,
    facecolor="none",
    edgecolor="yellow",
    linewidth=1,
    label="circle_scenario1",
)
gpd.GeoSeries(
    gdf_landfall_valid_2[gdf_landfall_valid_2["sid"] == "2013306N07162"][
        "circle"
    ]
).plot(
    ax=ax, facecolor="none", edgecolor="orange", linewidth=1, label="circle"
)
gpd.GeoSeries(
    gdf_landfall_valid_2[gdf_landfall_valid_2["sid"] == "2013306N07162"][
        "circle_scenario2"
    ]
).plot(
    ax=ax,
    facecolor="none",
    edgecolor="maroon",
    linewidth=1,
    label="circle_scenario2",
)
plt.axis("off")
plt.show()
```

```python
gdf_landfall_valid_1 = gdf_landfall_valid_1.copy()
gdf_landfall_valid_2 = gdf_landfall_valid_2.copy()

gdf_landfall_valid_1["year"] = gdf_landfall_valid_1["time"].dt.year
gdf_landfall_valid_2["year"] = gdf_landfall_valid_2["time"].dt.year

years_1 = gdf_landfall_valid_1.loc[
    gdf_landfall_valid_1["pop_ratio_scenario1"] >= 0.6, "year"
].unique()

years_2 = gdf_landfall_valid_2.loc[
    gdf_landfall_valid_2["pop_ratio_scenario2"] >= 0.6, "year"
].unique()

govt_method_years = sorted(set(years_1) | set(years_2))
```

```python
def calculate_return_period(
    intersecting_storms, wind_threshold_kts, start_date="1980-01-01"
):
    # Filter and copy to avoid SettingWithCopyWarning
    filtered_storms_1990 = intersecting_storms[
        (intersecting_storms["time"] >= start_date)
    ].copy()
    filtered_storms = filtered_storms_1990[
        filtered_storms_1990["wmo_wind"] > wind_threshold_kts
    ].copy()

    # Unique storms
    unique_storms = filtered_storms.drop_duplicates(subset=["sid"])
    num_storms = len(unique_storms)

    # Extract year and name
    filtered_storms.loc[:, "year"] = pd.to_datetime(
        filtered_storms["time"]
    ).dt.year
    unique_years = sorted(filtered_storms["year"].unique())
    unique_years = [int(yr) for yr in unique_years]
    unique_names = filtered_storms["name"].unique()

    # Return period
    start_year = pd.to_datetime(filtered_storms_1990["time"]).min().year
    end_year = pd.to_datetime(filtered_storms_1990["time"]).max().year
    num_years = end_year - start_year + 1
    return_period = num_years / num_storms if num_storms > 0 else None

    # Print results
    print(
        f"Number of unique storms for > {wind_threshold_kts:.1f} knots: {num_storms}"
    )
    print(f"Return period: {return_period:.1f} years over {num_years} years.")
    print(f"Unique years with qualifying storms: {len(unique_years)}")
    print(f"Years: {unique_years}")
    print(f"Unique storms: {len(unique_names)}")
    print(f"Storms: {unique_names}")

    return
```

For the whole country

```python
calculate_return_period(intersecting_storms, wind_threshold_kts_1)
```

```python
calculate_return_period(intersecting_storms, wind_threshold_kts_2)
```

```python
calculate_return_period(intersecting_storms, wind_threshold_kts_3)
```

```python
calculate_return_period(intersecting_storms, wind_threshold_kts_4)
```

For the 3 selected regions

```python
# checking which storm points are intersecting with the country
ibtracs_gdf.crs == phl_adm1_gdf.crs
intersecting_storms_sel = gpd.sjoin(
    ibtracs_gdf, phl_sel_adm1, how="inner", predicate="intersects"
)
```

```python
calculate_return_period(intersecting_storms_sel, wind_threshold_kts_1)
```

```python
calculate_return_period(intersecting_storms_sel, wind_threshold_kts_2)
```

```python
calculate_return_period(intersecting_storms_sel, wind_threshold_kts_3)
```

```python
calculate_return_period(intersecting_storms_sel, wind_threshold_kts_4)
```

Just a note: If we do the interpolation to 30 mins then the buffer may not really be needed.

```python
# checking why MAN-YI would not have been included with no buffer
bins = [0, 34, 48, 64, 100, 999]
labels = [
    "<34 kt",
    "34–47 kt",
    "48–63 kt",
    "64–99 kt",
    ">100 kt",
]
man_yi = ibtracs_gdf[ibtracs_gdf["name"] == "MAN-YI"].copy()
man_yi["wind_bin"] = pd.cut(man_yi["wmo_wind"], bins=bins, labels=labels)
# Spatial join to find landfall points
man_yi_land = gpd.sjoin(
    man_yi, phl_adm1_gdf, how="inner", predicate="intersects"
)

fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)
man_yi.plot(
    ax=ax,
    column="wind_bin",
    cmap="YlOrRd",
    legend=True,
    edgecolor="none",
    alpha=0.7,
    markersize=10,
)
man_yi_land.plot(
    ax=ax,
    color="black",
    markersize=20,
    marker="x",
    label="On land",
)
plt.axis("off")
plt.title("Typhoon Man-Yi Wind Speed (kt) in 2024")
plt.ylim(5, 20)
plt.xlim(115, 130)
```

```python
# interactions
years = list(range(1980, 2024))
readiness_64kt = [
    1981,
    1982,
    1983,
    1984,
    1986,
    1987,
    1988,
    1990,
    1993,
    1994,
    1995,
    1998,
    2004,
    2006,
    2008,
    2012,
    2013,
    2014,
    2015,
    2016,
    2019,
    2020,
    2021,
]
readiness_74kt = [
    1981,
    1982,
    1984,
    1987,
    1990,
    1993,
    1995,
    1998,
    2004,
    2006,
    2008,
    2012,
    2013,
    2014,
    2015,
    2016,
    2019,
    2020,
    2021,
]
ws_method_years = [1984, 1987, 2013, 2020, 2021]

df = pd.DataFrame(
    {
        "Year": years,
        "Wind Speed 100kt": [
            "Yes" if y in ws_method_years else "No" for y in years
        ],
    }
)

model_output_years = [2013, 2014, 2016, 2019, 2021]
df["510 Model"] = ["Yes" if y in model_output_years else "No" for y in years]
df["PDRA"] = ["Yes" if y in govt_method_years else "No" for y in years]
df["64kt Readiness"] = ["Yes" if y in readiness_64kt else "No" for y in years]
df["Current Readiness"] = [
    "Yes" if y in readiness_74kt else "No" for y in years
]
# Reshape for plotting
df_melted = df.melt(id_vars="Year", var_name="Method", value_name="Threshold")

# Plot
plt.figure(figsize=(25, 5), dpi=300)
ax = sns.heatmap(
    df_melted.pivot(index="Method", columns="Year", values="Threshold")
    == "Yes",
    cmap=["lightgrey", "darkred"],
    cbar=False,
    linewidths=0.5,
    linecolor="white",
)
plt.title("Threshold Exceedance by Year and Method")
plt.xlabel("Year")
plt.ylabel("Method")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
ax.title.set_fontsize(22)
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.tight_layout()
plt.show()
```

```python
# rainfall for days before, during, and after the storm
landfall_df = landfall_df[landfall_df["time"] >= "2000-06-01"]
IMERG_START_DATE = pd.to_datetime("2000-06-01")
extra_days = 1
dfs = []
for sid, row in landfall_df.set_index("sid").iterrows():
    landfall_date = row["time"]
    start_date = landfall_date - pd.Timedelta(days=extra_days)
    end_date = landfall_date + pd.Timedelta(days=extra_days)
    if pd.Timestamp(end_date) < IMERG_START_DATE:
        print(f"{row['name']} too early")
        continue
    df_in = imerg.fetch_imerg_data(
        phl_sel_adm1["ADM1_PCODE"], start_date, end_date
    )
    df_in["sid"] = sid
    dfs.append(df_in)

imerg_df = pd.concat(dfs, ignore_index=True)
imerg_sum_df = imerg_df.groupby(["pcode", "sid"])["mean"].sum().reset_index()
imerg_sum_df = imerg_sum_df.rename(columns={"mean": "sum_mean_rain"})
```

```python
imerg_landfall_df = landfall_df.merge(imerg_sum_df)
```

```python
combined_df = imerg_landfall_df.merge(
    phl_sel_adm1.rename(columns={"ADM1_PCODE": "pcode"})[["pcode", "ADM1_EN"]]
)
```

```python
combined_df["year"] = combined_df["landfall_date"].apply(lambda x: x.year)
combined_df["nameseason"] = (
    combined_df["name"].str.capitalize()
    + " "
    + combined_df["year"].astype(str)
)
```

```python
def calculate_rp(group, col_name, total_seasons):
    group["rank"] = group[col_name].rank(ascending=False)
    group["rp"] = (total_seasons + 1) / group["rank"]
    return group


total_seasons = combined_df["year"].nunique() - 1
```

```python
rp = 3
col_name = "sum_mean_rain"
color = "crimson"

for pcode, group in combined_df.groupby("pcode"):
    fig, ax = plt.subplots(dpi=200)

    # calculate RP based only on complete seasons
    dff = group[group["year"] <= 2024].copy()
    dff = calculate_rp(dff, col_name, total_seasons)
    dff = dff.sort_values("rp")

    # interpolate return value
    rv = np.interp(rp, dff["rp"], dff[col_name])
    top_edge = dff[col_name].max() * 1.1
    right_edge = dff["wmo_wind"].max() + 10
    ax.plot(
        group["wmo_wind"],
        group[col_name],
        linestyle="none",
        marker=".",
        color="k",
    )
    ax.axhline(rv, linewidth=1, color=color)
    ax.axhspan(rv, top_edge, color=color, alpha=0.1)
    ax.annotate(
        f" 3-yr RP:\n {rv:.0f} mm",
        (right_edge, rv),
        va="center",
        color=color,
    )

    # annotate high rainfall events
    for nameseason, row in group.set_index("nameseason").iterrows():
        if row[col_name] > group[col_name].median():
            ax.annotate(
                f" {nameseason}",
                (row["wmo_wind"], row[col_name]),
                fontsize=8,
                va="center",
            )

    ax.legend().remove()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(right=right_edge)
    ax.set_ylim(bottom=0, top=top_edge)
    ax.set_title(group.iloc[0]["ADM1_EN"])
    ax.set_xlabel("Landfall wind speed (knots)")
    ax.set_ylabel(
        "Three-day total rainfall averaged across region,\ncentered on landfall date (mm) [IMERG]"
    )
```

```python
# those above 64 knots
# these to show
storms_to_show = gdf_landfall[
    (gdf_landfall["wmo_wind"] >= 80) & (gdf_landfall["time"] >= "2000-01-01")
].copy()
storms_to_show["storm_year"] = (
    storms_to_show["name"] + " " + storms_to_show["time"].dt.year.astype(str)
)
```

```python
readiness_storms = storms_to_show[
    storms_to_show["name"].isin(
        [
            "CLARA",
            "NANCY",
            "WAYNE",
            "AGNES",
            "BETTY",
            "ZEB",
            "CIMARON",
            "MEGI",
            "HAIYAN",
            "MERANTI",
            "MANGKHUT",
            "GONI",
            "RAI",
            "DOKSURI",
            "YINXING",
            "MAN-YI",
        ]
    )
]["storm_year"]
activation_wind_storms = storms_to_show[
    storms_to_show["name"].isin(["AGNES", "BETTY", "HAIYAN", "GONI", "RAI"])
]["storm_year"]
activation_pop_storms = storms_to_show[
    storms_to_show["sid"].isin(
        pd.concat(
            [
                gdf_landfall_valid_1.loc[
                    gdf_landfall_valid_1["pop_ratio_scenario1"] > 0.6, "sid"
                ],
                gdf_landfall_valid_2.loc[
                    gdf_landfall_valid_2["pop_ratio_scenario2"] > 0.6, "sid"
                ],
            ]
        ).unique()
    )
]["storm_year"]
observational_wind_storms = storms_to_show[
    storms_to_show["name"].isin(["AGNES", "BETTY", "HAIYAN", "GONI", "RAI"])
]["storm_year"]
observational_rain_storms = storms_to_show[
    storms_to_show["sid"].isin(
        imerg_landfall_df[
            (imerg_landfall_df["wmo_wind"] > 64)
            & (imerg_landfall_df["sum_mean_rain"] >= 200)
        ]["sid"]
    )
]["storm_year"]
```

```python
pd.concat(
    [
        gdf_landfall_valid_1.loc[
            gdf_landfall_valid_1["pop_ratio_scenario1"] > 0.6, "sid"
        ],
        gdf_landfall_valid_2.loc[
            gdf_landfall_valid_2["pop_ratio_scenario2"] > 0.6, "sid"
        ],
    ]
).unique()
```

```python
gdf_landfall[gdf_landfall["sid"] == "2022299N11134"]
```

```python
activation_pop_storms
```

```python
len(
    pd.concat(
        [
            gdf_landfall_valid_1.loc[
                gdf_landfall_valid_1["pop_ratio_scenario1"] > 0.6, "sid"
            ],
            gdf_landfall_valid_2.loc[
                gdf_landfall_valid_2["pop_ratio_scenario2"] > 0.6, "sid"
            ],
        ]
    ).unique()
)
```

```python
45 / len(
    pd.concat(
        [
            gdf_landfall_valid_1.loc[
                gdf_landfall_valid_1["pop_ratio_scenario1"] > 0.6, "sid"
            ],
            gdf_landfall_valid_2.loc[
                gdf_landfall_valid_2["pop_ratio_scenario2"] > 0.6, "sid"
            ],
        ]
    ).unique()
)
```

```python
imerg_landfall_df[
    (imerg_landfall_df["wmo_wind"] > 64)
    & (imerg_landfall_df["sum_mean_rain"] >= 200)
]["name"].unique()
```

```python
imerg_landfall_df[
    (imerg_landfall_df["wmo_wind"] > 64)
    & (imerg_landfall_df["sum_mean_rain"] >= 300)
]["name"].unique()
```

```python
25 / len(
    imerg_landfall_df[
        (imerg_landfall_df["wmo_wind"] > 64)
        & (imerg_landfall_df["sum_mean_rain"] >= 300)
    ]["name"].unique()
)
```

```python
imerg_landfall_df[imerg_landfall_df["name"] == "RAI"]
```

```python
gdf_landfall_valid_2[gdf_landfall_valid_2["name"] == "HAIYAN"]
```

```python
storm_df = pd.DataFrame(
    {
        "Storm": storms_to_show["storm_year"],
        "Readiness": storms_to_show["storm_year"]
        .isin(readiness_storms)
        .map({True: "Yes", False: ""}),
        "Activation(Speed)": storms_to_show["storm_year"]
        .isin(activation_wind_storms)
        .map({True: "Yes", False: ""}),
        "Activation(Pop)": storms_to_show["storm_year"]
        .isin(activation_pop_storms)
        .map({True: "Yes", False: ""}),
        "Observational(Speed)": storms_to_show["storm_year"]
        .isin(observational_wind_storms)
        .map({True: "Yes", False: ""}),
        "Observational(Rainfall)": storms_to_show["storm_year"]
        .isin(observational_rain_storms)
        .map({True: "Yes", False: ""}),
    }
)

storm_df
```
