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

# Review of Landfall Wind Speed and Rainfall


Reviewing rainfall thresholds and the landfall wind speeds

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from pathlib import Path
import os
import glob
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
import re
```

```python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
```

```python
trigger_dir = (
    Path(AA_DATA_DIR)
    / "public/exploration/phl/trigger_performance"
)
main_dir = Path(AA_DATA_DIR) / "public/exploration/phl"
```

```python
zip_path = (
    Path(AA_DATA_DIR)
    / "public"
    / "raw"
    / "phl"
    / "cod_ab"
    / "phl_adm_psa_namria_20231106_shp.zip"
)
phl_adm1_gdf = gpd.read_file(
    f"zip://{zip_path}!phl_admbnda_adm1_psa_namria_20231106.shp"
)
```

```python
landfall_df = pd.read_csv(
    Path(AA_DATA_DIR)
    / "public"
    / "raw"
    / "phl"
    / "landfall_storms.csv"
)
landfall_df.columns
```

```python
phl_sel_adm1 = phl_adm1_gdf[
    phl_adm1_gdf["ADM1_PCODE"].isin(["PH02", "PH03", "PH05", "PH08", "PH16"])
]
fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

phl_adm1_gdf.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.5)

phl_sel_adm1[phl_sel_adm1["ADM1_PCODE"].isin(["PH02", "PH03"])].plot(
    ax=ax, color="lightgreen", edgecolor="black", alpha=0.5
)

phl_sel_adm1[phl_sel_adm1["ADM1_PCODE"].isin(["PH05", "PH08", "PH16"])].plot(
    ax=ax, color="teal", edgecolor="black", alpha=0.5
)

for idx, row in phl_sel_adm1.iterrows():
    ax.annotate(
        row["ADM1_EN"],
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        ha="center",
        fontsize=9,
        color="black",
    )

plt.title(
    "ADM1 Regions in Current Framework: II, V, VIII, XIII\nWFP in Region II and III"
)
plt.axis("off")
plt.show()
```

```python
# Readiness: the storm has a wind speed of at least 177 km/h (96 knots)
wind_threshold_kph_1 = 177  # 118 or 136 or 177
wind_threshold_kts_1 = wind_threshold_kph_1 * 0.539957
wind_threshold_mps_1 = wind_threshold_kph_1 / 3.6
wind_threshold_kts_1
```

```python
# Activation/Observational: the storm has a wind speed of at least 185 km/h (100 knots)
wind_threshold_kph_2 = 185
wind_threshold_kts_2 = wind_threshold_kph_2 * 0.539957
wind_threshold_mps_2 = wind_threshold_kph_2 / 3.6
wind_threshold_kts_2
```

```python
# Rainfall
rain_threshold_mm_1 = 300
min_rainfall_speed = 64
```

```python
# looking at readiness
# hindcast data
readiness_max_leadtime = 120
readiness_min_leadtime = 72
csv_path = main_dir / "ecmwf_hindcast/csv/"
csv_files = glob.glob(os.path.join(csv_path, "*_all.csv"))
# Collect all filtered data
filtered_dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    if "lon" not in df.columns or "lat" not in df.columns:
        continue

    storm_name = os.path.basename(file).replace("_all.csv", "")
    print(f"Processing storm: {storm_name}")
    df["storm"] = storm_name

    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=phl_sel_adm1.crs)

    # Identify lead time and wind speed columns
    lead_col = next(
        (col for col in ["lead_time", "vhr"] if col in gdf_points.columns),
        None,
    )
    wind_col = next(
        (
            col
            for col in ["wind_speed", "speed", "cyc_speed"]
            if col in gdf_points.columns
        ),
        None,
    )

    if (
        lead_col
        and wind_col
        and {"forecast_time", "time"}.issubset(gdf_points.columns)
    ):
        gdf_points[lead_col] = pd.to_numeric(
            gdf_points[lead_col], errors="coerce"
        )
        gdf_points[wind_col] = pd.to_numeric(
            gdf_points[wind_col], errors="coerce"
        )
        if gdf_points[wind_col].max(skipna=True) < wind_threshold_mps_1:
            continue
        # Group by forecast_time and time, take median wind speed and first geometry
        grouped = gdf_points.groupby(
            ["forecast_time", "time"], as_index=False
        ).agg({lead_col: "first", wind_col: "median", "geometry": "first"})
        grouped = gpd.GeoDataFrame(
            grouped, geometry="geometry", crs=gdf_points.crs
        )

        # Step 1: Sort chronologically
        sorted_df = grouped.sort_values(["forecast_time", "time"]).copy()

        # Step 2: Tag whether point is on land
        sorted_df = sorted_df.copy()
        sorted_df.loc[:, "on_land"] = sorted_df.geometry.within(
            phl_sel_adm1.union_all()
        )

        # Step 3: Find landfall time per forecast
        first_landfall = (
            sorted_df[sorted_df["on_land"]]
            .groupby("forecast_time", as_index=False)
            .first()[["forecast_time", "time"]]
            .rename(columns={"time": "landfall_time"})
        )

        # Step 4: Merge landfall time back
        merged = sorted_df.merge(
            first_landfall, on="forecast_time", how="inner"
        )
        merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
        merged["landfall_time"] = pd.to_datetime(
            merged["landfall_time"], errors="coerce"
        )

        # Step 5: Keep points within 12h before landfall
        merged["hours_before_landfall"] = (
            merged["landfall_time"] - merged["time"]
        ).dt.total_seconds() / 3600
        pre_landfall = merged[
            (merged["hours_before_landfall"] >= 0)
            & (merged["hours_before_landfall"] <= 12)
        ]

        # Step 6: Take the point with the highest wind in this window
        intersecting = (
            pre_landfall.sort_values([wind_col], ascending=False)
            .groupby("forecast_time", as_index=False)
            .first()
        )
        # Apply readiness filter
        condition = (
            (intersecting[lead_col] >= readiness_min_leadtime)
            & (intersecting[lead_col] <= readiness_max_leadtime)
            & (intersecting[wind_col] >= wind_threshold_mps_1)
        )
        result = intersecting[condition]
        print(f"Filtered {len(result)} records for storm {storm_name}")

        if not result.empty:
            result["storm"] = storm_name
            filtered_dfs.append(result)

# Combine all results
if len(filtered_dfs) == 0:
    print("No data found for the specified criteria.")
else:
    readiness_df = pd.concat(filtered_dfs, ignore_index=True)
```

```python
# the return period
(2024 - 2008 + 1) / len(readiness_df["storm"].unique())
```

```python
# Action Trigger:
output_path = trigger_dir / "model_results_dir_test/"
pattern = os.path.join(output_path, "*]_TRIGGER_LEVEL.csv")
```

```python
# rainfall for days before, during, and after the storm
landfall_df = landfall_df[landfall_df["time"] >= "2000-06-01"]
IMERG_START_DATE = pd.to_datetime("2000-06-01")
extra_days = 1
dfs = []
for sid, row in landfall_df.set_index("sid").iterrows():
    landfall_date = pd.to_datetime(row["time"])
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
imerg_landfall_df = landfall_df.merge(
    imerg_sum_df, left_on=["sid", "ADM1_PCODE"], right_on=["sid", "pcode"]
)
imerg_landfall_df[
    (imerg_landfall_df["wmo_wind"] > 48)
    & (imerg_landfall_df["sum_mean_rain"] >= 300)
]
```

```python
(2024 - 2000 + 1) / (
    imerg_landfall_df[
        (imerg_landfall_df["wmo_wind"] > 64)
        & (imerg_landfall_df["sum_mean_rain"] >= 300)
    ]["sid"].nunique()
)
```

```python
imerg_landfall_df["landfall_date"] = pd.to_datetime(
    imerg_landfall_df["landfall_date"], errors="coerce"
)
imerg_landfall_df["year"] = imerg_landfall_df["landfall_date"].dt.year
imerg_landfall_df["label"] = (
    imerg_landfall_df["name"].str.title()
    + " "
    + imerg_landfall_df["year"].astype(str)
)
```

```python
adm1_list = imerg_landfall_df["ADM1_EN"].dropna().unique()[:4]

fig, axs = plt.subplots(2, 2, figsize=(12, 14))
axs = axs.flatten()

for i, adm in enumerate(adm1_list):
    subset = imerg_landfall_df[imerg_landfall_df["ADM1_EN"] == adm].dropna(
        subset=["sum_mean_rain"]
    )

    subset = subset.sort_values(by="sum_mean_rain", ascending=True)

    axs[i].barh(subset["label"], subset["sum_mean_rain"], color="skyblue")
    axs[i].set_title(f"{adm}")
    axs[i].set_xlabel("Rainfall (mm)")
    axs[i].set_ylabel("Storm")
    axs[i].grid(True, axis="x")

plt.suptitle("Rainfall by Storm for Selected ADM1 Regions", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```

```python
combined_df = imerg_landfall_df.merge(
    phl_sel_adm1.rename(columns={"ADM1_PCODE": "pcode"})[["pcode", "ADM1_EN"]]
)
combined_df["year"] = combined_df["landfall_date"].apply(lambda x: x.year)
combined_df["nameseason"] = (
    combined_df["name"].str.capitalize()
    + " "
    + combined_df["year"].astype(str)
)


def calculate_rp(group, col_name, total_seasons):
    group["rank"] = group[col_name].rank(ascending=False)
    group["rp"] = (total_seasons + 1) / group["rank"]
    return group


total_seasons = combined_df["year"].nunique() - 1

rp = 3
col_name = "sum_mean_rain"
color = "crimson"

for pcode, group in combined_df.groupby("pcode"):
    fig, ax = plt.subplots(dpi=200)

    top_edge = group["sum_mean_rain"].max() * 1.1
    right_edge = group["wmo_wind"].max() + 10

    # Scatter plot
    ax.plot(
        group["wmo_wind"],
        group["sum_mean_rain"],
        linestyle="none",
        marker=".",
        color="k",
    )

    # Annotate high rainfall events
    for nameseason, row in group.set_index("nameseason").iterrows():
        # if row["sum_mean_rain"] > group["sum_mean_rain"].median():
        ax.annotate(
            f" {nameseason}",
            (row["wmo_wind"], row["sum_mean_rain"]),
            fontsize=7,
            va="center",
        )

    ax.legend().remove()
    ax.axhline(
        300,
        color="dodgerblue",
        linewidth=1,
        linestyle="--",
        label="300 mm rain",
    )
    ax.axvline(
        48, color="firebrick", linewidth=1, linestyle="--", label="48 kt wind"
    )

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
storms_to_show = landfall_df[
    (landfall_df["wmo_wind"] >= 85) & (landfall_df["time"] >= "2000-01-01")
].copy()
storms_to_show["time"] = pd.to_datetime(storms_to_show["time"])
storms_to_show["storm_year"] = (
    storms_to_show["name"] + " " + storms_to_show["time"].dt.year.astype(str)
)
# readiness_storms = readiness_df["storm"].unique()
activation_wind_storms = ["HAGUPIT", "KAMMURI", "BOPHA", "HAIYAN", "RAI"]
observational_wind_storms = [
    "BETTY",
    "ZEB",
    "MEGI",
    "HAIYAN",
    "MERANTI",
    "MANGKHUT",
    "GONI",
    "RAI",
    "DOKSURI",
]
observational_rain_storms = imerg_landfall_df[
    (imerg_landfall_df["sum_mean_rain"] >= 300)
    & (imerg_landfall_df["wmo_wind"] > 64)
    & (imerg_landfall_df["pcode"] != "PH03")
]["name"].unique()
storm_df = pd.DataFrame(
    {
        "Storm": storms_to_show["storm_year"],  # e.g., "RAI 2021"
        "Activation(Speed)": storms_to_show["name"]
        .isin(activation_wind_storms)
        .map({True: "Yes", False: ""}),
        "Observational(Speed)": storms_to_show["name"]
        .isin(observational_wind_storms)
        .map({True: "Yes", False: ""}),
        "Observational(Rainfall)": storms_to_show["name"]
        .isin(observational_rain_storms)
        .map({True: "Yes", False: ""}),
    }
)

storm_df
```

```python

input_folder = trigger_dir 
output_file = "compiled_model_output.csv"

# Find all matching files
files = glob.glob(os.path.join(input_folder, "*TRIGGER_LEVEL.csv"))

compiled_data = []

for file in files:
    filename = os.path.basename(file)

    # Extract trigger name (before CERF) and ISO3 code (in brackets)
    trigger_match = re.search(r"^(.*?)CERF", filename)
    iso3_match = re.search(r"\[(.*?)\]", filename)

    if not trigger_match or not iso3_match:
        continue  # Skip files that don't match expected pattern

    trigger_name = trigger_match.group(1).strip()
    iso3 = iso3_match.group(1).strip()

    df = pd.read_csv(file)
    df["trigger_name"] = trigger_name
    df["iso3"] = iso3
    compiled_data.append(df)

# Combine all into one DataFrame and save
if compiled_data:
    final_df = pd.concat(compiled_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
else:
    print("No valid files found.")
```
