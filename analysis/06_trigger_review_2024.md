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

# Trigger Review


This notebook review the return periods of the current trigger and the proposed triggers.

```python
%load_ext jupyter_black
```

```python
from pathlib import Path
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src.constants import *
```

```python
rng = np.random.default_rng(12345)

trigger_dir = (
    Path(AA_DATA_DIR)
    / "public/exploration/phl/trigger_performance"
)
ecmwf_forecast_dir = (
    Path(AA_DATA_DIR)
    / "public/exploration/phl/ecmwf_hindcast/csv"
)
```

```python
phl_codab = gpd.read_file(
    f"zip://{Path(AA_DATA_DIR) / 'public/raw/phl/cod_ab/phl_adm_psa_namria_20231106_shp.zip'}!phl_admbnda_adm1_psa_namria_20231106.shp"
)
```

```python
phl_regions = phl_codab.dissolve(
    by="ADM0_PCODE"
)  # [phl_codab["ADM1_PCODE"].isin(["PH05", "PH08", "PH13"])]
```

```python
phl_regions.plot()
```

```python
desired_typhoons = [
    "bopha",
    "fengshen",
    "haiyan",
    "hagupit",
    "phanfone",
    "nock-ten",
    "kammuri",
    "rai",
]
```

```python
filtered_files = [
    file
    for file in os.listdir(ecmwf_forecast_dir)
    if any(typhoon in file.lower() for typhoon in desired_typhoons)
]
```

```python
for filename in filtered_files:
    if filename.endswith(".csv"):
        file_path = os.path.join(ecmwf_forecast_dir, filename)
        df = pd.read_csv(file_path)
        df_med = (
            df.groupby(["forecast_time", "lead_time", "time"])
            .median()
            .reset_index()
        )
        df_med["year"] = pd.to_datetime(df_med["forecast_time"]).dt.year
        df_med["geometry"] = df_med.apply(
            lambda row: Point(row["lon"], row["lat"]), axis=1
        )

        # Split DataFrame by timestamp and process each year separately
        step_ls = []
        for year, df_year in df_med.groupby("year"):
            df_year["landfall"] = df_year["geometry"].apply(
                lambda point: point.within(phl_regions.geometry)
            )
```

```python
results = []

for filename in os.listdir(ecmwf_forecast_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(ecmwf_forecast_dir, filename)
        df = pd.read_csv(file_path)
        df["year"] = pd.to_datetime(df["forecast_time"]).dt.year
        df["geometry"] = df.apply(
            lambda row: Point(row["lon"], row["lat"]), axis=1
        )

        # Split DataFrame by timestamp and process each year separately
        step_ls = []
        for year, df_year in df.groupby("year"):
            df_year["landfall"] = df_year["geometry"].apply(
                lambda point: point.within(phl_regions.geometry)
            )
            df_control = df_year[df_year["mtype"] == "forecast"]
            point_of_landfall = df_control.index[
                df_control["landfall"] == True
            ].tolist()[0]
            df_year["time_until_landfall"] = (
                pd.to_datetime(df_year["time"], format="%Y/%m/%d, %H:%M:%S")
                - pd.to_datetime(
                    df_year["time"][point_of_landfall],
                    format="%Y/%m/%d, %H:%M:%S",
                )
            ).dt.total_seconds() / 3600
            for step, df_step in df_year.groupby("forecast_time"):
                time_step = df_step["forecast_time"].unique()
                df_step_med = (
                    df_step.groupby("time_until_landfall")
                    .median()
                    .reset_index()
                )
                # years_with_speed_gt_40 = df_year[df_year["speed"] > 40]["year"].unique()
                current_readiness_checks = df_step_med[
                    (df_step_med["time_until_landfall"] >= 96)
                    & (df_step_med["time_until_landfall"] <= 168)
                    & (df_step_med["speed"] >= 43.9)
                ]
                proposed_readiness_checks = df_step_med[
                    (df_step_med["time_until_landfall"] >= 96)
                    & (df_step_med["time_until_landfall"] <= 168)
                    & (df_step_med["speed"] >= 37.8)
                ]
                proposed_readiness_checks_no_readiness = df_step_med[
                    (df_step_med["time_until_landfall"] >= 72)
                    & (df_step_med["time_until_landfall"] <= 168)
                    & (df_step_med["speed"] >= 37.8)
                ]
                activation_checks = df_step_med[
                    (df_step_med["time_until_landfall"] >= 72)
                    & (df_step_med["time_until_landfall"] <= 168)
                ]

                # Determine readiness
                current_readiness = current_readiness_checks.shape[0] > 0
                proposed_readiness = proposed_readiness_checks.shape[0] > 0
                proposed_readiness_no_readiness = (
                    proposed_readiness_checks_no_readiness.shape[0] > 0
                )

                results.append(
                    {
                        "filename": filename,
                        "current_readiness": current_readiness,
                        "proposed_readiness": proposed_readiness,
                        "proposed_readiness_no_readiness": proposed_readiness_no_readiness,
                        "forecast_time": step,
                        # "max_time_until_landfall": (df_step_med[["time_until_landfall"]]).max().item(),
                        # "min_time_until_landfall": (df_step_med[["time_until_landfall"]]).min().item(),
                        "time_until_landfall_okay_72_168": (
                            any(
                                activation_checks[
                                    "time_until_landfall"
                                ].between(72, 168)
                            )
                        ),
                        "time_until_landfall_okay_96_168": (
                            any(
                                activation_checks[
                                    "time_until_landfall"
                                ].between(96, 168)
                            )
                        ),
                    }
                )
```

```python
results_df = pd.DataFrame(results)
results_df["year"] = pd.to_datetime(results_df["forecast_time"]).dt.year
```

```python
results_df["proposed_readiness_test"] = (
    results_df["proposed_readiness"]
    == results_df["proposed_readiness_no_readiness"]
)
results_df[results_df["proposed_readiness_test"] == False]
```

```python
year_ls = []
for year, df_year in results_df.groupby(["filename", "year"]):
    for ts, df_ts in df_year.groupby(["forecast_time"]):
        df_filter = df_year[
            (df_year["forecast_time"] <= ts) & (df_year["lt_okay"])
        ]
        df_ts["current_activation"] = any(df_filter["current_readiness"])
        df_ts["proposed_activation"] = any(df_filter["proposed_readiness"])

        year_ls.append(df_ts)
```

```python
results_df = pd.concat(year_ls)
```

```python
results_df["timestamp"] = pd.to_datetime(
    results_df["forecast_time"], format="%Y/%m/%d, %H:%M:%S"
).dt.strftime("%Y%m%d%H%M%S")
```

```python
results_df
```

```python
results_df.to_csv(
    trigger_dir / "results.csv",
    index=False,
)
```

```python
yr_len = max(results_df["year"]) - min(results_df["year"]) + 1
yr_len
```

```python
yr_len / (
    results_df.groupby("year")
    .sum({"current_readiness", "proposed_readiness"})
    .gt(0)
    .sum()
)
```

```python
# how many times readiness is reached under the current methodology
current_readiness_freq = (
    results_df.groupby(["filename", "year"])["current_readiness"]
    .apply(lambda x: (x == True).sum())
    .reset_index(name="Readiness_Frequency")
)
# RP
(
    "The current return period of readiness is 1-in-"
    + str(
        round(
            yr_len
            / sum(
                current_readiness_freq.groupby(["year"])[
                    "Readiness_Frequency"
                ].sum()
                > 0
            ),
            1,
        )
    )
    + " years."
)
```

```python
current_readiness_freq.to_csv(
    trigger_dir / "current_readiness_freq.csv",
    index=False,
)
```

```python
# how many times readiness is reached under the proposed methodology
# how many times readiness is reached under the current methodology
proposed_readiness_freq = (
    results_df.groupby(["filename", "year"])["proposed_readiness_no_readiness"]
    .apply(lambda x: (x == True).sum())
    .reset_index(name="Readiness_Frequency")
)
# RP
(
    "The proposed return period of readiness is 1-in-"
    + str(
        round(
            yr_len
            / sum(
                proposed_readiness_freq.groupby(["year"])[
                    "Readiness_Frequency"
                ].sum()
                > 0
            ),
            1,
        )
    )
    + " years."
)
```

```python
proposed_readiness_freq.to_csv(
    trigger_dir / "proposed_readiness_freq.csv",
    index=False,
)
```

```python
current_activated = results_df[results_df["current_readiness"]]
proposed_activated = results_df[results_df["proposed_readiness"]]
proposed_activated_no_readiness = results_df[
    results_df["proposed_readiness_no_readiness"]
]
```

```python
current_activated["processed_timestamps"] = (
    '["'
    + current_activated["filename"]
    .apply(lambda x: x.split("_")[0])
    .str.upper()
    + '"]="'
    + current_activated["timestamp"]
    + '"'
)
```

```python
file_ls = []
for file, df_file in current_activated.groupby("filename"):
    w = (
        '["'
        + file.split("_")[0].upper()
        + '"]="'
        + " ".join(df_file["timestamp"])
        + '"'
    )
    file_ls.append(w)
file_df = pd.DataFrame(file_ls)
```

```python
file_df.to_csv(
    trigger_dir / "current_readiness.csv",
    index=False,
)
```

```python
file_ls = []
for file, df_file in proposed_activated.groupby("filename"):
    w = (
        '["'
        + file.split("_")[0].upper()
        + '"]="'
        + " ".join(df_file["timestamp"])
        + '"'
    )
    file_ls.append(w)
file_df = pd.DataFrame(file_ls)
```

```python
file_df.to_csv(
    trigger_dir / "proposed_readiness.csv",
    index=False,
)
```

```python
file_ls = []
for file, df_file in proposed_activated_no_readiness.groupby("filename"):
    w = (
        '["'
        + file.split("_")[0].upper()
        + '"]="'
        + " ".join(df_file["timestamp"])
        + '"'
    )
    file_ls.append(w)
file_df = pd.DataFrame(file_ls)
file_df.to_csv(
    trigger_dir / "proposed_activation_no_readiness.csv",
    index=False,
)
```

```python
current_activated["processed_timestamps"] = (
    '["'
    + current_activated["filename"]
    .apply(lambda x: x.split("_")[0])
    .str.upper()
    + '"]="'
    + current_activated["timestamp"]
    + '"'
).to_csv(
    trigger_dir / "current_activated.csv",
    index=False,
)
proposed_activated["processed_timestamps"] = (
    '["'
    + proposed_activated["filename"]
    .apply(lambda x: x.split("_")[0])
    .str.upper()
    + '"]="'
    + proposed_activated["timestamp"]
    + '"'
).to_csv(
    trigger_dir / "proposed_activated.csv",
    index=False,
)
```

```python
current_model_results_dir = trigger_dir / "model_run_results_test/current/"
proposed_model_results_dir = trigger_dir / "model_run_results_test/proposed/"

current_results = pd.DataFrame()
for filename in current_model_results_dir.glob("*CERF_TRIGGER_LEVEL.csv"):
    year = str(filename.name).split("_")[1][:4]
    file_df = pd.read_csv(filename)
    file_df["year"] = year
    # results["year"] = year
    current_results = pd.concat([current_results, file_df], ignore_index=True)
# Check if it triggered
current_results["Activation reached"] = (
    (current_results["80k"] >= 0.5)
    | (current_results["50k"] >= 0.6)
    | (current_results["30k"] >= 0.7)
    | (current_results["10k"] >= 0.8)
    | (current_results["5k"] >= 0.95)
)
current_results.to_csv(
    trigger_dir / "current_files.csv",
    index=False,
)
current_freq = (
    current_results.groupby("year")["Activation reached"].any().sum()
) / yr_len
```

```python
current_results.groupby(["Typhoon_name", "year"])[
    "Activation reached"
].any().reset_index().to_csv(
    trigger_dir / "current_activation.csv",
    index=False,
)
```

```python
# RP
(
    "The current return period of activation after readiness is 1-in-"
    + str(round(1 / current_freq, 1))
    + " years."
)
```

```python
proposed_results = pd.DataFrame()
for filename in proposed_model_results_dir.glob("*CERF_TRIGGER_LEVEL.csv"):
    year = str(filename.name).split("_")[1][:4]
    file_df = pd.read_csv(filename)
    file_df["year"] = year
    # results["year"] = year
    proposed_results = pd.concat(
        [proposed_results, file_df], ignore_index=True
    )
# Check if it triggered
proposed_results["Activation reached"] = (
    (proposed_results["50k"] >= 0.5)
    | (proposed_results["30k"] >= 0.6)
    | (proposed_results["15k"] >= 0.7)
    | (proposed_results["6k"] >= 0.8)
    | (proposed_results["3k"] >= 0.95)
)
proposed_results.to_csv(
    trigger_dir / "proposed_files.csv",
    index=False,
)
proposed_freq = (
    proposed_results.groupby("year")["Activation reached"].any().sum()
) / yr_len
```

```python
proposed_results.groupby(["Typhoon_name", "year"])[
    "Activation reached"
].any().reset_index().to_csv(
    trigger_dir / "proposed_activation.csv",
    index=False,
)
```

```python
# RP
(
    "The proposed return period of activation is 1-in-"
    + str(round(1 / proposed_freq, 1))
    + " years."
)
```
