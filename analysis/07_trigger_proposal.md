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

# Trigger Proposal


Reviewing new thresholds based on the feedback of the proposed ones.

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
trigger_dir = (
    Path(AA_DATA_DIR)
    / "public/exploration/phl/trigger_performance"
)
```

```python
current_model_results_dir = trigger_dir / "model_run_results_test/current/"
proposed_model_results_dir = trigger_dir / "model_run_results_test/proposed/"
proposed_model_results_dir_test = trigger_dir / "model_results_dir_test/"
```

```python
yr_len = 17
```

```python
proposed_results = pd.DataFrame()
for filename in proposed_model_results_dir.glob("*CERF_TRIGGER_LEVEL.csv"):
    year = str(filename.name).split("_")[1][:4]
    time = str(filename.name).split("_")[1]
    file_df = pd.read_csv(filename)
    file_df["year"] = year
    file_df["time"] = time
    # results["year"] = year
    proposed_results = pd.concat(
        [proposed_results, file_df], ignore_index=True
    )

```

```python
# Check if it triggered
proposed_results["Activation reached"] = (
    (proposed_results["50k"] >= 0.5)
    | (proposed_results["30k"] >= 0.6)
    | (proposed_results["15k"] >= 0.7)
)

proposed_freq = (
    proposed_results.groupby("year")["Activation reached"].any().sum()
) / yr_len
# RP
(
    "The proposed return period of activation is 1-in-"
    + str(round(1 / proposed_freq, 1))
    + " years."
)
```

```python
proposed_results[proposed_results["Activation reached"]]
```

```python
# Check if each threshold is reached individually
proposed_results["50k_triggered"] = proposed_results["50k"] >= 0.5
proposed_results["30k_triggered"] = proposed_results["30k"] >= 0.6
proposed_results["15k_triggered"] = proposed_results["15k"] >= 0.8
proposed_results["6k_triggered"] = proposed_results["6k"] >= 1
proposed_results["3k_triggered"] = proposed_results["3k"] >= 1
# Calculate the frequency and return period for each threshold
thresholds = ["50k", "30k", "15k", "6k", "3k"]
return_periods = {}

for threshold in thresholds:
    freq = proposed_results.groupby("year")[f"{threshold}_triggered"].any().sum()/yr_len
    return_periods[threshold] = round(1 / freq, 1) if freq > 0 else float('inf')

# Output the proposed return period for each threshold
for threshold, rp in return_periods.items():
    if rp == float('inf'):
        print(f"The proposed return period of activation for {threshold} is more than the dataset range (infrequent occurrence).")
    else:
        print(f"The proposed return period of activation for {threshold} is 1-in-{rp} years.")
```

```python
# Option 1
# Check if it triggered
proposed_results["Activation reached"] = (
    (proposed_results["50k"] >= 0.5)
    | (proposed_results["30k"] >= 0.6)
    | (proposed_results["15k"] >= 0.7)
    | (proposed_results["6k"] >= 0.85)
    | (proposed_results["3k"] >= 1)
)

proposed_freq = (
    proposed_results.groupby("year")["Activation reached"].any().sum()
) / yr_len
# RP
(
    "The proposed return period of activation is 1-in-"
    + str(round(1 / proposed_freq, 1))
    + " years."
)
```

```python
proposed_results = pd.DataFrame()
for filename in proposed_model_results_dir_test.glob("*CERF_TRIGGER_LEVEL.csv"):
    year = str(filename.name).split("_")[1][:4]
    time = str(filename.name).split("_")[1]
    file_df = pd.read_csv(filename)
    file_df["year"] = year
    file_df["time"] = time
    # results["year"] = year
    proposed_results = pd.concat(
        [proposed_results, file_df], ignore_index=True
    )

```

```python
proposed_results.to_csv(proposed_model_results_dir_test / "proposed_8k.csv", index=False)
```

```python
# Option 1
# Check if it triggered
proposed_results["Activation reached"] = (
    (proposed_results["8k"] >= 0.85)
)

proposed_freq = (
    proposed_results.groupby("year")["Activation reached"].any().sum()
) / yr_len
# RP
(
    "The proposed return period of activation is 1-in-"
    + str(round(1 / proposed_freq, 1))
    + " years."
)
```

```python
proposed_results[proposed_results["Activation reached"]]
```
