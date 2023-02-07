---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: pa-aa-phl-storms
    language: python
    name: pa-aa-phl-storms
---

# Historical damage all typhoons

To measure trigger performance, we want to know for which
historical typhoons CERF would have wanted to activate.
Therefore, we want to know how much damage the typhoons caused
in our regions of interest, and also the targeted municipalities.

```python
%load_ext jupyter_black
```

```python
from pathlib import Path
import os

import pandas as pd
import geopandas as gpd
```

```python
MAIN_DIR = Path(os.environ["AA_DATA_DIR"]) / "public/exploration/phl/"
INPUT_DIR = MAIN_DIR / "data_from_repo"
IMPACT_DATA_FILENAME = "data/IMpact_data_philipines_SEP_2021.csv"
```

```python
# Readin the latest impact data from Aki, used to train the model
df = pd.read_csv(INPUT_DIR / IMPACT_DATA_FILENAME)
# Clean it a bit
df["typhoon"] = df["typhoon"].str.upper()
df.columns = df.columns.str.lower()
df = (
    df[["pcode", "typhoon", "year", "totally", "partially"]]
    .drop_duplicates(subset=["pcode", "typhoon", "year"])
    .rename(columns={"pcode": "adm3_pcode"})
)
df
```

```python
# Read in the regions of interest
# keep only the pcodes
filename_geo = INPUT_DIR / "data-raw/phl_admin3_simpl.geojson"
gdf = gpd.read_file(filename_geo)[["adm3_pcode", "adm1_pcode"]]
df = df.merge(gdf, how="left", on="adm3_pcode")
df
```

```python
# Regions of interest and their pcodes
ROI = {
    "5": "PH050000000",
    "8": "PH080000000",
    "13": "PH160000000",
}
# Municipalities of interest and their pcodes
df_mun = pd.read_csv(
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRZNX0UwZ_KFqxMLjumqL3RRc6KW8ljNpnFRgwYKNvn8cW-2APYpE4QRCCzonaH7A/pub?gid=172173510&single=true&output=csv"
).rename(columns={"Muny_City Code": "adm3_pcode"})
df_mun
```

```python
# Get the total damage in regions 5, 8, and 13
df_regional_damage = (
    df.loc[df["adm1_pcode"].isin(ROI.values())]
    .groupby(["typhoon", "year", "adm1_pcode"])
    .sum()
    .reset_index(level="adm1_pcode")
)
df_regional_damage
df_regional_damage.to_csv("typhoon_regional_damage.csv")
```

```python
df_regional_damage.groupby(["typhoon", "year"]).sum().to_csv(
    "typhoon_regional_damage_total.csv"
)
```

```python
df_municipal_damage = (
    df.merge(df_mun, how="left", on="adm3_pcode")
    .dropna(subset="Scenario")
    .groupby(["typhoon", "year", "Scenario"])
    .sum()
    .reset_index(level=["Scenario"])
)
df_municipal_damage["Scenario"] = df_municipal_damage["Scenario"].astype(int)
df_municipal_damage
df_municipal_damage.to_csv("typhoon_scenario_damage.csv")
```
