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

# Scatter plot

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar

from src.constants import *
from src.datasources import codab, imerg
```

## Load data

### CODAB

```python
adm0 = codab.load_codab_from_blob()
```

## Historical stats

Calculated in `ibtracs_imerg_agg.ipynb` then merged with EM-DAT in `emdat.ipynb`

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs_imerg_emdat_stats.parquet"
df_stats_raw = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats["Total Affected"] = df_stats["Total Affected"].fillna(0)
df_stats = df_stats[df_stats["season"] < 2025]
```

## Check correlations

```python
df_stats.corr(numeric_only=True)["Total Affected"].plot.bar()
```

```python
df_stats_with_impact = df_stats[df_stats["Total Affected"] > 0]
```

```python
df_stats_with_impact.corr(numeric_only=True)["Total Affected"].plot.bar()
```

Seems like `q80_roll2` is best for rainfall. We can just check that by running a regression with wind and swapping in the various rainfall columns to see which performs best.

```python
target = "Total Affected"  # your single target column
fixed_col = "wind_speed_max"  # the one you definitely want
other_cols = [
    x for x in df_stats.columns if "roll" in x
]  # the ones you want to test

results = []

for other in other_cols:
    X = df_stats[[fixed_col, other]]
    X = sm.add_constant(X)  # adds intercept term
    y = df_stats[target]

    model = sm.OLS(y, X).fit()
    r2 = model.rsquared

    results.append(
        {
            "other_col": other,
            "r2": r2,
        }
    )

results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print(results_df)
```

Yeah seems like `q80_roll2` is best but honestly none of these $R^{2}$ values are very good.


## Load IMERG

```python
# get IMERG
query = """
SELECT *
FROM public.imerg
WHERE pcode = 'PH'
"""
with stratus.get_engine(stage="prod").connect() as con:
    df_imerg = pd.read_sql(query, con)
```

Check to see the most recent dates we have.

```python
df_imerg.sort_values("valid_date").iloc[-10:]
```

### Get recent IMERG rasters

```python
# grabbing for the most recent dates that seem to be from Fung-Wong
# the max date can be updated once we get more rainfall data
imerg_dates = pd.date_range("2025-11-08", "2025-11-08")
```

```python
da_imerg_raw = imerg.open_imerg_raster_dates(imerg_dates)
```

```python
da_imerg = da_imerg_raw.rio.clip(adm0.geometry)
```

```python
da_imerg_roll2 = da_imerg.rolling(date=2, min_periods=1).sum()
```

```python
da_imerg_roll2_q = da_imerg_roll2.quantile(0.8, dim=["x", "y"])
```

## Get Fung-Wong current values

```python
fungwong_imerg = float(da_imerg_roll2_q.max())
```

```python
fungwong_imerg
```

```python
# from https://agora.ex.nii.ac.jp/digital-typhoon/summary/wnp/l/202526.html.en
fungwong_wind = 85
```

## Plot

```python
def plot_stats(
    wind_col="wind_speed_max",
    rain_col="q80_roll2",
    impact_col="Total Affected",
):
    df_plot = df_stats_with_impact.copy()
    # cerf_color = "crimson"
    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)

    ymax = df_plot[rain_col].max() * 1.1
    xmax = df_plot[wind_col].max() * 1.1

    bubble_sizes = df_plot[impact_col].fillna(0)
    bubble_sizes_scaled = bubble_sizes / bubble_sizes.max() * 2000

    ax.scatter(
        df_plot[wind_col],
        df_plot[rain_col],
        s=bubble_sizes_scaled,
        # c=df_stats["cerf"].apply(lambda x: cerf_color if x else "k"),
        alpha=0.3,
        edgecolor="none",
        zorder=1,
    )

    for _, row in df_plot.iterrows():
        ax.annotate(
            row["name"].capitalize() + "\n" + str(row["season"]),
            (row[wind_col], row[rain_col]),
            ha="center",
            va="center",
            fontsize=4,
            # color=cerf_color if row["cerf"] == True else "k",
            # zorder=10 if row["cerf"] else 9,
            alpha=0.8,
        )

    ax.scatter(
        [fungwong_wind],
        [fungwong_imerg],
        marker="x",
        color=CHD_GREEN,
        linewidths=3,
        s=100,
    )
    ax.annotate(
        f"Fung-Wong\nlandfall wind speed\nand rainfall up to {imerg_dates.max().date()}",
        xy=(fungwong_wind, fungwong_imerg),  # point to highlight
        xytext=(
            90,
            300,
        ),  # position of the label
        va="center",
        ha="center",
        color=CHD_GREEN,
        fontweight="normal",
        fontsize=8,
        arrowprops=dict(
            arrowstyle="-",  # simple line, no arrowhead
            color=CHD_GREEN,  # same color as text
            lw=0.5,  # fine line
            alpha=0.8,
        ),
    )

    legend_text = (
        "    Size of bubble proportional to\n"
        "    total number of people affected [EM-DAT]"
    )
    ax.annotate(
        legend_text,
        (0, 390),
        va="top",
        fontsize=6,
        fontstyle="italic",
        color="grey",
    )

    ylabel = (
        "Two-day rainfall, 80th percentile over whole country (mm) [IMERG]"
        if rain_col == "q80_roll2"
        else rain_col
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(
        "Max. wind speed while within 230 km of Philippines (knots) [IBTrACS]"
    )

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)

    ax.set_title("Philippines: historical rainfall and wind speed")

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    return fig, ax
```

```python
plot_stats()
```

```python

```
