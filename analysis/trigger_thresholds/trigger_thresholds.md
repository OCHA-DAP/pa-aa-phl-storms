# OCHA trigger thresholds

This notebook is for determining trigger thresholds for 
the OCHA Philippines trigger

```python
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme as gev
from scipy.interpolate import interp1d

rng = np.random.default_rng(12345)
```

```python
# The regions of interest and their pcodes
ROI = {
    "5": "PH050000000", 
    "8": "PH080000000", 
    "13": "PH160000000", 
}
```

## Reading in the data

```python
# Read in the typhoon info
filename = "../../IBF-Typhoon-model/data/loos_table_ibtrcs.csv"
df_typhoon = pd.read_csv(filename, index_col=0)
df_typhoon["year"] = df_typhoon["typhoon"].apply(lambda x: int(x[:4]))
# Drop 2022 because it will mess up the RP
df_typhoon = df_typhoon.loc[df_typhoon["year"] != 2022]
df_typhoon
```

```python
# Check the number of years is complete
typhoon_year_list = df_typhoon.year.unique()
nyears = len(typhoon_year_list)
typhoon_year_list
```

```python
# Read in the geo info
filename_geo = "../../IBF-Typhoon-model/data/gis_data/phl_admin3_simpl.geojson"
gdf = gpd.read_file(filename_geo)
gdf
```

## Make a combined data frame with damage totals

```python
# Merge them
df_comb = (pd.merge(df_typhoon, gdf[["adm3_pcode", "adm1_pcode"]], 
              left_on="Mun_Code", right_on="adm3_pcode", how="left")
      .drop(columns=["adm3_pcode"])
     )
df_comb
```

```python
# Group by admin1 and typhoon, and sum, then take max of each year
df_max = (df_comb.groupby(['adm1_pcode', 'typhoon'])
      .agg({'No_Totally_DMG_BLD': sum,'year': 'first'})
     .groupby(['adm1_pcode', 'year']).max()
     .rename(columns={'No_Totally_DMG_BLD': 'max_damage_event'})
     .reset_index()
             )
df_max
```

```python
# Now do the same but taking the ROI as a whole

df_max_roi_input = {
    "5,8,13": ["5", "8", "13"],
    "5,8": ["5", "8"],
    "8,13": ["8", "13"],
}

df_max_roi = {}
for key, value in df_max_roi_input.items():

    df_max_roi[key] = (df_comb.loc[df_comb['adm1_pcode'].isin([ROI[v] for v in value])]
              .groupby('typhoon')
              .agg({'No_Totally_DMG_BLD': sum,'year': 'first'})
              .groupby(['year']).max()
              .rename(columns={'No_Totally_DMG_BLD': 'max_damage_event'})
              .reset_index()
             )

df_max_roi


```

## Get the return periods

```python
def get_rp_analytical(
    df_rp: pd.DataFrame,
    rp_var: str,
    show_plots: bool = False,
    plot_title: str = "",
    extend_factor: int = 1,
):
    """
    :param df_rp: DataFrame where the index is the year, and the rp_var
    column contains the maximum value per year
    :param rp_var: The column with the quantity to be evaluated
    :param show_plots: Show the histogram with GEV distribution overlaid
    :param plot_title: The title of the plot
    :param extend_factor: Extend the interpolation range in case you want to
    calculate a relatively high return period
    :return: Interpolated function that gives the quantity for a
    given return period
    """
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    rp_var_values = df_rp[rp_var]
    shape, loc, scale = gev.fit(
        rp_var_values,
        loc=rp_var_values.median(),
        scale=rp_var_values.median() / 2,
    )
    x = np.linspace(
        rp_var_values.min(),
        rp_var_values.max() * extend_factor,
        100 * extend_factor,
    )
    if show_plots:
        fig, ax = plt.subplots()
        ax.hist(rp_var_values, density=True, bins=20)
        ax.plot(x, gev.pdf(x, shape, loc, scale))
        ax.set_title(plot_title)
        plt.show()
    y = gev.cdf(x, shape, loc, scale)
    y = 1 / (1 - y)
    return interp1d(y, x)


def get_rp_empirical(df_rp: pd.DataFrame, rp_var: str):
    """
    :param df_rp: DataFrame where the index is the year, and the rp_var
    column contains the maximum value per year
    :param rp_var: The column
    with the quantity to be evaluated
    :return: Interpolated function
    that gives the quantity for a give return period
    """
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    n = len(df_rp)
    df_rp["rank"] = np.arange(n) + 1
    df_rp["exceedance_probability"] = df_rp["rank"] / (n + 1)
    df_rp["rp"] = 1 / df_rp["exceedance_probability"]
    return interp1d(df_rp["rp"], df_rp[rp_var])



def get_rp_df(
    df: pd.DataFrame,
    rp_var: str,
    years: list = None,
    method: str = "analytical",
    show_plots: bool = False,
    extend_factor: int = 1,
    round_rp: bool = True,
) -> pd.DataFrame:
    """
    Function to get the return periods, either empirically or
    analytically See the `glofas/utils.py` to do this with a xarray
    dataset instead of a dataframe
    :param df: Dataframe with data to compute rp on
    :param rp_var: column name to compute return period on
    :param years: Return period years to compute
    :param method: Either "analytical" or "empirical"
    :param show_plots: If method is analytical, can show the histogram and GEV
    distribution overlaid
    :param extend_factor: If method is analytical, can extend the interpolation
    range to reach higher return periods
    :param round_rp: if True, round the rp values, else return original values
    :return: Dataframe with return period years as index and stations as
    columns
    """
    if years is None:
        years = [1.5, 2, 3, 5]
    df_rps = pd.DataFrame(columns=["rp"], index=years)
    if method == "analytical":
        f_rp = get_rp_analytical(
            df_rp=df,
            rp_var=rp_var,
            show_plots=show_plots,
            extend_factor=extend_factor,
        )
    elif method == "empirical":
        f_rp = get_rp_empirical(
            df_rp=df,
            rp_var=rp_var,
        )
    else:
        print(f"{method} is not a valid keyword for method")
        return None
    df_rps["rp"] = f_rp(years)
    if round_rp:
        df_rps["rp"] = np.round(df_rps["rp"])
    return df_rps

```

```python
def add_rps_to_df(df, df_rp, region_key, years=None, method="analytical", show_plots=False):
    print(f"Running for region {region_key}...")
    if years is None:
        years = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2, 3, 3.5,  4, 4.5, 5, 5.5]
    df = df.copy()
    if len(df) < nyears:
        missing_years = set(typhoon_year_list) - set(df.year)
        print(f"Warning: need to fill in missing years {missing_years}")
        for year in missing_years:
            df = df.append({"year": year,
                       "max_damage_event": 0}, ignore_index=True)
    df = df.set_index(df.year)
    df_rp_tmp = (get_rp_df(df, "max_damage_event", years=years, extend_factor=5,
                          method=method, show_plots=show_plots)
                 .reset_index()
                 .rename(columns={"rp": "max_damage", "index": "rp"})
                )
    df_rp_tmp["region"] = region_key
    # Now bootstrap resample
    #df_results = bootstrap_resample(df, years, method)
    # Add results to final RP DF
    df_rp = pd.concat([df_rp, df_rp_tmp], ignore_index=True)
    print("...done")
    return df_rp

def bootstrap_resample(df, years, method, n_bootstrap=100):
    df_results = pd.DataFrame()
    for i in range(n_bootstrap):
        df_rs = df.sample(frac=1, replace=True, 
                            random_state=rng.bit_generator)
        try:
            df_rp_tmp = (get_rp_df(df_rs, "max_damage_event", years=years, extend_factor=5,
                          method=method, show_plots=False)
                 .reset_index()
                 .rename(columns={"rp": "max_damage", "index": "rp"})
                )
            df_results = df_results.append(df_rp_tmp, ignore_index=True)
        except ValueError:
            continue
    df_results = df_results.dropna().groupby("rp").quantile([0.05, .95])
    print(df_results)
    return df_results
```

```python
df_rp = pd.DataFrame()


# Only do for regions 5, 8, and 13
for region_key, region_value in ROI.items():
    df_rp = add_rps_to_df(df_max.loc[df_max["adm1_pcode"] == region_value], df_rp, region_key)

# Then for all 3 regions
for region_key, df_region in df_max_roi.items():
    df_rp = add_rps_to_df(df_region, df_rp, region_key)


df_rp

```

```python
df_rp.to_csv("rp_damage_per_region.csv", index=False)
```

```python

```
