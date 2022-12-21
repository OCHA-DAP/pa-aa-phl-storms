# Readiness trigger performance

Reads in the readiness analysis by Joseph, does some BSRS,
and gets the stats

```python
%load_ext jupyter_black
```

```python
from pathlib import Path
import os

import numpy as np
import pandas as pd
```

```python
rng = np.random.default_rng(12345)

trigger_dir = (
    Path(os.environ["AA_DATA_DIR"])
    / "public/exploration/phl/trigger_performance"
)
```

## Reading in the readiness data

```python
trigger_filename = trigger_dir / "Readiness trigger performance.xlsx"
df_readiness = pd.read_excel(trigger_filename)
```

```python
# Convert scenarios to strings
for cname in ["Activation scenario", "Readiness scenario"]:
    df_readiness[cname] = df_readiness[cname].fillna(0).astype(int).astype(str)
# Make sure Y/N is capitalized
for cname in ["Should have activated", "Readiness reached"]:
    df_readiness[cname] = df_readiness[cname].str.upper()
```

```python
# fill in confusion matrix
df_readiness["Confusion matrix"] = "TN"
df_readiness.loc[
    (df_readiness["Should have activated"] == "Y")
    & (df_readiness["Readiness reached"] == "N"),
    "Confusion matrix",
] = "FN"
df_readiness.loc[
    (df_readiness["Should have activated"] == "Y")
    & (df_readiness["Readiness reached"] == "Y")
    & (
        df_readiness["Activation scenario"]
        == df_readiness["Readiness scenario"]
    ),
    "Confusion matrix",
] = "TP"
# Dataset has no FPs so skipping those
```

```python
# Rename confusion matrix column
df_readiness = df_readiness[
    ["International name", "Year", "Confusion matrix", "Should have activated"]
].rename(
    columns={
        "International name": "typhoon",
        "Year": "year",
        "Confusion matrix": "readiness",
    }
)
# Capitlize typhoon names
df_readiness["typhoon"] = (
    df_readiness["typhoon"].str.upper().str.strip().str.replace(" ", "-")
)

df_readiness
```

## Activation trigger

```python
model_results_dir = trigger_dir / "model_run_results"
results = pd.DataFrame()
for filename in model_results_dir.glob("*CERF_TRIGGER_LEVEL.csv"):
    typhoon = str(filename.name).split("_")[0]
    results = pd.concat([results, pd.read_csv(filename)], ignore_index=True)
results
```

```python
# Check if it triggered
results["Activation reached"] = (
    (results["80k"] >= 0.5)
    | (results["50k"] >= 0.6)
    | (results["30k"] >= 0.7)
    | (results["10k"] >= 0.8)
    | (results["5k"] >= 0.95)
)
results
```

```python
# Join with readinessas
df_activation = pd.merge(
    left=df_readiness.rename(columns={"International name": "typhoon"}),
    right=results[["Typhoon_name", "Activation reached"]].rename(
        columns={"Typhoon_name": "typhoon"}
    ),
    how="left",
    on="typhoon",
)

# count activations
df_activation.loc[
    (df_activation["Should have activated"] == "Y")
    & df_activation["Activation reached"],
    "activation",
] = "TP"

df_activation.loc[
    (df_activation["Should have activated"] == "Y")
    & (df_activation["Activation reached"] == False),
    "activation",
] = "FN"

# Here assuming that there are no false positives -- pretty safe assumption for now
# (but I will go back and run on more historical typhoons)
df_activation.loc[
    (df_activation["Should have activated"] == "N")
    & (
        (df_activation["Activation reached"] == False)
        # Remove this after running on more
        | (df_activation["Activation reached"].isna())
    ),
    "activation",
] = "TN"


df_activation
```

## Base metrics

```python
def calc_far(TP, FP):
    return FP / (TP + FP)


def calc_var(TP, FP):
    return TP / (TP + FP)


def calc_det(TP, FN):
    return TP / (TP + FN)


def calc_mis(TP, FN):
    return FN / (TP + FN)


def calc_acc(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def calc_atv(TP, TN, FP, FN):
    return (TP + FP) / (TP + TN + FP + FN)
```

```python
def calc_df_base(df):
    df_new = (
        df.sample(n=len(df), replace=True, random_state=rng.bit_generator)
        .apply(pd.value_counts)
        .fillna(0)
    )
    # Some realizations are missing certain counts
    for count in ["FN", "FP", "TN", "TP"]:
        if count not in df_new.index:
            df_new.loc[count] = 0
    return (
        df_new.apply(
            lambda x: {
                "far": calc_far(x.TP, x.FP),
                "var": calc_var(x.TP, x.FP),
                "det": calc_det(x.TP, x.FN),
                "mis": calc_mis(x.TP, x.FN),
                "acc": calc_acc(x.TP, x.TN, x.FP, x.FN),
                "atv": calc_atv(x.TP, x.TN, x.FP, x.FN),
                "nTP": x.TP.sum(),
                "nFP": x.FP.sum(),
                "nFN": x.FN.sum(),
            },
            result_type="expand",
        )
        .melt(ignore_index=False)
        .reset_index()
        .rename(columns={"index": "metric", "variable": "trigger"})
        .assign(point="central")
    )


df_base = calc_df_base(df)
df_base
```

## Bootstrap resample

```python
n_bootstrap = 1_000  # 10,000 takes about ...

# Copied from Niger
def get_df_bootstrap(df, n_bootstrap=1_000):
    # Create a bootstrapped DF
    df_all_bootstrap = pd.DataFrame()
    for i in range(n_bootstrap):
        df_new = (
            df.sample(n=len(df), replace=True, random_state=rng.bit_generator)
            .apply(pd.value_counts)
            .fillna(0)
        )
        # Some realizations are missing certain counts
        for count in ["FN", "FP", "TN", "TP"]:
            if count not in df_new.index:
                df_new.loc[count] = 0
        df_new = (
            df_new.apply(
                lambda x: {
                    "far": calc_far(x.TP, x.FP),
                    "var": calc_var(x.TP, x.FP),
                    "det": calc_det(x.TP, x.FN),
                    "mis": calc_mis(x.TP, x.FN),
                    "acc": calc_acc(x.TP, x.TN, x.FP, x.FN),
                    "atv": calc_atv(x.TP, x.TN, x.FP, x.FN),
                    "nTP": x.TP.sum(),
                    "nFP": x.FP.sum(),
                    "nFN": x.FN.sum(),
                },
                result_type="expand",
            )
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        df_all_bootstrap = pd.concat(
            [df_all_bootstrap, df_new], ignore_index=True
        )
    return df_all_bootstrap


df_all_bootstrap = get_df_bootstrap(df, n_bootstrap)
df_all_bootstrap
```

## Confidence intervals

Below we will use the rule of three since we don't have any FNs.

Using the rule of three, we know our sample could have an FN rate of:

$p = 1 - (1 - CI)^{1/n}$

where CI is the confidence interval (0.95 or 0.68) and $n$
is the sample size.

So we need to compute any metrics that are undefined due to
missing FN by hand.

```python
def rate(CI, n):
    return 1 - (1 - CI) ** (1 / n)


replacement_metrics = {"det": {}, "mis": {}}

n = len(df)
nTP = df_base.loc[df_base["metric"] == "nTP", "value"].values[0]

for CI in [0.68, 0.95]:
    replacement_metrics["det"][CI] = calc_det(TP=nTP / n, FN=rate(CI, n))
    replacement_metrics["mis"][CI] = calc_mis(TP=nTP / n, FN=rate(CI, n))

replacement_metrics
```

```python
# Calculate the quantiles over the bootstrapped df
def calc_ci(
    df_bootstrap, df_base, replace_fn_metrics=None, save_filename_suffix=None
):
    df_grouped = df_bootstrap.groupby("metric")
    for ci in [0.68, 0.95]:
        df_ci = df_base.copy()
        points = {"low_end": (1 - ci) / 2, "high_end": 1 - (1 - ci) / 2}
        for point, ci_val in points.items():
            df = (
                df_grouped.quantile(ci_val)
                .melt(ignore_index=False)
                .reset_index()
                .rename(columns={"variable": "trigger"})
            )
            df["point"] = point
            df_ci = df_ci.append(df, ignore_index=True)
        # Special case for trigger1 mis and det
        df_ci.loc[
            (df_ci.metric == "det")
            # & (df_ci.trigger.isin(["Trigger1", "framework-min"]))
            & (df_ci.point == "low_end"),
            "value",
        ] = replacement_metrics["det"][ci]
        df_ci.loc[
            (df_ci.metric == "mis")
            # & (df_ci.trigger.isin(["Trigger1", "framework-min"]))
            & (df_ci.point == "high_end"),
            "value",
        ] = (
            1 - replacement_metrics["mis"][ci]
        )
        # Save file
        output_filename = f"phl_perf_metrics_table_ci_{ci}.csv"
        df_ci.to_csv(trigger_dir / output_filename, index=False)


calc_ci(df_all_bootstrap, df_base, replace_fn_metrics=True)
```
