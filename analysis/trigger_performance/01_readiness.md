# Readiness trigger performance

Reads in the readiness analysis by Joseph, does some BSRS,
and gets the stats

```python
%load_ext jupyter_black
```

```python
from pathlib import Path
import os
import warnings

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
df_readiness
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
    # year = str(filename.name).split("_")[1][:4]
    # results["year"] = year
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
df_activation
```

```python
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


df_activation.loc[
    (df_activation["Should have activated"] == "N")
    & (
        (df_activation["Activation reached"] == False)
        & (df_activation["readiness"] == "TN")
    ),
    "activation",
] = "TN"


df_activation.loc[
    (df_activation["Should have activated"] == "N")
    & ((df_activation["Activation reached"] == True)),
    "activation",
] = "FP"


df_activation
```

```python
# Final DF - compute performance of full framework
df = df_activation.copy()

df.loc[
    (df["readiness"] == "TP") & (df["activation"] == "TP"),
    "framework",
] = "TP"

df.loc[
    (df["readiness"] == "TN") & (df["activation"] == "TN"),
    "framework",
] = "TN"

df.loc[
    ((df["readiness"] == "FN") | (df["activation"] == "FN"))
    & (~df["activation"].isna()),
    "framework",
] = "FN"

# Only activation has FP
df.loc[
    df["activation"] == "FP",
    "framework",
] = "FP"

df
```

```python
# Drop unneeded columns for BSRS
trigger_names = ["readiness", "activation", "framework"]
df = df[trigger_names]
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
df_new = (
    df.sample(n=len(df), replace=True, random_state=rng.bit_generator)
    .apply(pd.value_counts)
    .fillna(0)
)
for count in ["FN", "FP", "TN", "TP"]:
    if count not in df_new.index:
        df_new.loc[count] = 0
df_new
```

```python
def calc_df_base(df):
    df_new = df.apply(pd.value_counts).fillna(0)
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
trigger = "activation"
df_new = (
    df[[trigger]]
    .dropna()
    .sample(
        n=sum(~df[trigger].isna()),
        replace=True,
        random_state=rng.bit_generator,
    )
    .apply(pd.value_counts)
)
df_new

# Fill in counts missing in original dataset as 0
for count in ["FN", "FP", "TN", "TP"]:
    if count not in df_new.index:
        df_new.loc[count] = 0

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
).melt(ignore_index=False, var_name="trigger").reset_index().rename(
    columns={"index": "metric"}
)
```

```python
n_bootstrap = 10_000  # 10,000 takes about 4 mins

# Adapted from Niger
def get_df_bootstrap(df, n_bootstrap=1_000):
    # Lots of divide by 0 warnings, turn off
    warnings.filterwarnings(action="ignore")
    # Create a bootstrapped DF
    df_all_bootstrap = pd.DataFrame()
    for trigger in df.columns:
        i = 0
        while i < n_bootstrap:
            df_new = (
                df[[trigger]]
                .dropna()
                .sample(
                    n=sum(~df[trigger].isna()),
                    replace=True,
                    random_state=rng.bit_generator,
                )
                .apply(pd.value_counts)
            )
            # If a particular metric is missing that was in the
            # original sample, need to redraw
            if set(df_new.index) != set(df[trigger].dropna().unique()):
                continue
            i += 1
            # Fill in counts missing in original dataset as 0
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
                .melt(ignore_index=False, var_name="trigger")
                .reset_index()
                .rename(columns={"index": "metric"})
            )
            df_all_bootstrap = pd.concat(
                [df_all_bootstrap, df_new], ignore_index=True
            )
    warnings.filterwarnings(action="default")
    return df_all_bootstrap


df_all_bootstrap = get_df_bootstrap(df, n_bootstrap)
df_all_bootstrap
```

## Confidence intervals

Below we will use the rule of three since we don't have any FPs.

Using the rule of three, we know our sample could have an FP rate of:

$p = 1 - (1 - CI)^{1/n}$

where CI is the confidence interval (0.95 or 0.68) and $n$
is the sample size.

So we need to compute any metrics that are undefined due to
missing FP by hand.

```python
CIs = [0.68, 0.95]


def rate(CI, n):
    return 1 - (1 - CI) ** (1 / n)


replacement_metrics = pd.DataFrame()

n = len(df)
nTP = df_base.loc[df_base["metric"] == "nTP", "value"].values[0]

for CI in CIs:
    # for trigger in trigger_names:
    for trigger in ["readiness"]:
        n = sum(~df[trigger].isna())
        nTP = nTP = df_base.loc[
            (df_base["metric"] == "nTP") & (df_base["trigger"] == trigger),
            "value",
        ].values[0]
        replacement_metrics = pd.concat(
            [
                replacement_metrics,
                pd.DataFrame(
                    {
                        "metric": ["var", "far"],
                        "trigger": trigger,
                        "CI": CI,
                        "value": [
                            calc_var(TP=nTP / n, FP=rate(CI, n)),
                            1 - calc_far(TP=nTP / n, FP=rate(CI, n)),
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

replacement_metrics
```

```python
# Calculate the quantiles over the bootstrapped df
def calc_ci(
    df_bootstrap, df_base, replace_fn_metrics=None, save_filename_suffix=None
):
    df_grouped = df_bootstrap.groupby(["metric", "trigger"])
    for CI in CIs:
        df_ci = df_base.copy()
        points = {"low_end": (1 - CI) / 2, "high_end": 1 - (1 - CI) / 2}
        for point, ci_val in points.items():
            df = df_grouped.quantile(ci_val).reset_index()
            df["point"] = point
            df_ci = df_ci.append(df, ignore_index=True)
        # Special case for trigger1 mis and det
        # for trigger in trigger_names:
        for trigger in ["readiness"]:
            for metric, point in zip(("var", "far"), ("low_end", "high_end")):
                df_ci.loc[
                    (df_ci.metric == metric)
                    & (df_ci.trigger == trigger)
                    & (df_ci.point == point),
                    "value",
                ] = replacement_metrics.loc[
                    (replacement_metrics.metric == metric)
                    & (replacement_metrics.trigger == trigger)
                    & (replacement_metrics.CI == CI),
                    "value",
                ].values[
                    0
                ]
        # Save file
        output_filename = f"phl_perf_metrics_table_ci_{CI}.csv"
        df_ci.to_csv(trigger_dir / output_filename, index=False)


calc_ci(df_all_bootstrap, df_base, replace_fn_metrics=True)
```
