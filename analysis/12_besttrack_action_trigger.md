# 12 — Action Trigger: Best Track Hindcast

Evaluates a housing-damage-based action trigger for the Philippines AA Storms programme
using IBTrACS best-track hindcast results. The trigger asks: **how many houses in a target
region are predicted to be damaged by a typhoon?** This is a post-landfall hindcast
assessment - it tells us what the damage model predicts *after* applying observed track and
intensity, providing the baseline for calibrating per-region action thresholds.

---

## Setup

| Parameter | Value |
|---|---|
| Data sources | Three combined hindcast runs (see below) |
| Record | 2006-2024 (19 years) |
| Storms | 90 total (74 corrected + 1 updated + 15 old) |
| Regions | PH02 Region II, PH03 Region III, PH05 Region V, PH08 Region VIII |
| Damage metric | Median predicted houses damaged (per-region sum from admin-3 damage tables) |
| Binary flags | 500-house increments from 500 to 40,000 houses |
| Target RP | 3 years (TARGET_SEAS = 6 seasons in 19 years) |

---

## Data Sources

Three hindcast runs are combined in priority order. A storm is taken from the
highest-priority source that contains it:

| Priority | Blob path | Storms | Coverage |
|---|---|---|---|
| 1 (primary) | `all_storms_hindcast_results/besttrack_500increments_corrected/` | 74 | 2006-2024 |
| 2 | `all_storms_hindcast_results/besttrack_500increments/` | +1 | 2011 (NESAT) |
| 3 (old) | `besttrack_hindcast_results/fine_500_increments/` | +15 | 2006-2020 |

The 15 storms restored from the old run include Fengshen (2008), Conson (2010),
Bopha (2012), Usagi and Utor (2013), Jangmi, Lingling, and Rammasun (2014),
Goni, Linfa, and Noul (2015), Meranti and Tokage (2016), and Vongfong (2020).

---

## Calibrated Thresholds

Five tiers, calibrated so that each region individually achieves RP ~3 yr at Tier 3.

**Constraint:** RII > max(RV, RVIII) and RIII > max(RV, RVIII) at every tier.
RII and RIII may differ from each other.

| Tier | Region II | Region III | Region V | Region VIII |
|------|-----------|------------|----------|-------------|
| 1 | 2,000 | 2,000 | 1,500 | 500 |
| 2 | 4,000 | 4,000 | 3,500 | 1,500 |
| **3 (target)** | **6,000** | **6,000** | **5,000** | **2,500** |
| 4 | 9,000 | 9,000 | 6,000 | 4,000 |
| 5 | 21,000 | 21,000 | 20,500 | 7,500 |

Values are median predicted houses damaged (sum across admin-3 units per region).
The composite trigger fires if **any** region meets or exceeds its threshold.

---

## Trigger Logic

- **Per-region trigger:** fire if median predicted damage >= region threshold for that tier
- **Composite trigger:** fire if any region fires (union of activated seasons)
- **Return period:** RECORD_YEARS / n_seasons_activated
- **Composite RP at Tier 3:** driven by whichever region activates most frequently;
  Region VIII adds unique seasons (Fengshen 2008, Haiyan 2013) not captured by Regions II, III, V

---

## Design Considerations

- **Tier 3 is the target action tier.** Each region is calibrated individually to RP ~3 yr.
  The composite RP at Tier 3 will be lower than 3 yr because different regions activate
  in different seasons.

- **Binary flag range (500-40,000 houses).** All Tier 3 thresholds fall within this range,
  enabling direct lookup from the 500-increment CSV. Earlier calibrations based on
  per-storm damage tables required thresholds above 40,000 houses for Region V and were
  incompatible with the binary flag format.

- **RII and RIII constraint.** Both Region II and Region III thresholds must exceed
  max(RV, RVIII) at every tier, but they are not required to be equal to each other.
  Current thresholds set them equal pending independent per-region calibration.

- **Old run storms.** The 15 storms restored from the old run (2006-2020) were processed
  with an earlier model version. Their binary flags are directly comparable in format but
  may reflect slightly different wind-field or vulnerability assumptions.

- **Damage table coverage.** The per-storm damage tables (used for the regional summary
  statistics in the notebook) cover only the old run storms (up to ~2020). Storms from the
  corrected and updated runs (2021-2024) do not yet have per-storm damage table outputs in
  the blob.
