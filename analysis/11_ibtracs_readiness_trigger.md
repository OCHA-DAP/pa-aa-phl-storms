# Readiness Trigger — IBTrACS Analysis (2026 Revision)

**Notebook:** `11_ibtracs_readiness_trigger.ipynb`

Tests the readiness trigger threshold against historical IBTrACS Western Pacific track data for the four target regions of the revised 2026 trigger framework.

---

## Target Regions

| Code | Region |
|------|--------|
| PH02 | Region II (Cagayan Valley) |
| PH03 | Region III (Central Luzon) |
| PH05 | Region V (Bicol Region) |
| PH08 | Region VIII (Eastern Visayas) |

---

## Data and Methodology

| Item | Detail |
|------|--------|
| Source | IBTrACS Western Pacific (`storms.ibtracs_tracks_geo`) |
| Record | 2000–2024 (25 years) |
| Track points | 22,115 across 604 storms |
| Wind speed column | WMO 10-minute sustained winds |
| Spatial check | Consecutive track points connected into LineString segments; storm counted only if a segment crosses the actual land boundary of a target region (no buffer) |
| Activation rule | Once per season — if multiple storms qualify in the same year, it counts as one activation |

**Wind speed conversion:**
> 177 kph (1-min) × 0.88 = **156 kph (10-min) = 84.1 kt** — applied to IBTrACS data

---

## Return Period Results

### 177 kph (1-min) / 156 kph (10-min) — Current readiness threshold

24 storms made landfall at threshold across 25 years. Applying the once-per-season rule gives **15 activated seasons**.

| Region | Seasons | Return period (yrs) |
|--------|---------|---------------------|
| Region II (Cagayan Valley) | 8 | 3.1 |
| Region V (Bicol Region) | 7 | 3.6 |
| Region VIII (Eastern Visayas) | 5 | 5.0 |
| Region III (Central Luzon) | 4 | 6.2 |
| **Overall (any region)** | **15** | **1.7** |

At 177 kph the overall return period is 1-in-1.7 years — too frequent for most AA budgets.

### 192 kph (1-min) / 169 kph (10-min) — PAGASA Typhoon (test threshold)

| Region | Seasons | Return period (yrs) |
|--------|---------|---------------------|
| Region VIII (Eastern Visayas) | 3 | 8.3 |
| Region II (Cagayan Valley) | 2 | 12.5 |
| Region V (Bicol Region) | 2 | 12.5 |
| Region III (Central Luzon) | 0 | — |
| **Overall (any region)** | **7** | **3.6** |

192 kph (1-min) corresponds to 91 kt (10-min WMO), within the PAGASA **Typhoon** category (64–99 kt). This is the threshold at which the overall return period first reaches 1-in-3 years.

---

## Threshold Sensitivity

A wind speed vs return period curve was computed by sweeping thresholds from 60–120 kt (10-min) with the once-per-season rule applied throughout.

| Threshold | 1-min kph | 10-min kt | Seasons | Overall RP |
|-----------|-----------|-----------|---------|------------|
| Current readiness | 177 | 84 | 15 | 1.7 yrs |
| **1-in-3 year target** | **192** | **91** | **7** | **3.6 yrs** |
| PAGASA Super Typhoon | ~210 | 100 | 5 | 5.0 yrs |

---

## Key Storms at 177 kph Threshold

Selected storms making landfall at or above the readiness threshold, ordered by intensity:

| Storm | Year | Region | Wind at landfall (kph, 10-min) |
|-------|------|--------|-------------------------------|
| Haiyan | 2013 | Region VIII | 218 |
| Megi | 2010 | Region II | 213 |
| Goni | 2020 | Region V | 204 |
| Haima | 2016 | Region II | 176 |
| Mangkhut | 2018 | Region II | 185 |
| Noul | 2015 | Region II | 176 |
| Rai | 2021 | Region VIII | 185 |
| Man-yi | 2024 | Regions II, III, V | 190 |

---

## Design Considerations

- **Region II (Cagayan Valley)** is the most frequently exposed region, appearing in 8 of the 15 activated seasons. Its northeast-facing Pacific coast is the primary landfall corridor for intense westward-tracking typhoons.
- **Multiple storms in one season** occurred in 7 of the 25 years (notably 2006, 2015, and 2016 each had three qualifying storms). The once-per-season rule means these count as one activation.
- **192 kph (1-min) / 91 kt** is the empirically derived 1-in-3 year threshold for this geography. It falls within the PAGASA Typhoon category and provides a clean narrative: readiness activates for intense typhoons making direct landfall in the target regions.
