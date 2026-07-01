# Trigger Revision — MHIBFEWS (Philippines Storms)

**Notebook:** `10_trigger_revision_MHIBFEWS.ipynb`

This analysis explores revising the anticipatory action trigger for the Philippines storms programme using the MHIBFEWS framework. The central question is whether **municipality-level wind exposure** — derived from historical IBTrACS track data — can serve as a meaningful trigger metric alongside or instead of the existing 510-model damage thresholds.

---

## Data Sources

| Source | Coverage | Use |
|--------|----------|-----|
| IBTrACS (WP basin) | 1980–2024, 62,823 track points, 1,127 storms | Storm tracks and wind speed |
| Philippine CODAB (admin 1 & 3) | Current | Spatial boundaries |
| WorldPop 2026 (1 km) | Philippines | Population exposure |
| Combined input data | Per municipality per typhoon | Rainfall, wind, track features |
| CERF allocations | Selected storms | Financial response benchmark |

---

## Exposure Methods

Two approaches were tested to define a storm's wind footprint:

### Option A — Fixed Buffer (100 km, ≥ 80 kt sustained)

IBTrACS `gust_speed` has **0% historical coverage**, so the 194 km/h gust threshold cannot be applied directly. The equivalent sustained wind is derived using a gust factor of 1.3:

> 194 ÷ 1.3 ÷ 1.852 ≈ **80.6 knots (149 km/h)**

Each track point at or above this threshold is buffered by 100 km. Circles per storm are dissolved before the spatial join with admin-3 municipalities.

- Storms with exposure ≥ 1 municipality (full record): **72**
- Storms from 2006 onwards: **37**

### Option B — IBTrACS 64-knot Wind Radii (asymmetric)

IBTrACS reports per-quadrant (NE/SE/SW/NW) radii of 64-knot winds in nautical miles. Only the `usa_quadrant_radius_64` column has usable coverage (7.8%, 4,877 of 62,823 points). Coverage is too sparse to use alone.

### Option B with Mean Infill — Primary Method

To maximise coverage, missing quadrant values are filled with the per-quadrant mean across all track points that have valid radii. All track points ≥ 64 kt are included.

**Mean 64-kt radii used for infill:** NE 41 nm / SE 38 nm / SW 35 nm / NW 38 nm

| | Count | Share |
|---|---|---|
| Track points ≥ 80 kt | 5,706 | — |
| Observed radii | 2,827 | 49.5% |
| Mean-filled | 2,879 | 50.5% |
| Storms with exposure ≥ 1 municipality (2006–2024) | 39 | — |

This is the method used in all downstream analyses.

---

## Key Results

### Top Storms by Municipality Exposure (2006–2024, wind radii + mean infill)

| Rank | Storm | Year | Max wind (km/h) | Municipalities exposed | People exposed |
|------|-------|------|-----------------|----------------------|----------------|
| 1 | Kammuri | 2019 | 166.7 | 235 | 18,878,933 |
| 2 | Megi | 2010 | 231.5 | 223 | 9,990,291 |
| 3 | Rai | 2021 | 194.5 | 217 | 15,712,429 |
| 4 | Haiyan | 2013 | 231.5 | 203 | 10,780,971 |
| 5 | Mangkhut | 2018 | 203.7 | 182 | 6,223,082 |
| 6 | Haima | 2016 | 213.0 | 175 | 6,255,757 |
| 7 | Durian | 2006 | 194.5 | 109 | 7,049,563 |
| 8 | Melor | 2015 | 175.9 | 109 | 6,351,791 |
| 9 | Nock-ten | 2016 | 194.5 | 107 | 9,629,332 |
| 10 | Hagupit | 2014 | 213.0 | 104 | 3,895,893 |
| 11 | Man-yi | 2024 | 194.5 | 96 | — |

Total Philippine population (WorldPop 2026): **142,965,110**

### CERF Allocations for Reference Storms

| Storm | CERF Allocation (USD) |
|-------|-----------------------|
| Durian (2006) | 855,843 |
| Haiyan (2013) | 25,284,204 |
| Koppu (2015) | 1,512,074 |
| Goni (2020) | 3,113,674 |
| Rai (2021) | 11,974,601 |

---

## Return Period Analysis

Method: **Weibull plotting position** — event-based, not annual maxima.

> RP = (N + 1) / rank, where N = years in record, rank = position sorted worst to least severe

- Record: **19 years (2006–2024)**
- Storms ranked: **39**

Each storm is one entry regardless of year, so RP can be less than 1 for frequent events. Return periods are computed for both municipality exposure and people exposed.

---

## Case Study: Typhoon Man-yi (2024)

Under Option A (fixed buffer), Man-yi is the **second most impactful storm on record** with 272 municipalities exposed. Under the wind radii method it ranks 11th (96 municipalities), reflecting the difference between a uniform 100 km circle and the actual asymmetric wind field.

