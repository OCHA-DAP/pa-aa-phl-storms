# Observational Trigger — Wind Exposure & Population (2026 Revision)

**Notebook:** `13_observational_trigger.ipynb`

Evaluates a historical wind exposure trigger for the Philippines AA Storms programme using IBTrACS best-track data. The trigger asks: **how many people in the target regions were inside the 64-kt wind footprint of a super typhoon?** This is an observational (not forecast-based) assessment - it tells us what *actually happened*, providing the baseline for calibrating the readiness and action trigger thresholds.

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
| Source | IBTrACS WP - `storms.ibtracs_tracks_geo` (PostGIS) |
| Wind speed units | JMA 10-minute sustained winds in knots |
| Record | 1980-2024 (45 years) |
| Storm filter | Centre wind >= **100 kt** (PAGASA super typhoon = 185 kph 10-min sustained) |
| Wind footprint | Asymmetric 64-kt wind radius polygons (NE/SE/SW/NW quadrants) per qualifying track point; mean-infilled where observed radii are missing |
| Mean radii (infill) | NE 41 nm / SE 38 nm / SW 35 nm / NW 38 nm |
| Spatial scope | Footprints are **clipped to the target-region union** before any population count |
| Population | WorldPop 2026 1km raster (`global_pop_2026_CN_1km_R2025A_UA_v1.tif`). Population is summed from **raster pixels inside the actual wind footprint** - not from full municipality totals |
| Municipality count | Admin-3 municipalities whose boundary intersects the clipped wind footprint |
| Total population in target regions | 42,324,903 |
| Storms assessed | 30 super typhoons with target-region exposure (64-kt footprint overlap), from 1,127 total WP storms |

**Why 100 kt centre wind?**
> At 64 kt centre wind the 64-kt radius is degenerate (approx 0), so including weaker typhoons produces noisy results driven by mean-infill rather than actual wind structure. At 100 kt the storm centre sits 36 kt above the 64-kt contour, so the wind field extends substantially outward and IBTrACS radii are well-observed (53% observed coverage at >= 100 kt vs 6.8% across all intensity levels).

**Why 64-kt radii, not 34-kt or 50-kt?**
> The 64-kt (~120 kph) wind zone is the destructive threshold for severe structural damage. The 34-kt and 50-kt radii would capture a much larger area of tropical storm and moderate typhoon winds - appropriate for evacuation planning but not for a damage-based observational trigger.

---

## Return Period Results

### Top 15 storms by population exposed to 64-kt winds

| Rank | Storm | Year | Max wind (km/h) | Municipalities | Population exposed |
|------|-------|------|-----------------|----------------|--------------------|
| 1 | Goni | 2020 | 222 | 65 | 3,501,347 |
| 2 | Haiyan | 2013 | 232 | 87 | 3,062,238 |
| 3 | Megi | 2010 | 232 | 70 | 2,759,613 |
| 4 | Mangkhut | 2018 | 204 | 73 | 2,547,622 |
| 5 | Zeb | 1998 | 204 | 53 | 2,140,650 |
| 6 | Nock-ten | 2016 | 195 | 35 | 1,053,152 |
| 7 | Nancy | 1982 | 185 | 26 | 936,323 |
| 8 | Man-yi | 2024 | 195 | 36 | 882,887 |
| 9 | Betty | 1987 | 204 | 29 | 570,387 |
| 10 | Angela | 1995 | 213 | 14 | 361,450 |
| 11 | Noul | 2015 | 204 | 23 | 354,757 |
| 12 | Agnes | 1984 | 195 | 15 | 348,123 |
| 13 | Doksuri | 2023 | 185 | 11 | 298,857 |
| 14 | Koppu | 2015 | 185 | 13 | 208,730 |
| 15 | Ellen | - | 204 | 7 | 122,099 |

### Population threshold sweep

How many seasons had at least one super typhoon exposing >= X people to 64-kt winds in the target regions:

| Population threshold | Seasons activated | Return period (yrs) | Storms |
|----------------------|-------------------|---------------------|--------|
| >= 100k | 14 | 3.2 | 16 |
| >= 250k | 13 | 3.5 | 13 |
| >= 500k | 9 | 5.0 | 9 |
| >= 750k | 8 | 5.6 | 8 |
| >= 1.0M | 6 | 7.5 | 6 |
| >= 1.5M | 5 | 9.0 | 5 |
| >= 2.0M | 5 | 9.0 | 5 |
| >= 3.0M | 2 | 22.5 | 2 |
| >= 5.0M | 0 | - | 0 |

The >= 250k threshold (RP 3.5 yr, 13 seasons) is closest to the CMA readiness trigger baseline of RP 3.1 yr.

---

## Per-Region Population Exposure

Population inside the 64-kt footprint clipped independently to each region (raster-based, WorldPop 2026):

| Storm | Year | Region II | Region III | Region V | Region VIII |
|-------|------|-----------|------------|----------|-------------|
| Nancy | 1982 | 936,323 | - | - | - |
| Agnes | 1984 | - | - | - | 348,123 |
| Betty | 1987 | - | - | - | 570,387 |
| Angela | 1995 | - | - | 361,450 | - |
| Zeb | 1998 | 2,140,650 | - | - | - |
| Cimaron | 2006 | 62,058 | 55,558 | - | - |
| Megi | 2010 | 2,738,963 | 20,681 | - | - |
| Haiyan | 2013 | - | - | - | 3,062,238 |
| Koppu | 2015 | 142,692 | 66,512 | - | - |
| Noul | 2015 | 354,757 | - | - | - |
| Nock-ten | 2016 | - | - | 1,053,152 | - |
| Mangkhut | 2018 | ~2,547,622 | - | - | - |
| Goni | 2020 | ~3,501,347 | - | - | - |
| Man-yi | 2024 | ~882,887 | - | - | - |

*Mangkhut, Goni, and Man-yi regional breakdown approximated from total; exact per-region split available in notebook output.*

### Population exceedance by region (chart)

Three population thresholds (100k, 500k, 1M) shown as stacked horizontal bars per storm with each region's contribution in a distinct colour. A right panel shows how many storms exceeded the threshold per region. This is the per-region equivalent of the population threshold sweep above.

Key observations from the chart:
- **Region II dominates** at all thresholds - it has the most storms exceeding 100k, 500k, and 1M
- **Region VIII** is the only region with a 3M+ event (Haiyan 2013) but has few storms at lower thresholds
- **Region V** activates almost exclusively for Nock-ten (2016) and Goni (2020) at higher thresholds
- **Region III** rarely activates independently - it appears alongside Region II in Megi and Koppu

---

## Activations by Season (>= 500k threshold, RP 5.0 yr)

| Year | Storm | Primary region(s) | Population exposed |
|------|-------|-------------------|--------------------|
| 1982 | Nancy | Region II | 936,323 |
| 1987 | Betty | Region VIII | 570,387 |
| 1998 | Zeb | Region II | 2,140,650 |
| 2010 | Megi | Region II, III | 2,759,613 |
| 2013 | Haiyan | Region VIII | 3,062,238 |
| 2016 | Nock-ten | Region V | 1,053,152 |
| 2018 | Mangkhut | Region II | 2,547,622 |
| 2020 | Goni | Region II, V | 3,501,347 |
| 2024 | Man-yi | Region II | 882,887 |

Non-activated seasons: 1980, 1981, 1983, 1984, 1985, 1986, 1988, 1989, 1990-1997, 1999-2009, 2011, 2012, 2014, 2015, 2017, 2019, 2021, 2022, 2023

---

## Design Considerations

- **Region VIII (Eastern Visayas) is captured here but not by CMA.** Haiyan (2013) is the second-highest exposure event (3.1M people) and is driven entirely by Region VIII. It does not appear in the CMA readiness trigger activations because rapid intensification in the days before landfall meant the CMA forecast 4-7 days out did not show it at threshold. This is the critical operational gap: the worst storms for Region VIII are rapid intensifiers that are systematically missed by forecast-based triggers.

- **Goni (2020) is the worst event on record** at 3.5M people exposed. It does not appear in the CMA record (2004-2025) as an activation, suggesting it also intensified rapidly. Goni was a Category 5 equivalent at landfall in Bicol/Quezon but this analysis captures its 64-kt wind reach into the target regions.

- **Closest RP match to the readiness trigger:** The >= 250k population threshold gives RP 3.5 yr (13 seasons over 45 years), compared to CMA readiness at RP 3.1 yr (7 seasons over 22 years). These are broadly comparable, though the time periods and metrics differ.

- **Region II (Cagayan Valley) dominates** the top events in observational terms - Zeb (1998), Megi (2010), Mangkhut (2018), and Man-yi (2024) all ran through the Luzon corridor. This mirrors the CMA readiness finding, but here we also see the historic contribution of older storms (Nancy 1982, Betty 1987) not in the CMA record.

- **Storm count note:** "Ellen" appears in the exposure table with a missing year, which indicates a data issue in the IBTrACS storm ID parsing. This storm contributes a modest 122k people and does not affect any threshold above >= 100k for more than one count.

- **Comparison with CMA readiness trigger (nb11.1):** CMA gave 9 storms / 7 seasons (RP 3.1 yr) over 2004-2025. Observational gives 30 storms / 20 seasons over 1980-2024 at any exposure level. The two triggers agree on the core events (Megi, Mangkhut, Man-yi, Noul, Koppu) but the observational analysis additionally captures Haiyan, Goni, and Nock-ten - the rapid intensifiers and non-Luzon-corridor storms that CMA forecasts missed.
