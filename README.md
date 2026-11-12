# Cyclone Warning for Fishers

**Rapid-intensification cyclone warnings for India's fishing communities, in their own language.**
HS202 Human Geography and Societal Needs · Group 18 · IIT Ropar · based on SIH 2026 problem 26070 (MoES / IMD)

Live site: **https://nikhil280607-oss.github.io/cyclone-warning-hs202/**

The system reads India Meteorological Department (IMD) storm records every 3 hours, predicts whether a cyclone will
strengthen by 30 knots or more in the next 24 hours (rapid intensification, RI), finds the Indian coast it is heading
for, and writes a short action-first SMS in that coast's language (Odia, Bengali, Gujarati, Tamil, Telugu, Marathi,
Malayalam, Kannada, plus English).

## Results (replay on every storm, as if the system ran live)

| | 2019–2025 (test) | 2013–2018 (check) |
|---|---|---|
| Cyclones heading for India | 26 | 17 |
| Warned before arrival | **25** | **15** |
| Severe cyclones (48 kt+) warned | **19 of 19** | 10 of 11 |
| Median time from first SMS to arrival | **63 h** | 66 h |
| SMS sent (at most one a day per storm) | 64 (vs 396 at every update) | 38 |

RI model on 76 unseen test storms (2019–2025), thresholds chosen on 2013–2018:

| Model | Recall | Precision | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| **Logistic regression (used for warnings)** | **0.81** | 0.17 | 0.22 | **0.84** |
| Random forest | 0.65 | 0.21 | 0.22 | 0.82 |
| Gradient boosting | 0.84 | 0.12 | 0.19 | 0.79 |
| Neural network (MLP) | 0.89 | 0.12 | 0.23 | 0.82 |
| LR + RF ensemble, extra inputs (high-confidence mode) | 0.65 | 0.19 | **0.27** | 0.84 |
| Rule: +10 kt in last 12 h | 0.63 | 0.18 | 0.14 | 0.72 |
| Random guess | – | – | 0.06 | 0.50 |

The high-confidence mode sent **no false warnings** in either period (22 of 26 test cyclones warned).

## Satellite (INSAT-3DR, MOSDAC)

447 INSAT-3DR IMAGER TIR1 brightness-temperature images (L1C ASIA_MER, 4 km) of six cyclones (Fani, Amphan, Tauktae,
Yaas, Biparjoy, Michaung) were processed with `src/mosdac_features.py`: for each image, the storm centre from IMD's
best track is located on the Mercator grid and cloud-top statistics are measured within 200 km and 500 km.
For Fani (2019) the cold cloud shield within 500 km reached its largest extent (66% of the area below 220 K) at the
onset of rapid intensification on 29 April. Across the six storms, times followed by RI had a larger cold-cloud share
within 500 km (58% vs 40%) and colder mean cloud tops (221 K vs 233 K). Only 7 of the 430 matched images fall in RI
periods, so these are case-study results; `results/satellite/` holds every number.

## Reading test with native readers

A Google Form showed the warning SMS in eight languages (Bengali, Odia, Tamil, Telugu, Gujarati, Marathi, Malayalam,
Kannada) and was shared through IIT Ropar student groups for each state. 105 people responded (12–13 Nov 2026;
40% have lived in a coastal district). 90% chose the correct action (return to shore immediately), 84% rated the
urgency clear (4 or 5 of 5, mean 4.27), 94% found the information at least mostly sufficient, 86% rated it effective as
an official alert (mean 4.30) and 76% said they would act on it (19% maybe). Raw export: `results/survey/form_responses.csv`.

## Run it

```
pip install -r requirements.txt
python3 src/run_replay.py data/observations.json data/storms.json data/india_coastal_states.geojson results
python3 src/compare_models.py data/observations.json data/storms.json results/models
python3 src/make_figures.py data/observations.json data/storms.json data/india_coastal_states.geojson data/ne_50m_admin_0_countries.geojson results results/models figures
```
Satellite features from MOSDAC INSAT-3DR files: `python3 src/mosdac_features.py FOLDER_WITH_H5` (run from a folder holding `storm_tracks.csv`).

## Repository

- `src/prep.py` IMD records → model rows (inputs known at time T, RI target from real timestamps ±3 h)
- `src/model.py` logistic-regression RI model, F2-chosen threshold
- `src/warn.py` alert rule, coast at risk, 9-language SMS templates, SMS length (GSM-7 / Unicode)
- `src/run_replay.py` full replay on 2013–2018 and 2019–2025 storms
- `src/compare_models.py`, `src/improve_model.py`, `src/replay_ensemble.py` model comparison, CV, ablation, ensemble
- `src/make_figures.py`, `src/build_site_data.py` figures and website data
- `src/mosdac_features.py` INSAT-3DR TIR1 brightness-temperature features around the storm centre
- `src/satellite_analysis.py`, `src/sat_figures.py` satellite case-study analysis and figures
- `src/survey_analysis.py` reading-test summary by language and coastal background
- `results/sms_cost_by_state.csv` SMS volume and cost per fisher household by state
- `index.html` the live site · `results/` all outputs · `figures/` report figures · `data/` inputs

## Data and acknowledgements

- IMD RSMC New Delhi best track 1982–2025, via the open stormtrack/imdtrack project. Data © India Meteorological Department.
- INSAT-3DR IMAGER data: **MOSDAC, Space Applications Centre (ISRO)**.
- Boundaries: Natural Earth.
- In the replay, each storm's real track stands in for IMD's official track forecast.

## Team (Group 18)

Indukuri Nikhil Rama Manikanta Varma (team leader) · Jaskaran Kora · Shourya Pastor · Arjun S · Rishi Datt Gupta ·
Mistry Kush Hiteshkumar · Shams Kalappatt · Lakshita Meena · Simran

Course: HS202, IIT Ropar · Instructor: Dr. Amritesh · TA: Mr. Rahul Kumar
