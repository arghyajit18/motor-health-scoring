# Motor Health Scoring from Current and Vibration Signatures

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://motor-health-scoring-taxicqlsuzs3kbbmnyusy6.streamlit.app/)

**Live demo:** https://motor-health-scoring-taxicqlsuzs3kbbmnyusy6.streamlit.app/

Fleet-level motor condition monitoring: daily short current and vibration recordings are turned into spectral fault indicators, scored 0-100, and ranked with shop-floor actions.

## Outcomes

Validated on real rig data (Paderborn University bearing dataset, 4 bearings at 1500 rpm / 0.7 Nm / 1000 N, 320 one-second windows):

- Detection precision 1.00 with recall 0.90 against labeled healthy and inner-race-damage bearings
- Both damaged bearings flagged Alert with bearing defect as the suspected cause; both healthy bearings stay Healthy
- Current spectrum alone was inconclusive on this rig (recall near zero in testing), so the score fuses phase-current indicators with accelerometer band energy, where damage shows a 5-6x elevation across 400-900 Hz

The synthetic 12-motor development set (seeded rotor, bearing, eccentricity and unbalance faults over 45 days, kept under `backup_synthetic/`) scores precision 0.84 with recall 1.00 and about 12 days warning before full severity.

## Pipeline

```
data/generate_motor_data.py -> waveforms.npz, readings_meta.csv (synthetic 12 x 45 at 2 kHz)
data/load_paderborn.py      -> same layout from real .mat recordings (resample 64 kHz to 2 kHz,
                               third phase reconstructed, supply frequency estimated per file)
clean_features.py           -> waves_clean.npz, clean_meta.csv (missing/flat removal, spike clipping)
features.py                 -> motor_features.csv (FFT: RMS, crest, THD, rotor sidebands,
                               eccentricity and bearing bands, unbalance, vibration RMS/crest/high-band)
scoring.py                  -> health_trend.csv, fleet_ranking.csv, alerts.csv, metrics.json
eda.py                      -> outputs/*.png (trend, spectra, status, heatmap)
dashboard/app.py            -> Streamlit fleet monitor with what-if simulator
```

Run in order (synthetic, or real after placing bearing files in `data/paderborn_rars/`):

```bash
pip install -r requirements.txt
python data/generate_motor_data.py   # or: python data/load_paderborn.py
python clean_features.py
python features.py
python scoring.py
python eda.py
streamlit run dashboard/app.py
```

## Service (FastAPI + Docker)

The scoring logic lives in importable functions (`features.extract_feature_row`,
`scoring.fit_baseline` / `scoring.score_features`), and `api.py` serves them:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

`POST /score` takes one waveform reading plus metadata and returns the same
result the batch pipeline would produce:

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d @- <<'EOF'
{"phase_a": [...], "phase_b": [...], "phase_c": [...], "fs": 2000.0,
 "vibration": [...],
 "metadata": {"motor_id": "M-05", "load_pct": 72.0, "temp_c": 58.0,
              "f0_hz": 50.0, "fr_hz": 24.0, "slip": 0.04}}
EOF
# {"health": 0.0, "status": "Critical", "suspected_cause": "Bearing defect
#  frequencies", "action": "Replace DE/NDE bearings at next planned stop", ...}
```

`GET /fleet` returns the latest fleet ranking, `GET /health` is the liveness
check. Docker image build and run:

```bash
docker build -t motor-health-scoring .
docker run -p 8000:8000 motor-health-scoring
```

## Method

A StandardScaler plus IsolationForest fitted on early healthy readings cross-checks a rule score built from baseline z-scores of each fault indicator. Health is 100 at the baseline median and decays exponentially with degradation, with bands at 80/60/40 for Healthy, Watch, Alert and Critical. The dominant contributor names the suspected cause, and each cause maps to one action (bearing replacement, rotor inspection and rewind planning, alignment check, supply and insulation check). Feature code takes per-reading supply frequency, rotor speed and slip from the metadata, so inverter-driven rigs and mains-fed motors run through the same path.

To run on plant clamp-meter or drive recordings instead, convert them to one waveform per motor per day in the generator's layout; the rest of the pipeline is unchanged.

## Tech stack

Python, NumPy FFT, pandas, SciPy resampling, scikit-learn (IsolationForest), matplotlib, seaborn, Streamlit, Plotly.
