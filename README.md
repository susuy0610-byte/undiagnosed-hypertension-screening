# Non-invasive machine learning screening for undiagnosed hypertension

Analysis code and manuscript files for a study developing non-invasive models to screen for undiagnosed hypertension in Korean adults (KNHANES 2014-2020), with temporal validation (KNHANES 2021-2022) and external validation (Health Survey for England 2018, 2019, 2021, 2022).

## Contents

- `manuscript/` - main manuscript, supplementary material (Multimedia Appendix 1), and TRIPOD+AI checklist (Multimedia Appendix 2)
- `scripts/` - dataset construction, model development and freezing, external validation, calibration/decision-curve analysis, fairness checks, and figure scripts
- `requirements-ml.txt` - Python dependencies

## Data

No participant-level data are included in this repository.

- KNHANES microdata are available from the Korea Disease Control and Prevention Agency (https://knhanes.kdca.go.kr).
- HSE microdata are available from the UK Data Service under an End User Licence.

Scripts expect the raw files under `data/` in the repository root; paths are resolved relative to the repository.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ml.txt
bash scripts/run_knhanes_primary_ml_pipeline.sh
```

Random seed: 20260722.
