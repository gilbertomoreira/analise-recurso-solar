# CLAUDE.md — SolarApp

Streamlit app for solar resource analysis (NASA POWER + Duffie & Beckman). UI/text in pt-BR; code,
function names and docstrings in English; SI units always explicit.

## Commands (use the project venv)
- Run app: `.\.venv\Scripts\python -m streamlit run src/solarapp/app.py`
- Deps: `requirements.txt` (runtime, used by Streamlit Cloud) and `requirements-dev.txt` (+pytest, responses, ruff)
- Tests: `.\.venv\Scripts\python -m pytest -q` · Lint: `.\.venv\Scripts\ruff check src tests`
- Run from the `SolarApp/` directory; `pyproject.toml` sets `pythonpath = ["src"]` for pytest.

## Layout and rules
- `physics/` = pure functions (no I/O). Angles in degrees, azimuth 0 = south, east negative.
  Each function documents its equation and D&B reference. Daily/monthly energy in kWh/m²/day.
- `data/` = I/O (NASA POWER, geocoding, cache, file import). Tests mock HTTP with `responses`;
  never hit the network in tests.
- `analysis.py` builds tables for UI/reports; `app.py` is UI only (E501 ignored there).
- Monthly tilted radiation uses Klein's mean-day method (`physics/tilted.py`).

## Gotchas
- The repo lives on Google Drive: imports are slow and OpenSSL hangs loading certifi's
  `cacert.pem`. `data/nasa_power.py::_SystemTrustAdapter` uses the OS trust store; keep it.
- Write files without BOM (PowerShell 5.1 `Set-Content -Encoding utf8` adds one and breaks TOML).
- NASA monthly responses include key `YYYY13` = annual mean; missing value is -999.
- Sibling files in the parent folder belong to an unrelated Garmin Connect IQ project.