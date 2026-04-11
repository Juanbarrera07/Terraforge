# TerraForge — Session Resume Prompt

Copy everything between the horizontal rules into a new Claude session.

---

You are a senior GeoAI engineer continuing an in-progress build.
Read the full context below before doing anything. Do NOT start coding immediately.

---

## PROJECT

**TerraForge Mining Intelligence** — a production Streamlit application that
automates and quality-controls a SAR + multispectral mining survey pipeline for
geospatial consulting firms. Reduces human error via validation gates, audit
logging, and drift monitoring.

**Root:** `/mnt/d/App_Geo/GeoRisk_Agent/TMI/`  
**Entry point:** `streamlit run app.py`  
**Tests:** `python3 -m pytest tests/ -v` → 101 tests, 0 failures  
**Environment:** micromamba env `terraforge`, Python 3.11.15, WSL2  

---

## CURRENT FILE STRUCTURE

```
TMI/
├── app.py                          # Streamlit UI — 6-page sidebar, Page 1 functional
├── config/
│   └── pipeline_config.yaml        # All thresholds — edit here, app reloads on restart
├── pipeline/
│   ├── audit.py                    # ISO-timestamped JSON audit log per run_id
│   ├── config_loader.py            # YAML loader with hard-coded defaults fallback
│   ├── coregister.py               # CoregistrationResult + RMSE gate + AROSICS stub
│   ├── ingest.py                   # Sensor detection, date parsing, ZIP, metadata table
│   ├── preprocess.py               # DOS1 atmospheric correction + Lee speckle filter
│   ├── raster_io.py                # Windowed I/O — iter_windows, get_meta, write_raster
│   ├── session.py                  # Session state schema + stage unlock logic
│   └── validate.py                 # All validation gates + orchestrator
├── tests/
│   ├── conftest.py                 # make_raster factory fixture + make_layer helper
│   ├── test_coregister.py          # 23 tests
│   ├── test_ingest.py              # 22 tests
│   ├── test_preprocess.py          # 26 tests
│   └── test_validate.py            # 30 tests
├── docs/
│   ├── project_state.md            # Full phase-by-phase system documentation
│   └── resume_prompt.md            # This file
├── requirements-conda.txt          # conda-forge geo stack
├── requirements.txt                # pip-only packages
├── logs/                           # Runtime: {run_id}.json audit logs
├── models/                         # Runtime: trained model .pkl files
├── outputs/                        # Runtime: COG GeoTIFFs, reports, ZIPs
└── tmp/                            # Runtime: uploaded file staging
```

---

## COMPLETED PHASES

### Phase 1 — Scaffold ✅
`config_loader.py`, `session.py`, `audit.py`, `pipeline_config.yaml`

- YAML config with defaults fallback
- Session state schema: `run_id`, `raw_data`, `validation_results`, `preprocessed`,
  `features`, `model`, `classified`, `accuracy`, `audit_log`, `pipeline_unlocked`
- Stage unlock chain: `ingestion → preprocessing → features → classification → postprocess → export`
- Audit logger writes every gate/decision to `./logs/{run_id}.json` with ISO 8601 UTC timestamp

### Phase 2A — Light Ingestion ✅
`raster_io.py`, `ingest.py`

- `raster_io.py`: `get_meta()` (zero pixel reads), `iter_windows()`, `read_window()`,
  `read_overview()` (thumbnail only), `compute_overlap_pct()`, `write_raster()` (tiled deflate)
- **Design rule enforced throughout:** no function loads a full raster array
- `ingest.py`: Sentinel-1/2, Landsat 7/8/9, DEM sensor detection; date extraction
  from 7 tag formats + filename patterns; ZIP unwrapping; chunked 128 KiB file writes

### Phase 2B — Validation Gates ✅
`validate.py`

| Check | Severity | Critical |
|-------|----------|----------|
| `crs_match` | error | Yes — blocks pipeline |
| `resolution_compatibility` | warn | No |
| `date_proximity` | warn | No |
| `phenology_risk` | warn | No |
| `minimum_overlap` | error | Yes — blocks pipeline |
| `band_count_{layer}` | warn/error | No |

- `ValidationResult` frozen dataclass; `run_all_validations()` orchestrator
- `has_critical_failures()` drives the stage unlock gate in the UI

### Phase 2 UI — Streamlit Page 1 ✅
`app.py`

- 6-page sidebar navigation; pages 2–6 locked as stubs
- Page 1: three upload sections (SAR/Optical/DEM), ingest + validate pipeline,
  ✅/⚠️/❌ status badges, metadata dataframe, summary metrics, stage unlock button
- All events written to audit log; `session.unlock_stage("preprocessing")` called on pass

### Phase 3 — Preprocessing + Coregistration Interface ✅
`preprocess.py`, `coregister.py`

**DOS1 atmospheric correction (optical):**
- Two-pass windowed: Pass 1 finds per-band dark object value (min non-zero, non-nodata);
  Pass 2 subtracts and clips to `[0, dtype_max]`; nodata pixels preserved

**Lee speckle filter (SAR):**
- Integral-image local statistics: O(n) memory, no scipy, independent of kernel size
- Tile-by-tile with overlap padding (`kernel // 2` px) — eliminates edge artifacts
- Noise model: `noise_var = (band_mean / ENL)²`; Lee weight: `w = local_var / (local_var + noise_var)`
- Both functions accept `progress: Callable[[int, int], None]` for Streamlit progress bars

**Coregistration (`coregister.py`):**
- `CoregistrationResult` — frozen dataclass; `shift_map` excluded from equality
- `apply_rmse_gate(result, threshold)` — returns new instance; nulls `corrected_path` on failure
- `_arosics_available()` — runtime detection; activates real AROSICS backend automatically once installed
- `_run_arosics_coreg()` is fully written and waiting; currently blocked only by AROSICS install
- `get_shift_report()` returns JSON-serialisable dict for audit log

---

## KEY TECHNICAL RULES (never violate these)

1. **No full raster reads.** All pixel access goes through `iter_windows()` /
   `read_window()` from `raster_io.py`. `read_overview()` is the only exception
   and is restricted to preview/thumbnail use.

2. **conda-forge for all native geo libs.** Never `pip install` any of:
   `gdal`, `rasterio`, `fiona`, `shapely`, `pyproj`, `geopandas`, `arosics`,
   `scikit-image`, `scikit-learn`, `xgboost`, `scipy`.
   Pure-Python packages (`streamlit`, `pytest`, `imbalanced-learn`, `whitebox`,
   `rio-cogeo`, `pystac`, `reportlab`, `plotly`) use pip.

3. **Pipeline modules must not import Streamlit.** The UI layer lives in `app.py`
   only. Progress callbacks use `Callable[[int, int], None]` — the UI passes a
   lambda; tests pass `None`.

4. **Write tests alongside each module** — not deferred. Every public function
   in a pipeline module has test coverage. Fixtures use synthetic rasters
   (created in `tmp_path`); no external data files required.

5. **Modular and minimal.** No speculative abstractions. Functions are small and
   single-purpose. No placeholder stubs unless the interface contract requires one
   (e.g., AROSICS stub — justified because the full interface must exist before
   the backend is installed).

6. **`ValidationResult` and `CoregistrationResult` are frozen dataclasses.**
   Use `dataclasses.replace()` for updates; never mutate in place.

---

## PIPELINE CONFIG THRESHOLDS

```yaml
coreg_rmse_threshold: 0.5       # pixels — hard stop above this
min_oa_threshold: 0.80          # block export if OA below this
min_minority_f1: 0.70           # block export if minority F1 below this
drift_alert_pct: 20             # % class area change vs previous run
min_mapping_unit_ha: 0.5        # sieve filter default
default_k_folds: 5
smote_auto_threshold_pct: 10    # auto-suggest SMOTE below this class %
max_date_gap_days: 30
min_overlap_pct: 80.0
max_resolution_ratio: 2.0
phenology_month_gap: 2
```

---

## INSTALLED PACKAGES (terraforge env)

**conda-forge (installed):** gdal, rasterio, fiona, shapely, pyproj, geopandas,
numpy, pandas, pyyaml, folium  
**pip (installed):** streamlit 1.56.0, pytest 9.0.3, pytest-cov 7.1.0  
**Not yet installed:** arosics, scipy, scikit-image, scikit-learn, xgboost,
imbalanced-learn, whitebox, rio-cogeo, pystac, reportlab, plotly

---

## NEXT PHASE TO IMPLEMENT

**Phase 4 — Feature Engineering**

New file: `pipeline/features.py`  
New test: `tests/test_features.py`  
New conda dep required: `scikit-image` (for GLCM)

Features to compute (all windowed, band arithmetic on pre-loaded tiles):

| Feature | Input bands | Formula |
|---------|------------|---------|
| NDVI | NIR, Red | `(NIR - Red) / (NIR + Red)` |
| NDWI | Green, NIR | `(Green - NIR) / (Green + NIR)` |
| BSI | SWIR, Red, NIR, Blue | `((SWIR+Red)-(NIR+Blue)) / ((SWIR+Red)+(NIR+Blue))` |
| NDRE | RedEdge, Red | `(RedEdge - Red) / (RedEdge + Red)` |
| VARI | Green, Red, Blue | `(Green - Red) / (Green + Red - Blue)` |
| SAR ratio | VV, VH | `VV / VH` (log scale optional) |
| GLCM contrast | any band | integral-image GLCM per tile |
| GLCM homogeneity | any band | integral-image GLCM per tile |
| GLCM entropy | any band | integral-image GLCM per tile |
| Slope | DEM | gradient magnitude in CRS units |
| Aspect | DEM | gradient direction in degrees |

Output: multi-band GeoTIFF feature stack written via `raster_io.write_raster()`  
Additional outputs: per-feature correlation matrix (flag r > 0.95), feature names list

Band index mapping must be configurable (different sensors have different band orders).
Use a `BandMap` dataclass or dict: `{"nir": 4, "red": 3, "green": 2, "blue": 1, ...}`

GLCM should use a pure-numpy integral-image approach consistent with `preprocess.py`
(avoid scikit-image if the integral-image variant is feasible; use scikit-image as
fallback for correctness).

---

## BEHAVIOUR RULES FOR THIS SESSION

- Read `docs/project_state.md` first for full detail on any completed phase.
- Always inspect existing files before modifying them.
- Propose the implementation plan for Phase 4 and wait for approval before writing code.
- Work one phase at a time; confirm completion before starting the next.
- Run `python3 -m pytest tests/ -v` after each phase and fix all failures before proceeding.
- Keep responses concise and high-signal — no padding.

---
