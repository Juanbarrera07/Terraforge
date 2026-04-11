# TerraForge Mining Intelligence — Project State

**Last updated:** 2026-04-10  
**Python:** 3.11.15 (micromamba env `terraforge`)  
**Test suite:** 101 tests · 0 failures · 0.63 s  

---

## 1. What This System Is

A production-ready Streamlit application that automates and quality-controls a
SAR + multispectral mining survey pipeline for geospatial consulting firms.

Core value: replaces manual error-prone steps with automated validation gates,
quality checks, and a full audit trail — every decision is logged with an ISO
timestamp and written to `./logs/{run_id}.json`.

**Entry point:** `streamlit run app.py`

---

## 2. Repository Structure

```
TMI/
├── app.py                          # Streamlit entry point — 6-page sidebar UI
├── config/
│   └── pipeline_config.yaml        # All thresholds (edit here; app reloads on restart)
├── pipeline/
│   ├── __init__.py
│   ├── audit.py                    # ISO-timestamped JSON audit logger
│   ├── config_loader.py            # YAML loader with hard-coded defaults fallback
│   ├── coregister.py               # Coregistration interface + RMSE gate
│   ├── ingest.py                   # Light ingestion + sensor detection + date parsing
│   ├── preprocess.py               # DOS1 atmospheric correction + Lee speckle filter
│   ├── raster_io.py                # Windowed raster I/O (no full-raster reads)
│   ├── session.py                  # Streamlit session state schema + stage unlock logic
│   └── validate.py                 # All validation gates + orchestrator
├── tests/
│   ├── conftest.py                 # Synthetic raster fixtures (make_raster factory)
│   ├── test_coregister.py          # 23 tests
│   ├── test_ingest.py              # 22 tests
│   ├── test_preprocess.py          # 26 tests
│   └── test_validate.py            # 30 tests
├── docs/
│   └── project_state.md            # This file
├── logs/                           # Runtime: {run_id}.json audit logs
├── models/                         # Runtime: {timestamp}_model.pkl
├── outputs/                        # Runtime: COG GeoTIFFs, reports, ZIPs
├── tmp/                            # Runtime: uploaded file staging
├── requirements-conda.txt          # conda-forge geo stack (GDAL, rasterio, etc.)
└── requirements.txt                # pip-only pure-Python packages
```

---

## 3. Installation

```bash
# Step 1 — native geo stack (conda-forge ONLY — never pip for these)
micromamba install -n terraforge -c conda-forge \
  gdal rasterio fiona shapely pyproj geopandas numpy pandas pyyaml --yes

# Step 2 — pure-Python packages
pip install -r requirements.txt   # streamlit, pytest, pytest-cov

# Run app
streamlit run app.py

# Run tests
python3 -m pytest tests/ -v
```

> **Rule:** Never install `gdal`, `rasterio`, `fiona`, `shapely`, `pyproj`, or
> `geopandas` via pip. ABI conflicts cause silent CRS failures and PROJ datum
> grid path errors on WSL.

---

## 4. Configuration

All pipeline thresholds live in `config/pipeline_config.yaml`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `coreg_rmse_threshold` | `0.5` | Hard stop if coregistration RMSE > this (pixels) |
| `min_oa_threshold` | `0.80` | Block export if overall accuracy < this |
| `min_minority_f1` | `0.70` | Block export if minority class F1 < this |
| `drift_alert_pct` | `20` | Flag class area change > this % vs previous run |
| `min_mapping_unit_ha` | `0.5` | Sieve filter default (ha) |
| `default_k_folds` | `5` | Cross-validation folds |
| `smote_auto_threshold_pct` | `10` | Auto-suggest SMOTE if any class < this % of total |
| `max_date_gap_days` | `30` | Warn if acquisition date gap > this (days) |
| `min_overlap_pct` | `80.0` | Error if spatial overlap < this % |
| `max_resolution_ratio` | `2.0` | Warn if GSD ratio between layers > this |
| `phenology_month_gap` | `2` | Warn if layers span > this many calendar months |

---

## 5. Phase Status

### Phase 1 — Scaffold + Config + Session + Audit ✅

**Files:** `pipeline/config_loader.py`, `pipeline/session.py`, `pipeline/audit.py`,
`config/pipeline_config.yaml`, `pipeline/__init__.py`

- YAML config loading with defaults fallback
- Canonical session state schema (`run_id`, `raw_data`, `validation_results`,
  `preprocessed`, `features`, `model`, `classified`, `accuracy`, `audit_log`,
  `pipeline_unlocked`)
- Stage-based pipeline unlock system (`ingestion → preprocessing → features →
  classification → postprocess → export`)
- Audit logger: every gate, decision, and parameter change written to
  `./logs/{run_id}.json` with ISO 8601 UTC timestamp
- 8-char hex run IDs generated via `uuid.uuid4()`

---

### Phase 2A — Data Ingestion (Light) ✅

**Files:** `pipeline/raster_io.py`, `pipeline/ingest.py`

**`raster_io.py` — Core windowed I/O module:**
- `get_meta()` — full metadata without any pixel reads
- `iter_windows()` — generates non-overlapping tile `Window` objects
- `read_window()` — reads a single tile (bands, rows, cols)
- `read_overview()` — decimated thumbnail for preview only
- `compute_overlap_pct()` — spatial overlap from bounding boxes
- `write_raster()` — tiled, deflate-compressed GeoTIFF writer
- Design rule: no function loads a full raster array; all reads are windowed

**`ingest.py` — Light ingestion:**
- Sensor auto-detection: Sentinel-1/2, Landsat 8/9/7, DEM, fallback heuristics
- Acquisition date extraction from TIFF tags and filename patterns (7 formats)
- ZIP handling: extracts `.tif` files from SAR ZIP archives
- Chunked file writes (128 KiB) to avoid loading large files into memory
- `build_metadata_table()` for Streamlit `st.dataframe()` display

---

### Phase 2B — Validation Gates ✅

**File:** `pipeline/validate.py`

| Check | Severity | Blocks pipeline |
|-------|----------|-----------------|
| `crs_match` | error | Yes |
| `resolution_compatibility` | warn | No |
| `date_proximity` | warn | No |
| `phenology_risk` | warn | No |
| `minimum_overlap` | error | Yes |
| `band_count_{layer}` | warn/error | No |

- `ValidationResult` frozen dataclass with `status`, `is_critical`, `blocks_pipeline`
- `run_all_validations()` orchestrator returns flat dict keyed by check name
- `has_critical_failures()` for pipeline gate evaluation
- `validation_summary()` for UI metrics display

---

### Streamlit UI — Page 1 (Data Ingestion) ✅

**File:** `app.py`

- 6-page sidebar navigation; pages 2–6 locked until prerequisites complete
- Page 1 fully functional:
  - Three upload sections (SAR / Optical / DEM) with ZIP support
  - "Ingest & Validate" button triggers Phase 2A + 2B pipeline
  - Validation checklist with ✅/⚠️/❌ status badges per check
  - Metadata table (`st.dataframe`) with CRS, bands, GSD, date, extent
  - Summary metrics (passed / warnings / errors)
  - "Unlock Preprocessing" button (only appears when all critical checks pass)
  - All events written to audit log
- Pages 2–6: locked stubs showing next-phase description

---

### Phase 3 — Preprocessing + Coregistration Interface ✅

**Files:** `pipeline/preprocess.py`, `pipeline/coregister.py`

**`preprocess.py` — DOS1 + Lee filter:**

DOS1 atmospheric correction (optical):
- Two-pass windowed algorithm — zero full-raster loads
- Pass 1: scan all tiles, find per-band minimum non-zero non-nodata pixel
- Pass 2: subtract dark object value, clip to `[0, dtype_max]`
- Nodata pixels preserved unchanged
- Progress callback support for Streamlit progress bars

Lee speckle filter (SAR):
- Integral-image local statistics — O(n) memory, independent of kernel size
- Overlap-padded tile processing — eliminates edge artifacts between tiles
- Noise variance model: `noise_var = (band_mean / ENL)²`
- Lee weight: `w = local_var / (local_var + noise_var)`
- Filtered: `local_mean + w * (pixel - local_mean)`
- `kernel_size` must be odd ≥ 3; validated at entry
- Nodata pixels restored after filtering

**`coregister.py` — AROSICS stub interface:**

`CoregistrationResult` frozen dataclass fields:
- `shift_x_px`, `shift_y_px`, `shift_magnitude` — shift vector
- `rmse` — registration error in pixels
- `gate_passed` — set by `apply_rmse_gate()`
- `corrected_path` — output path (set to `None` when gate fails)
- `is_stub` — `True` until real AROSICS is activated
- `shift_map` — optional 2-D shift heatmap array (excluded from equality)

Gate logic:
- `apply_rmse_gate(result, threshold)` — returns new frozen result; `corrected_path`
  is nulled on gate failure
- `_arosics_available()` — runtime detection; activates real backend automatically
- `_run_arosics_coreg()` is fully implemented and ready; blocked only by AROSICS install
- `get_shift_report()` returns JSON-serialisable dict for audit log and UI display

---

## 6. Test Suite

```
tests/test_ingest.py      22 tests   Sensor detection, date parsing, ZIP, metadata table
tests/test_validate.py    30 tests   All gate checks, CRS, overlap, resolution, dates, band counts
tests/test_preprocess.py  26 tests   Integral-image stats, DOS1 two-pass, Lee filter, callbacks
tests/test_coregister.py  23 tests   Gate logic, result immutability, report serialisation
─────────────────────────────────────────────────────────────────────────────
Total                    101 tests   0 failures · 0.63 s
```

All tests use synthetic rasters generated in `tmp_path` fixtures — no external
data files required. Tests are isolated from Streamlit (no `st.*` imports in
pipeline modules).

---

## 7. Quality Gates (implemented)

| Gate | Location | Threshold | Action on failure |
|------|----------|-----------|-------------------|
| CRS match | `validate.py` | N/A | Block pipeline unlock |
| Spatial overlap | `validate.py` | `min_overlap_pct` (80%) | Block pipeline unlock |
| Coregistration RMSE | `coregister.py` | `coreg_rmse_threshold` (0.5 px) | Null `corrected_path`; require user confirmation in UI |

**Implemented but pending UI wiring (Phases 5–6):**
- OA threshold gate (`min_oa_threshold` = 0.80)
- Minority class F1 gate (`min_minority_f1` = 0.70)
- Drift monitor gate (`drift_alert_pct` = 20%)

---

## 8. Pending Phases

### Phase 4 — Feature Engineering
**Modules:** `pipeline/features.py`  
**Conda deps needed:** `scikit-image` (GLCM texture)

Planned features:
- Spectral indices: NDVI, NDWI, BSI, NDRE, VARI (windowed, band arithmetic)
- SAR backscatter ratio: VV/VH
- GLCM texture per-tile: contrast, homogeneity, entropy (integral-image variant)
- DEM derivatives: slope, aspect (windowed gradient)
- Feature stack output: multi-band GeoTIFF
- Correlation matrix for redundancy detection (r > 0.95 flagged)

### Phase 5 — ML Classification
**Modules:** `pipeline/classify.py`  
**Conda deps needed:** `scikit-learn`, `xgboost`  
**Pip deps needed:** `imbalanced-learn`

- Random Forest / XGBoost / Ensemble
- Class imbalance auto-detection + SMOTE (threshold: `smote_auto_threshold_pct`)
- k-fold cross-validation with configurable k
- OA and Kappa coefficient
- Quality gate: OA < 0.80 or minority F1 < 0.70 → block export
- Model serialisation: `./models/{timestamp}_model.pkl`

### Phase 6 — Post-processing + Validation
**Modules:** `pipeline/postprocess.py`, `pipeline/validate.py` (extensions)  
**Pip deps needed:** `whitebox`

- Sieve filter (MMU in ha → pixel count conversion)
- Morphological closing
- Accuracy assessment with reference points (CSV lat/lon/class)
- Drift monitor: compare class area % with `previous_class_areas` in session state

### Phase 7 — Export System
**Modules:** `pipeline/export.py`  
**Pip deps needed:** `rio-cogeo`, `pystac`, `reportlab`

- Cloud-Optimised GeoTIFF output via `rio-cogeo`
- STAC item JSON generation
- PDF accuracy report (auto-filled with run metadata)
- ZIP bundle with all outputs + full audit log
- Filename convention: `{run_id}_{stage}_{YYYYMMDD}.tif`

### Phase 8 — Streamlit UI (Pages 2–6)
Wire all pipeline modules to the remaining 5 pages:
- Page 2: preprocessing controls + coregistration report + RMSE gate UI
- Page 3: feature selection, correlation matrix, importance chart
- Page 4: model training, confusion matrix, accuracy metrics
- Page 5: classified map, accuracy assessment, drift monitor
- Page 6: export package builder + ZIP download

### Phase 9 — Testing (Phases 4–8)
- `test_features.py`, `test_classify.py`, `test_postprocess.py`, `test_export.py`
- pytest-cov report targeting ≥ 80% coverage on pipeline modules

### Phase 10 — Performance Optimisation
- Profile memory usage on 1 GB+ rasters
- Add `@st.cache_data` / `@st.cache_resource` decorators at UI boundaries
- Batch mode: accept folder of scenes, queue processing, status table

---

## 9. Known Assumptions & Constraints

- All input rasters must be in the same projected CRS before upload (no auto-reprojection)
- SAR preprocessing (pyroSAR/snappy) is deferred — raw SAR GeoTIFFs are assumed as input
- AROSICS coregistration is stubbed; real results require `micromamba install arosics -c conda-forge`
- Large file uploads are memory-buffered by Streamlit before reaching `ingest_file()`;
  practical limit ~500 MB per file (configurable via `upload_size_limit_mb`)
- Lee filter `kernel_size` should stay ≤ 11 for typical block sizes; larger kernels
  require increasing `block_size` to maintain valid overlap padding

---

## 10. Audit Log Format

Every significant event is written to `./logs/{run_id}.json`:

```json
[
  {
    "timestamp": "2026-04-10T14:32:01.123456+00:00",
    "run_id": "A3F7C91B",
    "event_type": "run_start",
    "details": { "action": "new_run_via_button" }
  },
  {
    "timestamp": "2026-04-10T14:32:15.987654+00:00",
    "run_id": "A3F7C91B",
    "event_type": "gate",
    "details": {
      "gate": "ingestion_validation",
      "summary": { "ok": 4, "warn": 1, "error": 0 },
      "checks": { "crs_match": "ok", "minimum_overlap": "ok", "..." : "..." }
    },
    "decision": "proceed"
  }
]
```

Event types: `run_start` · `ingestion` · `gate` · `decision` · `param_change` · `error`  
Gate decisions: `proceed` · `block` · `override`
