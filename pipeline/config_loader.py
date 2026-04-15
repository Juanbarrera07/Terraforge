"""
Pipeline configuration loader.

Loads pipeline_config.yaml and merges with hard-coded defaults so the app
always has safe values even if the YAML is missing or incomplete.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"

_DEFAULTS: dict[str, Any] = {
    # Coregistration
    "coreg_rmse_threshold": 0.5,
    # ML gates
    "min_oa_threshold": 0.80,
    "min_minority_f1": 0.70,
    # Drift monitor
    "drift_alert_pct": 20,
    # Post-processing
    "min_mapping_unit_ha": 0.5,
    # Drone mode overrides
    "drone_pixel_res_threshold_m": 1.0,
    "drone_median_kernel_max": 7,
    "drone_morpho_iterations": 2,
    "drone_sieve_min_px": 100,
    # Drone adaptive smoothing (used by drone_adaptive_params)
    "drone_smooth_target_fine_m": 0.3,
    "drone_smooth_target_medium_m": 1.5,
    "drone_smooth_target_coarse_m": 3.0,
    "drone_morpho_target_m": 0.75,
    "drone_sieve_target_m2": 1.0,
    "drone_max_median_kernel": 31,
    "drone_max_morpho_kernel": 11,
    # Training
    "default_k_folds": 5,
    "smote_auto_threshold_pct": 10,
    # Ingestion validation
    "max_date_gap_days": 30,
    "min_overlap_pct": 80.0,
    "max_resolution_ratio": 2.0,
    "phenology_month_gap": 2,
    # Feature engineering
    "corr_flag_threshold": 0.95,
    # SAR preprocessing
    "sar_enl_default": 1.0,
    "sar_enl_sentinel1_iw": 4.9,
    # File handling
    "upload_size_limit_mb": 500,
    "tmp_dir": "tmp",
}


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """
    Load config from YAML and merge over defaults.

    Missing keys fall back to _DEFAULTS, so callers can always rely on a
    complete config dict without guarding individual keys.
    """
    loaded: dict[str, Any] = {}
    if path.exists():
        with open(path) as fh:
            loaded = yaml.safe_load(fh) or {}

    return {**_DEFAULTS, **loaded}
