"""
Phase 2B — Validation gates.

Every check returns a ValidationResult (or list thereof).
The run_all_validations() orchestrator collects all results into a flat dict.
Callers use has_critical_failures() to decide whether to block pipeline progression.

Severity levels
---------------
"ok"    — check passed
"warn"  — soft failure; user is warned but may proceed
"error" — hard failure; is_critical=True blocks pipeline progression

Critical checks (hard-block on failure)
----------------------------------------
- crs_match
- minimum_overlap
- shapefile_alignment (CRS missing, overlap < 50%, < 2 classes)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from pipeline.raster_io import compute_overlap_pct

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    check:       str
    status:      str              # "ok" | "warn" | "error"
    message:     str
    is_critical: bool = False
    detail:      Optional[dict]  = field(default=None, compare=False)

    @property
    def passed(self) -> bool:
        return self.status == "ok"

    @property
    def blocks_pipeline(self) -> bool:
        return self.is_critical and self.status == "error"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crs_key(crs) -> Optional[str]:
    """Return a stable string key for a CRS object; None if CRS is missing."""
    if crs is None:
        return None
    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        pass
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()[:80]


# ── Individual checks ─────────────────────────────────────────────────────────

def check_crs_match(layers: dict[str, dict]) -> ValidationResult:
    """All layers must share the same CRS."""
    check = "crs_match"
    if len(layers) < 2:
        return ValidationResult(check, "ok", "Single layer — no CRS comparison needed.")

    crs_map = {name: _crs_key(layer["meta"].get("crs")) for name, layer in layers.items()}
    missing  = [k for k, v in crs_map.items() if v is None]
    unique   = {v for v in crs_map.values() if v is not None}

    if missing:
        return ValidationResult(
            check, "error",
            f"Missing CRS on layer(s): {', '.join(missing)}. Assign a CRS before uploading.",
            is_critical=True,
            detail={"crs_map": crs_map},
        )
    if len(unique) > 1:
        return ValidationResult(
            check, "error",
            f"CRS mismatch across layers: {crs_map}. Reproject all layers to a common CRS.",
            is_critical=True,
            detail={"crs_map": crs_map},
        )
    return ValidationResult(
        check, "ok",
        f"All layers share {list(unique)[0]}.",
        detail={"crs_map": crs_map},
    )


def check_resolution_compatibility(
    layers: dict[str, dict],
    max_ratio: float = 2.0,
) -> ValidationResult:
    """Warn if any two layers differ in GSD by more than max_ratio."""
    check = "resolution_compatibility"
    if len(layers) < 2:
        return ValidationResult(check, "ok", "Single layer — no resolution comparison needed.")

    res_map = {
        name: abs(float(layer["meta"]["res"][0]))
        for name, layer in layers.items()
    }
    vals = list(res_map.values())
    min_r, max_r = min(vals), max(vals)
    ratio = max_r / min_r if min_r > 0 else float("inf")

    if ratio > max_ratio:
        return ValidationResult(
            check, "warn",
            (
                f"Resolution ratio {ratio:.1f}× exceeds threshold ({max_ratio}×). "
                f"Coregistration accuracy may be reduced. Values: {res_map}"
            ),
            detail={"res_map": res_map, "ratio": round(ratio, 3)},
        )
    return ValidationResult(
        check, "ok",
        f"Resolution ratio {ratio:.1f}× is within threshold ({max_ratio}×).",
        detail={"res_map": res_map, "ratio": round(ratio, 3)},
    )


def check_date_proximity(
    layers: dict[str, dict],
    max_gap_days: int = 30,
    phenology_month_gap: int = 2,
) -> list[ValidationResult]:
    """
    Return up to two ValidationResults:
      1. date_proximity  — error/warn on acquisition date gap
      2. phenology_risk  — warn if layers span > phenology_month_gap months
    """
    results: list[ValidationResult] = []
    dates: dict[str, date] = {
        name: layer["date"]
        for name, layer in layers.items()
        if layer.get("date") is not None
    }

    # Not enough parseable dates — soft warning
    if len(dates) < 2:
        results.append(ValidationResult(
            "date_proximity", "warn",
            "Acquisition date could not be parsed for all layers. "
            "Verify temporal alignment manually.",
        ))
        return results

    date_vals = list(dates.values())
    max_gap = max(
        abs((a - b).days)
        for i, a in enumerate(date_vals)
        for b in date_vals[i + 1:]
    )

    if max_gap > max_gap_days:
        results.append(ValidationResult(
            "date_proximity", "warn",
            f"Maximum acquisition date gap: {max_gap} days (threshold: {max_gap_days} days). "
            "Large temporal gaps may introduce change artefacts.",
            detail={"dates": {k: str(v) for k, v in dates.items()}, "max_gap_days": max_gap},
        ))
    else:
        results.append(ValidationResult(
            "date_proximity", "ok",
            f"All layers acquired within {max_gap} days of each other.",
            detail={"max_gap_days": max_gap},
        ))

    # Phenological warning
    months = [d.month for d in dates.values()]
    month_span = max(months) - min(months)
    if month_span > phenology_month_gap:
        results.append(ValidationResult(
            "phenology_risk", "warn",
            f"Seasonal variation risk — layers span {month_span} calendar months. "
            "Verify NDVI and vegetation features for phenological bias.",
            detail={"months": months, "span": month_span},
        ))

    return results


def check_minimum_overlap(
    layers: dict[str, dict],
    min_pct: float = 80.0,
) -> ValidationResult:
    """
    Verify pairwise spatial overlap between all layers.
    Overlap is measured as a percentage of the smaller layer's bounding area.

    NOTE: Bounds are assumed to be in the same CRS.  Run check_crs_match first.
    """
    check = "minimum_overlap"
    if len(layers) < 2:
        return ValidationResult(check, "ok", "Single layer — no overlap check needed.")

    names       = list(layers.keys())
    bounds_list = [layers[n]["meta"]["bounds"] for n in names]

    pair_results: list[tuple[str, str, float]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pct = compute_overlap_pct(bounds_list[i], bounds_list[j])
            pair_results.append((names[i], names[j], pct))

    min_overlap = min(p for _, _, p in pair_results)
    failing = [(a, b, f"{p:.1f}%") for a, b, p in pair_results if p < min_pct]

    if failing:
        return ValidationResult(
            check, "error",
            f"Insufficient spatial overlap. Pairs below {min_pct}%: {failing}",
            is_critical=True,
            detail={"pairs": [(a, b, round(p, 2)) for a, b, p in pair_results]},
        )
    return ValidationResult(
        check, "ok",
        f"All layer pairs overlap ≥ {min_overlap:.1f}% (threshold: {min_pct}%).",
        detail={"pairs": [(a, b, round(p, 2)) for a, b, p in pair_results]},
    )


def check_band_counts(layers: dict[str, dict]) -> list[ValidationResult]:
    """
    Validate that band counts are consistent with the declared layer type.
    Returns one ValidationResult per layer.
    """
    results: list[ValidationResult] = []
    for name, layer in layers.items():
        count = layer["meta"]["count"]
        ltype = layer["layer_type"]
        check = f"band_count_{name}"

        if ltype == "sar":
            if count not in (1, 2, 4):
                results.append(ValidationResult(
                    check, "warn",
                    f"SAR layer '{name}' has {count} bands. "
                    "Expected 1 (single-pol), 2 (dual-pol), or 4 (quad-pol).",
                ))
            else:
                results.append(ValidationResult(
                    check, "ok",
                    f"'{name}': {count} SAR band(s) — valid.",
                ))
        elif ltype == "dem":
            if count != 1:
                results.append(ValidationResult(
                    check, "warn",
                    f"DEM layer '{name}' has {count} bands; expected 1.",
                ))
            else:
                results.append(ValidationResult(check, "ok", f"'{name}': 1 DEM band — valid."))
        elif ltype == "optical":
            if count < 3:
                results.append(ValidationResult(
                    check, "error",
                    f"Optical layer '{name}' has only {count} band(s). "
                    "Minimum 3 bands required for spectral analysis.",
                    is_critical=False,
                ))
            else:
                results.append(ValidationResult(
                    check, "ok",
                    f"'{name}': {count} optical band(s) — valid.",
                ))
        else:
            results.append(ValidationResult(
                check, "ok",
                f"'{name}': {count} band(s) — type '{ltype}' (no specific band check).",
            ))

    return results


# ── Label alignment gate ──────────────────────────────────────────────────────

def check_label_alignment(
    label_meta: dict,
    feature_meta: dict,
    tolerance_px: float = 0.5,
) -> ValidationResult:
    """
    Verify that a label raster is spatially aligned with the feature stack.

    Three sub-checks are performed in order; the first failure short-circuits:

    1. **CRS match** — both rasters must share the same coordinate reference system.
    2. **Resolution match** — ``|label_res - feat_res| ≤ tolerance_px × feat_res``.
    3. **Origin match** — Euclidean distance between the top-left corners of the
       two rasters must be ≤ ``tolerance_px × feat_res`` (in CRS units).

    A misaligned label causes silently wrong training data: the model trains
    without error but with labels offset from the corresponding feature pixels.

    Parameters
    ----------
    label_meta   : Metadata dict as returned by ``raster_io.get_meta()`` for
                   the label raster.
    feature_meta : Metadata dict as returned by ``raster_io.get_meta()`` for
                   the feature stack raster.
    tolerance_px : Allowed misalignment in pixels (default 0.5).  Applied to
                   both resolution and origin checks.

    Returns
    -------
    ValidationResult with ``is_critical=True`` on any failure.

    Notes
    -----
    This function is intentionally **not** wired into ``run_all_validations()``
    — it is called explicitly from ``page_classification.py`` when the user
    uploads label.tif (Phase 5 UI).
    """
    check = "label_alignment"

    # ── 1. CRS ────────────────────────────────────────────────────────────────
    label_crs_key = _crs_key(label_meta.get("crs"))
    feat_crs_key  = _crs_key(feature_meta.get("crs"))

    if label_crs_key is None or feat_crs_key is None:
        missing = []
        if label_crs_key is None:
            missing.append("label")
        if feat_crs_key is None:
            missing.append("feature stack")
        return ValidationResult(
            check, "error",
            f"CRS is missing on: {', '.join(missing)}. "
            "Assign a CRS before running alignment check.",
            is_critical=True,
        )

    if label_crs_key != feat_crs_key:
        return ValidationResult(
            check, "error",
            f"CRS mismatch: label has {label_crs_key}, "
            f"feature stack has {feat_crs_key}. "
            "Reproject label.tif to match the feature stack CRS before training.",
            is_critical=True,
            detail={"label_crs": label_crs_key, "feature_crs": feat_crs_key},
        )

    # ── 2. Resolution ─────────────────────────────────────────────────────────
    feat_res  = abs(float(feature_meta["res"][0]))
    label_res = abs(float(label_meta["res"][0]))
    res_tol   = tolerance_px * feat_res

    if abs(label_res - feat_res) > res_tol:
        return ValidationResult(
            check, "error",
            f"Resolution mismatch: label {label_res:.4f} vs "
            f"feature stack {feat_res:.4f} CRS units/px "
            f"(tolerance: {tolerance_px:.1f} px = {res_tol:.4f} units). "
            "Resample label.tif to match the feature stack resolution.",
            is_critical=True,
            detail={"label_res": label_res, "feat_res": feat_res,
                    "tolerance_units": round(res_tol, 6)},
        )

    # ── 3. Origin (top-left corner) ───────────────────────────────────────────
    lt = label_meta["transform"]
    ft = feature_meta["transform"]
    dist     = math.sqrt((lt.c - ft.c) ** 2 + (lt.f - ft.f) ** 2)
    max_dist = tolerance_px * feat_res

    if dist > max_dist:
        return ValidationResult(
            check, "error",
            f"Spatial origin mismatch: label ({lt.c:.2f}, {lt.f:.2f}) vs "
            f"feature stack ({ft.c:.2f}, {ft.f:.2f}) — "
            f"offset {dist:.4f} units exceeds "
            f"{tolerance_px:.1f} px = {max_dist:.4f} units. "
            "Align label.tif to the feature stack grid before training.",
            is_critical=True,
            detail={"label_origin": (lt.c, lt.f), "feat_origin": (ft.c, ft.f),
                    "distance": round(dist, 6), "max_dist": round(max_dist, 6)},
        )

    return ValidationResult(
        check, "ok",
        f"Label raster is spatially aligned with the feature stack "
        f"(CRS: {feat_crs_key}, res: {feat_res:.4f}, "
        f"origin offset: {dist:.4f} units).",
        detail={"distance": round(dist, 6), "feat_res": feat_res},
    )


def check_shapefile_alignment(
    shp_meta:     dict,
    feature_meta: dict,
) -> ValidationResult:
    """
    Validate that a shapefile is usable as training data for the feature stack.

    Three sub-checks are performed in order; the first failure short-circuits:

    1. **CRS availability** — both the shapefile and the feature stack must have
       a valid CRS so that reprojection to a common system is possible.
    2. **Bounding-box overlap** — the shapefile extent (in the feature CRS) must
       overlap at least 50 % of the feature raster's extent.
    3. **Class count** — at least 2 unique classes must be present in the
       shapefile's class column.

    ``shp_meta`` is constructed by ``training.build_shapefile_meta`` and contains:

    * ``"crs"``          — original CRS of the shapefile (``rasterio.crs.CRS`` or None)
    * ``"bounds"``       — ``rasterio.coords.BoundingBox`` in the feature CRS
    * ``"class_labels"`` — sorted list of unique integer class labels

    Parameters
    ----------
    shp_meta     : Shapefile metadata dict (from ``training.build_shapefile_meta``).
    feature_meta : Feature stack metadata dict (from ``raster_io.get_meta``).

    Returns
    -------
    ValidationResult with ``is_critical=True`` on any failure.
    """
    check = "shapefile_alignment"

    shp_crs_key  = _crs_key(shp_meta.get("crs"))
    feat_crs_key = _crs_key(feature_meta.get("crs"))

    # ── 1. CRS availability ───────────────────────────────────────────────────
    if shp_crs_key is None:
        return ValidationResult(
            check, "error",
            "Shapefile has no CRS.  Assign a coordinate reference system "
            "before using it for training.",
            is_critical=True,
            detail={"shp_crs": None, "feature_crs": feat_crs_key},
        )
    if feat_crs_key is None:
        return ValidationResult(
            check, "error",
            "Feature stack has no CRS.  Cannot determine reprojection target.",
            is_critical=True,
            detail={"shp_crs": shp_crs_key, "feature_crs": None},
        )

    # ── 2. Bounding-box overlap (shp bounds already in feature CRS) ───────────
    shp_bounds  = shp_meta.get("bounds")
    feat_bounds = feature_meta.get("bounds")

    if shp_bounds is None or feat_bounds is None:
        return ValidationResult(
            check, "error",
            "Cannot compute spatial overlap: missing bounds in metadata.",
            is_critical=True,
        )

    overlap = compute_overlap_pct(shp_bounds, feat_bounds)
    if overlap < 50.0:
        return ValidationResult(
            check, "error",
            f"Shapefile extent overlaps only {overlap:.1f}% of the feature raster "
            f"(minimum: 50%).  Verify that the shapefile covers the study area.",
            is_critical=True,
            detail={"overlap_pct": round(overlap, 2)},
        )

    # ── 3. Class count ────────────────────────────────────────────────────────
    class_labels = shp_meta.get("class_labels", [])
    n_classes    = len(class_labels)
    if n_classes < 2:
        return ValidationResult(
            check, "error",
            f"Only {n_classes} unique class(es) found in the shapefile.  "
            "At least 2 classes are required for supervised classification.",
            is_critical=True,
            detail={"n_classes": n_classes, "class_labels": class_labels},
        )

    return ValidationResult(
        check, "ok",
        f"Shapefile is aligned with the feature stack "
        f"(overlap: {overlap:.1f}%, CRS: {shp_crs_key} → {feat_crs_key}, "
        f"{n_classes} classes: {class_labels}).",
        detail={
            "overlap_pct":  round(overlap, 2),
            "shp_crs":      shp_crs_key,
            "feature_crs":  feat_crs_key,
            "n_classes":    n_classes,
            "class_labels": class_labels,
        },
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_validations(
    layers: dict[str, dict],
    config: dict,
) -> dict[str, ValidationResult]:
    """
    Run all validation checks and return a flat dict keyed by check name.

    Execution order matters: CRS check runs before overlap (overlap result is
    only meaningful when CRS is shared).
    """
    results: dict[str, ValidationResult] = {}

    def _add(r: ValidationResult | list[ValidationResult]) -> None:
        if isinstance(r, list):
            for item in r:
                results[item.check] = item
        else:
            results[r.check] = r

    _add(check_crs_match(layers))
    _add(check_resolution_compatibility(
        layers,
        max_ratio=float(config.get("max_resolution_ratio", 2.0)),
    ))
    _add(check_date_proximity(
        layers,
        max_gap_days=int(config.get("max_date_gap_days", 30)),
        phenology_month_gap=int(config.get("phenology_month_gap", 2)),
    ))
    _add(check_minimum_overlap(
        layers,
        min_pct=float(config.get("min_overlap_pct", 80.0)),
    ))
    _add(check_band_counts(layers))

    return results


def has_critical_failures(results: dict[str, ValidationResult]) -> bool:
    """Return True if any result is both critical and errored."""
    return any(r.blocks_pipeline for r in results.values())


def validation_summary(results: dict[str, ValidationResult]) -> dict[str, int]:
    """Return counts of ok / warn / error results."""
    counts = {"ok": 0, "warn": 0, "error": 0}
    for r in results.values():
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


# =============================================================================
# Drone-specific validation
# =============================================================================

def check_drone_inputs(
    raster_path: "str | Path",
    cfg:         "dict | None" = None,
) -> "list[ValidationResult]":
    """
    Validate a drone raster before feature extraction.

    Checks
    ------
    1. Band count: must be >= 3 (RGB minimum).
    2. Pixel resolution: warns if >= drone_pixel_res_threshold_m (likely satellite).
    3. Band scaling: if pixel_res < threshold and dtype is uint8, warns unless
       drone_rgb_scale_to_float is enabled in cfg.
    4. File size: warns when > 2 GB with estimate for tile processing time.
    5. Survey area: computes covered area in ha and returns it in the message.

    Parameters
    ----------
    raster_path : Path to the raster to validate.
    cfg         : Pipeline config dict.  Loads from yaml if None.

    Returns
    -------
    List of ValidationResult — one per check.  Empty list = nothing to validate
    (rasterio unavailable or file missing).
    """
    import math
    from pathlib import Path as _Path

    try:
        import rasterio
    except ImportError:
        return []

    if cfg is None:
        from pipeline.config_loader import load_config
        cfg = load_config()

    raster_path = _Path(raster_path)
    results: list[ValidationResult] = []

    if not raster_path.exists():
        results.append(ValidationResult(
            check="drone_file_exists",
            status="error",
            message=f"Drone raster not found: {raster_path}",
            blocks_pipeline=True,
        ))
        return results

    try:
        with rasterio.open(raster_path) as ds:
            n_bands    = ds.count
            dtype      = ds.dtypes[0] if n_bands > 0 else "unknown"
            width      = ds.width
            height     = ds.height
            crs        = ds.crs
            transform  = ds.transform
            file_bytes = raster_path.stat().st_size

        # Derive pixel resolution
        if crs and crs.is_projected:
            pixel_res_m = float(abs(transform.a))
        else:
            # Geographic CRS: approximate from degrees -> metres at equator
            pixel_res_m = float(abs(transform.a)) * 111_320.0

        drone_thresh = float(cfg.get("drone_pixel_res_threshold_m", 1.0))

    except Exception as exc:
        results.append(ValidationResult(
            check="drone_open",
            status="error",
            message=f"Cannot open drone raster: {exc}",
            blocks_pipeline=True,
        ))
        return results

    # 1. Band count
    if n_bands < 3:
        results.append(ValidationResult(
            check="drone_band_count",
            status="error",
            message=(
                f"Drone raster has only {n_bands} band(s). "
                "Minimum required is 3 (Red, Green, Blue)."
            ),
            blocks_pipeline=True,
        ))
    else:
        results.append(ValidationResult(
            check="drone_band_count",
            status="ok",
            message=f"Band count OK: {n_bands} band(s).",
            blocks_pipeline=False,
        ))

    # 2. Resolution check
    if pixel_res_m >= drone_thresh:
        results.append(ValidationResult(
            check="drone_resolution",
            status="warn",
            message=(
                f"Pixel resolution {pixel_res_m:.2f} m >= {drone_thresh} m. "
                "Drone mode expects sub-metre imagery. "
                "If this is satellite data, use satellite mode instead."
            ),
            blocks_pipeline=False,
        ))
    else:
        gsd_cm = pixel_res_m * 100.0
        results.append(ValidationResult(
            check="drone_resolution",
            status="ok",
            message=f"Drone GSD: {gsd_cm:.1f} cm/px ({pixel_res_m:.4f} m).",
            blocks_pipeline=False,
        ))

    # 3. Band scaling (uint8 without scale flag)
    if pixel_res_m < drone_thresh and dtype == "uint8":
        scale_flag = bool(cfg.get("drone_rgb_scale_to_float", False))
        if not scale_flag:
            results.append(ValidationResult(
                check="drone_band_scaling",
                status="warn",
                message=(
                    "Bands appear to be uint8 (0-255). "
                    "Spectral indices require float [0, 1]. "
                    "Enable 'drone_rgb_scale_to_float: true' in pipeline_config.yaml "
                    "or bands will not be scaled automatically."
                ),
                blocks_pipeline=False,
            ))
        else:
            results.append(ValidationResult(
                check="drone_band_scaling",
                status="ok",
                message="uint8 bands will be auto-scaled to [0, 1] (drone_rgb_scale_to_float: true).",
                blocks_pipeline=False,
            ))
    else:
        results.append(ValidationResult(
            check="drone_band_scaling",
            status="ok",
            message=f"Band dtype: {dtype} — no scaling required.",
            blocks_pipeline=False,
        ))

    # 4. File size + tiling estimate
    size_gb = file_bytes / (1024 ** 3)
    if size_gb > 2.0:
        block = int(cfg.get("drone_block_size", 256))
        n_tiles = math.ceil(height / block) * math.ceil(width / block)
        est_sec = n_tiles * 0.05   # ~50 ms/tile rough estimate
        results.append(ValidationResult(
            check="drone_file_size",
            status="warn",
            message=(
                f"Large file: {size_gb:.1f} GB ({n_tiles:,} tiles at {block}px). "
                f"Estimated processing time: ~{est_sec/60:.0f} min. "
                "Processing is fully windowed -- memory use stays constant."
            ),
            blocks_pipeline=False,
        ))
    else:
        results.append(ValidationResult(
            check="drone_file_size",
            status="ok",
            message=f"File size: {size_gb*1024:.0f} MB.",
            blocks_pipeline=False,
        ))

    # 5. Survey area
    if pixel_res_m > 0:
        area_ha = (width * height * pixel_res_m ** 2) / 10_000.0
        gsd_cm  = pixel_res_m * 100.0
        results.append(ValidationResult(
            check="drone_survey_area",
            status="ok",
            message=(
                f"Survey area: {area_ha:.2f} ha "
                f"({width} x {height} px at {gsd_cm:.1f} cm/px GSD)."
            ),
            blocks_pipeline=False,
        ))

    return results
