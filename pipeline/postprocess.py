"""
Phase 6 — Post-processing and validation.

Provides spatial filtering (GDAL sieve, morphological close), class area
statistics, accuracy assessment against ground-truth points, and temporal
drift detection between classification runs.

Public API
----------
ClassAreaResult  — per-class pixel counts and areas in hectares
DriftResult      — per-class area change vs previous run
AccuracyResult   — accuracy metrics + point-count bookkeeping
QualityGateResult           — single 3-tier gate result (pass/warning/fail)
compute_class_areas         — windowed per-class pixel tally
sieve_filter                — MMU enforcement via GDAL SieveFilter (see warning)
morphological_close         — windowed morphological closing via overlap-padded tiles
assess_accuracy_from_points — accuracy from a reference CSV with lat/lon
check_drift                 — class-area drift between two runs
confidence_filter           — replace low-confidence pixels with local median (windowed)
median_smooth               — windowed median filter via scipy.ndimage
run_postprocess_chain       — 4-step chain: confidence_filter→median_smooth→morpho→sieve
run_quality_gates           — evaluate OA/Kappa/F1/confidence/nodata against 3-tier gates
has_gate_failures           — True if any gate result has status "fail"

Design rules
------------
- No Streamlit imports.
- All raster reads through iter_windows except where documented otherwise.
- sieve_filter is a deliberate exception: GDAL SieveFilter loads the full
  raster band into GDAL-managed memory.  See that function's docstring.
- morphological_close and confidence_filter use overlap-padded windowed reads
  to avoid full-raster allocation in Python.
- AccuracyResult records total/valid/discarded point counts with reason codes.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import rowcol

try:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    _GDAL_AVAILABLE = True
except ImportError:          # pragma: no cover
    _GDAL_AVAILABLE = False

try:
    from pyproj import Transformer
    _PYPROJ_AVAILABLE = True
except ImportError:          # pragma: no cover
    _PYPROJ_AVAILABLE = False

from pipeline import audit as _audit
from pipeline.config_loader import load_config
from pipeline.raster_io import DEFAULT_BLOCK, iter_windows


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassAreaResult:
    """
    Per-class pixel counts and area in hectares.

    Attributes
    ----------
    class_ids        : sorted list of class integer labels found in the raster
    pixel_counts     : {class_id: pixel_count}
    areas_ha         : {class_id: area in hectares}
    total_area_ha    : sum of all class areas (excluding nodata)
    pixel_res_m      : ground-sampling distance used for area conversion
    """
    class_ids:     list[int]
    pixel_counts:  dict[int, int]
    areas_ha:      dict[int, float]
    total_area_ha: float
    pixel_res_m:   float


@dataclass(frozen=True)
class DriftResult:
    """
    Class-area drift between a current and previous classification run.

    Attributes
    ----------
    pct_change      : {class_id: signed % change in area vs previous run}
    flagged_classes : class ids whose |pct_change| >= drift_alert_pct
    drift_alert_pct : threshold used
    """
    pct_change:      dict[int, float]
    flagged_classes: list[int]
    drift_alert_pct: float


@dataclass(frozen=True)
class AccuracyResult:
    """
    Accuracy metrics from reference ground-truth points.

    Attributes
    ----------
    oa               : overall accuracy [0, 1]
    kappa            : Cohen's kappa
    per_class_metrics: {class_id: {precision, recall, f1, support}}
    confusion_matrix : np.ndarray, shape (n_classes, n_classes); excluded from ==
    class_labels     : ordered class labels matching confusion_matrix rows/cols
    n_points         : total reference points read from CSV
    n_valid          : points successfully matched to a classified pixel
    n_discarded      : n_points - n_valid (points skipped for any reason)
    discard_reasons  : {reason_code: count} — keys defined below

    Discard reason codes
    --------------------
    "out_of_bounds"  : reprojected coordinate falls outside raster extent
    "nodata_pixel"   : raster pixel at that location is nodata
    "reproject_error": pyproj raised an exception for this point
    "missing_fields" : row was missing lat/lon/class column(s)
    """
    oa:               float
    kappa:            float
    per_class_metrics: dict[int, dict[str, float]]
    confusion_matrix: np.ndarray = field(compare=False)
    class_labels:     list[int]
    n_points:         int
    n_valid:          int
    n_discarded:      int
    discard_reasons:  dict[str, int]


# ── Public functions ──────────────────────────────────────────────────────────

def compute_class_areas(
    classified_path: str | Path,
    nodata:          int           = -1,
    pixel_res_m:     Optional[float] = None,
) -> ClassAreaResult:
    """
    Tally per-class pixel counts and compute areas in hectares.

    Reads the raster band-1 tile by tile; accumulates counts without loading
    the full array.

    Parameters
    ----------
    classified_path : int16/int32 classified raster (band 1).
    nodata          : pixel value to exclude from counts.
    pixel_res_m     : ground-sampling distance in metres.  If None, the
                      function reads it from the raster CRS/transform (only
                      valid for projected CRS).  Raises ValueError if it
                      cannot be determined.
    """
    classified_path = Path(classified_path)
    counts: dict[int, int] = {}

    with rasterio.open(classified_path) as ds:
        if pixel_res_m is None:
            if ds.crs and ds.crs.is_projected:
                pixel_res_m = float(abs(ds.transform.a))
            else:
                raise ValueError(
                    "pixel_res_m must be supplied explicitly for geographic CRS."
                )
        for win in iter_windows(ds):
            tile = ds.read(1, window=win)
            mask = tile != nodata
            vals, cnts = np.unique(tile[mask], return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                counts[v] = counts.get(v, 0) + c

    px_area_ha = (pixel_res_m ** 2) / 10_000.0
    areas_ha   = {k: v * px_area_ha for k, v in counts.items()}
    total      = sum(areas_ha.values())

    return ClassAreaResult(
        class_ids     = sorted(counts.keys()),
        pixel_counts  = counts,
        areas_ha      = areas_ha,
        total_area_ha = total,
        pixel_res_m   = pixel_res_m,
    )


def sieve_filter(
    classified_path: str | Path,
    out_path:        str | Path,
    mmu_ha:          float,
    pixel_res_m:     float,
    connectivity:    int = 4,
) -> tuple[Path, int]:
    """
    Remove patches smaller than the minimum mapping unit (MMU).

    .. warning::
        **This function is a deliberate exception to the windowed-I/O rule.**
        GDAL's SieveFilter operates on a full raster band loaded into
        GDAL-managed memory.  For very large rasters (> a few GB) without a
        COG/tile workflow, memory usage may be prohibitive.  The caller is
        responsible for ensuring the raster fits within available RAM or for
        splitting the raster into manageable tiles before calling this function.

    Parameters
    ----------
    classified_path : source int16 classified raster.
    out_path        : destination path for the filtered raster.
    mmu_ha          : minimum mapping unit in hectares; patches smaller than
                      this will be absorbed by adjacent classes.
    pixel_res_m     : ground-sampling distance in metres used to convert
                      mmu_ha to a pixel-count threshold.
    connectivity    : 4 (cardinal) or 8 (diagonal) pixel connectivity.

    Returns
    -------
    (out_path, threshold_pixels)  where threshold_pixels is the MMU expressed
    as a pixel count (the value passed to GDAL).
    """
    if not _GDAL_AVAILABLE:
        raise RuntimeError("GDAL (osgeo) is required for sieve_filter.")

    classified_path = Path(classified_path)
    out_path        = Path(out_path)

    px_area_ha        = (pixel_res_m ** 2) / 10_000.0
    threshold_pixels  = max(1, int(math.ceil(mmu_ha / px_area_ha)))

    # Copy source to destination first so SieveFilter edits dst in place.
    driver = gdal.GetDriverByName("GTiff")
    src_ds = gdal.Open(str(classified_path), gdal.GA_ReadOnly)
    dst_ds = driver.CreateCopy(str(out_path), src_ds, strict=0,
                               options=["COMPRESS=DEFLATE", "TILED=YES",
                                        "BLOCKXSIZE=512", "BLOCKYSIZE=512",
                                        "PREDICTOR=2"])
    src_ds = None  # close source

    src_band = dst_ds.GetRasterBand(1)
    # SieveFilter modifies src_band in place; mask_band=None → no explicit mask
    gdal.SieveFilter(
        src_band,
        None,           # mask band (None = no mask)
        src_band,       # destination band (same → in-place)
        threshold_pixels,
        connectivity,
    )
    src_band.FlushCache()
    dst_ds.FlushCache()
    dst_ds = None  # close + flush

    return out_path, threshold_pixels


def morphological_close(
    classified_path: str | Path,
    out_path:        str | Path,
    kernel_size:     int = 3,
    block_size:      int = DEFAULT_BLOCK,
) -> Path:
    """
    Apply morphological closing (dilate then erode by majority fill) to a
    classified raster using windowed, overlap-padded reads.

    Each window is expanded by ``kernel_size // 2`` pixels on all sides so
    that neighbourhood lookups at tile edges use real data rather than zeros.
    No full-raster array is ever allocated in Python.

    Parameters
    ----------
    classified_path : source classified raster (single band, integer dtype).
    out_path        : destination path for the closed raster.
    kernel_size     : side length of the square structuring element.
                      **Must be a positive odd integer ≥ 3.**

    Raises
    ------
    ValueError
        If kernel_size is not a positive odd integer.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer >= 3, got {kernel_size!r}."
        )

    classified_path = Path(classified_path)
    out_path        = Path(out_path)
    pad             = kernel_size // 2

    with rasterio.open(classified_path) as src:
        nodata_val = src.nodata if src.nodata is not None else -1
        profile    = src.profile.copy()
        profile.update(
            driver="GTiff",
            compress="deflate",
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            predictor=2,
        )
        height, width = src.height, src.width

        with rasterio.open(out_path, "w", **profile) as dst:
            for win in iter_windows(src, block_size):
                col_off = win.col_off
                row_off = win.row_off
                win_w   = win.width
                win_h   = win.height

                # Padded window — clipped to raster bounds
                p_col_off = max(0, col_off - pad)
                p_row_off = max(0, row_off - pad)
                p_col_end = min(width,  col_off + win_w + pad)
                p_row_end = min(height, row_off + win_h + pad)

                p_win = rasterio.windows.Window(
                    col_off = p_col_off,
                    row_off = p_row_off,
                    width   = p_col_end - p_col_off,
                    height  = p_row_end - p_row_off,
                )

                padded_tile = src.read(1, window=p_win).astype(np.int32)

                # ── Morphological close = dilate then erode ───────────────
                closed = _majority_filter(padded_tile, kernel_size, int(nodata_val))
                closed = _majority_filter(closed,      kernel_size, int(nodata_val))

                # Trim back to the original (unpadded) window
                r_start = row_off - p_row_off
                c_start = col_off - p_col_off
                result  = closed[r_start:r_start + win_h, c_start:c_start + win_w]

                dst.write(result.astype(profile["dtype"]), 1, window=win)

    return out_path


def assess_accuracy_from_points(
    classified_path: str | Path,
    ref_csv_path:    str | Path,
    lat_col:         str  = "lat",
    lon_col:         str  = "lon",
    class_col:       str  = "class",
    nodata:          int  = -1,
) -> AccuracyResult:
    """
    Compute accuracy metrics by sampling the classified raster at reference
    ground-truth point locations.

    Points are read from a CSV that must contain decimal-degree WGS-84
    latitude/longitude columns and an integer class column.  Coordinates are
    reprojected to the raster CRS via pyproj before sampling.

    Point bookkeeping
    -----------------
    Every input row is counted.  Points that cannot be matched to a valid
    classified pixel are discarded and counted by reason:

    ``out_of_bounds``   — reprojected coord outside the raster extent
    ``nodata_pixel``    — pixel at that location equals nodata
    ``reproject_error`` — pyproj raised an exception for this row
    ``missing_fields``  — row is missing one or more required columns

    Parameters
    ----------
    classified_path : classified int16/int32 raster (band 1).
    ref_csv_path    : path to reference CSV file.
    lat_col         : column name for latitude (WGS-84 decimal degrees).
    lon_col         : column name for longitude (WGS-84 decimal degrees).
    class_col       : column name for integer reference class.
    nodata          : pixel value treated as nodata (default -1).

    Returns
    -------
    AccuracyResult with oa, kappa, per_class_metrics, confusion_matrix,
    n_points, n_valid, n_discarded, and discard_reasons.

    Raises
    ------
    ImportError  if pyproj is not installed.
    ValueError   if fewer than 2 valid points are found.
    """
    if not _PYPROJ_AVAILABLE:
        raise ImportError("pyproj is required for assess_accuracy_from_points.")

    classified_path = Path(classified_path)
    ref_csv_path    = Path(ref_csv_path)

    discard: dict[str, int] = {
        "out_of_bounds":   0,
        "nodata_pixel":    0,
        "reproject_error": 0,
        "missing_fields":  0,
    }

    y_true: list[int] = []
    y_pred: list[int] = []
    n_points = 0

    with rasterio.open(classified_path) as ds:
        raster_crs  = ds.crs
        transform   = ds.transform
        raster_h    = ds.height
        raster_w    = ds.width

        # Build reprojector from WGS-84 → raster CRS
        if raster_crs is not None:
            transformer = Transformer.from_crs(
                "EPSG:4326", raster_crs.to_epsg() or raster_crs,
                always_xy=True,
            )
        else:
            transformer = None  # assume raster is already lat/lon

        with open(ref_csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                n_points += 1

                # ── 1. Check required fields ──────────────────────────────
                if lat_col not in row or lon_col not in row or class_col not in row:
                    discard["missing_fields"] += 1
                    continue
                try:
                    lat   = float(row[lat_col])
                    lon   = float(row[lon_col])
                    label = int(row[class_col])
                except (ValueError, TypeError):
                    discard["missing_fields"] += 1
                    continue

                # ── 2. Reproject to raster CRS ────────────────────────────
                try:
                    if transformer is not None:
                        x, y = transformer.transform(lon, lat)
                    else:
                        x, y = lon, lat
                except Exception:
                    discard["reproject_error"] += 1
                    continue

                # ── 3. Convert to pixel row/col ───────────────────────────
                try:
                    r, c = rowcol(transform, x, y)
                except Exception:
                    discard["out_of_bounds"] += 1
                    continue

                if not (0 <= r < raster_h and 0 <= c < raster_w):
                    discard["out_of_bounds"] += 1
                    continue

                # ── 4. Sample raster ──────────────────────────────────────
                win   = rasterio.windows.Window(c, r, 1, 1)
                pixel = int(ds.read(1, window=win)[0, 0])

                if pixel == nodata:
                    discard["nodata_pixel"] += 1
                    continue

                y_true.append(label)
                y_pred.append(pixel)

    n_valid     = len(y_true)
    n_discarded = n_points - n_valid

    if n_valid < 2:
        raise ValueError(
            f"Only {n_valid} valid point(s) found after discarding "
            f"{n_discarded} points — need at least 2 to compute accuracy."
        )

    return _compute_accuracy_metrics(
        np.array(y_true, dtype=np.int32),
        np.array(y_pred, dtype=np.int32),
        n_points    = n_points,
        n_valid     = n_valid,
        n_discarded = n_discarded,
        discard_reasons = discard,
    )


def check_drift(
    current_areas:  ClassAreaResult,
    previous_areas: ClassAreaResult,
    drift_alert_pct: Optional[float] = None,
    cfg:             Optional[dict]  = None,
) -> DriftResult:
    """
    Compare class areas between two classification runs and flag anomalies.

    Only classes present in *both* runs are compared.  Classes that appear
    or disappear entirely are not included in ``pct_change``; the caller
    should treat such cases as a data quality flag separately.

    Parameters
    ----------
    current_areas   : ClassAreaResult from the latest run.
    previous_areas  : ClassAreaResult from the reference run.
    drift_alert_pct : override threshold (%).  If None, read from cfg.
    cfg             : config dict (or None to load from file).

    Returns
    -------
    DriftResult with per-class signed % change and flagged class ids.
    """
    if drift_alert_pct is None:
        if cfg is None:
            cfg = load_config()
        drift_alert_pct = float(cfg.get("drift_alert_pct", 20))

    pct_change: dict[int, float] = {}
    for cls_id in current_areas.class_ids:
        prev_ha = previous_areas.areas_ha.get(cls_id)
        if prev_ha is None:
            continue
        curr_ha = current_areas.areas_ha[cls_id]
        if prev_ha == 0.0:
            # Avoid division by zero: treat emergence from zero as +inf → always flag
            pct_change[cls_id] = float("inf") if curr_ha > 0 else 0.0
        else:
            pct_change[cls_id] = (curr_ha - prev_ha) / prev_ha * 100.0

    flagged = [
        cls_id for cls_id, pct in pct_change.items()
        if abs(pct) >= drift_alert_pct
    ]

    return DriftResult(
        pct_change      = pct_change,
        flagged_classes = sorted(flagged),
        drift_alert_pct = drift_alert_pct,
    )


def confidence_filter(
    classified_path: str | Path,
    confidence_path: str | Path,
    threshold:       float,
    out_path:        str | Path,
    block_size:      int = DEFAULT_BLOCK,
) -> Path:
    """
    Replace low-confidence pixels with the median of their classified neighbourhood.

    Pixels where confidence < ``threshold`` (and confidence is not nodata) are
    replaced with the result of a 5×5 scipy median filter applied to the
    classified raster.  Processing uses overlap-padded windowed reads so
    tile-boundary pixels receive correct neighbourhood values.  Nodata pixels
    in the classified raster are never modified.

    Parameters
    ----------
    classified_path : Source classified raster (single band, integer dtype).
    confidence_path : Companion confidence raster (float32, same extent/CRS).
                      Nodata sentinel is expected to be -9999.0.
    threshold       : Pixels with confidence strictly below this value are replaced.
    out_path        : Destination path for the filtered raster.
    block_size      : Tile side length for windowed processing.

    Returns
    -------
    Path of the written output raster.
    """
    from scipy.ndimage import median_filter as _ndimage_median  # noqa: PLC0415

    _CONF_FILTER_SIZE = 5
    pad = _CONF_FILTER_SIZE // 2

    classified_path = Path(classified_path)
    confidence_path = Path(confidence_path)
    out_path        = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(classified_path) as cls_src:
        nodata_val = int(cls_src.nodata) if cls_src.nodata is not None else -1
        profile    = cls_src.profile.copy()
        height     = cls_src.height
        width      = cls_src.width

    profile.update(
        driver    = "GTiff",
        compress  = "deflate",
        predictor = 2,
        tiled     = True,
        blockxsize = block_size,
        blockysize = block_size,
    )

    with (
        rasterio.open(classified_path) as cls_src,
        rasterio.open(confidence_path) as conf_src,
        rasterio.open(out_path, "w", **profile) as dst,
    ):
        for win in iter_windows(cls_src, block_size):
            col_off = win.col_off
            row_off = win.row_off
            win_w   = win.width
            win_h   = win.height

            # Padded window — clipped to raster bounds
            p_col_off = max(0, col_off - pad)
            p_row_off = max(0, row_off - pad)
            p_col_end = min(width,  col_off + win_w + pad)
            p_row_end = min(height, row_off + win_h + pad)
            p_win = rasterio.windows.Window(
                col_off = p_col_off,
                row_off = p_row_off,
                width   = p_col_end - p_col_off,
                height  = p_row_end - p_row_off,
            )

            padded_cls  = cls_src.read(1, window=p_win).astype(np.int32)
            padded_conf = conf_src.read(1, window=p_win).astype(np.float32)

            # Compute 5×5 median of classified values for potential replacements
            padded_median = _ndimage_median(padded_cls, size=_CONF_FILTER_SIZE)

            # Replace: confidence is valid (>= 0) AND below threshold AND not nodata
            replace = (
                (padded_conf >= 0.0) &
                (padded_conf < threshold) &
                (padded_cls != nodata_val)
            )
            padded_result = np.where(replace, padded_median, padded_cls)

            # Trim back to the original (unpadded) window
            r_start = row_off - p_row_off
            c_start = col_off - p_col_off
            result  = padded_result[r_start:r_start + win_h, c_start:c_start + win_w]

            dst.write(result.astype(profile["dtype"]), 1, window=win)

    return out_path


def median_smooth(
    src_path:    str | Path,
    out_path:    str | Path,
    kernel_size: int = 3,
    block_size:  int = DEFAULT_BLOCK,
) -> Path:
    """
    Smooth a classified raster using a windowed median filter.

    Processing uses overlap-padded windowed reads so tile-boundary pixels
    receive correct neighbourhood values.  Nodata pixels are preserved.

    Parameters
    ----------
    src_path    : Source classified raster (single band, integer dtype).
    out_path    : Destination path for the smoothed raster.
    kernel_size : Side length of the square median-filter kernel.
                  Must be a positive odd integer ≥ 3.
    block_size  : Tile side length for windowed processing.

    Returns
    -------
    Path of the written output raster.

    Raises
    ------
    ValueError
        If kernel_size is not a positive odd integer.
    """
    from scipy.ndimage import median_filter as _ndimage_median  # noqa: PLC0415

    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer >= 3, got {kernel_size!r}."
        )

    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pad = kernel_size // 2

    with rasterio.open(src_path) as src:
        nodata_val = int(src.nodata) if src.nodata is not None else -1
        profile    = src.profile.copy()
        height     = src.height
        width      = src.width

    profile.update(
        driver     = "GTiff",
        compress   = "deflate",
        predictor  = 2,
        tiled      = True,
        blockxsize = block_size,
        blockysize = block_size,
    )

    with (
        rasterio.open(src_path) as src,
        rasterio.open(out_path, "w", **profile) as dst,
    ):
        for win in iter_windows(src, block_size):
            col_off = win.col_off
            row_off = win.row_off
            win_w   = win.width
            win_h   = win.height

            p_col_off = max(0, col_off - pad)
            p_row_off = max(0, row_off - pad)
            p_col_end = min(width,  col_off + win_w + pad)
            p_row_end = min(height, row_off + win_h + pad)
            p_win = rasterio.windows.Window(
                col_off = p_col_off,
                row_off = p_row_off,
                width   = p_col_end - p_col_off,
                height  = p_row_end - p_row_off,
            )

            padded      = src.read(1, window=p_win).astype(np.int32)
            nodata_mask = padded == nodata_val

            smoothed             = _ndimage_median(padded, size=kernel_size)
            smoothed[nodata_mask] = nodata_val  # restore nodata pixels

            r_start = row_off - p_row_off
            c_start = col_off - p_col_off
            result  = smoothed[r_start:r_start + win_h, c_start:c_start + win_w]

            dst.write(result.astype(profile["dtype"]), 1, window=win)

    return out_path


def estimate_chain_time(
    raster_path: Path,
    cfg: dict,
    pixel_res_m: float,
) -> dict[str, float]:
    """
    Estimate processing time per step based on pixel count and kernel sizes.

    Uses raster_io.get_meta() (zero pixel reads) to get dimensions.
    Returns {step_name: estimated_seconds}.
    If total > 300s, also returns key "warning" with a message string.

    Empirical coefficients (μs/pixel):
        confidence_filter  — 0.8
        median_smooth      — 1.2
        morphological_close— 0.6
        sieve_filter       — 0.4  (GDAL is faster)
    """
    from pipeline.raster_io import get_meta as _get_meta  # noqa: PLC0415

    meta       = _get_meta(raster_path)
    n_pixels   = meta["width"] * meta["height"]
    drone_mode = pixel_res_m < float(cfg.get("drone_pixel_res_threshold_m", 1.0))

    # In drone_mode confidence_filter is skipped — zero cost
    conf_coeff  = 0.0 if drone_mode and pixel_res_m < 0.5 else 0.8e-6
    median_coeff = 1.2e-6
    morpho_coeff = 0.6e-6
    sieve_coeff  = 0.4e-6

    morpho_iters = int(cfg.get("morphological_iterations", 1))
    if drone_mode:
        morpho_iters = int(cfg.get("drone_morpho_iterations", 2))

    estimates = {
        "confidence_filter":    n_pixels * conf_coeff,
        "median_smooth":        n_pixels * median_coeff,
        "morphological_close":  n_pixels * morpho_coeff * max(1, morpho_iters),
        "sieve_filter":         n_pixels * sieve_coeff,
    }

    total = sum(estimates.values())
    if total > 300:
        estimates["warning"] = (
            f"Large raster detected ({n_pixels:,} pixels). "
            f"Estimated total: {total:.0f}s (~{total/60:.1f} min)."
        )

    return estimates


def run_postprocess_chain(
    classified_path: str | Path,
    confidence_path: str | Path,
    cfg:             dict,
    out_dir:         str | Path,
    run_id:          str,
    progress:        Optional[Any] = None,
    pixel_res_m:     float = 10.0,
    drone_mode:      bool  = False,
    cancel_event:    Optional[Any] = None,
) -> dict:
    """
    Run the 4-step post-classification filtering chain.

    Steps
    -----
    1. ``confidence_filter``   — Replace low-confidence pixels with local median.
                                 Skipped in drone_mode when pixel_res_m < 0.5.
    2. ``median_smooth``       — Windowed median filter to reduce isolated pixels.
    3. ``morphological_close`` — Majority-vote morphological closing
                                 (repeated ``morphological_iterations`` times).
    4. ``sieve_filter``        — MMU enforcement via GDAL SieveFilter.

    Each step is logged to the audit trail via ``audit.log_event``.
    Between steps, ``cancel_event`` is checked if provided — if set, the chain
    stops and raises RuntimeError("Cancelled by user").

    Parameters
    ----------
    classified_path : Input classified raster (int16, single band).
    confidence_path : Companion confidence raster produced by predict_raster.
    cfg             : Pipeline config dict.  Keys used:
                      ``confidence_threshold``, ``median_filter_size``,
                      ``morphological_kernel_size``, ``morphological_iterations``,
                      ``min_mapping_unit_ha``, ``sieve_connectivity``.
                      Drone keys: ``drone_pixel_res_threshold_m``,
                      ``drone_median_kernel_max``, ``drone_morpho_iterations``,
                      ``drone_sieve_min_px``.
    out_dir         : Directory for intermediate and final outputs.
    run_id          : Active run identifier for audit logging.
    progress        : Optional callable(msg: str) invoked after each step.
    pixel_res_m     : Ground sampling distance in metres.  Auto-read from CRS
                      if the raster has a projected CRS.
    drone_mode      : If True, apply drone-optimised parameter overrides.
                      Auto-enabled if pixel_res_m < drone_pixel_res_threshold_m.
    cancel_event    : Optional threading.Event; chain stops between steps if set.

    Returns
    -------
    dict with keys:
        ``"confidence_filtered"``    — Path, output of step 1 (or input if skipped)
        ``"median_smoothed"``        — Path, output of step 2
        ``"morphologically_closed"`` — Path, output of step 3
        ``"final"``                  — Path, output of step 4 (sieve)

    Raises
    ------
    RuntimeError : If GDAL is not available or cancel_event is set.
    ValueError   : If the raster has a non-projected CRS (pixel_res_m cannot
                   be derived automatically).
    """
    if not _GDAL_AVAILABLE:
        raise RuntimeError(
            "GDAL (osgeo) is required for the full post-processing chain "
            "(sieve_filter step)."
        )

    classified_path = Path(classified_path)
    confidence_path = Path(confidence_path)
    out_dir         = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_threshold = float(cfg.get("confidence_threshold",    0.6))
    median_k       = int(cfg.get("median_filter_size",         3))
    morpho_k       = int(cfg.get("morphological_kernel_size",  3))
    morpho_iters   = int(cfg.get("morphological_iterations",   1))
    mmu_ha         = float(cfg.get("min_mapping_unit_ha",      0.5))
    connectivity   = int(cfg.get("sieve_connectivity",         4))

    drone_threshold_m = float(cfg.get("drone_pixel_res_threshold_m", 1.0))

    with rasterio.open(classified_path) as ds:
        if ds.crs and ds.crs.is_projected:
            pixel_res_m = float(abs(ds.transform.a))
        else:
            raise ValueError(
                "Cannot derive pixel_res_m for a non-projected CRS. "
                "Reproject the classified raster to a projected CRS first."
            )

    # Auto-detect drone mode
    if pixel_res_m < drone_threshold_m:
        drone_mode = True

    # Apply drone-mode parameter overrides
    skip_confidence_filter = False
    if drone_mode:
        drone_median_max  = int(cfg.get("drone_median_kernel_max",  7))
        drone_morpho_iter = int(cfg.get("drone_morpho_iterations",  2))
        drone_sieve_min   = int(cfg.get("drone_sieve_min_px",       100))

        # Cap median kernel
        median_k = max(3, min(drone_median_max, median_k))
        if pixel_res_m < 0.1:
            import warnings as _warnings
            _warnings.warn(
                f"Very high resolution drone data ({pixel_res_m:.3f} m/px). "
                f"Median kernel capped at {median_k}px.",
                RuntimeWarning,
                stacklevel=2,
            )

        # More passes, smaller kernel
        morpho_k     = 3
        morpho_iters = drone_morpho_iter

        # Guarantee minimum MMU of drone_sieve_min_px pixels
        min_mmu_ha = (pixel_res_m ** 2) * drone_sieve_min / 10_000.0
        mmu_ha     = max(mmu_ha, min_mmu_ha)

        # Skip confidence filter for very-high-resolution drones
        if pixel_res_m < 0.5:
            skip_confidence_filter = True
            _audit.log_event(
                run_id, "gate",
                {"stage": "chain_confidence_filter",
                 "skipped": True,
                 "reason": f"drone_mode + pixel_res_m={pixel_res_m:.3f} < 0.5"},
                decision="proceed",
            )

    results: dict = {}

    # ── Step 1: confidence filter ─────────────────────────────────────────────
    if skip_confidence_filter:
        # Pass the original classified raster through unchanged
        conf_filtered = classified_path
    else:
        conf_filtered = out_dir / f"{run_id}_chain_conf_filtered.tif"
        confidence_filter(classified_path, confidence_path, conf_threshold, conf_filtered)
        _audit.log_event(
            run_id, "gate",
            {"stage": "chain_confidence_filter", "threshold": conf_threshold,
             "output": str(conf_filtered)},
            decision="proceed",
        )
    results["confidence_filtered"] = conf_filtered
    if progress is not None:
        progress("confidence_filter complete")

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled by user")

    # ── Step 2: median smooth ─────────────────────────────────────────────────
    median_smoothed = out_dir / f"{run_id}_chain_median_smoothed.tif"
    bs_median = 256 if (drone_mode and pixel_res_m < 1.0) else DEFAULT_BLOCK
    median_smooth(conf_filtered, median_smoothed, kernel_size=median_k,
                  block_size=bs_median)
    results["median_smoothed"] = median_smoothed
    _audit.log_event(
        run_id, "gate",
        {"stage": "chain_median_smooth", "kernel_size": median_k,
         "drone_mode": drone_mode, "output": str(median_smoothed)},
        decision="proceed",
    )
    if progress is not None:
        progress("median_smooth complete")

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled by user")

    # ── Step 3: morphological close (repeated morpho_iters times) ────────────
    morpho_input = median_smoothed
    bs_morpho = 256 if (drone_mode and pixel_res_m < 1.0) else DEFAULT_BLOCK
    for i in range(max(1, morpho_iters)):
        morpho_out = out_dir / f"{run_id}_chain_morpho_{i}.tif"
        morphological_close(morpho_input, morpho_out, kernel_size=morpho_k,
                            block_size=bs_morpho)
        morpho_input = morpho_out
    results["morphologically_closed"] = morpho_input
    _audit.log_event(
        run_id, "gate",
        {"stage": "chain_morphological_close",
         "kernel_size": morpho_k, "iterations": morpho_iters,
         "drone_mode": drone_mode, "output": str(morpho_input)},
        decision="proceed",
    )
    if progress is not None:
        progress("morphological_close complete")

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled by user")

    # ── Step 4: sieve filter ──────────────────────────────────────────────────
    sieved_out = out_dir / f"{run_id}_chain_sieved.tif"
    sieve_filter(
        morpho_input, sieved_out,
        mmu_ha       = mmu_ha,
        pixel_res_m  = pixel_res_m,
        connectivity = connectivity,
    )
    results["final"] = sieved_out
    _audit.log_event(
        run_id, "gate",
        {"stage": "chain_sieve_filter",
         "mmu_ha": mmu_ha, "pixel_res_m": pixel_res_m,
         "connectivity": connectivity,
         "output": str(sieved_out)},
        decision="proceed",
    )
    if progress is not None:
        progress("sieve_filter complete")

    return results


# ── 3-tier quality gate ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class QualityGateResult:
    """
    Result of a single 3-tier quality gate evaluation.

    Attributes
    ----------
    metric_name    : Human-readable name of the metric evaluated.
    value          : Measured value.
    status         : "pass" | "warning" | "fail"
    threshold_pass : Value required for PASS (meaning depends on higher_is_better).
    threshold_fail : Value that triggers FAIL.
    message        : Human-readable explanation of the gate outcome.
    """
    metric_name:    str
    value:          float
    status:         str    # "pass" | "warning" | "fail"
    threshold_pass: float
    threshold_fail: float
    message:        str


def _eval_gate(
    metric_name:      str,
    value:            float,
    threshold_pass:   float,
    threshold_fail:   float,
    higher_is_better: bool = True,
) -> QualityGateResult:
    """
    Evaluate ``value`` against pass/warning/fail thresholds.

    For higher-is-better metrics (OA, Kappa, F1, confidence):
        pass    ← value >= threshold_pass
        warning ← threshold_fail <= value < threshold_pass
        fail    ← value < threshold_fail

    For lower-is-better metrics (nodata coverage fraction):
        pass    ← value <= threshold_pass
        warning ← threshold_pass < value <= threshold_fail
        fail    ← value > threshold_fail
    """
    if higher_is_better:
        if value >= threshold_pass:
            status  = "pass"
            message = f"{metric_name}: {value:.4f} ≥ {threshold_pass:.4f} (PASS)"
        elif value >= threshold_fail:
            status  = "warning"
            message = (
                f"{metric_name}: {value:.4f} below pass threshold "
                f"({threshold_pass:.4f}) — above fail threshold "
                f"({threshold_fail:.4f}) (WARNING)"
            )
        else:
            status  = "fail"
            message = (
                f"{metric_name}: {value:.4f} < {threshold_fail:.4f} (FAIL)"
            )
    else:
        if value <= threshold_pass:
            status  = "pass"
            message = f"{metric_name}: {value:.2%} ≤ {threshold_pass:.2%} (PASS)"
        elif value <= threshold_fail:
            status  = "warning"
            message = (
                f"{metric_name}: {value:.2%} exceeds pass threshold "
                f"({threshold_pass:.2%}) (WARNING)"
            )
        else:
            status  = "fail"
            message = (
                f"{metric_name}: {value:.2%} > {threshold_fail:.2%} (FAIL)"
            )

    return QualityGateResult(
        metric_name    = metric_name,
        value          = value,
        status         = status,
        threshold_pass = threshold_pass,
        threshold_fail = threshold_fail,
        message        = message,
    )


def run_quality_gates(
    model_result,
    accuracy_result,
    class_areas,
    confidence_stats,
    cfg:        dict,
    nodata_pct: Optional[float] = None,
) -> list[QualityGateResult]:
    """
    Evaluate classification quality across 3-tier PASS / WARNING / FAIL gates.

    All thresholds are read from ``cfg``; no hardcoded values.

    Parameters
    ----------
    model_result     : ClassificationResult (OA, kappa, minority_f1 attributes).
    accuracy_result  : AccuracyResult or None (independent accuracy assessment).
    class_areas      : ClassAreaResult or None (used for context only; nodata
                       fraction is passed separately via ``nodata_pct``).
    confidence_stats : dict with at least a ``"mean"`` key, or None.
    cfg              : Pipeline config dict.
    nodata_pct       : Fraction of raster pixels that are nodata [0, 1], or None
                       to skip the nodata coverage gate.

    Returns
    -------
    list[QualityGateResult] — one entry per gate evaluated.  The order is:
    OA (training), Kappa, Minority F1, OA (independent), Mean Confidence,
    Nodata Coverage.  Gates are skipped (not appended) when the required input
    is None.
    """
    results: list[QualityGateResult] = []

    # ── Read thresholds from cfg ───────────────────────────────────────────────
    oa_pass    = float(cfg.get("quality_gate_oa_pass",          0.90))
    oa_fail    = float(cfg.get("quality_gate_oa_fail",          0.80))
    kappa_pass = float(cfg.get("quality_gate_kappa_pass",       0.80))
    kappa_fail = float(cfg.get("quality_gate_kappa_fail",       0.65))
    f1_pass    = float(cfg.get("quality_gate_f1_pass",          0.70))
    f1_fail    = float(cfg.get("quality_gate_f1_fail",          0.50))
    conf_pass  = float(cfg.get("quality_gate_confidence_pass",  0.75))
    conf_fail  = float(cfg.get("quality_gate_confidence_fail",  0.60))
    nd_pass    = float(cfg.get("quality_gate_nodata_pass",       0.05))
    nd_fail    = float(cfg.get("quality_gate_nodata_fail",       0.10))

    # ── 1. Overall Accuracy (training model) ──────────────────────────────────
    if model_result is not None:
        oa = getattr(model_result, "oa", None)
        if oa is not None:
            results.append(_eval_gate(
                "Overall Accuracy (training)", float(oa), oa_pass, oa_fail
            ))

    # ── 2. Cohen's Kappa (training model) ─────────────────────────────────────
    if model_result is not None:
        kappa = getattr(model_result, "kappa", None)
        if kappa is not None:
            results.append(_eval_gate(
                "Cohen's Kappa", float(kappa), kappa_pass, kappa_fail
            ))

    # ── 3. Minority F1 (training model) ───────────────────────────────────────
    if model_result is not None:
        f1 = getattr(model_result, "minority_f1", None)
        if f1 is not None:
            results.append(_eval_gate(
                "Minority F1", float(f1), f1_pass, f1_fail
            ))

    # ── 4. Overall Accuracy (independent accuracy assessment, if run) ──────────
    if accuracy_result is not None:
        ind_oa = getattr(accuracy_result, "oa", None)
        if ind_oa is not None:
            results.append(_eval_gate(
                "Overall Accuracy (independent)", float(ind_oa), oa_pass, oa_fail
            ))

    # ── 5. Mean confidence ────────────────────────────────────────────────────
    if confidence_stats is not None:
        mean_conf = confidence_stats.get("mean")
        if mean_conf is not None:
            results.append(_eval_gate(
                "Mean Confidence", float(mean_conf), conf_pass, conf_fail
            ))

    # ── 6. Nodata coverage ────────────────────────────────────────────────────
    if nodata_pct is not None:
        results.append(_eval_gate(
            "Nodata Coverage", float(nodata_pct), nd_pass, nd_fail,
            higher_is_better=False,
        ))

    return results


def has_gate_failures(results: list[QualityGateResult]) -> bool:
    """Return True if any gate in ``results`` has status ``"fail"``."""
    return any(r.status == "fail" for r in results)


# ── Private helpers ───────────────────────────────────────────────────────────

def _majority_filter(
    tile:     np.ndarray,
    k:        int,
    nodata:   int,
) -> np.ndarray:
    """
    Replace each pixel with the majority value in its k×k neighbourhood.

    Nodata pixels are treated as absent when computing the majority; if the
    entire neighbourhood is nodata the pixel retains its original value.
    This is a simple O(N·k²) implementation — acceptable for tile-level use.
    """
    pad    = k // 2
    rows, cols = tile.shape
    result = tile.copy()

    # Pad the tile with nodata so edge pixels have full neighbourhoods
    padded = np.full((rows + 2 * pad, cols + 2 * pad), nodata, dtype=tile.dtype)
    padded[pad:pad + rows, pad:pad + cols] = tile

    for r in range(rows):
        for c in range(cols):
            neighbourhood = padded[r:r + k, c:c + k].ravel()
            valid = neighbourhood[neighbourhood != nodata]
            if valid.size == 0:
                continue
            # majority = mode
            vals, cnts = np.unique(valid, return_counts=True)
            result[r, c] = int(vals[np.argmax(cnts)])

    return result


def _compute_accuracy_metrics(
    y_true:         np.ndarray,
    y_pred:         np.ndarray,
    n_points:       int,
    n_valid:        int,
    n_discarded:    int,
    discard_reasons: dict[str, int],
) -> AccuracyResult:
    """Compute OA, kappa, per-class metrics, and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    oa    = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm    = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class: dict[int, dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   float(support[i]),
        }

    return AccuracyResult(
        oa               = oa,
        kappa            = kappa,
        per_class_metrics = per_class,
        confusion_matrix = cm,
        class_labels     = labels,
        n_points         = n_points,
        n_valid          = n_valid,
        n_discarded      = n_discarded,
        discard_reasons  = discard_reasons,
    )
