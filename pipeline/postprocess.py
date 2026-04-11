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
compute_class_areas     — windowed per-class pixel tally
sieve_filter            — MMU enforcement via GDAL SieveFilter (see warning)
morphological_close     — windowed morphological closing via overlap-padded tiles
assess_accuracy_from_points — accuracy from a reference CSV with lat/lon
check_drift             — class-area drift between two runs

Design rules
------------
- No Streamlit imports.
- All raster reads through iter_windows except where documented otherwise.
- sieve_filter is a deliberate exception: GDAL SieveFilter loads the full
  raster band into GDAL-managed memory.  See that function's docstring.
- morphological_close uses overlap-padded windowed reads to avoid full-raster
  allocation in Python.
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
