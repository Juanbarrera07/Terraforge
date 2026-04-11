"""
Phase 4 — Feature engineering.

Computes a per-pixel spectral + terrain feature stack from preprocessed rasters.
All processing is windowed — no full raster loads at any point.

Texture statistics (GLCM contrast, homogeneity, entropy) are computed per tile
via a pure-NumPy tile-level approach and returned as an aggregated summary in
FeatureResult.texture_summary.  They are NOT written into the feature raster;
the output GeoTIFF contains only spatially-continuous per-pixel features.

Public API
----------
BandMap          — maps logical band names to 1-indexed band positions
FeatureResult    — frozen dataclass returned by compute_features()
compute_features — main entry point

Design rules (consistent with preprocess.py)
--------------------------------------------
- All raster reads use iter_windows; list(iter_windows(...)) is never called.
- No full-raster arrays at any point.
- No Streamlit imports.  Progress callbacks: (current_tile: int, total_tiles: int).
- Output nodata is always a finite float (never NaN in the file).
- Internal calculations use NaN for invalid pixels; NaN is restored to
  file_nodata before every write.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio

from pipeline.config_loader import load_config
from pipeline.raster_io import DEFAULT_BLOCK, iter_windows

ProgressCallback = Optional[Callable[[int, int], None]]


# ── BandMap ───────────────────────────────────────────────────────────────────

@dataclass
class BandMap:
    """
    1-indexed band positions for a source raster.  None means the band is absent.

    Features are computed only when all required bands are present.  The ``dsm``
    field is reserved for future use; no extra logic is applied to it.
    """
    nir:     Optional[int] = None
    red:     Optional[int] = None
    green:   Optional[int] = None
    blue:    Optional[int] = None
    swir:    Optional[int] = None
    rededge: Optional[int] = None
    vv:      Optional[int] = None
    vh:      Optional[int] = None
    dem:     Optional[int] = None
    dsm:     Optional[int] = None


# ── FeatureResult ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureResult:
    """
    Immutable result returned by compute_features().

    ``correlation_matrix`` is excluded from equality comparisons (consistent
    with CoregistrationResult.shift_map).

    Fields
    ------
    feature_path           : Path of the written feature-stack GeoTIFF.
    feature_names          : Ordered list of feature names matching band order.
    feature_stats          : Per-feature summary statistics (min, max, mean,
                             std, valid_pct).
    texture_summary        : Aggregated tile-level GLCM statistics, or None if
                             no band was available for GLCM computation.
    correlation_matrix     : (n × n) Pearson correlation matrix (excluded from
                             equality).
    high_correlation_pairs : Pairs (name_a, name_b, r) where |r| > threshold.
    """
    feature_path:           Path
    feature_names:          list[str]
    feature_stats:          dict[str, dict[str, float]]
    texture_summary:        Optional[dict[str, dict[str, float]]]
    correlation_matrix:     np.ndarray = field(compare=False)
    high_correlation_pairs: list[tuple[str, str, float]]


# ── Band arithmetic helpers ───────────────────────────────────────────────────

def _safe_ratio(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """
    Element-wise a / b.  Where b == 0 or either operand is non-finite, return
    ``fill`` instead.  IEEE NaN propagation is suppressed by the explicit mask.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (b != 0) & np.isfinite(b) & np.isfinite(a),
            a / b,
            fill,
        )


def _ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index: (NIR − Red) / (NIR + Red)."""
    return _safe_ratio(nir - red, nir + red, fill=np.nan)


def _ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Water Index: (Green − NIR) / (Green + NIR)."""
    return _safe_ratio(green - nir, green + nir, fill=np.nan)


def _bsi(
    swir: np.ndarray, red: np.ndarray, nir: np.ndarray, blue: np.ndarray
) -> np.ndarray:
    """Bare Soil Index: ((SWIR+Red) − (NIR+Blue)) / ((SWIR+Red) + (NIR+Blue))."""
    num = (swir + red) - (nir + blue)
    den = (swir + red) + (nir + blue)
    return _safe_ratio(num, den, fill=np.nan)


def _ndre(rededge: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Red-Edge Index: (RedEdge − Red) / (RedEdge + Red)."""
    return _safe_ratio(rededge - red, rededge + red, fill=np.nan)


def _vari(green: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Visible Atmospherically Resistant Index: (G − R) / (G + R − B)."""
    return _safe_ratio(green - red, green + red - blue, fill=np.nan)


def _sar_ratio(
    vv: np.ndarray, vh: np.ndarray, log_scale: bool = False
) -> np.ndarray:
    """
    SAR backscatter ratio VV / VH.

    Parameters
    ----------
    log_scale : If True, return 10 * log10(VV/VH).  Pixels where the linear
                ratio is ≤ 0 are set to NaN.
    """
    ratio = _safe_ratio(vv, vh, fill=np.nan)
    if log_scale:
        ratio = np.where(ratio > 0, 10.0 * np.log10(ratio), np.nan)
    return ratio


def _slope(dem: np.ndarray, res: float) -> np.ndarray:
    """
    Slope magnitude (rise / run, dimensionless) computed via numpy.gradient.

    Edge pixels use one-sided finite differences; interior pixels use central
    differences.  This is sufficient for classification input.
    """
    safe_res = max(res, 1e-9)
    dy, dx = np.gradient(dem, safe_res, safe_res)
    return np.sqrt(dx ** 2 + dy ** 2)


def _aspect(dem: np.ndarray, res: float) -> np.ndarray:
    """
    Aspect — direction of steepest ascent in degrees [0, 360).

    0° = East, counter-clockwise convention (consistent with atan2 output).
    """
    safe_res = max(res, 1e-9)
    dy, dx = np.gradient(dem, safe_res, safe_res)
    return np.degrees(np.arctan2(-dy, dx)) % 360.0


# ── GLCM texture (tile-level, pure NumPy) ────────────────────────────────────

def _glcm_features(tile: np.ndarray, levels: int = 16) -> dict[str, float]:
    """
    Compute GLCM-derived texture statistics for a single 2D tile.

    Algorithm
    ---------
    1. Quantise the tile to ``levels`` grey levels.
    2. Build the (levels × levels) co-occurrence matrix from horizontal
       (dx=1) pixel-pair co-occurrences.
    3. Symmetrise and normalise to a probability matrix P.
    4. Derive contrast, homogeneity, and entropy from P.

    Returns a dict with keys ``contrast``, ``homogeneity``, ``entropy``.
    A uniform tile returns contrast=0, homogeneity=1, entropy=0 immediately.
    """
    t_min = float(np.nanmin(tile))
    t_max = float(np.nanmax(tile))

    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return {"contrast": 0.0, "homogeneity": 1.0, "entropy": 0.0}

    q = np.clip(
        ((tile - t_min) / (t_max - t_min) * (levels - 1)).astype(np.int32),
        0,
        levels - 1,
    )

    glcm = np.zeros((levels, levels), dtype=np.float64)
    # Horizontal co-occurrences: pixel at column j paired with pixel at column j+1
    np.add.at(glcm, (q[:, :-1].ravel(), q[:, 1:].ravel()), 1)
    glcm = glcm + glcm.T  # symmetrise

    total = glcm.sum()
    if total == 0:
        return {"contrast": 0.0, "homogeneity": 1.0, "entropy": 0.0}

    P = glcm / total
    i_idx = np.arange(levels, dtype=np.float64)
    ii, jj = np.meshgrid(i_idx, i_idx, indexing="ij")
    diff = ii - jj

    contrast    = float(np.sum(P * diff ** 2))
    homogeneity = float(np.sum(P / (1.0 + np.abs(diff))))
    entropy     = float(-np.sum(P * np.log2(P + 1e-12)))

    return {"contrast": contrast, "homogeneity": homogeneity, "entropy": entropy}


# ── Feature stack internals ───────────────────────────────────────────────────

def _active_features(band_map: BandMap) -> list[str]:
    """Return the ordered list of feature names that can be computed from band_map."""
    bm = band_map
    feats: list[str] = []
    if bm.nir     and bm.red:                           feats.append("ndvi")
    if bm.green   and bm.nir:                           feats.append("ndwi")
    if bm.swir    and bm.red and bm.nir and bm.blue:    feats.append("bsi")
    if bm.rededge and bm.red:                           feats.append("ndre")
    if bm.green   and bm.red and bm.blue:               feats.append("vari")
    if bm.vv      and bm.vh:                            feats.append("sar_ratio")
    if bm.dem:                                          feats.append("slope")
    if bm.dem:                                          feats.append("aspect")
    return feats


def _count_windows(ds: rasterio.DatasetReader, block_size: int) -> int:
    """Total tile count for a dataset — used to drive progress callbacks."""
    return math.ceil(ds.height / block_size) * math.ceil(ds.width / block_size)


def _band_as_float(
    data: np.ndarray,
    band_idx: int,
    src_nodata: Optional[float],
) -> np.ndarray:
    """
    Extract a 1-indexed band from a (bands, H, W) array as float64.
    Source nodata pixels are converted to NaN.
    """
    b = data[band_idx - 1].astype(np.float64)
    if src_nodata is not None:
        b[b == src_nodata] = np.nan
    return b


def _compute_tile_features(
    data:         np.ndarray,
    band_map:     BandMap,
    res:          float,
    active_feats: list[str],
    src_nodata:   Optional[float],
) -> np.ndarray:
    """
    Compute per-pixel feature bands for one tile.

    Parameters
    ----------
    data         : (n_src_bands, H, W) float64 read from source raster.
    active_feats : Ordered feature names from _active_features().
    src_nodata   : Source nodata value (converted to NaN internally).

    Returns
    -------
    (n_features, H, W) float64 — NaN where nodata or result is indeterminate.
    """
    H, W = data.shape[1], data.shape[2]
    bm = band_map

    def _b(idx: int) -> np.ndarray:  # noqa: E731
        return _band_as_float(data, idx, src_nodata)

    out = np.empty((len(active_feats), H, W), dtype=np.float64)

    for fi, name in enumerate(active_feats):
        if name == "ndvi":
            out[fi] = _ndvi(_b(bm.nir), _b(bm.red))          # type: ignore[arg-type]
        elif name == "ndwi":
            out[fi] = _ndwi(_b(bm.green), _b(bm.nir))        # type: ignore[arg-type]
        elif name == "bsi":
            out[fi] = _bsi(
                _b(bm.swir), _b(bm.red), _b(bm.nir), _b(bm.blue)  # type: ignore[arg-type]
            )
        elif name == "ndre":
            out[fi] = _ndre(_b(bm.rededge), _b(bm.red))      # type: ignore[arg-type]
        elif name == "vari":
            out[fi] = _vari(_b(bm.green), _b(bm.red), _b(bm.blue))  # type: ignore[arg-type]
        elif name == "sar_ratio":
            out[fi] = _sar_ratio(_b(bm.vv), _b(bm.vh))       # type: ignore[arg-type]
        elif name == "slope":
            out[fi] = _slope(_b(bm.dem), res)                 # type: ignore[arg-type]
        elif name == "aspect":
            out[fi] = _aspect(_b(bm.dem), res)                # type: ignore[arg-type]

    return out


def _select_glcm_band(
    data:       np.ndarray,
    band_map:   BandMap,
    src_nodata: Optional[float],
) -> Optional[np.ndarray]:
    """
    Return a 2D float64 array suitable for GLCM computation.

    Priority: red > nir > vv > band 1.
    NaN pixels (from nodata) are replaced with the band mean so they do not
    contaminate the co-occurrence histogram.
    Returns None if no valid pixels exist.
    """
    # Priority selection — all indices are 1-based and ≥ 1, so ``or`` is safe
    idx = band_map.red or band_map.nir or band_map.vv
    if idx is None:
        if data.shape[0] == 0:
            return None
        idx = 1  # band-1 fallback

    if idx > data.shape[0]:
        return None

    band = _band_as_float(data, idx, src_nodata)
    finite = np.isfinite(band)
    if not finite.any():
        return None

    band[~finite] = float(band[finite].mean())
    return band


# ── Streaming correlation (Pass 2) ────────────────────────────────────────────

def _streaming_correlation(
    feat_path:   Path,
    n_features:  int,
    file_nodata: float,
    block_size:  int,
) -> np.ndarray:
    """
    Compute the (n × n) Pearson correlation matrix from the feature raster
    using streaming sum / sum-of-squares / cross-product accumulators.

    Pixels that are nodata or non-finite in ANY feature band are excluded from
    all pairs (listwise deletion), ensuring all off-diagonal entries are computed
    on a consistent sample.

    Returns an identity matrix if fewer than 2 valid pixels are found.
    """
    count  = 0
    sums   = np.zeros(n_features, dtype=np.float64)
    sumsq  = np.zeros(n_features, dtype=np.float64)
    cross  = np.zeros((n_features, n_features), dtype=np.float64)

    with rasterio.open(feat_path) as ds:
        for window in iter_windows(ds, block_size):
            data = ds.read(window=window).astype(np.float64)   # (n_feat, H, W)
            flat = data.reshape(n_features, -1)                # (n_feat, H*W)

            # Listwise valid mask
            valid = np.ones(flat.shape[1], dtype=bool)
            for fi in range(n_features):
                valid &= np.isfinite(flat[fi]) & (flat[fi] != file_nodata)

            pix = flat[:, valid]   # (n_features, n_valid)
            nv  = pix.shape[1]
            if nv == 0:
                continue

            count += nv
            sums  += pix.sum(axis=1)
            sumsq += (pix ** 2).sum(axis=1)
            cross += pix @ pix.T

    if count < 2:
        return np.eye(n_features, dtype=np.float64)

    mean = sums / count
    var  = np.maximum(0.0, sumsq / count - mean ** 2)
    std  = np.sqrt(var)

    corr = np.ones((n_features, n_features), dtype=np.float64)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            cov_ij = cross[i, j] / count - mean[i] * mean[j]
            denom  = std[i] * std[j]
            r      = cov_ij / denom if denom > 1e-12 else 0.0
            corr[i, j] = corr[j, i] = r

    return corr


# ── Public API ────────────────────────────────────────────────────────────────

def active_features(band_map: BandMap) -> list[str]:
    """
    Return the ordered list of feature names computable from *band_map*.

    This is the public entry point for the UI layer to preview which features
    are available before calling compute_features().
    """
    return _active_features(band_map)


def compute_features(
    src_path:         str | Path,
    out_path:         str | Path,
    band_map:         BandMap,
    block_size:       int               = DEFAULT_BLOCK,
    progress:         ProgressCallback  = None,
    cfg:              Optional[dict]    = None,
    enabled_features: Optional[list[str]] = None,
) -> FeatureResult:
    """
    Compute a per-pixel feature stack from a preprocessed raster.

    Features written to the output GeoTIFF (when required bands are present):

        ndvi       (NIR, Red)
        ndwi       (Green, NIR)
        bsi        (SWIR, Red, NIR, Blue)
        ndre       (RedEdge, Red)
        vari       (Green, Red, Blue)
        sar_ratio  (VV, VH)
        slope      (DEM)
        aspect     (DEM)

    GLCM texture statistics (contrast, homogeneity, entropy) are computed
    per tile and returned in ``FeatureResult.texture_summary``; they are NOT
    written to the feature raster.

    Two-pass algorithm
    ------------------
    Pass 1 — for each tile:
        compute feature bands → write float32 tile to output GeoTIFF
        accumulate per-feature statistics (min/max/sum/sum²/count)
        compute tile-level GLCM and append to running list

    Pass 2 — for each tile of the written feature raster:
        accumulate streaming sums/sum²/cross-products for correlation

    Parameters
    ----------
    src_path         : Preprocessed source raster.
    out_path         : Output feature-stack path.
    band_map         : Maps logical band names to 1-indexed positions in src_path.
    block_size       : Tile side length for windowed processing.
    progress         : Optional callback(current_tile, total_tiles) — Pass 1 only.
    cfg              : Config dict; loaded from pipeline_config.yaml if None.
    enabled_features : Explicit allow-list of feature names to compute.  When
                       None (default), all features derivable from band_map are
                       computed.  Names not supported by the current band_map are
                       silently ignored.

    Returns
    -------
    FeatureResult
    """
    # Callers that already hold a config dict pass it directly; load_config() is
    # the fallback only when no dict is provided.  The .get() guard handles the
    # edge case where a partial dict is passed without this key.
    if cfg is None:
        cfg = load_config()
    corr_thresh = float(cfg.get("corr_flag_threshold", 0.95))

    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    possible_feats = _active_features(band_map)
    if enabled_features is not None:
        active_feats = [f for f in possible_feats if f in enabled_features]
    else:
        active_feats = possible_feats
    if not active_feats:
        raise ValueError(
            "BandMap provides no computable features. "
            "Set at least two band fields (e.g. nir + red for NDVI)."
        )
    n_features = len(active_feats)

    with rasterio.open(src_path) as ds:
        src_profile = ds.profile.copy()
        src_nodata  = ds.nodata
        res         = float(abs(ds.transform.a))
        n_tiles     = _count_windows(ds, block_size)

    # Output nodata: inherit if valid (finite), otherwise use -9999.0
    if src_nodata is not None:
        try:
            _nd = float(src_nodata)
            file_nodata: float = _nd if np.isfinite(_nd) else -9999.0
        except (TypeError, ValueError):
            file_nodata = -9999.0
    else:
        file_nodata = -9999.0

    out_profile = src_profile.copy()
    out_profile.update({
        "count":      n_features,
        "dtype":      "float32",
        "nodata":     file_nodata,
        "compress":   "deflate",
        "predictor":  3,   # floating-point horizontal differencing (TIFF spec §14.4)
        "tiled":      True,
        "blockxsize": block_size,
        "blockysize": block_size,
    })

    # ── Pass 1: compute + write features ──────────────────────────────────────
    stats_acc: dict[str, dict] = {
        f: {"min": math.inf, "max": -math.inf, "sum": 0.0, "sum2": 0.0, "count": 0}
        for f in active_feats
    }
    glcm_lists: dict[str, list[float]] = {
        "contrast": [], "homogeneity": [], "entropy": []
    }
    total_pixels = 0
    tile_num     = 0

    with rasterio.open(src_path) as src, rasterio.open(out_path, "w", **out_profile) as dst:
        for window in iter_windows(src, block_size):
            raw = src.read(window=window).astype(np.float64)
            H, W = raw.shape[1], raw.shape[2]
            total_pixels += H * W

            feat_tile = _compute_tile_features(
                raw, band_map, res, active_feats, src_nodata
            )

            # Accumulate per-feature statistics from valid (finite) pixels
            for fi, fname in enumerate(active_feats):
                fb    = feat_tile[fi]
                valid = np.isfinite(fb)
                if valid.any():
                    vals = fb[valid]
                    acc  = stats_acc[fname]
                    acc["min"]    = min(acc["min"],   float(vals.min()))
                    acc["max"]    = max(acc["max"],   float(vals.max()))
                    acc["sum"]   += float(vals.sum())
                    acc["sum2"]  += float((vals ** 2).sum())
                    acc["count"] += int(valid.sum())

            # Tile-level GLCM on a representative band
            glcm_band = _select_glcm_band(raw, band_map, src_nodata)
            if glcm_band is not None:
                tx = _glcm_features(glcm_band)
                for k in glcm_lists:
                    glcm_lists[k].append(tx[k])

            # Replace internal NaN with file_nodata before writing
            out_tile = np.where(
                np.isfinite(feat_tile), feat_tile, file_nodata
            ).astype(np.float32)
            dst.write(out_tile, window=window)

            tile_num += 1
            if progress is not None:
                progress(tile_num, n_tiles)

    # ── Finalise feature_stats ─────────────────────────────────────────────────
    feature_stats: dict[str, dict[str, float]] = {}
    for fname in active_feats:
        acc = stats_acc[fname]
        n   = acc["count"]
        if n > 0:
            mean = acc["sum"] / n
            std  = math.sqrt(max(0.0, acc["sum2"] / n - mean ** 2))
            vp   = 100.0 * n / total_pixels if total_pixels > 0 else 0.0
            feature_stats[fname] = {
                "min":       acc["min"],
                "max":       acc["max"],
                "mean":      mean,
                "std":       std,
                "valid_pct": vp,
            }
        else:
            feature_stats[fname] = {
                "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "valid_pct": 0.0
            }

    # ── Texture summary ────────────────────────────────────────────────────────
    texture_summary: Optional[dict[str, dict[str, float]]] = None
    if glcm_lists["contrast"]:
        texture_summary = {}
        for metric, vals_list in glcm_lists.items():
            arr = np.array(vals_list, dtype=np.float64)
            texture_summary[metric] = {
                "mean": float(arr.mean()),
                "std":  float(arr.std()),
                "min":  float(arr.min()),
                "max":  float(arr.max()),
            }

    # ── Pass 2: streaming correlation ─────────────────────────────────────────
    corr_matrix = _streaming_correlation(
        out_path, n_features, file_nodata, block_size
    )

    # ── High-correlation pairs ─────────────────────────────────────────────────
    # Strictly greater than (>) — pairs exactly at the threshold are not flagged.
    # Matches "flag pairs with |r| above this" in pipeline_config.yaml.
    high_corr: list[tuple[str, str, float]] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r = float(corr_matrix[i, j])
            if abs(r) > corr_thresh:
                high_corr.append((active_feats[i], active_feats[j], r))

    return FeatureResult(
        feature_path=out_path,
        feature_names=active_feats,
        feature_stats=feature_stats,
        texture_summary=texture_summary,
        correlation_matrix=corr_matrix,
        high_correlation_pairs=high_corr,
    )
