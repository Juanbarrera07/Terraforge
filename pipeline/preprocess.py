"""
Phase 3 — Optical and SAR preprocessing.

DOS1 atmospheric correction (optical)
--------------------------------------
Two-pass windowed algorithm — no full raster load at any point.

  Pass 1: Scan all tiles to find per-band dark object value
          (minimum non-zero, non-nodata pixel).
  Pass 2: Subtract the dark object value from every pixel; clip to [0, dtype_max].

Lee speckle filter (SAR)
------------------------
Classic Lee filter applied band-by-band, tile-by-tile with overlap padding
to eliminate edge artifacts between tiles.

Local statistics (mean, variance) are computed via integral images using
pure NumPy — O(n) memory, no scipy dependency, compatible with large rasters.

Noise variance is estimated from a representative window sample and the
ENL (Equivalent Number of Looks) parameter.

Design rules
------------
- All raster reads go through windowed reads (raster_io.iter_windows).
- Output is always written as tiled, deflate-compressed GeoTIFF.
- Progress callbacks (current_tile, total_tiles) allow the UI layer
  to drive a progress bar without this module importing Streamlit.
- No in-memory full-raster arrays at any point.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio

from pipeline.raster_io import DEFAULT_BLOCK, iter_windows

ProgressCallback = Optional[Callable[[int, int], None]]

_DTYPE_MAX: dict[str, float] = {
    "uint8":   255.0,
    "uint16":  65535.0,
    "uint32":  4294967295.0,
    "int16":   32767.0,
    "int32":   2147483647.0,
    "float32": 3.4e+38,
    "float64": 1.8e+308,
}


# ── Integral-image local statistics ──────────────────────────────────────────

def _integral_image(arr: np.ndarray) -> np.ndarray:
    """
    Compute the 2D cumulative sum table (integral image) of a float64 array.

    Shape of output: (arr.shape[0] + 1, arr.shape[1] + 1).
    The extra leading row/col of zeros allows boundary-free box-sum queries.
    """
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    np.cumsum(arr, axis=0, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
    return ii


def _box_sums(ii: np.ndarray, k: int, out_h: int, out_w: int) -> np.ndarray:
    """
    Vectorised box-filter sums for a k×k window using an integral image.

    The integral image `ii` is assumed to be built from an array that has
    already been padded by `k//2` on all four sides, so the output covers
    every pixel of the *original* (un-padded) array.

    Parameters
    ----------
    ii    : Integral image of the padded array, shape (out_h+k, out_w+k).
    k     : Window side length (should be odd and ≥ 3).
    out_h : Height of the original (un-padded) array.
    out_w : Width  of the original (un-padded) array.

    Returns
    -------
    ndarray of shape (out_h, out_w) — box filter sum at each pixel.
    """
    # Row / col index vectors for the output grid
    r = np.arange(out_h, dtype=np.int32)
    c = np.arange(out_w, dtype=np.int32)

    # Four corners of each k×k box in the integral image
    D = ii[r[:, None] + k, c[None, :] + k]   # bottom-right
    B = ii[r[:, None],     c[None, :] + k]   # top-right
    C = ii[r[:, None] + k, c[None, :]]       # bottom-left
    A = ii[r[:, None],     c[None, :]]       # top-left

    return D - B - C + A


def _local_mean_var(
    tile: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local mean and local variance for every pixel in a 2D tile.

    Uses integral images for O(n) complexity — independent of window size.
    The tile is pre-padded with reflect mode so the output has the same
    shape as the input.

    Parameters
    ----------
    tile : 2-D float64 array of shape (H, W).
    k    : Square window side length (must be odd, ≥ 3).

    Returns
    -------
    (local_mean, local_var) — both shape (H, W).
    """
    H, W = tile.shape
    pad  = k // 2
    n    = float(k * k)

    padded = np.pad(tile.astype(np.float64), pad, mode="reflect")

    ii  = _integral_image(padded)
    ii2 = _integral_image(padded ** 2)

    sums  = _box_sums(ii,  k, H, W)
    sums2 = _box_sums(ii2, k, H, W)

    local_mean = sums  / n
    # Var = E[X²] - E[X]²; clamp negatives from floating-point rounding
    local_var  = np.maximum(0.0, sums2 / n - local_mean ** 2)

    return local_mean, local_var


# ── DOS1 helpers ──────────────────────────────────────────────────────────────

def _find_dark_object_values(
    src_path: Path,
    nodata: Optional[float],
    block_size: int,
) -> np.ndarray:
    """
    Pass 1 of DOS1: scan all tiles and return per-band minimum non-zero,
    non-nodata pixel values as a 1-D array of length band_count.

    Uses a heap-based partial sort approach: maintains the global minimum
    non-zero value seen across all tiles without storing all pixel values.
    """
    with rasterio.open(src_path) as ds:
        n_bands = ds.count
        dark    = np.full(n_bands, np.inf, dtype=np.float64)

        for window in iter_windows(ds, block_size):
            data = ds.read(window=window).astype(np.float64)  # (bands, H, W)

            for b in range(n_bands):
                band = data[b]
                mask = band > 0  # exclude zeros
                if nodata is not None:
                    mask &= band != nodata
                if mask.any():
                    dark[b] = min(dark[b], band[mask].min())

    # If a band had no valid pixels, use 0 (no correction)
    dark[~np.isfinite(dark)] = 0.0
    return dark


# ── Public API ────────────────────────────────────────────────────────────────

def dos1_atmospheric_correction(
    src_path: str | Path,
    out_path: str | Path,
    block_size: int = DEFAULT_BLOCK,
    progress: ProgressCallback = None,
) -> Path:
    """
    Apply DOS1 (Dark Object Subtraction 1) atmospheric correction.

    Algorithm
    ---------
    1. Find the minimum non-zero pixel per band (dark object value).
    2. Subtract the dark object value from every pixel.
    3. Clip the result to [0, dtype_max].

    Nodata pixels are preserved unchanged in the output.

    Parameters
    ----------
    src_path   : Input optical raster path.
    out_path   : Output corrected raster path.
    block_size : Tile size for windowed processing.
    progress   : Optional callback (current_tile: int, total_tiles: int).

    Returns
    -------
    Path of the written output raster.
    """
    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as ds:
        profile  = ds.profile.copy()
        nodata   = ds.nodata
        dtype    = str(ds.dtypes[0])
        dmax     = _DTYPE_MAX.get(dtype, 1.0)
        windows  = list(iter_windows(ds, block_size))
        n_tiles  = len(windows)

    # Pass 1 — find dark object values
    dark = _find_dark_object_values(src_path, nodata, block_size)

    # Pass 2 — apply correction and write
    profile.update(
        driver="GTiff",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
    )

    with rasterio.open(src_path) as src, rasterio.open(out_path, "w", **profile) as dst:
        for i, window in enumerate(windows):
            data = src.read(window=window).astype(np.float64)  # (bands, H, W)

            for b in range(src.count):
                band = data[b]
                if nodata is not None:
                    valid_mask = band != nodata
                    band[valid_mask] = np.clip(
                        band[valid_mask] - dark[b], 0.0, dmax
                    )
                    data[b] = band
                else:
                    data[b] = np.clip(band - dark[b], 0.0, dmax)

            dst.write(data.astype(profile["dtype"]), window=window)

            if progress is not None:
                progress(i + 1, n_tiles)

    return out_path


def lee_speckle_filter(
    src_path: str | Path,
    out_path: str | Path,
    kernel_size: int = 7,
    enl: float = 1.0,
    block_size: int = DEFAULT_BLOCK,
    progress: ProgressCallback = None,
) -> Path:
    """
    Apply the Lee speckle filter to a SAR raster (all bands).

    Algorithm
    ---------
    For each pixel:
        w          = local_var / (local_var + noise_var)
        filtered   = local_mean + w * (pixel - local_mean)

    where:
        noise_var  = (image_mean / ENL)²   (SAR multiplicative noise model)
        local_mean = mean of k×k neighbourhood
        local_var  = variance of k×k neighbourhood

    Tiles are processed with `kernel_size // 2` overlap padding to avoid
    edge discontinuities between adjacent output tiles.

    Parameters
    ----------
    src_path    : Input SAR raster path.
    out_path    : Output filtered raster path.
    kernel_size : Square filter window size (must be odd, ≥ 3).
    enl         : Equivalent Number of Looks. Higher → less filtering.
                  Typical values: 1 (single-look), 2–4 (multi-look).
    block_size  : Tile size for windowed processing.
    progress    : Optional callback (current_tile: int, total_tiles: int).

    Returns
    -------
    Path of the written output raster.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if kernel_size < 3:
        raise ValueError(f"kernel_size must be ≥ 3, got {kernel_size}")

    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pad = kernel_size // 2

    with rasterio.open(src_path) as ds:
        profile = ds.profile.copy()
        nodata  = ds.nodata
        height  = ds.height
        width   = ds.width
        n_bands = ds.count
        windows = list(iter_windows(ds, block_size))
        n_tiles = len(windows)

    # Estimate per-band image mean from a central window (fast, good enough for noise model)
    band_means = _estimate_band_means(src_path, nodata)

    profile.update(
        driver="GTiff",
        dtype="float32",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
    )

    with rasterio.open(src_path) as src, rasterio.open(out_path, "w", **profile) as dst:
        for i, window in enumerate(windows):
            # Expand window bounds by `pad` pixels on each side
            col_off = max(0, window.col_off - pad)
            row_off = max(0, window.row_off - pad)
            col_end = min(width,  window.col_off + window.width  + pad)
            row_end = min(height, window.row_off + window.height + pad)

            padded_win = rasterio.windows.Window(
                col_off=col_off,
                row_off=row_off,
                width=col_end  - col_off,
                height=row_end - row_off,
            )

            # Inner tile position within the padded read
            inner_col = window.col_off - col_off
            inner_row = window.row_off - row_off
            inner_h   = window.height
            inner_w   = window.width

            padded_data = src.read(window=padded_win).astype(np.float64)  # (bands, pH, pW)

            out_data = np.empty((n_bands, inner_h, inner_w), dtype=np.float32)

            for b in range(n_bands):
                tile = padded_data[b]

                # Replace nodata with band mean to avoid filter contamination
                if nodata is not None:
                    nodata_mask = tile == nodata
                    tile = tile.copy()
                    tile[nodata_mask] = band_means[b]

                local_mean, local_var = _local_mean_var(tile, kernel_size)

                # Noise variance from SAR multiplicative model
                noise_var = (band_means[b] / max(enl, 1e-6)) ** 2

                # Lee weight — avoid division by zero
                denom  = local_var + noise_var
                weight = np.where(denom > 0, local_var / denom, 0.0)

                filtered = local_mean + weight * (tile - local_mean)

                # Extract inner (non-padded) region
                inner = filtered[inner_row: inner_row + inner_h,
                                 inner_col: inner_col + inner_w]

                # Restore nodata pixels
                if nodata is not None:
                    nodata_inner = nodata_mask[inner_row: inner_row + inner_h,
                                              inner_col: inner_col + inner_w]
                    inner[nodata_inner] = nodata

                out_data[b] = inner.astype(np.float32)

            dst.write(out_data, window=window)

            if progress is not None:
                progress(i + 1, n_tiles)

    return out_path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_band_means(
    src_path: Path,
    nodata: Optional[float],
    n_windows: int = 4,
) -> np.ndarray:
    """
    Estimate per-band mean from the first `n_windows` tiles.
    Used for Lee filter noise variance estimation.
    Returns 1-D array of length band_count.
    """
    with rasterio.open(src_path) as ds:
        n_bands = ds.count
        sums    = np.zeros(n_bands, dtype=np.float64)
        counts  = np.zeros(n_bands, dtype=np.float64)

        for j, window in enumerate(iter_windows(ds, DEFAULT_BLOCK)):
            if j >= n_windows:
                break
            data = ds.read(window=window).astype(np.float64)
            for b in range(n_bands):
                band = data[b]
                mask = np.isfinite(band)
                if nodata is not None:
                    mask &= band != nodata
                sums[b]   += band[mask].sum()
                counts[b] += mask.sum()

    means = np.where(counts > 0, sums / counts, 1.0)
    # Guard against zero means (e.g. all-zero SAR bands)
    means = np.where(means > 0, means, 1.0)
    return means


def count_windows(src_path: Path, block_size: int = DEFAULT_BLOCK) -> int:
    """Return the total number of processing tiles for a raster. Used by UI progress bars."""
    with rasterio.open(src_path) as ds:
        return math.ceil(ds.height / block_size) * math.ceil(ds.width / block_size)
