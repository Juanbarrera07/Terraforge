"""
Windowed / chunked raster I/O utilities.

Design rules
------------
- Never load a full raster array into memory.  All pixel reads go through
  read_window() or iter_windows().
- Metadata is always extracted via get_meta() — zero pixel reads.
- write_raster() always outputs tiled, deflate-compressed GeoTIFFs.
- read_overview() is the only function allowed to load a full (decimated) array,
  and only for preview/thumbnail purposes.

Typical workflow
----------------
    meta = get_meta(path)
    with rasterio.open(path) as ds:
        for win in iter_windows(ds):
            arr = read_window(ds, win)   # shape (bands, rows, cols)
            ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

import numpy as np
import rasterio
import rasterio.enums
from rasterio.windows import Window

DEFAULT_BLOCK: int = 512  # tile side in pixels


# ── Metadata ─────────────────────────────────────────────────────────────────

def get_meta(path: str | Path) -> dict:
    """
    Return dataset metadata without reading any pixel data.

    Keys
    ----
    driver, dtype, crs, crs_epsg, transform, width, height,
    count, bounds, res, nodata, tags
    """
    with rasterio.open(path) as ds:
        epsg = None
        if ds.crs:
            try:
                epsg = ds.crs.to_epsg()
            except Exception:
                pass
        return {
            "driver":    ds.driver,
            "dtype":     str(ds.dtypes[0]),
            "crs":       ds.crs,
            "crs_epsg":  epsg,
            "transform": ds.transform,
            "width":     ds.width,
            "height":    ds.height,
            "count":     ds.count,
            "bounds":    ds.bounds,
            "res":       ds.res,      # (xres, yres) in CRS units
            "nodata":    ds.nodata,
            "tags":      ds.tags(),
        }


def get_gsd_m(path: str | Path) -> Optional[float]:
    """
    Return the pixel size in metres for projected CRS, None for geographic CRS.
    Uses the absolute x pixel size from the affine transform.
    """
    with rasterio.open(path) as ds:
        if ds.crs and ds.crs.is_projected:
            return float(abs(ds.transform.a))
    return None


# ── Windowed reads ────────────────────────────────────────────────────────────

def iter_windows(
    ds: rasterio.DatasetReader,
    block_size: int = DEFAULT_BLOCK,
) -> Generator[Window, None, None]:
    """
    Yield non-overlapping tile windows that cover the full dataset extent.
    Windows at the edges are clipped to the dataset boundary.
    """
    for row_off in range(0, ds.height, block_size):
        row_h = min(block_size, ds.height - row_off)
        for col_off in range(0, ds.width, block_size):
            col_w = min(block_size, ds.width - col_off)
            yield Window(col_off=col_off, row_off=row_off,
                         width=col_w, height=row_h)


def read_window(
    ds: rasterio.DatasetReader,
    window: Window,
    bands: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Read a raster window and return an ndarray of shape (bands, rows, cols).

    Parameters
    ----------
    bands : 1-indexed list of band indices; None reads all bands.
    """
    if bands is None:
        return ds.read(window=window)
    return ds.read(bands, window=window)


def read_overview(
    path: str | Path,
    max_px: int = 512,
) -> np.ndarray:
    """
    Read a decimated thumbnail of shape (bands, rows, cols) capped at max_px
    on the longest side.  Uses existing overviews when available.

    Intended for preview display only — do NOT use for analysis.
    """
    with rasterio.open(path) as ds:
        scale = max(ds.height, ds.width) / max_px
        out_h = max(1, int(ds.height / scale))
        out_w = max(1, int(ds.width / scale))
        return ds.read(
            out_shape=(ds.count, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )


# ── Spatial helpers ───────────────────────────────────────────────────────────

def compute_overlap_pct(
    bounds_a: rasterio.coords.BoundingBox,
    bounds_b: rasterio.coords.BoundingBox,
) -> float:
    """
    Return the overlap as a percentage of the *smaller* dataset's area.

    Both BoundingBox objects must be in the same CRS.
    Returns 0.0 when there is no overlap.
    """
    x_min = max(bounds_a.left,   bounds_b.left)
    x_max = min(bounds_a.right,  bounds_b.right)
    y_min = max(bounds_a.bottom, bounds_b.bottom)
    y_max = min(bounds_a.top,    bounds_b.top)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    inter  = (x_max - x_min) * (y_max - y_min)
    area_a = (bounds_a.right - bounds_a.left) * (bounds_a.top - bounds_a.bottom)
    area_b = (bounds_b.right - bounds_b.left) * (bounds_b.top - bounds_b.bottom)
    smaller = min(area_a, area_b)

    return 100.0 * inter / smaller if smaller > 0 else 0.0


# ── Write helpers ─────────────────────────────────────────────────────────────

def write_raster(
    path: str | Path,
    data: np.ndarray,
    ref_meta: dict,
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
) -> Path:
    """
    Write an ndarray to a tiled, deflate-compressed GeoTIFF.

    Parameters
    ----------
    data      : Shape (bands, rows, cols) or (rows, cols) for single-band.
    ref_meta  : Metadata dict from get_meta() — used for CRS, transform.
    dtype     : Override output dtype; defaults to data.dtype.
    nodata    : Override nodata value.

    Returns
    -------
    Path of written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]  # (1, rows, cols)

    meta = {
        "driver":     "GTiff",
        "crs":        ref_meta["crs"],
        "transform":  ref_meta["transform"],
        "count":      data.shape[0],
        "height":     data.shape[1],
        "width":      data.shape[2],
        "dtype":      dtype or str(data.dtype),
        "nodata":     nodata if nodata is not None else ref_meta.get("nodata"),
        "compress":   "deflate",
        "predictor":  2,
        "tiled":      True,
        "blockxsize": DEFAULT_BLOCK,
        "blockysize": DEFAULT_BLOCK,
    }

    with rasterio.open(out, "w", **meta) as dst:
        dst.write(data)

    return out
