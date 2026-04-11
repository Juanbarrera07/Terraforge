"""
Shared pytest fixtures for TerraForge tests.

All raster fixtures are created in-process using rasterio — no external files
required.  The `make_raster` factory covers the full range of CRS, resolution,
band count, and spatial extent combinations needed by all test modules.
"""
from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


# ── Raster factory ────────────────────────────────────────────────────────────

RasterFactory = Callable[..., Path]


@pytest.fixture
def make_raster(tmp_path: Path) -> RasterFactory:
    """
    Returns a factory function that creates synthetic GeoTIFFs in tmp_path.

    Usage
    -----
    path = make_raster(
        filename="sar.tif",
        crs_epsg=32633,
        res=10.0,
        bands=2,
        width=200,
        height=200,
        x_origin=500000.0,   # top-left easting
        y_origin=5000200.0,  # top-left northing
    )
    """
    def _factory(
        filename: str = "test.tif",
        crs_epsg: int | None = 32633,     # UTM zone 33N; None → no CRS
        res: float = 10.0,
        bands: int = 1,
        width: int = 200,
        height: int = 200,
        x_origin: float = 500_000.0,
        y_origin: float = 5_000_200.0,
        dtype: str = "float32",
        nodata: float = -9999.0,
        tags: dict | None = None,
    ) -> Path:
        path = tmp_path / filename
        transform = from_origin(x_origin, y_origin, res, res)
        data = np.random.default_rng(42).random((bands, height, width)).astype(dtype)

        profile: dict = {
            "driver":    "GTiff",
            "dtype":     dtype,
            "width":     width,
            "height":    height,
            "count":     bands,
            "transform": transform,
            "nodata":    nodata,
        }
        if crs_epsg is not None:
            profile["crs"] = CRS.from_epsg(crs_epsg)

        with rasterio.open(path, "w", **profile) as ds:
            ds.write(data)
            if tags:
                ds.update_tags(**tags)

        return path

    return _factory


# ── Layer dict builder (mirrors ingest.py output) ────────────────────────────

def make_layer(
    path: Path,
    layer_type: str = "optical",
    sensor: str = "Sentinel-2",
    acq_date: date | None = None,
) -> dict:
    """Build a synthetic layer dict as returned by ingest.ingest_file()."""
    from pipeline.raster_io import get_meta
    meta = get_meta(path)
    return {
        "path":       path,
        "filename":   path.name,
        "sensor":     sensor,
        "layer_type": layer_type,
        "date":       acq_date,
        "meta":       meta,
    }


# ── ZIP fixture helper ────────────────────────────────────────────────────────

def wrap_in_zip(tif_path: Path, zip_path: Path) -> Path:
    """Create a ZIP archive containing a single TIF. Returns zip_path."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tif_path, arcname=tif_path.name)
    return zip_path
