"""
Tests for pipeline/preprocess.py (Phase 3).

Strategy
--------
All tests use small synthetic rasters (≤ 64×64) written to tmp_path
so the full windowed pipeline runs end-to-end without touching real data.

Covered
-------
_local_mean_var     : correctness of integral-image local statistics
_find_dark_object_values : per-band dark object scan
dos1_atmospheric_correction : two-pass algorithm, output properties
lee_speckle_filter  : output shape/CRS, variance reduction, edge handling
_estimate_band_means: mean estimation from sampled windows
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from pipeline.preprocess import (
    _estimate_band_means,
    _find_dark_object_values,
    _local_mean_var,
    dos1_atmospheric_correction,
    lee_speckle_filter,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _write_raster(
    path: Path,
    data: np.ndarray,          # (bands, H, W)
    res: float = 10.0,
    crs_epsg: int = 32633,
    dtype: str = "float32",
    nodata: float | None = None,
) -> Path:
    """Write a synthetic raster from an ndarray. Returns path."""
    bands, H, W = data.shape
    transform = from_origin(500_000.0, 5_000_000.0 + H * res, res, res)
    profile = {
        "driver":    "GTiff",
        "dtype":     dtype,
        "width":     W,
        "height":    H,
        "count":     bands,
        "crs":       CRS.from_epsg(crs_epsg),
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data.astype(dtype))
    return path


# ── _local_mean_var ───────────────────────────────────────────────────────────

class TestLocalMeanVar:
    def test_uniform_array_var_is_zero(self):
        arr = np.ones((20, 20), dtype=np.float64) * 5.0
        mean, var = _local_mean_var(arr, k=5)
        np.testing.assert_allclose(mean, 5.0, atol=1e-10)
        np.testing.assert_allclose(var,  0.0, atol=1e-10)

    def test_output_shape_matches_input(self):
        arr = np.random.default_rng(0).random((30, 40))
        mean, var = _local_mean_var(arr, k=7)
        assert mean.shape == (30, 40)
        assert var.shape  == (30, 40)

    def test_known_mean(self):
        # 5×5 array all equal to row index — local 3×3 mean at interior is exact
        arr = np.tile(np.arange(5, dtype=np.float64), (5, 1)).T  # col values = col index
        mean, _ = _local_mean_var(arr, k=3)
        # Centre pixel (2,2): window cols [1,2,3] → mean = 2.0
        assert abs(mean[2, 2] - 2.0) < 1e-10

    def test_variance_non_negative(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((50, 50))
        _, var = _local_mean_var(arr, k=5)
        assert (var >= 0).all()

    def test_larger_window_smoother_mean(self):
        # For white noise, a larger averaging window produces a local_mean map
        # with lower spatial variance (central limit theorem: Var[mean] ∝ 1/n).
        # Test the variance OF the local_mean values, not local_var itself.
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((64, 64))
        mean_small, _ = _local_mean_var(arr, k=3)
        mean_large, _ = _local_mean_var(arr, k=9)
        # Larger window → local means vary less spatially
        assert float(np.var(mean_large)) < float(np.var(mean_small))


# ── _find_dark_object_values ──────────────────────────────────────────────────

class TestFindDarkObjectValues:
    def test_returns_per_band_minimum(self, tmp_path):
        # Band 1 minimum non-zero = 2.0, band 2 = 5.0
        data = np.array([
            [[0, 2, 10],
             [4, 6,  8]],   # band 1
            [[5, 9, 15],
             [7, 11, 3]],   # band 2 — min is 3.0
        ], dtype="float32").reshape(2, 2, 3)
        path = _write_raster(tmp_path / "src.tif", data)

        dark = _find_dark_object_values(path, nodata=None, block_size=512)
        assert dark[0] == pytest.approx(2.0)
        assert dark[1] == pytest.approx(3.0)

    def test_excludes_nodata(self, tmp_path):
        # nodata = -9999; the real minimum non-zero is 5.0
        data = np.array([[[-9999, 5, 10]]], dtype="float32")
        path = _write_raster(tmp_path / "src.tif", data, nodata=-9999.0)
        dark = _find_dark_object_values(path, nodata=-9999.0, block_size=512)
        assert dark[0] == pytest.approx(5.0)

    def test_excludes_zeros(self, tmp_path):
        data = np.array([[[0, 0, 7, 3]]], dtype="float32")
        path = _write_raster(tmp_path / "src.tif", data)
        dark = _find_dark_object_values(path, nodata=None, block_size=512)
        assert dark[0] == pytest.approx(3.0)

    def test_all_zero_band_returns_zero(self, tmp_path):
        data = np.zeros((1, 4, 4), dtype="float32")
        path = _write_raster(tmp_path / "src.tif", data)
        dark = _find_dark_object_values(path, nodata=None, block_size=512)
        assert dark[0] == pytest.approx(0.0)


# ── dos1_atmospheric_correction ───────────────────────────────────────────────

class TestDos1AtmosphericCorrection:
    def test_output_file_created(self, tmp_path):
        data = np.random.default_rng(0).uniform(10, 100, (3, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        out  = tmp_path / "dos1.tif"
        result = dos1_atmospheric_correction(src, out, block_size=16)
        assert result == out
        assert out.exists()

    def test_preserves_shape_bands_crs(self, tmp_path):
        data = np.random.default_rng(1).uniform(5, 80, (4, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data, crs_epsg=32633)
        out  = tmp_path / "dos1.tif"
        dos1_atmospheric_correction(src, out, block_size=16)

        with rasterio.open(src) as s, rasterio.open(out) as d:
            assert d.count   == s.count
            assert d.width   == s.width
            assert d.height  == s.height
            assert d.crs.to_epsg() == s.crs.to_epsg()

    def test_no_negative_values_in_output(self, tmp_path):
        # Ensure clipping to 0 works correctly
        data = np.random.default_rng(2).uniform(1, 50, (2, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        out  = tmp_path / "dos1.tif"
        dos1_atmospheric_correction(src, out, block_size=16)

        with rasterio.open(out) as ds:
            arr = ds.read()
        assert (arr >= 0).all()

    def test_dark_object_becomes_near_zero(self, tmp_path):
        # Build band with known minimum = 10. After DOS1, minimum → 0.
        data = np.random.default_rng(3).uniform(10, 100, (1, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        out  = tmp_path / "dos1.tif"
        dos1_atmospheric_correction(src, out, block_size=16)

        with rasterio.open(out) as ds:
            result = ds.read(1)
        assert result.min() == pytest.approx(0.0, abs=1e-4)

    def test_nodata_pixels_unchanged(self, tmp_path):
        # nodata pixel should pass through as-is
        data = np.full((1, 4, 4), 50.0, dtype="float32")
        data[0, 0, 0] = -9999.0   # nodata pixel
        src = _write_raster(tmp_path / "src.tif", data, nodata=-9999.0)
        out = tmp_path / "dos1.tif"
        dos1_atmospheric_correction(src, out, block_size=16)

        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert arr[0, 0] == pytest.approx(-9999.0)

    def test_progress_callback_called(self, tmp_path):
        data = np.random.default_rng(4).uniform(1, 50, (2, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        out  = tmp_path / "dos1.tif"

        calls: list[tuple[int, int]] = []
        dos1_atmospheric_correction(src, out, block_size=16, progress=lambda c, t: calls.append((c, t)))

        assert len(calls) > 0
        # Last call's current == total
        assert calls[-1][0] == calls[-1][1]


# ── lee_speckle_filter ────────────────────────────────────────────────────────

class TestLeeSpeckleFilter:
    def test_output_file_created(self, tmp_path):
        rng  = np.random.default_rng(10)
        data = rng.uniform(0, 1, (2, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        out  = tmp_path / "lee.tif"
        result = lee_speckle_filter(src, out, kernel_size=5, block_size=16)
        assert result == out
        assert out.exists()

    def test_preserves_shape_bands_crs(self, tmp_path):
        rng  = np.random.default_rng(11)
        data = rng.uniform(0, 1, (2, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data, crs_epsg=32633)
        out  = tmp_path / "lee.tif"
        lee_speckle_filter(src, out, kernel_size=5, block_size=16)

        with rasterio.open(src) as s, rasterio.open(out) as d:
            assert d.count  == s.count
            assert d.width  == s.width
            assert d.height == s.height
            assert d.crs.to_epsg() == s.crs.to_epsg()

    def test_reduces_local_variance(self, tmp_path):
        """Filtered image must have lower mean local variance than input (speckle reduced)."""
        rng  = np.random.default_rng(12)
        # Simulate speckled SAR: correlated signal + multiplicative noise
        signal = np.ones((1, 64, 64), dtype="float32") * 0.5
        noise  = rng.gamma(1.0, 1.0, (1, 64, 64)).astype("float32")
        data   = signal * noise
        src    = _write_raster(tmp_path / "src.tif", data)
        out    = tmp_path / "lee.tif"
        lee_speckle_filter(src, out, kernel_size=7, enl=1.0, block_size=64)

        with rasterio.open(src) as s, rasterio.open(out) as d:
            src_arr = s.read(1).astype(np.float64)
            out_arr = d.read(1).astype(np.float64)

        src_var = float(np.var(src_arr))
        out_var = float(np.var(out_arr))
        assert out_var < src_var, (
            f"Lee filter did not reduce variance: input {src_var:.6f}, output {out_var:.6f}"
        )

    def test_even_kernel_raises(self, tmp_path):
        rng  = np.random.default_rng(13)
        data = rng.uniform(0, 1, (1, 16, 16)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        with pytest.raises(ValueError, match="odd"):
            lee_speckle_filter(src, tmp_path / "out.tif", kernel_size=4)

    def test_kernel_less_than_3_raises(self, tmp_path):
        rng  = np.random.default_rng(14)
        data = rng.uniform(0, 1, (1, 16, 16)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        with pytest.raises(ValueError, match="≥ 3"):
            lee_speckle_filter(src, tmp_path / "out.tif", kernel_size=1)

    def test_progress_callback_called(self, tmp_path):
        rng  = np.random.default_rng(15)
        data = rng.uniform(0, 1, (1, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        out  = tmp_path / "lee.tif"

        calls: list[tuple[int, int]] = []
        lee_speckle_filter(src, out, kernel_size=3, block_size=16,
                           progress=lambda c, t: calls.append((c, t)))

        assert len(calls) > 0
        assert calls[-1][0] == calls[-1][1]

    def test_nodata_pixels_restored(self, tmp_path):
        rng  = np.random.default_rng(16)
        data = rng.uniform(0.1, 1.0, (1, 16, 16)).astype("float32")
        data[0, 0, 0] = -9999.0
        src = _write_raster(tmp_path / "src.tif", data, nodata=-9999.0)
        out = tmp_path / "lee.tif"
        lee_speckle_filter(src, out, kernel_size=3, block_size=16)

        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert arr[0, 0] == pytest.approx(-9999.0)

    def test_no_nan_in_output(self, tmp_path):
        rng  = np.random.default_rng(17)
        data = rng.uniform(0, 1, (1, 32, 32)).astype("float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        out  = tmp_path / "lee.tif"
        lee_speckle_filter(src, out, kernel_size=7, block_size=16)

        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert not np.isnan(arr).any()


# ── _estimate_band_means ──────────────────────────────────────────────────────

class TestEstimateBandMeans:
    def test_returns_one_value_per_band(self, tmp_path):
        data = np.ones((3, 32, 32), dtype="float32") * np.array([1, 2, 3])[:, None, None]
        src  = _write_raster(tmp_path / "src.tif", data)
        means = _estimate_band_means(src, nodata=None)
        assert len(means) == 3

    def test_constant_band_mean(self, tmp_path):
        data = np.full((1, 32, 32), 7.5, dtype="float32")
        src  = _write_raster(tmp_path / "src.tif", data)
        means = _estimate_band_means(src, nodata=None)
        assert means[0] == pytest.approx(7.5, rel=0.05)

    def test_excludes_nodata_from_mean(self, tmp_path):
        data = np.full((1, 32, 32), 10.0, dtype="float32")
        data[0, :, :16] = -9999.0   # half nodata
        src  = _write_raster(tmp_path / "src.tif", data, nodata=-9999.0)
        means = _estimate_band_means(src, nodata=-9999.0)
        assert means[0] == pytest.approx(10.0, rel=0.05)
