"""
Phase 6 — Post-processing and validation tests.

All rasters are created in-process; no external files required.
The make_raster conftest fixture is reused for generic float rasters;
helper functions build classified (int16) rasters directly.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from pipeline.postprocess import (
    AccuracyResult,
    ClassAreaResult,
    DriftResult,
    _majority_filter,
    assess_accuracy_from_points,
    check_drift,
    compute_class_areas,
    confidence_filter,
    median_smooth,
    morphological_close,
    run_postprocess_chain,
    sieve_filter,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_classified(
    tmp_path:   Path,
    data:       np.ndarray,          # 2-D int array (rows, cols)
    res:        float = 10.0,
    nodata:     int   = -1,
    crs_epsg:   int   = 32633,
    filename:   str   = "classified.tif",
) -> Path:
    """Write a single-band int16 classified raster to tmp_path."""
    path      = tmp_path / filename
    height, width = data.shape
    transform = from_origin(500_000.0, 5_000_200.0, res, res)
    profile   = {
        "driver":    "GTiff",
        "dtype":     "int16",
        "width":     width,
        "height":    height,
        "count":     1,
        "transform": transform,
        "nodata":    nodata,
        "crs":       CRS.from_epsg(crs_epsg) if crs_epsg else None,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data.astype(np.int16)[np.newaxis, :, :])
    return path


def _make_csv(
    tmp_path: Path,
    rows:     list[dict],
    filename: str = "ref.csv",
) -> Path:
    """Write a reference CSV with arbitrary columns."""
    path = tmp_path / filename
    if not rows:
        path.write_text("lat,lon,class\n")
        return path
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _pixel_to_wgs84(
    col: int, row: int,
    transform,
    src_epsg: int = 32633,
) -> tuple[float, float]:
    """Return (lat, lon) in WGS-84 for the *centre* of a pixel."""
    from pyproj import Transformer
    # pixel centre in projected CRS
    x = transform.c + (col + 0.5) * transform.a
    y = transform.f + (row + 0.5) * transform.e   # e is negative

    tr = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lon, lat = tr.transform(x, y)
    return lat, lon


# ── compute_class_areas ───────────────────────────────────────────────────────

class TestComputeClassAreas:

    def test_pixel_counts_correct(self, tmp_path: Path) -> None:
        data = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int16)
        path = _make_classified(tmp_path, data)
        result = compute_class_areas(path, nodata=-1, pixel_res_m=10.0)
        assert result.pixel_counts[1] == 3
        assert result.pixel_counts[2] == 3

    def test_nodata_excluded(self, tmp_path: Path) -> None:
        data = np.array([[1, -1, 2], [-1, 2, 2]], dtype=np.int16)
        path = _make_classified(tmp_path, data)
        result = compute_class_areas(path, nodata=-1, pixel_res_m=10.0)
        assert -1 not in result.pixel_counts
        assert sum(result.pixel_counts.values()) == 4

    def test_areas_ha_correct(self, tmp_path: Path) -> None:
        # 4 pixels of class 1 at 10 m resolution → 4 * 100 m² = 400 m² = 0.04 ha
        data = np.ones((2, 2), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        result = compute_class_areas(path, nodata=-1, pixel_res_m=10.0)
        assert math.isclose(result.areas_ha[1], 0.04, rel_tol=1e-9)

    def test_total_area_ha_is_sum(self, tmp_path: Path) -> None:
        data = np.array([[1, 2], [3, 3]], dtype=np.int16)
        path = _make_classified(tmp_path, data)
        result = compute_class_areas(path, nodata=-1, pixel_res_m=10.0)
        assert math.isclose(
            result.total_area_ha,
            sum(result.areas_ha.values()),
            rel_tol=1e-9,
        )

    def test_auto_pixel_res_from_crs(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        path = _make_classified(tmp_path, data, res=20.0, crs_epsg=32633)
        result = compute_class_areas(path, nodata=-1)   # no explicit pixel_res_m
        assert result.pixel_res_m == pytest.approx(20.0)

    def test_geographic_crs_raises_without_explicit_res(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        path = _make_classified(tmp_path, data, crs_epsg=4326)
        with pytest.raises(ValueError, match="pixel_res_m must be supplied"):
            compute_class_areas(path, nodata=-1)

    def test_class_ids_sorted(self, tmp_path: Path) -> None:
        data = np.array([[3, 1, 2]], dtype=np.int16)
        path = _make_classified(tmp_path, data)
        result = compute_class_areas(path, nodata=-1, pixel_res_m=10.0)
        assert result.class_ids == [1, 2, 3]

    def test_all_nodata_returns_empty_counts(self, tmp_path: Path) -> None:
        data = np.full((5, 5), -1, dtype=np.int16)
        path = _make_classified(tmp_path, data)
        result = compute_class_areas(path, nodata=-1, pixel_res_m=10.0)
        assert result.pixel_counts == {}
        assert result.total_area_ha == pytest.approx(0.0)


# ── sieve_filter ──────────────────────────────────────────────────────────────

class TestSieveFilter:

    @pytest.fixture(autouse=True)
    def _require_gdal(self) -> None:
        pytest.importorskip("osgeo.gdal", reason="GDAL not available")

    def test_threshold_pixels_calculation(self, tmp_path: Path) -> None:
        """1 ha at 10 m resolution = 100 pixels."""
        data = np.ones((50, 50), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        _, thresh = sieve_filter(path, tmp_path / "out.tif",
                                 mmu_ha=1.0, pixel_res_m=10.0)
        assert thresh == 100

    def test_threshold_pixels_ceil(self, tmp_path: Path) -> None:
        """0.5 ha at 10 m = 50 pixels; 0.3 ha at 10 m → ceil(30) = 30."""
        data = np.ones((20, 20), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        _, thresh = sieve_filter(path, tmp_path / "out.tif",
                                 mmu_ha=0.3, pixel_res_m=10.0)
        assert thresh == 30

    def test_output_file_created(self, tmp_path: Path) -> None:
        data = np.ones((20, 20), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        out, _ = sieve_filter(path, tmp_path / "sieved.tif",
                               mmu_ha=0.001, pixel_res_m=10.0)
        assert out.exists()

    def test_small_patch_removed(self, tmp_path: Path) -> None:
        """A 2×2 patch of class 2 in a field of class 1 should be absorbed
        when mmu_ha is set above the patch size.

        10 m pixels → each pixel = (10²)/10000 = 0.01 ha.
        The patch has 4 pixels = 0.04 ha.
        mmu_ha=0.05 → threshold = ceil(0.05/0.01) = 5 pixels > 4 → patch removed.
        """
        data = np.ones((20, 20), dtype=np.int16)
        data[9:11, 9:11] = 2       # 4-pixel (2×2) patch
        path = _make_classified(tmp_path, data)
        out, thresh = sieve_filter(path, tmp_path / "sieved.tif",
                                   mmu_ha=0.05, pixel_res_m=10.0)
        assert thresh == 5
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert 2 not in arr


# ── morphological_close ───────────────────────────────────────────────────────

class TestMorphologicalClose:

    def test_even_kernel_raises(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        with pytest.raises(ValueError, match="odd integer"):
            morphological_close(path, tmp_path / "out.tif", kernel_size=4)

    def test_kernel_size_1_raises(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        with pytest.raises(ValueError, match="odd integer"):
            morphological_close(path, tmp_path / "out.tif", kernel_size=1)

    def test_kernel_size_zero_raises(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        with pytest.raises(ValueError, match="odd integer"):
            morphological_close(path, tmp_path / "out.tif", kernel_size=0)

    def test_float_kernel_raises(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        with pytest.raises(ValueError, match="odd integer"):
            morphological_close(path, tmp_path / "out.tif", kernel_size=3.0)  # type: ignore[arg-type]

    def test_output_file_created(self, tmp_path: Path) -> None:
        data = np.ones((20, 20), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        out = morphological_close(path, tmp_path / "closed.tif", kernel_size=3)
        assert out.exists()

    def test_uniform_raster_unchanged(self, tmp_path: Path) -> None:
        data = np.full((20, 20), 1, dtype=np.int16)
        path = _make_classified(tmp_path, data)
        out  = morphological_close(path, tmp_path / "closed.tif", kernel_size=3)
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert np.all(arr == 1)

    def test_isolated_pixel_filled(self, tmp_path: Path) -> None:
        """A single isolated pixel surrounded by another class should be
        overwritten by the majority (closing fills holes)."""
        data = np.full((11, 11), 1, dtype=np.int16)
        data[5, 5] = 2   # lone different pixel
        path = _make_classified(tmp_path, data, nodata=-1)
        out  = morphological_close(path, tmp_path / "closed.tif", kernel_size=3)
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert arr[5, 5] == 1


# ── _majority_filter (internal) ───────────────────────────────────────────────

class TestMajorityFilter:

    def test_uniform_tile_unchanged(self) -> None:
        tile   = np.ones((5, 5), dtype=np.int32)
        result = _majority_filter(tile, k=3, nodata=-1)
        assert np.all(result == 1)

    def test_all_nodata_preserved(self) -> None:
        tile   = np.full((5, 5), -1, dtype=np.int32)
        result = _majority_filter(tile, k=3, nodata=-1)
        assert np.all(result == -1)

    def test_single_outlier_replaced_by_majority(self) -> None:
        tile      = np.full((5, 5), 1, dtype=np.int32)
        tile[2, 2] = 99
        result    = _majority_filter(tile, k=3, nodata=-1)
        # Neighbourhood of (2,2) has 8 ones and 1×99 → majority is 1
        assert result[2, 2] == 1


# ── assess_accuracy_from_points ───────────────────────────────────────────────

class TestAssessAccuracyFromPoints:

    @pytest.fixture(autouse=True)
    def _require_pyproj(self) -> None:
        pytest.importorskip("pyproj", reason="pyproj not available")

    def _build_scenario(
        self,
        tmp_path: Path,
        class_data: np.ndarray,
        pixel_cols: list[int],
        pixel_rows: list[int],
        ref_classes: list[int],
    ) -> tuple[Path, Path]:
        """
        Create a classified raster and a matching reference CSV.

        pixel_cols/rows are 0-indexed pixel addresses within the raster.
        ref_classes are the ground-truth labels written to the CSV (may differ
        from the raster values to simulate errors).
        """
        raster_path = _make_classified(tmp_path, class_data, res=10.0, nodata=-1)

        with rasterio.open(raster_path) as ds:
            transform = ds.transform
            epsg      = ds.crs.to_epsg()

        rows_csv = []
        for col, row, cls in zip(pixel_cols, pixel_rows, ref_classes):
            lat, lon = _pixel_to_wgs84(col, row, transform, src_epsg=epsg)
            rows_csv.append({"lat": lat, "lon": lon, "class": cls})

        csv_path = _make_csv(tmp_path, rows_csv)
        return raster_path, csv_path

    def test_perfect_accuracy(self, tmp_path: Path) -> None:
        data = np.array([[1, 1, 2, 2],
                         [1, 1, 2, 2],
                         [3, 3, 4, 4],
                         [3, 3, 4, 4]], dtype=np.int16)
        # Reference points match raster exactly
        raster, csv_path = self._build_scenario(
            tmp_path, data,
            pixel_cols  = [0, 2, 0, 2],
            pixel_rows  = [0, 0, 2, 2],
            ref_classes = [1, 2, 3, 4],
        )
        result = assess_accuracy_from_points(raster, csv_path)
        assert result.oa == pytest.approx(1.0)
        assert result.n_valid == 4
        assert result.n_discarded == 0

    def test_partial_accuracy(self, tmp_path: Path) -> None:
        data = np.array([[1, 2], [1, 2]], dtype=np.int16)
        raster, csv_path = self._build_scenario(
            tmp_path, data,
            pixel_cols  = [0, 1],
            pixel_rows  = [0, 0],
            ref_classes = [1, 1],   # second point wrong (raster=2, ref=1)
        )
        result = assess_accuracy_from_points(raster, csv_path)
        assert result.oa == pytest.approx(0.5)
        assert result.n_valid == 2

    def test_out_of_bounds_point_discarded(self, tmp_path: Path) -> None:
        """A point that reprojects to coordinates outside the raster extent
        must be counted as out_of_bounds and not crash the function."""
        data = np.ones((5, 5), dtype=np.int16)
        raster_path = _make_classified(tmp_path, data, filename="cls2.tif")

        with rasterio.open(raster_path) as ds:
            transform = ds.transform
            epsg      = ds.crs.to_epsg()

        # Two valid in-extent points (need ≥2 valid for no ValueError)
        lat1, lon1 = _pixel_to_wgs84(1, 1, transform, src_epsg=epsg)
        lat2, lon2 = _pixel_to_wgs84(2, 2, transform, src_epsg=epsg)
        # One point guaranteed out of extent: WGS-84 (0, -170) reprojects to
        # a very large easting far outside the small 5×5 test raster.
        csv_path = _make_csv(tmp_path, [
            {"lat": lat1, "lon": lon1,  "class": 1},
            {"lat": lat2, "lon": lon2,  "class": 1},
            {"lat": 0.0,  "lon": -170.0,"class": 1},
        ])
        result = assess_accuracy_from_points(raster_path, csv_path)
        assert result.n_points == 3
        assert result.n_valid  == 2
        assert result.n_discarded == 1
        assert result.discard_reasons["out_of_bounds"] == 1

    def test_nodata_pixel_discarded(self, tmp_path: Path) -> None:
        data = np.array([[1, -1, 1],
                         [1,  1, 1]], dtype=np.int16)
        raster_path = _make_classified(tmp_path, data, filename="nd.tif")
        with rasterio.open(raster_path) as ds:
            transform = ds.transform
            epsg      = ds.crs.to_epsg()
        # pixel (1,0) is nodata=-1
        lat_nd, lon_nd = _pixel_to_wgs84(1, 0, transform, src_epsg=epsg)
        lat_ok, lon_ok = _pixel_to_wgs84(0, 0, transform, src_epsg=epsg)
        lat_ok2, lon_ok2 = _pixel_to_wgs84(2, 0, transform, src_epsg=epsg)
        csv_path = _make_csv(tmp_path, [
            {"lat": lat_nd,  "lon": lon_nd,  "class": 1},
            {"lat": lat_ok,  "lon": lon_ok,  "class": 1},
            {"lat": lat_ok2, "lon": lon_ok2, "class": 1},
        ])
        result = assess_accuracy_from_points(raster_path, csv_path, nodata=-1)
        assert result.discard_reasons["nodata_pixel"] == 1
        assert result.n_valid == 2

    def test_missing_fields_discarded(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        raster_path = _make_classified(tmp_path, data, filename="mf.tif")
        with rasterio.open(raster_path) as ds:
            transform = ds.transform
            epsg      = ds.crs.to_epsg()
        lat, lon = _pixel_to_wgs84(2, 2, transform, src_epsg=epsg)
        lat2, lon2 = _pixel_to_wgs84(3, 3, transform, src_epsg=epsg)
        csv_path = _make_csv(tmp_path, [
            {"lat": lat,  "lon": lon,  "class": 1},
            {"lat": lat2, "lon": lon2, "class": 1},
            {"lat": "",   "lon": "",   "class": 1},   # bad coords → missing_fields
        ])
        result = assess_accuracy_from_points(raster_path, csv_path)
        assert result.discard_reasons["missing_fields"] == 1
        assert result.n_points == 3
        assert result.n_valid  == 2

    def test_fewer_than_2_valid_raises(self, tmp_path: Path) -> None:
        data = np.ones((5, 5), dtype=np.int16)
        raster_path = _make_classified(tmp_path, data, filename="fv.tif")
        csv_path = _make_csv(tmp_path, [{"lat": 999.0, "lon": 999.0, "class": 1}])
        with pytest.raises(ValueError, match="at least 2"):
            assess_accuracy_from_points(raster_path, csv_path)

    def test_result_fields_populated(self, tmp_path: Path) -> None:
        data = np.ones((10, 10), dtype=np.int16)
        raster_path = _make_classified(tmp_path, data, filename="pop.tif")
        with rasterio.open(raster_path) as ds:
            transform = ds.transform
            epsg      = ds.crs.to_epsg()
        lats_lons = [_pixel_to_wgs84(c, r, transform, src_epsg=epsg)
                     for c, r in [(1, 1), (2, 2), (3, 3)]]
        csv_path = _make_csv(tmp_path, [
            {"lat": ll[0], "lon": ll[1], "class": 1} for ll in lats_lons
        ])
        result = assess_accuracy_from_points(raster_path, csv_path)
        assert isinstance(result, AccuracyResult)
        assert result.n_points == 3
        assert result.n_valid  == 3
        assert isinstance(result.confusion_matrix, np.ndarray)
        assert result.oa == pytest.approx(1.0)


# ── check_drift ───────────────────────────────────────────────────────────────

class TestCheckDrift:

    def _make_area_result(
        self,
        areas_ha: dict[int, float],
        pixel_res_m: float = 10.0,
    ) -> ClassAreaResult:
        counts = {k: int(v * 10_000 / (pixel_res_m ** 2)) for k, v in areas_ha.items()}
        return ClassAreaResult(
            class_ids     = sorted(areas_ha.keys()),
            pixel_counts  = counts,
            areas_ha      = areas_ha,
            total_area_ha = sum(areas_ha.values()),
            pixel_res_m   = pixel_res_m,
        )

    def test_no_drift_no_flags(self) -> None:
        curr = self._make_area_result({1: 100.0, 2: 200.0})
        prev = self._make_area_result({1: 100.0, 2: 200.0})
        result = check_drift(curr, prev, drift_alert_pct=20.0)
        assert result.flagged_classes == []

    def test_flagged_above_threshold(self) -> None:
        curr = self._make_area_result({1: 130.0, 2: 200.0})
        prev = self._make_area_result({1: 100.0, 2: 200.0})
        result = check_drift(curr, prev, drift_alert_pct=20.0)
        assert 1 in result.flagged_classes
        assert 2 not in result.flagged_classes

    def test_pct_change_sign_and_value(self) -> None:
        curr = self._make_area_result({1: 80.0})
        prev = self._make_area_result({1: 100.0})
        result = check_drift(curr, prev, drift_alert_pct=30.0)
        assert result.pct_change[1] == pytest.approx(-20.0)

    def test_threshold_is_inclusive(self) -> None:
        """Exactly at threshold → flagged (>= comparison)."""
        curr = self._make_area_result({1: 120.0})
        prev = self._make_area_result({1: 100.0})
        result = check_drift(curr, prev, drift_alert_pct=20.0)
        assert 1 in result.flagged_classes

    def test_class_absent_in_previous_skipped(self) -> None:
        curr = self._make_area_result({1: 100.0, 5: 50.0})
        prev = self._make_area_result({1: 100.0})
        result = check_drift(curr, prev, drift_alert_pct=20.0)
        assert 5 not in result.pct_change   # class 5 new → not compared

    def test_zero_previous_area_flagged(self) -> None:
        """Class that was absent (0 ha) but now present → flagged as inf change."""
        curr = self._make_area_result({1: 50.0})
        prev = self._make_area_result({1: 0.0})
        result = check_drift(curr, prev, drift_alert_pct=20.0)
        assert 1 in result.flagged_classes
        assert math.isinf(result.pct_change[1])

    def test_uses_config_threshold(self) -> None:
        """When drift_alert_pct is None, load from cfg dict."""
        curr = self._make_area_result({1: 125.0})
        prev = self._make_area_result({1: 100.0})
        result = check_drift(curr, prev, drift_alert_pct=None,
                             cfg={"drift_alert_pct": 20})
        assert 1 in result.flagged_classes
        assert result.drift_alert_pct == pytest.approx(20.0)


# ── Helpers for confidence tests ──────────────────────────────────────────────

def _make_confidence(
    tmp_path:  Path,
    data:      np.ndarray,      # 2-D float array (rows, cols)
    res:       float = 10.0,
    nodata:    float = -9999.0,
    crs_epsg:  int   = 32633,
    filename:  str   = "confidence.tif",
) -> Path:
    """Write a single-band float32 confidence raster to tmp_path."""
    path      = tmp_path / filename
    height, width = data.shape
    transform = from_origin(500_000.0, 5_000_200.0, res, res)
    profile   = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     width,
        "height":    height,
        "count":     1,
        "transform": transform,
        "nodata":    nodata,
        "crs":       CRS.from_epsg(crs_epsg) if crs_epsg else None,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data.astype(np.float32)[np.newaxis, :, :])
    return path


# ── confidence_filter ─────────────────────────────────────────────────────────

class TestConfidenceFilter:

    @pytest.fixture(autouse=True)
    def _require_scipy(self) -> None:
        pytest.importorskip("scipy", reason="scipy not available")

    def test_low_confidence_pixel_replaced(self, tmp_path: Path) -> None:
        """A single low-confidence pixel in a field of class 1 should be
        replaced by the local median (which is also class 1)."""
        data_cls = np.ones((15, 15), dtype=np.int16)
        data_cls[7, 7] = 2                              # lone class-2 pixel
        cls_path = _make_classified(tmp_path, data_cls, nodata=-1)

        # Confidence: 0.9 everywhere except the class-2 pixel (0.2 < threshold)
        data_conf = np.full((15, 15), 0.9, dtype=np.float32)
        data_conf[7, 7] = 0.2
        conf_path = _make_confidence(tmp_path, data_conf)

        out_path = tmp_path / "filtered.tif"
        confidence_filter(cls_path, conf_path, threshold=0.6, out_path=out_path)

        assert out_path.exists()
        with rasterio.open(out_path) as ds:
            result = ds.read(1)

        # Low-confidence pixel replaced with median of 5×5 neighbourhood (all 1s)
        assert result[7, 7] == 1

    def test_high_confidence_pixels_unchanged(self, tmp_path: Path) -> None:
        """Pixels with confidence >= threshold must not be modified."""
        data_cls  = np.array([[1, 2], [3, 4]], dtype=np.int16)
        data_conf = np.full((2, 2), 0.95, dtype=np.float32)
        cls_path  = _make_classified(tmp_path, data_cls)
        conf_path = _make_confidence(tmp_path, data_conf)

        out_path = tmp_path / "unchanged.tif"
        confidence_filter(cls_path, conf_path, threshold=0.6, out_path=out_path)

        with rasterio.open(out_path) as ds:
            result = ds.read(1)
        assert np.array_equal(result, data_cls.astype(result.dtype))

    def test_nodata_pixels_preserved(self, tmp_path: Path) -> None:
        """Nodata pixels in the classified raster must never be modified."""
        data_cls = np.array([[1, -1, 1], [1, 1, 1]], dtype=np.int16)
        # All confidence low — but nodata pixel at (0,1) must stay nodata
        data_conf = np.full((2, 3), 0.1, dtype=np.float32)
        cls_path  = _make_classified(tmp_path, data_cls, nodata=-1)
        conf_path = _make_confidence(tmp_path, data_conf)

        out_path = tmp_path / "nodata.tif"
        confidence_filter(cls_path, conf_path, threshold=0.6, out_path=out_path)

        with rasterio.open(out_path) as ds:
            result = ds.read(1)
        assert result[0, 1] == -1

    def test_confidence_nodata_not_treated_as_low(self, tmp_path: Path) -> None:
        """Confidence nodata (-9999.0) must not trigger a replacement because
        -9999.0 is not >= 0."""
        data_cls  = np.ones((5, 5), dtype=np.int16)
        data_cls[2, 2] = 2
        data_conf = np.full((5, 5), 0.9, dtype=np.float32)
        data_conf[2, 2] = -9999.0   # nodata confidence, not "low" confidence
        cls_path  = _make_classified(tmp_path, data_cls)
        conf_path = _make_confidence(tmp_path, data_conf)

        out_path = tmp_path / "confnd.tif"
        confidence_filter(cls_path, conf_path, threshold=0.6, out_path=out_path)

        with rasterio.open(out_path) as ds:
            result = ds.read(1)
        assert result[2, 2] == 2   # unchanged — conf nodata ≠ low confidence

    def test_output_file_created(self, tmp_path: Path) -> None:
        data_cls  = np.ones((10, 10), dtype=np.int16)
        data_conf = np.full((10, 10), 0.8, dtype=np.float32)
        cls_path  = _make_classified(tmp_path, data_cls)
        conf_path = _make_confidence(tmp_path, data_conf)
        out_path  = tmp_path / "created.tif"
        confidence_filter(cls_path, conf_path, threshold=0.6, out_path=out_path)
        assert out_path.exists()


# ── median_smooth ─────────────────────────────────────────────────────────────

class TestMedianSmooth:

    @pytest.fixture(autouse=True)
    def _require_scipy(self) -> None:
        pytest.importorskip("scipy", reason="scipy not available")

    def test_even_kernel_raises(self, tmp_path: Path) -> None:
        path = _make_classified(tmp_path, np.ones((10, 10), dtype=np.int16))
        with pytest.raises(ValueError, match="odd integer"):
            median_smooth(path, tmp_path / "out.tif", kernel_size=4)

    def test_kernel_size_1_raises(self, tmp_path: Path) -> None:
        path = _make_classified(tmp_path, np.ones((10, 10), dtype=np.int16))
        with pytest.raises(ValueError, match="odd integer"):
            median_smooth(path, tmp_path / "out.tif", kernel_size=1)

    def test_output_file_created(self, tmp_path: Path) -> None:
        data = np.ones((20, 20), dtype=np.int16)
        path = _make_classified(tmp_path, data)
        out  = median_smooth(path, tmp_path / "smoothed.tif", kernel_size=3)
        assert out.exists()

    def test_uniform_raster_unchanged(self, tmp_path: Path) -> None:
        data = np.full((20, 20), 3, dtype=np.int16)
        path = _make_classified(tmp_path, data)
        out  = median_smooth(path, tmp_path / "smoothed.tif", kernel_size=3)
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert np.all(arr == 3)

    def test_isolated_pixel_reduced(self, tmp_path: Path) -> None:
        """An isolated class-2 pixel surrounded by class-1 pixels should be
        smoothed to class 1 by a 3×3 median (8 ones vs 1 two → median = 1)."""
        data       = np.full((15, 15), 1, dtype=np.int16)
        data[7, 7] = 2
        path = _make_classified(tmp_path, data, nodata=-1)
        out  = median_smooth(path, tmp_path / "smoothed.tif", kernel_size=3)
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert arr[7, 7] == 1

    def test_nodata_preserved(self, tmp_path: Path) -> None:
        """Nodata pixels must remain nodata after smoothing."""
        data       = np.ones((10, 10), dtype=np.int16)
        data[5, 5] = -1
        path = _make_classified(tmp_path, data, nodata=-1)
        out  = median_smooth(path, tmp_path / "smoothed.tif", kernel_size=3)
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert arr[5, 5] == -1


# ── run_postprocess_chain ─────────────────────────────────────────────────────

class TestRunPostprocessChain:

    @pytest.fixture(autouse=True)
    def _require_deps(self) -> None:
        pytest.importorskip("scipy",    reason="scipy not available")
        pytest.importorskip("osgeo.gdal", reason="GDAL not available")

    def _build_inputs(
        self, tmp_path: Path, size: int = 30
    ) -> tuple[Path, Path]:
        """Create a minimal classified + confidence raster pair."""
        data_cls = np.ones((size, size), dtype=np.int16)
        # Small patch of class 2 for sieve to potentially absorb
        data_cls[2:5, 2:5] = 2

        data_conf = np.full((size, size), 0.9, dtype=np.float32)
        # A few low-confidence pixels to exercise confidence_filter
        data_conf[10:13, 10:13] = 0.3

        cls_path  = _make_classified(tmp_path, data_cls, res=10.0, nodata=-1)
        conf_path = _make_confidence(tmp_path, data_conf)
        return cls_path, conf_path

    def test_chain_produces_final_output(self, tmp_path: Path) -> None:
        cls_path, conf_path = self._build_inputs(tmp_path)
        cfg = {
            "confidence_threshold":    0.6,
            "median_filter_size":      3,
            "morphological_kernel_size": 3,
            "morphological_iterations":  1,
            "min_mapping_unit_ha":     0.001,
            "sieve_connectivity":      4,
        }
        result = run_postprocess_chain(
            classified_path = cls_path,
            confidence_path = conf_path,
            cfg             = cfg,
            out_dir         = tmp_path / "chain_out",
            run_id          = "test_run",
        )
        assert "final" in result
        assert Path(result["final"]).exists()

    def test_chain_produces_all_intermediates(self, tmp_path: Path) -> None:
        cls_path, conf_path = self._build_inputs(tmp_path)
        cfg = {
            "confidence_threshold":    0.6,
            "median_filter_size":      3,
            "morphological_kernel_size": 3,
            "morphological_iterations":  1,
            "min_mapping_unit_ha":     0.001,
            "sieve_connectivity":      4,
        }
        result = run_postprocess_chain(
            cls_path, conf_path, cfg,
            out_dir = tmp_path / "chain_out2",
            run_id  = "test_run2",
        )
        for key in ("confidence_filtered", "median_smoothed",
                    "morphologically_closed", "final"):
            assert key in result, f"Missing key: {key}"
            assert Path(result[key]).exists(), f"File not found: {result[key]}"

    def test_chain_progress_callback_called(self, tmp_path: Path) -> None:
        cls_path, conf_path = self._build_inputs(tmp_path)
        cfg = {
            "confidence_threshold":    0.6,
            "median_filter_size":      3,
            "morphological_kernel_size": 3,
            "morphological_iterations":  1,
            "min_mapping_unit_ha":     0.001,
            "sieve_connectivity":      4,
        }
        messages: list[str] = []
        run_postprocess_chain(
            cls_path, conf_path, cfg,
            out_dir  = tmp_path / "chain_out3",
            run_id   = "test_run3",
            progress = messages.append,
        )
        assert len(messages) == 4   # one message per step

    def test_chain_non_projected_crs_raises(self, tmp_path: Path) -> None:
        """A raster with geographic CRS must raise ValueError."""
        data_cls  = np.ones((10, 10), dtype=np.int16)
        data_conf = np.full((10, 10), 0.9, dtype=np.float32)
        cls_path  = _make_classified(tmp_path, data_cls,  crs_epsg=4326)
        conf_path = _make_confidence(tmp_path, data_conf, crs_epsg=4326)
        with pytest.raises(ValueError, match="non-projected"):
            run_postprocess_chain(
                cls_path, conf_path, {},
                out_dir = tmp_path / "err",
                run_id  = "err_run",
            )
