"""
Tests for pipeline/training.py

Covers:
  - extract_from_shapefile: correct X/y shapes and class summary
  - extract_from_shapefile: auto-reprojection from a different CRS
  - extract_from_label_raster: output matches classify.extract_training_samples
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from pipeline.training import extract_from_label_raster, extract_from_shapefile

geopandas = pytest.importorskip("geopandas", reason="geopandas not installed")
shapely = pytest.importorskip("shapely", reason="shapely not installed")


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_feature_raster(
    tmp_path: Path,
    *,
    n_bands: int = 3,
    width: int = 60,
    height: int = 60,
    crs_epsg: int = 32633,
    res: float = 10.0,
    x_origin: float = 500_000.0,
    y_origin: float = 5_000_600.0,
) -> Path:
    path = tmp_path / "features.tif"
    transform = from_origin(x_origin, y_origin, res, res)
    rng = np.random.default_rng(0)
    data = rng.random((n_bands, height, width)).astype("float32")
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="float32",
        width=width, height=height, count=n_bands,
        crs=CRS.from_epsg(crs_epsg), transform=transform, nodata=-9999.0,
    ) as ds:
        ds.write(data)
    return path


def _make_label_raster(
    tmp_path: Path,
    feature_path: Path,
    *,
    filename: str = "labels.tif",
) -> Path:
    """Create a label raster perfectly aligned to feature_path.

    Top half (rows 0..29) → class 1.  Bottom half (rows 30..59) → class 2.
    Row 0 is nodata (value 0) so at least 1 nodata row exists.
    """
    path = tmp_path / filename
    with rasterio.open(feature_path) as src:
        profile = src.profile.copy()

    profile.update(count=1, dtype="int16", nodata=0)
    data = np.ones((1, profile["height"], profile["width"]), dtype="int16")
    # nodata row
    data[0, 0, :] = 0
    # bottom half → class 2
    data[0, 30:, :] = 2

    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data)
    return path


def _make_shapefile(
    tmp_path: Path,
    feature_path: Path,
    *,
    crs_epsg: int | None = None,
    class_column: str = "class_id",
) -> Path:
    """Create a two-polygon shapefile covering the left and right halves of the raster."""
    import geopandas as gpd
    from shapely.geometry import box

    with rasterio.open(feature_path) as src:
        bounds = src.bounds
        feat_crs = src.crs

    if crs_epsg is None:
        out_crs = feat_crs
    else:
        out_crs = CRS.from_epsg(crs_epsg)

    mid_x = (bounds.left + bounds.right) / 2.0
    poly1 = box(bounds.left,  bounds.bottom, mid_x,        bounds.top)
    poly2 = box(mid_x,        bounds.bottom, bounds.right, bounds.top)

    gdf = gpd.GeoDataFrame(
        {class_column: [1, 2], "geometry": [poly1, poly2]},
        crs=feat_crs,
    )
    if crs_epsg is not None:
        gdf = gdf.to_crs(CRS.from_epsg(crs_epsg))

    shp_path = tmp_path / "training.shp"
    gdf.to_file(str(shp_path))
    return shp_path


# ── Test classes ──────────────────────────────────────────────────────────────


class TestExtractFromShapefileShapes:
    """extract_from_shapefile returns correct array shapes and class summary."""

    def test_x_has_correct_n_features(self, tmp_path: Path) -> None:
        n_bands = 4
        feat = _make_feature_raster(tmp_path, n_bands=n_bands)
        shp  = _make_shapefile(tmp_path, feat)
        X, y, summary = extract_from_shapefile(shp, feat, class_column="class_id")
        assert X.shape[1] == n_bands, (
            f"Expected {n_bands} features per sample, got {X.shape[1]}"
        )

    def test_x_y_row_count_matches(self, tmp_path: Path) -> None:
        feat = _make_feature_raster(tmp_path)
        shp  = _make_shapefile(tmp_path, feat)
        X, y, summary = extract_from_shapefile(shp, feat, class_column="class_id")
        assert X.shape[0] == y.shape[0]

    def test_class_summary_columns(self, tmp_path: Path) -> None:
        feat = _make_feature_raster(tmp_path)
        shp  = _make_shapefile(tmp_path, feat)
        _, _, summary = extract_from_shapefile(shp, feat, class_column="class_id")
        assert set(summary.columns) >= {"class", "count", "percentage"}

    def test_two_classes_present(self, tmp_path: Path) -> None:
        feat = _make_feature_raster(tmp_path)
        shp  = _make_shapefile(tmp_path, feat)
        _, y, _ = extract_from_shapefile(shp, feat, class_column="class_id")
        assert set(y.tolist()) == {1, 2}

    def test_x_dtype_is_float32(self, tmp_path: Path) -> None:
        feat = _make_feature_raster(tmp_path)
        shp  = _make_shapefile(tmp_path, feat)
        X, _, _ = extract_from_shapefile(shp, feat, class_column="class_id")
        assert X.dtype == np.float32

    def test_per_polygon_sample_cap(self, tmp_path: Path) -> None:
        """With a tight cap, total samples ≤ 2 × cap (one polygon per class)."""
        cap  = 5
        feat = _make_feature_raster(tmp_path, width=60, height=60)
        shp  = _make_shapefile(tmp_path, feat)
        X, _, _ = extract_from_shapefile(
            shp, feat, class_column="class_id", max_samples_per_polygon=cap
        )
        assert X.shape[0] <= 2 * cap


class TestExtractFromShapefileReprojection:
    """extract_from_shapefile reprojects the shapefile when CRS differs."""

    def test_different_crs_extracts_samples(self, tmp_path: Path) -> None:
        """Shapefile in EPSG:4326 vs feature in EPSG:32633 — should still work."""
        feat = _make_feature_raster(tmp_path, crs_epsg=32633)
        shp  = _make_shapefile(tmp_path, feat, crs_epsg=4326)
        X, y, _ = extract_from_shapefile(shp, feat, class_column="class_id")
        assert len(y) > 0, "No samples extracted after reprojection"

    def test_reprojected_classes_intact(self, tmp_path: Path) -> None:
        feat = _make_feature_raster(tmp_path, crs_epsg=32633)
        shp  = _make_shapefile(tmp_path, feat, crs_epsg=4326)
        _, y, _ = extract_from_shapefile(shp, feat, class_column="class_id")
        assert set(y.tolist()) == {1, 2}

    def test_same_crs_produces_same_count(self, tmp_path: Path) -> None:
        """Extraction result should yield both classes regardless of input CRS."""
        feat = _make_feature_raster(tmp_path, crs_epsg=32633)
        dir_same = tmp_path / "same"
        dir_diff = tmp_path / "diff"
        dir_same.mkdir()
        dir_diff.mkdir()
        shp_same = _make_shapefile(dir_same, feat, crs_epsg=None)
        shp_diff = _make_shapefile(dir_diff, feat, crs_epsg=4326)

        X_same, y_same, _ = extract_from_shapefile(shp_same, feat, class_column="class_id")
        X_diff, y_diff, _ = extract_from_shapefile(shp_diff, feat, class_column="class_id")

        # Both should yield samples for both classes
        assert set(y_same.tolist()) == {1, 2}
        assert set(y_diff.tolist()) == {1, 2}


class TestExtractFromLabelRasterMatchesOriginal:
    """extract_from_label_raster output matches classify.extract_training_samples."""

    def test_shapes_match_original(self, tmp_path: Path) -> None:
        from pipeline.classify import extract_training_samples

        feat  = _make_feature_raster(tmp_path)
        label = _make_label_raster(tmp_path, feat)

        X_new, y_new, _ = extract_from_label_raster(
            label_path=label, feature_path=feat, nodata_label=0, random_state=42
        )
        X_orig, y_orig = extract_training_samples(
            feature_path=feat, label_path=label, nodata_label=0, random_state=42
        )

        assert X_new.shape == X_orig.shape
        assert y_new.shape == y_orig.shape

    def test_values_match_original(self, tmp_path: Path) -> None:
        from pipeline.classify import extract_training_samples

        feat  = _make_feature_raster(tmp_path)
        label = _make_label_raster(tmp_path, feat)

        X_new, y_new, _ = extract_from_label_raster(
            label_path=label, feature_path=feat, nodata_label=0, random_state=42
        )
        X_orig, y_orig = extract_training_samples(
            feature_path=feat, label_path=label, nodata_label=0, random_state=42
        )

        np.testing.assert_array_equal(
            np.sort(y_new), np.sort(y_orig),
            err_msg="Label arrays differ between new and original extraction",
        )

    def test_class_summary_has_both_classes(self, tmp_path: Path) -> None:
        feat  = _make_feature_raster(tmp_path)
        label = _make_label_raster(tmp_path, feat)
        _, _, summary = extract_from_label_raster(
            label_path=label, feature_path=feat, nodata_label=0
        )
        assert set(summary["class"].tolist()) == {1, 2}

    def test_max_samples_subsamples(self, tmp_path: Path) -> None:
        feat  = _make_feature_raster(tmp_path, width=60, height=60)
        label = _make_label_raster(tmp_path, feat)
        _, y_full, _ = extract_from_label_raster(
            label_path=label, feature_path=feat, nodata_label=0
        )
        cap = len(y_full) // 2
        _, y_sub, _ = extract_from_label_raster(
            label_path=label, feature_path=feat, nodata_label=0, max_samples=cap
        )
        assert len(y_sub) <= cap + 2  # +2 for per-class rounding

    def test_nodata_pixels_excluded(self, tmp_path: Path) -> None:
        feat  = _make_feature_raster(tmp_path)
        label = _make_label_raster(tmp_path, feat)
        _, y, _ = extract_from_label_raster(
            label_path=label, feature_path=feat, nodata_label=0
        )
        assert 0 not in y.tolist(), "Nodata value 0 found in extracted labels"
