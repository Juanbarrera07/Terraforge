"""
Phase 4 — Feature engineering tests.

All rasters are synthetic, created in tmp_path via the make_raster fixture
from conftest.py.  No external data files required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio

from pipeline.features import (
    BandMap,
    FeatureResult,
    _active_features,
    _aspect,
    _bsi,
    _glcm_features,
    _ndre,
    _ndvi,
    _ndwi,
    _safe_ratio,
    _sar_ratio,
    _select_glcm_band,
    _slope,
    _vari,
    compute_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def full_stack(make_raster):
    """
    9-band raster with logical band layout:
        1=Blue, 2=Green, 3=Red, 4=NIR, 5=SWIR, 6=RedEdge, 7=VV, 8=VH, 9=DEM
    Synthetic values from RNG; all in (0, 1) — valid for index computation.
    """
    return make_raster(
        filename="full_stack.tif",
        bands=9,
        width=64,
        height=64,
        res=10.0,
        dtype="float32",
        nodata=-9999.0,
    )


@pytest.fixture
def full_band_map():
    return BandMap(
        blue=1, green=2, red=3, nir=4, swir=5, rededge=6, vv=7, vh=8, dem=9
    )


# ── BandMap ───────────────────────────────────────────────────────────────────

def test_bandmap_all_defaults_are_none():
    bm = BandMap()
    for attr in ("nir", "red", "green", "blue", "swir", "rededge", "vv", "vh", "dem", "dsm"):
        assert getattr(bm, attr) is None, f"Expected {attr} to be None"


def test_bandmap_field_assignment():
    bm = BandMap(nir=4, red=3, green=2, blue=1)
    assert bm.nir == 4
    assert bm.red == 3
    assert bm.green == 2
    assert bm.blue == 1
    assert bm.swir is None


def test_bandmap_dsm_field_exists():
    bm = BandMap(dsm=2)
    assert bm.dsm == 2
    assert bm.dem is None  # unrelated field unchanged


# ── _safe_ratio ───────────────────────────────────────────────────────────────

def test_safe_ratio_normal_division():
    a = np.array([6.0, 10.0])
    b = np.array([3.0, 5.0])
    np.testing.assert_allclose(_safe_ratio(a, b), [2.0, 2.0])


def test_safe_ratio_zero_denominator_uses_fill():
    a = np.array([1.0, 2.0])
    b = np.array([0.0, 2.0])
    result = _safe_ratio(a, b, fill=-99.0)
    assert result[0] == pytest.approx(-99.0)
    assert result[1] == pytest.approx(1.0)


def test_safe_ratio_nan_fill_propagates():
    result = _safe_ratio(np.array([1.0]), np.array([0.0]), fill=np.nan)
    assert np.isnan(result[0])


# ── Spectral indices ──────────────────────────────────────────────────────────

def test_ndvi_known_values():
    nir = np.array([[0.8]])
    red = np.array([[0.2]])
    expected = (0.8 - 0.2) / (0.8 + 0.2)   # 0.6
    np.testing.assert_allclose(_ndvi(nir, red), [[expected]])


def test_ndvi_zero_denominator_is_nan():
    result = _ndvi(np.array([[0.0]]), np.array([[0.0]]))
    assert np.isnan(result[0, 0])


def test_ndwi_known_values():
    green = np.array([[0.6]])
    nir   = np.array([[0.2]])
    expected = (0.6 - 0.2) / (0.6 + 0.2)   # 0.5
    np.testing.assert_allclose(_ndwi(green, nir), [[expected]])


def test_bsi_known_values():
    swir = np.array([[0.5]])
    red  = np.array([[0.3]])
    nir  = np.array([[0.4]])
    blue = np.array([[0.1]])
    num      = (0.5 + 0.3) - (0.4 + 0.1)   # 0.3
    den      = (0.5 + 0.3) + (0.4 + 0.1)   # 1.3
    np.testing.assert_allclose(_bsi(swir, red, nir, blue), [[num / den]], rtol=1e-6)


def test_ndre_known_values():
    re  = np.array([[0.7]])
    red = np.array([[0.3]])
    expected = (0.7 - 0.3) / (0.7 + 0.3)   # 0.4
    np.testing.assert_allclose(_ndre(re, red), [[expected]])


def test_vari_known_values():
    g = np.array([[0.5]])
    r = np.array([[0.3]])
    b = np.array([[0.2]])
    expected = (0.5 - 0.3) / (0.5 + 0.3 - 0.2)   # 0.2 / 0.6
    np.testing.assert_allclose(_vari(g, r, b), [[expected]], rtol=1e-6)


def test_vari_zero_denominator_is_nan():
    # G + R − B = 0  when G=1, R=0, B=1  (exact zero, no FP rounding)
    result = _vari(np.array([[1.0]]), np.array([[0.0]]), np.array([[1.0]]))
    assert np.isnan(result[0, 0])


# ── SAR ratio ─────────────────────────────────────────────────────────────────

def test_sar_ratio_linear():
    vv = np.array([[10.0]])
    vh = np.array([[2.0]])
    np.testing.assert_allclose(_sar_ratio(vv, vh, log_scale=False), [[5.0]])


def test_sar_ratio_log_scale():
    vv = np.array([[100.0]])
    vh = np.array([[10.0]])
    # ratio=10, 10*log10(10) = 10
    np.testing.assert_allclose(_sar_ratio(vv, vh, log_scale=True), [[10.0]], rtol=1e-6)


def test_sar_ratio_zero_vh_log_is_nan():
    # VH=0 → ratio=NaN → log → NaN
    result = _sar_ratio(np.array([[1.0]]), np.array([[0.0]]), log_scale=True)
    assert np.isnan(result[0, 0])


# ── Slope and aspect ──────────────────────────────────────────────────────────

def test_slope_flat_dem_is_zero():
    dem = np.ones((10, 10), dtype=np.float64) * 200.0
    np.testing.assert_allclose(_slope(dem, res=10.0), np.zeros((10, 10)), atol=1e-10)


def test_slope_ramp_interior():
    # z = column_index * res → dz/dx = 1 m/m everywhere in interior
    dem = np.tile(np.arange(10, dtype=np.float64) * 10.0, (10, 1))
    result = _slope(dem, res=10.0)
    np.testing.assert_allclose(result[5, 5], 1.0, rtol=1e-6)


def test_aspect_values_in_valid_range():
    """Aspect must always be in [0, 360)."""
    rng = np.random.default_rng(0)
    dem = rng.random((30, 30)) * 500.0
    result = _aspect(dem, res=5.0)
    assert np.all(result >= 0.0)
    assert np.all(result < 360.0)


def test_aspect_flat_dem_defined():
    """Flat DEM: arctan2(0, 0) = 0; result should be finite and in [0, 360)."""
    dem    = np.ones((8, 8), dtype=np.float64)
    result = _aspect(dem, res=10.0)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0) and np.all(result < 360.0)


# ── GLCM ─────────────────────────────────────────────────────────────────────

def test_glcm_returns_expected_keys():
    tx = _glcm_features(np.random.default_rng(7).random((16, 16)))
    assert set(tx.keys()) == {"contrast", "homogeneity", "entropy"}


def test_glcm_uniform_tile():
    tile = np.ones((32, 32), dtype=np.float64) * 0.5
    tx   = _glcm_features(tile)
    # t_max == t_min → early return
    assert tx["contrast"]    == pytest.approx(0.0)
    assert tx["homogeneity"] == pytest.approx(1.0)
    assert tx["entropy"]     == pytest.approx(0.0)


def test_glcm_gradient_tile_has_positive_contrast_and_entropy():
    tile = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float64), (32, 1))
    tx   = _glcm_features(tile)
    assert tx["contrast"] > 0.0
    assert tx["entropy"]  > 0.0


# ── GLCM robustness ───────────────────────────────────────────────────────────

def test_select_glcm_band_all_nodata_returns_none():
    """_select_glcm_band must return None when every pixel in the selected band is nodata."""
    data = np.full((1, 8, 8), -9999.0, dtype=np.float64)
    result = _select_glcm_band(data, BandMap(red=1), src_nodata=-9999.0)
    assert result is None


def test_glcm_majority_nodata_produces_finite_texture_summary(make_raster, tmp_path):
    """
    When ~90% of source pixels are nodata, the GLCM should still complete using
    the remaining valid pixels, and texture_summary values must all be finite.
    """
    path = make_raster(
        "mostly_nd.tif", bands=2, width=32, height=32,
        dtype="float32", nodata=-9999.0,
    )
    with rasterio.open(path, "r+") as ds:
        data = ds.read()
        rng  = np.random.default_rng(99)
        mask = rng.random((32, 32)) < 0.90   # mark ~90% of pixels as nodata
        data[:, mask] = -9999.0
        ds.write(data)

    result = compute_features(path, tmp_path / "nd90_feat.tif", BandMap(nir=1, red=2))

    # texture_summary may be None only if every tile was all-nodata — unlikely at 90%
    # If it is present, all statistics must be finite
    if result.texture_summary is not None:
        for metric, stats in result.texture_summary.items():
            for stat_name, val in stats.items():
                assert np.isfinite(val), (
                    f"texture_summary[{metric!r}][{stat_name!r}] = {val} is not finite"
                )


# ── compute_features — end-to-end ─────────────────────────────────────────────

def test_output_file_created(full_stack, full_band_map, tmp_path):
    out = tmp_path / "features.tif"
    compute_features(full_stack, out, full_band_map)
    assert out.exists()


def test_output_band_count_matches_feature_names(full_stack, full_band_map, tmp_path):
    out    = tmp_path / "features.tif"
    result = compute_features(full_stack, out, full_band_map)
    assert len(result.feature_names) == 8  # all 8 features available
    with rasterio.open(out) as ds:
        assert ds.count == len(result.feature_names)


def test_output_spatial_properties_match_source(full_stack, full_band_map, tmp_path):
    out    = tmp_path / "features.tif"
    compute_features(full_stack, out, full_band_map)
    with rasterio.open(full_stack) as src, rasterio.open(out) as dst:
        assert dst.height == src.height
        assert dst.width  == src.width
        assert dst.crs    == src.crs


def test_partial_band_map_skips_unavailable_features(make_raster, tmp_path):
    path   = make_raster(filename="optical4.tif", bands=4, dtype="float32")
    bm     = BandMap(blue=1, green=2, red=3, nir=4)
    out    = tmp_path / "opt_feat.tif"
    result = compute_features(path, out, bm)

    assert "ndvi"      in result.feature_names
    assert "ndwi"      in result.feature_names
    assert "vari"      in result.feature_names
    assert "sar_ratio" not in result.feature_names
    assert "slope"     not in result.feature_names
    assert "bsi"       not in result.feature_names  # no SWIR in 4-band raster

    with rasterio.open(out) as ds:
        assert ds.count == len(result.feature_names)


def test_feature_stats_keys_and_structure(full_stack, full_band_map, tmp_path):
    result = compute_features(full_stack, tmp_path / "f.tif", full_band_map)
    for fname in result.feature_names:
        assert fname in result.feature_stats
        assert set(result.feature_stats[fname].keys()) == {
            "min", "max", "mean", "std", "valid_pct"
        }


def test_feature_stats_valid_pct_in_range(full_stack, full_band_map, tmp_path):
    result = compute_features(full_stack, tmp_path / "f.tif", full_band_map)
    for fname in result.feature_names:
        vp = result.feature_stats[fname]["valid_pct"]
        assert 0.0 <= vp <= 100.0


def test_texture_summary_keys_and_structure(full_stack, full_band_map, tmp_path):
    result = compute_features(full_stack, tmp_path / "f.tif", full_band_map)
    assert result.texture_summary is not None
    assert set(result.texture_summary.keys()) == {"contrast", "homogeneity", "entropy"}
    for metric_stats in result.texture_summary.values():
        assert set(metric_stats.keys()) == {"mean", "std", "min", "max"}


def test_correlation_matrix_is_symmetric(full_stack, full_band_map, tmp_path):
    result = compute_features(full_stack, tmp_path / "f.tif", full_band_map)
    corr   = result.correlation_matrix
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)


def test_correlation_matrix_diagonal_is_one(full_stack, full_band_map, tmp_path):
    result = compute_features(full_stack, tmp_path / "f.tif", full_band_map)
    np.testing.assert_allclose(np.diag(result.correlation_matrix), 1.0, atol=1e-6)


def test_high_corr_pair_flagged_for_identical_features(make_raster, tmp_path):
    """
    When nir=1 and rededge=1 both point to the same physical band, NDVI and
    NDRE compute (b1-b2)/(b1+b2) from identical inputs → r = 1.0 → flagged.
    """
    path   = make_raster("dup.tif", bands=3, dtype="float32")
    bm     = BandMap(nir=1, red=2, rededge=1)
    out    = tmp_path / "dup_feat.tif"
    result = compute_features(path, out, bm)

    assert "ndvi" in result.feature_names
    assert "ndre" in result.feature_names
    assert len(result.high_correlation_pairs) > 0
    # The pair must contain both names and r near 1.0
    names_in_pairs = {
        name
        for a, b, _ in result.high_correlation_pairs
        for name in (a, b)
    }
    assert "ndvi" in names_in_pairs
    assert "ndre" in names_in_pairs
    r_values = [abs(r) for _, _, r in result.high_correlation_pairs]
    assert all(r > 0.95 for r in r_values)


def test_feature_result_equality_excludes_correlation_matrix(
    full_stack, full_band_map, tmp_path
):
    """Two FeatureResults differing only in correlation_matrix must compare equal."""
    out    = tmp_path / "f.tif"
    r1     = compute_features(full_stack, out, full_band_map)
    r2     = FeatureResult(
        feature_path=r1.feature_path,
        feature_names=r1.feature_names,
        feature_stats=r1.feature_stats,
        texture_summary=r1.texture_summary,
        correlation_matrix=np.zeros_like(r1.correlation_matrix),
        high_correlation_pairs=r1.high_correlation_pairs,
    )
    assert r1 == r2


def test_nodata_pixels_preserved_in_output(make_raster, tmp_path):
    """Source nodata pixels must appear as nodata in every output feature band."""
    path = make_raster(
        "nd_src.tif", bands=4, width=32, height=32,
        dtype="float32", nodata=-9999.0,
    )
    # Overwrite pixel (0, 0) with nodata across all bands
    with rasterio.open(path, "r+") as ds:
        data = ds.read()
        data[:, 0, 0] = -9999.0
        ds.write(data)

    bm  = BandMap(blue=1, green=2, red=3, nir=4)
    out = tmp_path / "nd_feat.tif"
    compute_features(path, out, bm)

    with rasterio.open(out) as ds:
        feat = ds.read()
        nd   = ds.nodata
    # All feature bands at (row=0, col=0) must equal the file nodata value
    for bi in range(feat.shape[0]):
        assert feat[bi, 0, 0] == pytest.approx(nd), (
            f"Band {bi+1} at nodata pixel: expected {nd}, got {feat[bi, 0, 0]}"
        )


def test_output_nodata_is_finite(make_raster, tmp_path):
    """File nodata must never be NaN."""
    path   = make_raster("nd_finite.tif", bands=2, nodata=None)
    out    = tmp_path / "nd_finite_feat.tif"
    result = compute_features(path, out, BandMap(nir=1, red=2))
    with rasterio.open(out) as ds:
        assert ds.nodata is not None
        assert np.isfinite(float(ds.nodata))


def test_progress_callback_fires_correct_count(full_stack, full_band_map, tmp_path):
    """Callback must be called exactly n_tiles times with correct (current, total)."""
    calls: list[tuple[int, int]] = []

    def cb(current: int, total: int) -> None:
        calls.append((current, total))

    block = 32
    compute_features(
        full_stack, tmp_path / "prog.tif", full_band_map,
        block_size=block, progress=cb,
    )

    with rasterio.open(full_stack) as ds:
        expected = math.ceil(ds.height / block) * math.ceil(ds.width / block)

    assert len(calls) == expected
    assert all(t == expected for _, t in calls)
    assert [c for c, _ in calls] == list(range(1, expected + 1))


def test_corr_threshold_is_strictly_greater_than(make_raster, tmp_path):
    """
    A pair with |r| exactly equal to corr_flag_threshold must NOT be flagged.
    cfg is passed explicitly so the threshold is known at call time.
    """
    # Use a threshold of 1.0; identical features produce r=1.0 exactly.
    # With strict >, r=1.0 is NOT > 1.0 → pair must not appear.
    path   = make_raster("strict_thresh.tif", bands=3, dtype="float32")
    bm     = BandMap(nir=1, red=2, rededge=1)   # ndvi and ndre are identical
    out    = tmp_path / "strict.tif"
    result = compute_features(path, out, bm, cfg={"corr_flag_threshold": 1.0})
    # r = 1.0 is not strictly > 1.0, so no pairs should be flagged
    assert result.high_correlation_pairs == []


def test_empty_bandmap_raises_value_error(make_raster, tmp_path):
    path = make_raster("single.tif", bands=1)
    with pytest.raises(ValueError, match="no computable features"):
        compute_features(path, tmp_path / "f.tif", BandMap())


def test_active_features_ordering():
    """_active_features must return features in the canonical order."""
    bm     = BandMap(blue=1, green=2, red=3, nir=4, swir=5, rededge=6, vv=7, vh=8, dem=9)
    feats  = _active_features(bm)
    assert feats == ["ndvi", "ndwi", "bsi", "ndre", "vari", "sar_ratio", "slope", "aspect"]


def test_single_dem_band_produces_slope_and_aspect(make_raster, tmp_path):
    path   = make_raster("dem_only.tif", bands=1, dtype="float32")
    out    = tmp_path / "dem_feat.tif"
    result = compute_features(path, out, BandMap(dem=1))
    assert result.feature_names == ["slope", "aspect"]
    with rasterio.open(out) as ds:
        assert ds.count == 2
