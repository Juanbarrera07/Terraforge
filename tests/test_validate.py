"""
Tests for pipeline/validate.py (Phase 2B).

Covers
------
- check_crs_match:        match, mismatch, missing CRS
- check_resolution_compatibility: within ratio, exceeds ratio
- check_date_proximity:   within gap, exceeds gap, phenology warning
- check_minimum_overlap:  sufficient, insufficient (non-overlapping extents)
- check_band_counts:      SAR/DEM/optical band expectations
- has_critical_failures:  composite gate logic
- run_all_validations:    orchestrator returns all expected check keys
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pipeline.validate import (
    ValidationResult,
    check_band_counts,
    check_crs_match,
    check_date_proximity,
    check_label_alignment,
    check_minimum_overlap,
    check_resolution_compatibility,
    has_critical_failures,
    run_all_validations,
)
from tests.conftest import make_layer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _layers_from_paths(
    path_meta_list: list[tuple[Path, str, str]],
) -> dict[str, dict]:
    """Build a layers dict from (path, layer_type, key) tuples."""
    result = {}
    for path, ltype, key in path_meta_list:
        result[key] = make_layer(path, ltype)
    return result


# ── check_crs_match ───────────────────────────────────────────────────────────

class TestCheckCrsMatch:
    def test_matching_crs_ok(self, make_raster):
        a = make_raster("a.tif", crs_epsg=32633)
        b = make_raster("b.tif", crs_epsg=32633)
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_crs_match(layers)
        assert result.status == "ok"
        assert not result.blocks_pipeline

    def test_crs_mismatch_is_critical_error(self, make_raster):
        a = make_raster("a.tif", crs_epsg=32633)
        b = make_raster("b.tif", crs_epsg=4326)
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_crs_match(layers)
        assert result.status == "error"
        assert result.is_critical
        assert result.blocks_pipeline

    def test_missing_crs_is_critical_error(self, make_raster):
        a = make_raster("a.tif", crs_epsg=32633)
        b = make_raster("b_nocrs.tif", crs_epsg=None)
        layers = {
            "sar__a":  make_layer(a, "sar"),
            "opt__b":  make_layer(b, "optical"),
        }
        result = check_crs_match(layers)
        assert result.status == "error"
        assert result.is_critical

    def test_single_layer_ok(self, make_raster):
        a = make_raster("a.tif", crs_epsg=32633)
        layers = {"sar__a": make_layer(a, "sar")}
        result = check_crs_match(layers)
        assert result.status == "ok"


# ── check_resolution_compatibility ───────────────────────────────────────────

class TestCheckResolutionCompatibility:
    def test_same_resolution_ok(self, make_raster):
        a = make_raster("a.tif", res=10.0)
        b = make_raster("b.tif", res=10.0)
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_resolution_compatibility(layers, max_ratio=2.0)
        assert result.status == "ok"

    def test_within_ratio_ok(self, make_raster):
        a = make_raster("a.tif", res=10.0)
        b = make_raster("b.tif", res=20.0)
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_resolution_compatibility(layers, max_ratio=2.0)
        assert result.status == "ok"

    def test_exceeds_ratio_warns(self, make_raster):
        a = make_raster("a.tif", res=10.0)
        b = make_raster("b.tif", res=35.0)  # 3.5× ratio
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_resolution_compatibility(layers, max_ratio=2.0)
        assert result.status == "warn"
        assert not result.is_critical   # warn, not block

    def test_single_layer_ok(self, make_raster):
        a = make_raster("a.tif", res=10.0)
        layers = {"sar__a": make_layer(a, "sar")}
        result = check_resolution_compatibility(layers)
        assert result.status == "ok"


# ── check_date_proximity ──────────────────────────────────────────────────────

class TestCheckDateProximity:
    def _layers_with_dates(self, paths_dates: list[tuple[Path, date]]) -> dict[str, dict]:
        return {
            f"layer_{i}": {**make_layer(p), "date": d}
            for i, (p, d) in enumerate(paths_dates)
        }

    def test_within_gap_ok(self, make_raster):
        a = make_raster("a.tif")
        b = make_raster("b.tif")
        layers = self._layers_with_dates([
            (a, date(2022, 5, 1)),
            (b, date(2022, 5, 20)),   # 19-day gap
        ])
        results = check_date_proximity(layers, max_gap_days=30)
        prox = next(r for r in results if r.check == "date_proximity")
        assert prox.status == "ok"

    def test_exceeds_gap_warns(self, make_raster):
        a = make_raster("a.tif")
        b = make_raster("b.tif")
        layers = self._layers_with_dates([
            (a, date(2022, 1, 1)),
            (b, date(2022, 3, 15)),   # 73-day gap
        ])
        results = check_date_proximity(layers, max_gap_days=30)
        prox = next(r for r in results if r.check == "date_proximity")
        assert prox.status == "warn"
        assert not prox.is_critical

    def test_phenology_warning_fires(self, make_raster):
        a = make_raster("a.tif")
        b = make_raster("b.tif")
        layers = self._layers_with_dates([
            (a, date(2022, 1, 10)),
            (b, date(2022, 5, 10)),   # 4-month span
        ])
        results = check_date_proximity(layers, phenology_month_gap=2)
        checks = {r.check for r in results}
        assert "phenology_risk" in checks
        pheno = next(r for r in results if r.check == "phenology_risk")
        assert pheno.status == "warn"

    def test_no_phenology_when_same_month(self, make_raster):
        a = make_raster("a.tif")
        b = make_raster("b.tif")
        layers = self._layers_with_dates([
            (a, date(2022, 6, 1)),
            (b, date(2022, 6, 28)),
        ])
        results = check_date_proximity(layers, phenology_month_gap=2)
        checks = {r.check for r in results}
        assert "phenology_risk" not in checks

    def test_missing_dates_warns(self, make_raster):
        a = make_raster("a.tif")
        layers = {"layer_0": {**make_layer(a), "date": None}}
        results = check_date_proximity(layers)
        assert any(r.status == "warn" for r in results)


# ── check_minimum_overlap ─────────────────────────────────────────────────────

class TestCheckMinimumOverlap:
    def test_fully_overlapping_ok(self, make_raster):
        # Same origin → same extent → 100% overlap
        a = make_raster("a.tif", x_origin=500_000.0, y_origin=5_000_200.0, width=200, height=200, res=10.0)
        b = make_raster("b.tif", x_origin=500_000.0, y_origin=5_000_200.0, width=200, height=200, res=10.0)
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_minimum_overlap(layers, min_pct=80.0)
        assert result.status == "ok"

    def test_partial_overlap_ok(self, make_raster):
        # b shifted by 500 m east — still 75% overlap for 2000 m wide rasters
        a = make_raster("a.tif", x_origin=500_000.0, y_origin=5_002_000.0,
                        width=200, height=200, res=10.0)
        b = make_raster("b.tif", x_origin=500_500.0, y_origin=5_002_000.0,
                        width=200, height=200, res=10.0)
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_minimum_overlap(layers, min_pct=70.0)
        assert result.status == "ok"

    def test_no_overlap_critical_error(self, make_raster):
        # Place b completely east of a (no overlap)
        a = make_raster("a.tif", x_origin=500_000.0, y_origin=5_002_000.0,
                        width=100, height=100, res=10.0)
        b = make_raster("b.tif", x_origin=600_000.0, y_origin=5_002_000.0,
                        width=100, height=100, res=10.0)
        layers = {
            "sar__a": make_layer(a, "sar"),
            "opt__b": make_layer(b, "optical"),
        }
        result = check_minimum_overlap(layers, min_pct=80.0)
        assert result.status == "error"
        assert result.is_critical
        assert result.blocks_pipeline

    def test_single_layer_ok(self, make_raster):
        a = make_raster("a.tif")
        layers = {"sar__a": make_layer(a, "sar")}
        result = check_minimum_overlap(layers)
        assert result.status == "ok"


# ── check_band_counts ─────────────────────────────────────────────────────────

class TestCheckBandCounts:
    def test_sar_single_band_ok(self, make_raster):
        a = make_raster("a.tif", bands=1)
        layers = {"sar__a": {**make_layer(a, "sar")}}
        results = check_band_counts(layers)
        assert all(r.status == "ok" for r in results)

    def test_sar_dual_pol_ok(self, make_raster):
        a = make_raster("a.tif", bands=2)
        layers = {"sar__a": {**make_layer(a, "sar")}}
        results = check_band_counts(layers)
        assert all(r.status == "ok" for r in results)

    def test_sar_unexpected_band_count_warns(self, make_raster):
        a = make_raster("a.tif", bands=3)
        layers = {"sar__a": {**make_layer(a, "sar")}}
        results = check_band_counts(layers)
        assert any(r.status == "warn" for r in results)

    def test_dem_single_band_ok(self, make_raster):
        a = make_raster("a.tif", bands=1)
        layers = {"dem__a": {**make_layer(a, "dem")}}
        results = check_band_counts(layers)
        assert all(r.status == "ok" for r in results)

    def test_dem_multi_band_warns(self, make_raster):
        a = make_raster("a.tif", bands=2)
        layers = {"dem__a": {**make_layer(a, "dem")}}
        results = check_band_counts(layers)
        assert any(r.status == "warn" for r in results)

    def test_optical_3_bands_ok(self, make_raster):
        a = make_raster("a.tif", bands=3)
        layers = {"opt__a": {**make_layer(a, "optical")}}
        results = check_band_counts(layers)
        assert all(r.status == "ok" for r in results)

    def test_optical_1_band_error(self, make_raster):
        a = make_raster("a.tif", bands=1)
        layers = {"opt__a": {**make_layer(a, "optical")}}
        results = check_band_counts(layers)
        assert any(r.status == "error" for r in results)


# ── has_critical_failures ─────────────────────────────────────────────────────

class TestHasCriticalFailures:
    def test_no_failures_returns_false(self):
        results = {
            "crs_match": ValidationResult("crs_match", "ok", "All good."),
            "resolution_compatibility": ValidationResult("resolution_compatibility", "ok", "Fine."),
        }
        assert has_critical_failures(results) is False

    def test_warn_only_returns_false(self):
        results = {
            "date_proximity": ValidationResult("date_proximity", "warn", "Gap too large."),
        }
        assert has_critical_failures(results) is False

    def test_critical_error_returns_true(self):
        results = {
            "crs_match": ValidationResult(
                "crs_match", "error", "CRS mismatch.", is_critical=True
            ),
        }
        assert has_critical_failures(results) is True

    def test_non_critical_error_returns_false(self):
        results = {
            "band_count_opt__a": ValidationResult(
                "band_count_opt__a", "error", "Too few bands.", is_critical=False
            ),
        }
        assert has_critical_failures(results) is False


# ── run_all_validations orchestrator ─────────────────────────────────────────

class TestRunAllValidations:
    def test_returns_all_expected_checks(self, make_raster):
        a = make_raster("S2A_20220601_MSIL2A.tif", bands=4, crs_epsg=32633, res=10.0)
        b = make_raster("S1A_20220610_GRD.tif",    bands=2, crs_epsg=32633, res=20.0)

        from tests.conftest import make_layer as ml
        layers = {
            "optical__a": {**ml(a, "optical"), "date": date(2022, 6, 1)},
            "sar__b":     {**ml(b, "sar"),     "date": date(2022, 6, 10)},
        }
        config = {
            "max_resolution_ratio": 2.0,
            "max_date_gap_days":    30,
            "min_overlap_pct":      80.0,
            "phenology_month_gap":  2,
        }
        results = run_all_validations(layers, config)

        assert "crs_match" in results
        assert "resolution_compatibility" in results
        assert "date_proximity" in results
        assert "minimum_overlap" in results
        # band counts — one per layer
        assert any(k.startswith("band_count_") for k in results)

    def test_crs_mismatch_blocks_pipeline(self, make_raster):
        a = make_raster("a.tif", bands=4, crs_epsg=32633)
        b = make_raster("b.tif", bands=2, crs_epsg=4326)

        from tests.conftest import make_layer as ml
        layers = {
            "optical__a": ml(a, "optical"),
            "sar__b":     ml(b, "sar"),
        }
        config = {"max_resolution_ratio": 2.0, "max_date_gap_days": 30,
                  "min_overlap_pct": 80.0, "phenology_month_gap": 2}
        results = run_all_validations(layers, config)
        assert has_critical_failures(results) is True


# ── check_label_alignment ─────────────────────────────────────────────────────

class TestCheckLabelAlignment:
    """
    Tests for check_label_alignment().

    Uses lightweight in-memory meta dicts (no raster files needed) because
    check_label_alignment() operates purely on metadata fields: crs, transform,
    and res — exactly what raster_io.get_meta() returns.
    """

    @staticmethod
    def _meta(
        epsg: int | None = 32633,
        res: float = 10.0,
        x_origin: float = 500_000.0,
        y_origin: float = 5_000_200.0,
    ) -> dict:
        """Build a minimal meta dict compatible with check_label_alignment."""
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
        crs = CRS.from_epsg(epsg) if epsg is not None else None
        transform = from_origin(x_origin, y_origin, res, res)
        return {"crs": crs, "transform": transform, "res": (res, res)}

    def test_label_aligned_passes(self) -> None:
        """Identical metas — all three sub-checks pass."""
        meta = self._meta()
        result = check_label_alignment(meta, meta)
        assert result.status == "ok"
        assert result.check == "label_alignment"
        assert not result.is_critical

    def test_label_crs_mismatch_fails(self) -> None:
        """Different CRS → critical error."""
        label = self._meta(epsg=4326)
        feat  = self._meta(epsg=32633)
        result = check_label_alignment(label, feat)
        assert result.status == "error"
        assert result.is_critical
        assert "CRS mismatch" in result.message

    def test_label_resolution_mismatch_fails(self) -> None:
        """label_res=30 vs feat_res=10 — far outside 0.5 px tolerance → critical error."""
        label = self._meta(res=30.0)
        feat  = self._meta(res=10.0)
        result = check_label_alignment(label, feat)
        assert result.status == "error"
        assert result.is_critical
        assert "Resolution mismatch" in result.message

    def test_label_extent_mismatch_fails(self) -> None:
        """
        Origin displaced 6 m (> 0.5 px × 10 m/px = 5 m tolerance) → critical error.
        """
        feat  = self._meta()
        label = self._meta(x_origin=500_006.0)  # 6 m shift > 5 m tolerance
        result = check_label_alignment(label, feat)
        assert result.status == "error"
        assert result.is_critical
        assert "origin mismatch" in result.message.lower()

    def test_label_tolerance_passes(self) -> None:
        """
        Origin displaced 1 m (0.1 px at 10 m/px, well within 0.5 px tolerance) → ok.
        """
        feat  = self._meta()
        label = self._meta(x_origin=500_001.0)  # 1 m shift < 5 m tolerance
        result = check_label_alignment(label, feat)
        assert result.status == "ok"
        assert not result.is_critical
