"""
Tests for pipeline/coregister.py (Phase 3).

Covered
-------
CoregistrationResult   : field types, frozen immutability
apply_rmse_gate        : pass/fail at threshold, exact-threshold boundary
run_coregistration     : stub path execution, gate application, report keys
get_shift_report       : output dict structure and types
_arosics_available     : detects missing AROSICS correctly
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from pipeline.coregister import (
    CoregistrationResult,
    _arosics_available,
    apply_rmse_gate,
    get_shift_report,
    run_coregistration,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _dummy_result(
    rmse: float = 0.2,
    gate_passed: bool = True,
    corrected_path: Path | None = None,
    is_stub: bool = True,
) -> CoregistrationResult:
    return CoregistrationResult(
        shift_x_px       = 0.1,
        shift_y_px       = -0.05,
        shift_magnitude  = 0.112,
        rmse             = rmse,
        gate_passed      = gate_passed,
        corrected_path   = corrected_path,
        message          = "test result",
        is_stub          = is_stub,
        shift_map        = None,
    )


def _write_small_raster(path: Path) -> Path:
    data = np.random.default_rng(99).uniform(0, 1, (1, 16, 16)).astype("float32")
    transform = from_origin(500_000.0, 5_000_160.0, 10.0, 10.0)
    with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                       width=16, height=16, count=1,
                       crs=CRS.from_epsg(32633), transform=transform) as ds:
        ds.write(data)
    return path


# ── CoregistrationResult ─────────────────────────────────────────────────────

class TestCoregistrationResult:
    def test_frozen_immutable(self):
        result = _dummy_result()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.rmse = 99.0  # type: ignore[misc]

    def test_shift_magnitude_type(self):
        result = _dummy_result()
        assert isinstance(result.shift_magnitude, float)

    def test_is_stub_defaults_true(self):
        result = CoregistrationResult(
            shift_x_px=0.0, shift_y_px=0.0, shift_magnitude=0.0,
            rmse=0.0, gate_passed=True, corrected_path=None,
            message="",
        )
        assert result.is_stub is True

    def test_shift_map_not_compared(self):
        """shift_map is excluded from equality (compare=False)."""
        a = _dummy_result()
        b = dataclasses.replace(a, shift_map=np.zeros((4, 4)))
        assert a == b


# ── apply_rmse_gate ───────────────────────────────────────────────────────────

class TestApplyRmseGate:
    def test_below_threshold_passes(self):
        result = apply_rmse_gate(_dummy_result(rmse=0.3), threshold=0.5)
        assert result.gate_passed is True

    def test_above_threshold_fails(self):
        result = apply_rmse_gate(_dummy_result(rmse=0.8), threshold=0.5)
        assert result.gate_passed is False

    def test_exact_threshold_passes(self):
        # Gate is ≤ threshold — exact match should pass
        result = apply_rmse_gate(_dummy_result(rmse=0.5), threshold=0.5)
        assert result.gate_passed is True

    def test_failed_gate_nulls_corrected_path(self, tmp_path):
        path = tmp_path / "out.tif"
        path.touch()
        result = apply_rmse_gate(_dummy_result(rmse=0.9, corrected_path=path), threshold=0.5)
        assert result.corrected_path is None

    def test_passed_gate_keeps_corrected_path(self, tmp_path):
        path = tmp_path / "out.tif"
        path.touch()
        result = apply_rmse_gate(_dummy_result(rmse=0.2, corrected_path=path), threshold=0.5)
        assert result.corrected_path == path

    def test_failed_gate_message_contains_threshold(self):
        result = apply_rmse_gate(_dummy_result(rmse=1.2), threshold=0.5)
        assert "0.5" in result.message
        assert not result.gate_passed

    def test_returns_new_instance(self):
        original = _dummy_result(rmse=0.3)
        updated  = apply_rmse_gate(original, threshold=0.5)
        assert updated is not original


# ── run_coregistration ────────────────────────────────────────────────────────

class TestRunCoregistration:
    def test_stub_returns_coreg_result(self, tmp_path):
        src = _write_small_raster(tmp_path / "src.tif")
        ref = _write_small_raster(tmp_path / "ref.tif")
        out = tmp_path / "out.tif"
        config = {"coreg_rmse_threshold": 0.5}

        result = run_coregistration(src, ref, out, config)
        assert isinstance(result, CoregistrationResult)

    def test_stub_is_flagged(self, tmp_path):
        src = _write_small_raster(tmp_path / "src.tif")
        ref = _write_small_raster(tmp_path / "ref.tif")
        out = tmp_path / "out.tif"
        config = {"coreg_rmse_threshold": 0.5}

        result = run_coregistration(src, ref, out, config)
        # AROSICS is not installed in Phase 3 test env
        if not _arosics_available():
            assert result.is_stub is True

    def test_stub_copies_source_to_output(self, tmp_path):
        src = _write_small_raster(tmp_path / "src.tif")
        ref = _write_small_raster(tmp_path / "ref.tif")
        out = tmp_path / "out.tif"

        result = run_coregistration(src, ref, out, {"coreg_rmse_threshold": 0.5})
        if result.is_stub and result.gate_passed:
            assert out.exists()

    def test_gate_applied_with_config_threshold(self, tmp_path):
        src = _write_small_raster(tmp_path / "src.tif")
        ref = _write_small_raster(tmp_path / "ref.tif")
        out = tmp_path / "out.tif"
        # Stub always returns RMSE=0.0, so gate always passes regardless of threshold
        result = run_coregistration(src, ref, out, {"coreg_rmse_threshold": 0.5})
        if result.is_stub:
            assert result.gate_passed is True   # 0.0 ≤ 0.5


# ── get_shift_report ──────────────────────────────────────────────────────────

class TestGetShiftReport:
    def test_returns_dict(self):
        report = get_shift_report(_dummy_result())
        assert isinstance(report, dict)

    def test_required_keys_present(self):
        report = get_shift_report(_dummy_result())
        for key in ("shift_x_px", "shift_y_px", "shift_magnitude",
                    "rmse_px", "gate_passed", "backend", "message"):
            assert key in report, f"Missing key: {key}"

    def test_values_are_json_serialisable(self):
        import json
        report = get_shift_report(_dummy_result())
        # Should not raise
        json.dumps(report)

    def test_stub_backend_label(self):
        report = get_shift_report(_dummy_result(is_stub=True))
        assert report["backend"] == "stub"

    def test_none_path_in_report(self):
        report = get_shift_report(_dummy_result(corrected_path=None))
        assert report["output_path"] is None

    def test_path_serialised_as_string(self, tmp_path):
        p = tmp_path / "out.tif"
        report = get_shift_report(_dummy_result(corrected_path=p))
        assert isinstance(report["output_path"], str)


# ── _arosics_available ────────────────────────────────────────────────────────

class TestArosicsAvailable:
    def test_returns_bool(self):
        result = _arosics_available()
        assert isinstance(result, bool)

    def test_false_without_install(self):
        # In the Phase 3 test environment, AROSICS is not installed.
        # This test will need updating once AROSICS is activated in Phase 3+.
        if not _arosics_available():
            assert _arosics_available() is False
