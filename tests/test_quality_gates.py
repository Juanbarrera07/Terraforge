"""
tests/test_quality_gates.py — Unit tests for the 3-tier quality gate system.

Tests cover:
    test_all_pass        — all metrics above pass thresholds
    test_oa_warning      — OA falls in the warning band (between fail and pass)
    test_f1_fail_blocks  — minority F1 below fail threshold → has_gate_failures is True
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pipeline.postprocess import (
    QualityGateResult,
    has_gate_failures,
    run_quality_gates,
)


# ── Minimal fakes ─────────────────────────────────────────────────────────────

@dataclass
class _FakeModel:
    oa:          float
    kappa:       float
    minority_f1: float


@dataclass
class _FakeAccuracy:
    oa:    float
    kappa: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(
    oa_pass=0.90,   oa_fail=0.80,
    kappa_pass=0.80, kappa_fail=0.65,
    f1_pass=0.70,   f1_fail=0.50,
    conf_pass=0.75, conf_fail=0.60,
    nd_pass=0.05,   nd_fail=0.10,
) -> dict:
    return {
        "quality_gate_oa_pass":          oa_pass,
        "quality_gate_oa_fail":          oa_fail,
        "quality_gate_kappa_pass":       kappa_pass,
        "quality_gate_kappa_fail":       kappa_fail,
        "quality_gate_f1_pass":          f1_pass,
        "quality_gate_f1_fail":          f1_fail,
        "quality_gate_confidence_pass":  conf_pass,
        "quality_gate_confidence_fail":  conf_fail,
        "quality_gate_nodata_pass":      nd_pass,
        "quality_gate_nodata_fail":      nd_fail,
    }


_CONF_STATS_GOOD = {"mean": 0.85}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAllPass:
    """All metrics comfortably above their pass thresholds → every gate is PASS."""

    def test_all_pass_statuses(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = _CONF_STATS_GOOD,
            cfg              = _cfg(),
            nodata_pct       = 0.02,
        )

        assert len(results) > 0, "Expected at least one gate result"
        for r in results:
            assert r.status == "pass", (
                f"Expected PASS for {r.metric_name!r}, got {r.status!r} "
                f"(value={r.value}, pass_threshold={r.threshold_pass})"
            )

    def test_all_pass_no_failures(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = _CONF_STATS_GOOD,
            cfg              = _cfg(),
            nodata_pct       = 0.02,
        )
        assert not has_gate_failures(results)

    def test_returns_quality_gate_result_instances(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = None,
            cfg              = _cfg(),
        )
        for r in results:
            assert isinstance(r, QualityGateResult)

    def test_includes_independent_oa_when_accuracy_provided(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.75)
        acc     = _FakeAccuracy(oa=0.93, kappa=0.82)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = acc,
            class_areas      = None,
            confidence_stats = _CONF_STATS_GOOD,
            cfg              = _cfg(),
        )
        metric_names = [r.metric_name for r in results]
        assert any("independent" in n for n in metric_names), (
            "Expected an independent OA gate when accuracy_result is provided"
        )
        assert not has_gate_failures(results)


class TestOAWarning:
    """OA in the warning band (0.80 ≤ OA < 0.90) → WARNING, not PASS or FAIL."""

    def test_oa_warning_status(self):
        model   = _FakeModel(oa=0.85, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = _CONF_STATS_GOOD,
            cfg              = _cfg(),
            nodata_pct       = 0.02,
        )
        oa_result = next(
            r for r in results if "Overall Accuracy (training)" == r.metric_name
        )
        assert oa_result.status == "warning", (
            f"OA=0.85 should be WARNING (pass≥0.90, fail<0.80), got {oa_result.status!r}"
        )

    def test_warning_does_not_block(self):
        model   = _FakeModel(oa=0.85, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = _CONF_STATS_GOOD,
            cfg              = _cfg(),
            nodata_pct       = 0.02,
        )
        assert not has_gate_failures(results), (
            "A WARNING should not be treated as a failure by has_gate_failures()"
        )

    def test_oa_exactly_at_fail_threshold_is_warning(self):
        """OA exactly at the fail threshold (0.80) should be WARNING, not FAIL."""
        model   = _FakeModel(oa=0.80, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = None,
            cfg              = _cfg(),
        )
        oa_result = next(
            r for r in results if "Overall Accuracy (training)" == r.metric_name
        )
        assert oa_result.status == "warning"

    def test_oa_exactly_at_pass_threshold_is_pass(self):
        model   = _FakeModel(oa=0.90, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = None,
            cfg              = _cfg(),
        )
        oa_result = next(
            r for r in results if "Overall Accuracy (training)" == r.metric_name
        )
        assert oa_result.status == "pass"


class TestF1FailBlocks:
    """Minority F1 below the fail threshold → gate FAIL, has_gate_failures True."""

    def test_f1_below_fail_threshold_is_fail(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.40)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = _CONF_STATS_GOOD,
            cfg              = _cfg(),
            nodata_pct       = 0.02,
        )
        f1_result = next(
            r for r in results if "Minority F1" == r.metric_name
        )
        assert f1_result.status == "fail", (
            f"F1=0.40 should be FAIL (fail_threshold=0.50), got {f1_result.status!r}"
        )

    def test_f1_fail_triggers_has_gate_failures(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.40)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = _CONF_STATS_GOOD,
            cfg              = _cfg(),
            nodata_pct       = 0.02,
        )
        assert has_gate_failures(results), (
            "has_gate_failures() must return True when any gate is FAIL"
        )

    def test_f1_just_above_fail_threshold_is_warning(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.51)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = None,
            cfg              = _cfg(),
        )
        f1_result = next(
            r for r in results if "Minority F1" == r.metric_name
        )
        assert f1_result.status == "warning"

    def test_nodata_above_fail_threshold_is_fail(self):
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.75)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = None,
            cfg              = _cfg(),
            nodata_pct       = 0.15,   # > 10% fail threshold
        )
        nd_result = next(
            r for r in results if "Nodata Coverage" == r.metric_name
        )
        assert nd_result.status == "fail"
        assert has_gate_failures(results)

    def test_none_model_skips_model_gates(self):
        results = run_quality_gates(
            model_result     = None,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = None,
            cfg              = _cfg(),
        )
        assert results == [], "No model → no gates should be emitted"

    def test_gate_result_thresholds_match_cfg(self):
        cfg     = _cfg(f1_pass=0.80, f1_fail=0.60)
        model   = _FakeModel(oa=0.95, kappa=0.88, minority_f1=0.40)
        results = run_quality_gates(
            model_result     = model,
            accuracy_result  = None,
            class_areas      = None,
            confidence_stats = None,
            cfg              = cfg,
        )
        f1_result = next(r for r in results if "Minority F1" == r.metric_name)
        assert f1_result.threshold_pass == 0.80
        assert f1_result.threshold_fail == 0.60
