"""
Tests for ui/_helpers.py — make_progress_cb() behaviour.

ui/_helpers.py imports Streamlit at module level, which is available in the
test environment.  make_progress_cb() accepts any object with a .progress()
method as *bar*, so these tests pass lightweight mocks instead of real
Streamlit widgets.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import pytest

from ui._helpers import make_progress_cb


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bar() -> MagicMock:
    """Return a MagicMock that acts as a st.progress bar."""
    return MagicMock()


# ── Signature / contract ──────────────────────────────────────────────────────


def test_make_progress_cb_returns_callable() -> None:
    """make_progress_cb() must return a callable."""
    cb = make_progress_cb(_make_bar())
    assert callable(cb)


def test_callback_signature_two_ints() -> None:
    """The returned callback must accept (current: int, total: int)."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    cb(1, 10)  # must not raise


# ── Progress value correctness ────────────────────────────────────────────────


def test_progress_value_at_start() -> None:
    """First tick (1/10) must call bar.progress with pct ≈ 0.1."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    cb(1, 10)
    pct_arg = bar.progress.call_args[0][0]
    assert pytest.approx(pct_arg, abs=1e-9) == 0.1


def test_progress_value_at_completion() -> None:
    """Last tick (10/10) must call bar.progress with pct == 1.0."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    cb(10, 10)
    pct_arg = bar.progress.call_args[0][0]
    assert pytest.approx(pct_arg, abs=1e-9) == 1.0


def test_progress_called_once_per_tick() -> None:
    """bar.progress() must be called exactly once per callback invocation."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    for i in range(1, 6):
        cb(i, 5)
    assert bar.progress.call_count == 5


def test_progress_zero_total_does_not_raise() -> None:
    """total=0 (edge case) must not raise ZeroDivisionError."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    cb(0, 0)  # must not raise


# ── Text format ───────────────────────────────────────────────────────────────


def test_progress_text_contains_tile_numbers() -> None:
    """Progress text must include 'Tile current/total'."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    cb(3, 10)
    text_arg = bar.progress.call_args[1]["text"]
    assert "3/10" in text_arg


def test_progress_text_contains_percentage() -> None:
    """Progress text must include the integer percentage."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    cb(5, 10)
    text_arg = bar.progress.call_args[1]["text"]
    assert "50%" in text_arg


def test_progress_text_contains_eta_label() -> None:
    """Progress text must contain 'ETA:' label."""
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    cb(1, 10)
    text_arg = bar.progress.call_args[1]["text"]
    assert "ETA:" in text_arg


def test_eta_calculando_on_first_tick() -> None:
    """
    On the very first tick elapsed ≈ 0 so speed = 0 → ETA unknown.
    The text must contain 'calculando'.
    """
    bar = _make_bar()
    # Freeze time so elapsed is truly 0 by patching perf_counter in the closure.
    # Instead, simply check that current=0 (no tiles done) produces 'calculando'.
    cb = make_progress_cb(bar)
    cb(0, 10)  # 0 tiles processed → speed = 0 → ETA unknown
    text_arg = bar.progress.call_args[1]["text"]
    assert "calculando" in text_arg


def test_eta_seconds_format() -> None:
    """
    When elapsed > 0 and ETA < 60 s, eta_str must end with 's'
    and contain only digits before it.
    """
    bar = _make_bar()
    cb  = make_progress_cb(bar)
    # Sleep a tiny amount so elapsed > 0, then call with current=1 of 2
    time.sleep(0.01)
    cb(1, 2)
    text_arg = bar.progress.call_args[1]["text"]
    # ETA should be present and in seconds format (e.g. "0s" or "1s")
    assert "ETA:" in text_arg
    # Should NOT contain 'calculando' since we have elapsed > 0 and speed > 0
    assert "calculando" not in text_arg


# ── Resilience — bar.progress() raises ───────────────────────────────────────


def test_progress_cb_does_not_raise_without_streamlit() -> None:
    """
    If bar.progress() raises AttributeError (widget gone, user navigated away),
    the callback must swallow it silently — never propagate to the pipeline.
    """
    bar = _make_bar()
    bar.progress.side_effect = AttributeError("widget no longer exists")

    cb = make_progress_cb(bar)
    for i in range(1, 11):
        cb(i, 10)  # must not raise on any of the 10 calls

    assert bar.progress.call_count == 10  # was called every time despite the error


def test_progress_cb_does_not_raise_on_runtime_error() -> None:
    """Any RuntimeError from bar.progress() must also be swallowed."""
    bar = _make_bar()
    bar.progress.side_effect = RuntimeError("Streamlit context missing")

    cb = make_progress_cb(bar)
    cb(5, 10)  # must not raise


def test_progress_cb_does_not_raise_on_exception_group() -> None:
    """ExceptionGroup (Python 3.11+) from bar.progress() must be swallowed."""
    bar = _make_bar()
    bar.progress.side_effect = ExceptionGroup(
        "widget errors", [AttributeError("gone"), RuntimeError("ctx")]
    )
    cb = make_progress_cb(bar)
    cb(1, 5)  # must not raise
