"""
Tests for pipeline/session.py — focused on tmp cleanup logic.

_cleanup_run_tmp is a pure function (no Streamlit), testable directly.
new_run() is tested via a patched st.session_state mock.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.session import _cleanup_run_tmp, new_run


# ── _cleanup_run_tmp ──────────────────────────────────────────────────────────


def test_cleanup_removes_run_dir(tmp_path: Path) -> None:
    """Basic happy path: existing run dir is deleted."""
    run_dir = tmp_path / "AABBCCDD"
    run_dir.mkdir()
    (run_dir / "preprocessed.tif").touch()
    (run_dir / "features.npy").touch()

    _cleanup_run_tmp("AABBCCDD", tmp_path)

    assert not run_dir.exists()


def test_cleanup_nonexistent_is_safe(tmp_path: Path) -> None:
    """Calling cleanup on a non-existent run dir must not raise."""
    _cleanup_run_tmp("DEADBEEF", tmp_path)  # no directory created — must be silent


def test_cleanup_empty_run_id_is_safe(tmp_path: Path) -> None:
    """Empty string run_id returns immediately without touching anything."""
    _cleanup_run_tmp("", tmp_path)


def test_cleanup_security_traversal_skipped(tmp_path: Path) -> None:
    """run_id containing path traversal must be silently skipped."""
    # Place a file one level above tmp_path — must NOT be deleted.
    sensitive = tmp_path.parent / "sensitive_file.txt"
    sensitive.touch()

    _cleanup_run_tmp("../sensitive_file.txt", tmp_path)

    assert sensitive.exists(), "Traversal attack must not delete files outside tmp_dir"
    sensitive.unlink()  # cleanup


def test_cleanup_only_removes_target_subdir(tmp_path: Path) -> None:
    """Sibling directories inside tmp_dir must survive."""
    to_remove = tmp_path / "RUN00001"
    sibling   = tmp_path / "RUN00002"
    to_remove.mkdir()
    sibling.mkdir()
    (to_remove / "data.bin").touch()
    (sibling   / "data.bin").touch()

    _cleanup_run_tmp("RUN00001", tmp_path)

    assert not to_remove.exists()
    assert sibling.exists()


# ── new_run() integration ─────────────────────────────────────────────────────


def _make_session_state(run_id: str | None, cfg: dict | None) -> MagicMock:
    """Return a MagicMock that mimics st.session_state for new_run()."""
    state = MagicMock()
    store = {"run_id": run_id, "config": cfg}
    state.get.side_effect = lambda key, default=None: store.get(key, default)
    return state


def test_new_run_removes_old_tmp(tmp_path: Path) -> None:
    """new_run() deletes tmp/{old_run_id}/ before starting the new run."""
    old_run_id = "OLDRUN01"
    old_dir = tmp_path / old_run_id
    old_dir.mkdir()
    (old_dir / "features.npy").touch()

    cfg = {"tmp_dir": str(tmp_path)}
    fake_state = _make_session_state(old_run_id, cfg)

    with patch("pipeline.session.st") as mock_st:
        mock_st.session_state = fake_state
        new_id = new_run()

    assert not old_dir.exists(), "Old run tmp dir must be removed by new_run()"
    assert new_id != old_run_id


def test_new_run_preserves_logs(tmp_path: Path) -> None:
    """new_run() must not touch logs/ — only tmp/ is cleaned."""
    old_run_id = "LOGSTEST"
    old_dir = tmp_path / old_run_id
    old_dir.mkdir()

    # Simulate a logs directory at the same level as tmp_dir
    logs_dir = tmp_path.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"{old_run_id}.json"
    log_file.write_text('{"run_id": "LOGSTEST"}')

    cfg = {"tmp_dir": str(tmp_path)}
    fake_state = _make_session_state(old_run_id, cfg)

    with patch("pipeline.session.st") as mock_st:
        mock_st.session_state = fake_state
        new_run()

    assert log_file.exists(), "logs/{run_id}.json must not be touched by new_run()"
    # Cleanup
    log_file.unlink()
    logs_dir.rmdir()


def test_new_run_no_previous_run_is_safe(tmp_path: Path) -> None:
    """new_run() with no prior run_id in state must not raise."""
    fake_state = _make_session_state(None, {"tmp_dir": str(tmp_path)})

    with patch("pipeline.session.st") as mock_st:
        mock_st.session_state = fake_state
        new_id = new_run()

    assert isinstance(new_id, str) and len(new_id) == 8


def test_new_run_no_config_is_safe(tmp_path: Path) -> None:
    """new_run() with no config in state must not raise (skips cleanup)."""
    fake_state = _make_session_state("OLDRUN01", None)

    with patch("pipeline.session.st") as mock_st:
        mock_st.session_state = fake_state
        new_id = new_run()

    assert isinstance(new_id, str)


def test_new_run_returns_uppercase_hex(tmp_path: Path) -> None:
    """The returned run_id must be 8 uppercase hex characters."""
    fake_state = _make_session_state(None, {"tmp_dir": str(tmp_path)})

    with patch("pipeline.session.st") as mock_st:
        mock_st.session_state = fake_state
        run_id = new_run()

    assert len(run_id) == 8
    assert run_id == run_id.upper()
    assert all(c in "0123456789ABCDEF" for c in run_id)
