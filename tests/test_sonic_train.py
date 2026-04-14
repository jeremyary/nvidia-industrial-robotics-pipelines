# This project was developed with assistance from AI tools.
"""Tests for SONIC training entrypoint logic."""

from __future__ import annotations

import pytest

from wbc_pipeline.sonic.train import _find_experiment_dir


class TestFindExperimentDir:
    """Validate _find_experiment_dir locates the most recent experiment."""

    def test_returns_none_when_logs_dir_missing(self, tmp_path):
        """Returns None when logs_rl/ does not exist."""
        assert _find_experiment_dir(tmp_path) is None

    def test_returns_none_when_no_checkpoints(self, tmp_path):
        """Returns None when logs_rl/ exists but has no .pt files."""
        (tmp_path / "logs_rl" / "exp1").mkdir(parents=True)
        assert _find_experiment_dir(tmp_path) is None

    def test_finds_directory_with_last_pt(self, tmp_path):
        """Returns directory containing last.pt."""
        exp_dir = tmp_path / "logs_rl" / "experiment" / "run_001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "last.pt").touch()
        result = _find_experiment_dir(tmp_path)
        assert result == exp_dir

    def test_returns_most_recent_when_multiple(self, tmp_path):
        """Returns the most recently modified last.pt directory."""
        old_dir = tmp_path / "logs_rl" / "exp" / "old"
        old_dir.mkdir(parents=True)
        old_pt = old_dir / "last.pt"
        old_pt.touch()

        new_dir = tmp_path / "logs_rl" / "exp" / "new"
        new_dir.mkdir(parents=True)
        new_pt = new_dir / "last.pt"
        new_pt.touch()

        # Ensure new is actually newer by setting mtime
        import os
        import time

        old_time = time.time() - 100
        os.utime(old_pt, (old_time, old_time))

        result = _find_experiment_dir(tmp_path)
        assert result == new_dir


class TestRunEarlyExit:
    """Validate run() exits when S3 is not configured."""

    def test_exits_without_s3(self, monkeypatch):
        """run() calls sys.exit(1) when S3 is not configured."""
        monkeypatch.delenv("S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        from wbc_pipeline.sonic.train import run

        with pytest.raises(SystemExit, match="1"):
            run()
