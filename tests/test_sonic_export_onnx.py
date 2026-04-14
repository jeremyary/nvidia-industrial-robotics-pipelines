# This project was developed with assistance from AI tools.
"""Tests for SONIC ONNX export entrypoint logic."""

from __future__ import annotations

import pytest


class TestRunEarlyExit:
    """Validate run() exits when S3 is not configured."""

    def test_exits_without_s3(self, monkeypatch):
        """run() calls sys.exit(1) when S3 is not configured."""
        monkeypatch.delenv("S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        from wbc_pipeline.sonic.export_onnx import run

        with pytest.raises(SystemExit, match="1"):
            run()
