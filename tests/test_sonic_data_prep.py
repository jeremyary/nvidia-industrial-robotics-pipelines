# This project was developed with assistance from AI tools.
"""Tests for SONIC data preparation with manifest-based caching."""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from wbc_pipeline.config import S3ConfigBase
from wbc_pipeline.sonic.data_prep import _manifest_exists, _write_manifest, run


class _FakeS3:
    """Minimal S3 client mock for testing manifest logic."""

    def __init__(self, objects: dict | None = None):
        self._objects = objects or {}

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def get_object(self, Bucket, Key):
        if Key in self._objects:
            body = mock.MagicMock()
            body.read.return_value = self._objects[Key]
            return {"Body": body}
        raise self.exceptions.NoSuchKey(f"No such key: {Key}")

    def put_object(self, Bucket, Key, Body):
        self._objects[Key] = Body.encode() if isinstance(Body, str) else Body


class TestManifestExists:
    """Validate manifest lookup in S3."""

    def test_returns_none_when_no_manifest(self):
        """Returns None when manifest.json does not exist in S3."""
        s3 = _FakeS3()
        assert _manifest_exists(s3, "bucket", "bones-seed/processed") is None

    def test_returns_manifest_when_exists(self):
        """Returns parsed manifest dict when manifest.json exists."""
        manifest = {"hf_repo": "bones-studio/seed", "pkl_files": ["motion_lib.pkl"], "prefix": "bones-seed/processed"}
        s3 = _FakeS3({"bones-seed/processed/manifest.json": json.dumps(manifest).encode()})
        result = _manifest_exists(s3, "bucket", "bones-seed/processed")
        assert result == manifest

    def test_returns_none_on_unexpected_error(self):
        """Returns None if S3 raises an unexpected error."""
        s3 = mock.MagicMock()
        s3.exceptions = _FakeS3.exceptions
        s3.get_object.side_effect = RuntimeError("connection lost")
        assert _manifest_exists(s3, "bucket", "prefix") is None


class TestWriteManifest:
    """Validate manifest writing to S3."""

    def test_writes_manifest(self):
        """Writes manifest.json with expected fields."""
        s3 = _FakeS3()
        result = _write_manifest(s3, "bucket", "bones-seed/processed", ["motion_lib.pkl"], "bones-studio/seed")
        assert result["hf_repo"] == "bones-studio/seed"
        assert result["pkl_files"] == ["motion_lib.pkl"]
        stored = json.loads(s3._objects["bones-seed/processed/manifest.json"])
        assert stored == result


class TestRunCaching:
    """Validate that run() skips work when manifest exists."""

    def test_skips_when_manifest_exists(self):
        """Run returns immediately when manifest already in S3."""
        manifest = {"hf_repo": "bones-studio/seed", "pkl_files": ["motion_lib.pkl"], "prefix": "bones-seed/processed"}
        fake_s3 = _FakeS3({"bones-seed/processed/manifest.json": json.dumps(manifest).encode()})

        env = {
            "S3_ENDPOINT": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "S3_DATA_PREFIX": "bones-seed/processed",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(S3ConfigBase, "create_client", return_value=fake_s3),
        ):
            result = run(force=False)

        assert result == "bones-seed/processed"

    def test_force_ignores_manifest(self):
        """Run with force=True downloads even when manifest exists."""
        manifest = {"hf_repo": "bones-studio/seed", "pkl_files": ["motion_lib.pkl"], "prefix": "bones-seed/processed"}
        fake_s3 = _FakeS3({"bones-seed/processed/manifest.json": json.dumps(manifest).encode()})

        env = {
            "S3_ENDPOINT": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "S3_DATA_PREFIX": "bones-seed/processed",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(S3ConfigBase, "create_client", return_value=fake_s3),
            mock.patch("wbc_pipeline.sonic.data_prep._download_from_hf") as mock_dl,
            mock.patch("wbc_pipeline.sonic.data_prep._convert_csv_to_pkl", return_value=["motion_lib.pkl"]),
            mock.patch("wbc_pipeline.sonic.data_prep._upload_to_s3"),
        ):
            result = run(force=True)

        assert result == "bones-seed/processed"
        mock_dl.assert_called_once()

    def test_exits_when_s3_not_configured(self):
        """Run exits with error when S3 is not configured."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit, match="1"):
                run()
