# This project was developed with assistance from AI tools.
"""Shared compilation helpers for KFP pipeline tests."""

from __future__ import annotations

import tempfile

import yaml
from kfp import compiler


def compile_pipeline(pipeline_fn) -> dict:
    """Compile a KFP pipeline and return the first YAML document (pipeline spec)."""
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(pipeline_fn, f.name)
        f.seek(0)
        docs = list(yaml.safe_load_all(f.read()))
        return docs[0]


def compile_pipeline_full_yaml(pipeline_fn) -> str:
    """Compile a KFP pipeline and return the raw YAML string."""
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(pipeline_fn, f.name)
        f.seek(0)
        return f.read().decode()
