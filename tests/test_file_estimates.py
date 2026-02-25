"""Tests for file row estimation and grouping."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

from bonepick.data import file_estimates


def _write_jsonl(path: Path) -> None:
    path.write_bytes(b'{"text":"x","label":"ok"}\n')


def test_group_files_by_min_rows_rejects_invalid_params(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    _write_jsonl(sample)

    with pytest.raises(ValueError, match="min_rows_per_group must be > 0"):
        file_estimates.group_files_by_min_rows([sample], min_rows_per_group=0)

    with pytest.raises(ValueError, match="num_proc must be > 0"):
        file_estimates.group_files_by_min_rows([sample], min_rows_per_group=10, num_proc=0)


def test_group_files_by_min_rows_drops_trailing_empty_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "sample.jsonl"
    _write_jsonl(sample)

    monkeypatch.setattr(file_estimates, "estimate_rows_in_file", lambda *_args, **_kwargs: 10)

    groups = file_estimates.group_files_by_min_rows([sample], min_rows_per_group=10, num_proc=1)

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert groups[0][0].path == sample
    assert groups[0][0].est == 10


def test_group_files_by_min_rows_surfaces_non_first_failure_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow_file = tmp_path / "slow.jsonl"
    bad_file = tmp_path / "bad.jsonl"
    _write_jsonl(slow_file)
    _write_jsonl(bad_file)

    slow_started = threading.Event()
    unblock_slow = threading.Event()

    def _fake_estimate(path: Path, min_compressed_bytes: int = 131_072) -> int:  # noqa: ARG001
        if path == slow_file:
            slow_started.set()
            unblock_slow.wait(timeout=1.0)
            return 123
        assert slow_started.wait(timeout=0.5)
        raise ValueError("boom")

    monkeypatch.setattr(file_estimates, "estimate_rows_in_file", _fake_estimate)

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match=re.escape(f"Failed to estimate rows for {bad_file}: boom")):
        file_estimates.group_files_by_min_rows([slow_file, bad_file], min_rows_per_group=100, num_proc=2)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.7
