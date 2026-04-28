"""Tests for PandasDataLoader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flo_pro_adk.core.data.pandas_data_loader import PandasDataLoader


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
    return path


def test_load_reads_csv(csv_file: Path):
    loader = PandasDataLoader(csv_file)
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_caches(csv_file: Path):
    loader = PandasDataLoader(csv_file)
    assert loader.load() is loader.load()


def test_auto_snapshot(csv_file: Path, tmp_path: Path):
    snapshot_dir = tmp_path / "snapshots"
    loader = PandasDataLoader(csv_file, snapshot_dir=snapshot_dir)
    loader.load()
    snapshot_dirs = list(snapshot_dir.iterdir())
    assert len(snapshot_dirs) == 1
    assert (snapshot_dirs[0] / "data.csv").exists()


def test_snapshot_content_matches(csv_file: Path, tmp_path: Path):
    snapshot_dir = tmp_path / "snapshots"
    loader = PandasDataLoader(csv_file, snapshot_dir=snapshot_dir)
    original = loader.load()
    snapshot_csv = next(snapshot_dir.iterdir()) / "data.csv"
    restored = pd.read_csv(snapshot_csv)
    pd.testing.assert_frame_equal(original, restored)
