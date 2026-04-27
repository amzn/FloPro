"""Tests for InMemoryDataLoader."""

from __future__ import annotations

from flo_pro_adk.core.data.in_memory_data_loader import InMemoryDataLoader


def test_load_returns_data():
    assert InMemoryDataLoader({"key": "value"}).load() == {"key": "value"}


def test_load_returns_same_data_every_call():
    loader = InMemoryDataLoader([1, 2, 3])
    assert loader.load() == loader.load()


def test_snapshot_is_noop():
    loader = InMemoryDataLoader("data")
    loader.snapshot("run-123")
    assert loader.load() == "data"
