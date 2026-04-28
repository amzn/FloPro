"""Tests for flopro registration helpers."""

from __future__ import annotations

from flo_pro_adk.core.types.public_variable_id import (
    PublicVariableId,
)
from flo_pro_adk.flopro.registration import (
    FLOPRO_GROUP_NAME,
    flopro_var_metadata,
)


def test_single_group():
    pids = [
        PublicVariableId(asin="A1", vendor_code="V1", inbound_node="N1"),
        PublicVariableId(asin="A1", vendor_code="V1", inbound_node="N2"),
    ]
    metadata = flopro_var_metadata(pids, n_weeks=3)
    assert len(metadata) == 1
    assert FLOPRO_GROUP_NAME in metadata


def test_var_count():
    pids = [
        PublicVariableId(asin="A1", vendor_code="V1", inbound_node="N1"),
        PublicVariableId(asin="A2", vendor_code="V1", inbound_node="N1"),
    ]
    metadata = flopro_var_metadata(pids, n_weeks=4)
    df = metadata[FLOPRO_GROUP_NAME].var_metadata
    assert len(df) == 8  # 2 pids * 4 weeks


def test_metadata_columns():
    pids = [PublicVariableId(asin="A1", vendor_code="V1", inbound_node="N1")]
    metadata = flopro_var_metadata(pids, n_weeks=3)
    df = metadata[FLOPRO_GROUP_NAME].var_metadata
    assert set(df.columns) == {"asin", "vendor_code", "inbound_node", "week"}
    assert list(df["week"]) == [0, 1, 2]
    assert df["asin"].iloc[0] == "A1"
    assert df["vendor_code"].iloc[0] == "V1"
    assert df["inbound_node"].iloc[0] == "N1"
