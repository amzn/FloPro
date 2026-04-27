"""Flo Pro domain constants and helpers for agent registration.

Public variables use a single group with rich metadata. The var_metadata
DataFrame has asin, vendor_code, inbound_node, week columns so vendors
can map flat indices back to business dimensions.

Aligns with the API contract's public_variable_identifier:
    "asin_vendor_inbound_periods"
"""

from __future__ import annotations

import pandas as pd
from flo_pro_sdk.core.variables import (
    PublicVarGroupMetadata,
    PublicVarGroupName,
    PublicVarsMetadata,
)

from flo_pro_adk.core.types.public_variable_id import (
    PublicVariableId,
)

FLOPRO_GROUP_NAME = PublicVarGroupName("asin_vendor_inbound_periods")


def flopro_var_metadata(
    public_variable_ids: list[PublicVariableId],
    n_weeks: int,
) -> PublicVarsMetadata:
    """Build PublicVarsMetadata for Flo Pro agents.

    Single group with n_vars = len(public_variable_ids) * n_weeks variables.
    The var_metadata DataFrame maps each flat index to its business dimensions.

    Args:
        public_variable_ids: List of (asin, vendor_code, inbound_node) triples.
        n_weeks: Number of planning weeks.
    """
    rows = []
    for pid in public_variable_ids:
        for week in range(n_weeks):
            rows.append({
                "asin": pid["asin"],
                "vendor_code": pid["vendor_code"],
                "inbound_node": pid["inbound_node"],
                "week": week,
            })

    return {
        FLOPRO_GROUP_NAME: PublicVarGroupMetadata(
            name=FLOPRO_GROUP_NAME,
            var_metadata=pd.DataFrame(rows),
        ),
    }
