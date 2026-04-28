"""Public variable identity type — matches the API contract."""

from __future__ import annotations

from typing import TypedDict


class PublicVariableId(TypedDict):
    """Matches the API contract's public_variable_id."""

    asin: str
    vendor_code: str
    inbound_node: str


# Keys that identify a public variable — derived from PublicVariableId fields.
PUBLIC_VARIABLE_ID_KEYS: tuple[str, ...] = tuple(PublicVariableId.__annotations__)
