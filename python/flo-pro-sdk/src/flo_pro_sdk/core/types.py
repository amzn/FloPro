"""Shared type aliases for the CPPF framework."""

from typing import Union

# Aligns with https://docs.pydantic.dev/latest/api/types/#pydantic.types.JsonValue
JsonValue = Union[
    str, int, float, bool, None, list["JsonValue"], dict[str, "JsonValue"]
]

# Canonical definition — imported by state.py, var_layout.py, etc.
AgentId = str
