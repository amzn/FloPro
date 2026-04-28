"""Base exception class for the V-ADK framework."""

from __future__ import annotations


class VADKError(Exception):
    """Base exception for all V-ADK errors."""

    @property
    def error_code(self) -> str:
        return self.__class__.__name__
