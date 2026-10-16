"""Built-in diagnostics, in report order."""

from __future__ import annotations

from mldoctor.core.protocols import Diagnostic


def builtin_diagnostics() -> list[Diagnostic]:
    return []
