"""Ordered diagnostic registry.

Built-in diagnostics are registered in a fixed order so reports are deterministic.
Third-party diagnostics can be exposed through the ``mldoctor.diagnostics`` entry-point
group; they are loaded **only** when ``load_plugins`` is enabled, because loading a plugin
executes its code.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from importlib.metadata import entry_points

from mldoctor.core.protocols import Diagnostic

ENTRY_POINT_GROUP = "mldoctor.diagnostics"
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class RegistryError(ValueError):
    pass


class DiagnosticRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Diagnostic] = {}

    def register(self, diagnostic: Diagnostic, *, replace: bool = False) -> None:
        if not isinstance(diagnostic, Diagnostic):
            raise RegistryError(f"{diagnostic!r} does not implement the Diagnostic protocol")
        if not _ID_PATTERN.match(diagnostic.id):
            raise RegistryError(f"invalid diagnostic id '{diagnostic.id}' (use snake_case)")
        if diagnostic.id in self._items and not replace:
            raise RegistryError(f"diagnostic '{diagnostic.id}' is already registered")
        self._items[diagnostic.id] = diagnostic

    def get(self, diagnostic_id: str) -> Diagnostic:
        try:
            return self._items[diagnostic_id]
        except KeyError:
            raise RegistryError(
                f"unknown diagnostic '{diagnostic_id}'; available: {self.ids()}"
            ) from None

    def ids(self) -> list[str]:
        return list(self._items)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def select(self, ids: list[str] | None) -> list[Diagnostic]:
        """Diagnostics to run, in registry order (unknown ids raise ``RegistryError``)."""
        if ids is None:
            return list(self)
        wanted = set(ids)
        for diagnostic_id in wanted:
            self.get(diagnostic_id)
        return [d for d in self if d.id in wanted]

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> list[str]:
        loaded: list[str] = []
        for ep in sorted(entry_points(group=group), key=lambda e: e.name):
            obj = ep.load()
            diagnostic = obj() if isinstance(obj, type) else obj
            self.register(diagnostic)
            loaded.append(diagnostic.id)
        return loaded


def default_registry(load_plugins: bool = False) -> DiagnosticRegistry:
    from mldoctor.diagnostics import builtin_diagnostics

    registry = DiagnosticRegistry()
    for diagnostic in builtin_diagnostics():
        registry.register(diagnostic)
    if load_plugins:
        registry.load_entry_points()
    return registry
