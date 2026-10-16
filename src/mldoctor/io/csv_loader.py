"""Local CSV loading.

Only local paths are accepted; nothing is fetched over the network. Identifier columns
(row ID, group ID) are read as strings so that joins never depend on numeric type
inference (e.g. ``"007"`` is not silently converted to ``7``).
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mldoctor.io.validation import InputValidationError

_REMOTE_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://", "file://")


@dataclass(frozen=True)
class LoadedTable:
    """A loaded CSV plus provenance metadata (never raw content)."""

    role: str
    frame: pd.DataFrame
    path: Path | None
    sha256: str | None


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def resolve_local_path(path: str | Path, role: str) -> Path:
    text = str(path)
    if text.lower().startswith(_REMOTE_PREFIXES):
        raise InputValidationError(f"{role}: only local files are supported, got '{text}'")
    resolved = Path(text).expanduser()
    if not resolved.is_file():
        raise InputValidationError(f"{role}: file not found: {resolved}")
    return resolved


def load_csv(path: str | Path, role: str, string_columns: list[str] | None = None) -> LoadedTable:
    """Load a CSV file, reading ``string_columns`` (IDs, groups) as text.

    Raises:
        InputValidationError: missing file, empty file, duplicate header names or parse errors.
    """
    resolved = resolve_local_path(path, role)
    header = _read_header(resolved)
    if not header:
        raise InputValidationError(f"{role}: file is empty: {resolved}")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise InputValidationError(f"{role}: duplicate column names in header: {duplicates}")
    if any(name.strip() == "" for name in header):
        raise InputValidationError(f"{role}: header contains an empty column name")
    dtypes = {name: str for name in (string_columns or []) if name in header}
    try:
        frame = pd.read_csv(resolved, dtype=dtypes, low_memory=False, encoding="utf-8-sig")  # type: ignore[arg-type]
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise InputValidationError(f"{role}: could not parse CSV {resolved}: {exc}") from exc
    if frame.empty:
        raise InputValidationError(f"{role}: no data rows in {resolved}")
    return LoadedTable(role=role, frame=frame, path=resolved, sha256=file_sha256(resolved))


def from_frame(
    frame: pd.DataFrame, role: str, string_columns: list[str] | None = None
) -> LoadedTable:
    """Wrap an in-memory DataFrame (Python API). The caller's frame is copied, never mutated."""
    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError(f"{role}: expected a pandas DataFrame, got {type(frame)!r}")
    if frame.columns.duplicated().any():
        raise InputValidationError(f"{role}: duplicate column names")
    if frame.empty:
        raise InputValidationError(f"{role}: DataFrame has no rows")
    copy = frame.copy()
    copy.columns = [str(c) for c in copy.columns]
    for name in string_columns or []:
        if name in copy.columns:
            copy[name] = copy[name].map(lambda v: v if pd.isna(v) else str(v))
    copy = copy.reset_index(drop=True)
    return LoadedTable(role=role, frame=copy, path=None, sha256=None)
