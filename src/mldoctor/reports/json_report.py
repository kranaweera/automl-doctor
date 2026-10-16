"""Versioned JSON report export and import.

Output is strict JSON: ``allow_nan=False`` guarantees that no ``NaN``/``Infinity`` tokens
are written (schema validation already rejects them; this is a second line of defence).
Key order follows the schema definition, and diagnostics/findings are emitted in a
deterministic order.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from mldoctor.core.schemas import Report


class ReportFormatError(ValueError):
    """Raised when a JSON file is not a valid AutoML Doctor report."""


def report_to_json(report: Report, indent: int = 2) -> str:
    payload = report.model_dump(mode="json")
    return json.dumps(payload, indent=indent, allow_nan=False, ensure_ascii=False) + "\n"


def write_report(report: Report, path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = report_to_json(report)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


def _reject_constant(token: str) -> None:
    raise ReportFormatError(f"invalid JSON constant '{token}' (NaN/Infinity are not allowed)")


def load_report(path: str | Path) -> Report:
    file = Path(path).expanduser()
    if not file.is_file():
        raise ReportFormatError(f"report file not found: {file}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ReportFormatError(f"{file} is not valid JSON: {exc}") from exc
    try:
        return Report.model_validate(data)
    except ValidationError as exc:
        raise ReportFormatError(f"{file} does not match the report schema:\n{exc}") from exc
