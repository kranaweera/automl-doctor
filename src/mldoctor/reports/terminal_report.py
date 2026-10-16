"""Terminal rendering of reports (plain fallback; Rich layout added in milestone 5)."""

from __future__ import annotations

from rich.console import Console

from mldoctor.core.schemas import Report


def render_report(report: Report, console: Console | None = None) -> None:
    out = console or Console()
    s = report.summary
    out.print(
        f"AutoML Doctor {report.tool.version} — {report.run.task}: "
        f"{s.passed} passed, {s.warning} warning, {s.failed} failed, "
        f"{s.skipped} skipped, {s.error} error"
    )
    for result in report.diagnostics:
        detail = result.skip_reason or result.error_message or f"{len(result.findings)} finding(s)"
        out.print(f"  [{result.status}] {result.id}: {detail}")
