"""Rich terminal rendering of a validated :class:`~mldoctor.core.schemas.Report`.

Rendering is read-only: it never recomputes statistics, so the terminal view and the JSON
report always agree.
"""

from __future__ import annotations

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mldoctor.core.schemas import SEVERITY_ORDER, DiagnosticResult, Report, highest_severity

STATUS_STYLE = {
    "passed": "bold green",
    "warning": "bold yellow",
    "failed": "bold red",
    "skipped": "dim",
    "error": "bold magenta",
}
SEVERITY_STYLE = {
    "info": "cyan",
    "low": "yellow",
    "medium": "dark_orange",
    "high": "red",
    "critical": "bold white on red",
}


def _header(report: Report) -> Panel:
    ds = report.dataset
    lines = [
        f"[bold]Task[/] {report.run.task}    [bold]Seed[/] {report.run.seed}    "
        f"[bold]Command[/] {report.run.command}    "
        f"[bold]Evaluation target[/] {report.run.evaluation_target}",
        f"[bold]Train[/] {ds.train_rows} rows    [bold]Test[/] "
        f"{'—' if ds.test_rows is None else f'{ds.test_rows} rows'}    "
        f"[bold]Features[/] {ds.feature_count} ({ds.numeric_feature_count} numeric)    "
        f"[bold]Target[/] {ds.target}",
        f"[bold]Group[/] {ds.group_column or '—'}    "
        f"[bold]Test predictions[/] {'yes' if ds.has_test_predictions else 'no'}    "
        f"[bold]Train predictions[/] {'yes' if ds.has_train_predictions else 'no'}",
    ]
    title = f"AutoML Doctor {report.tool.version} · report schema {report.schema_version}"
    return Panel("\n".join(lines), title=title, title_align="left", box=box.ROUNDED)


def _summary(report: Report) -> Table:
    s = report.summary
    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Findings", justify="right")
    table.add_column("Highest severity")
    table.add_column("Notes", overflow="fold")
    for result in report.diagnostics:
        sev = highest_severity([f.severity for f in result.findings])
        note = result.skip_reason or result.error_message or ""
        if not note and result.skipped_subchecks:
            note = f"{len(result.skipped_subchecks)} sub-check(s) not run"
        table.add_row(
            result.title or result.id,
            Text(result.status, style=STATUS_STYLE[result.status]),
            str(len(result.findings)) if result.status not in ("skipped", "error") else "—",
            Text(sev, style=SEVERITY_STYLE[sev]) if sev else Text("—", style="dim"),
            Text(note, style="dim"),
        )
    caption = (
        f"{s.checks_total} checks: {s.passed} passed · {s.warning} warning · {s.failed} failed · "
        f"{s.skipped} skipped · {s.error} error"
    )
    table.caption = caption
    return table


def _findings(result: DiagnosticResult, min_severity: str) -> Panel | None:
    floor = SEVERITY_ORDER[min_severity]
    shown = [f for f in result.findings if SEVERITY_ORDER[f.severity] >= floor]
    if not shown and not result.skipped_subchecks:
        return None
    blocks: list[Text] = []
    for finding in shown:
        text = Text()
        text.append(f" {finding.severity.upper()} ", style=SEVERITY_STYLE[finding.severity])
        text.append(f" {finding.title} ", style="bold")
        text.append(f"[{finding.code}]\n", style="dim")
        text.append(f"  {finding.message}\n")
        text.append("  → ", style="green")
        text.append(f"{finding.recommendation}\n", style="green")
        blocks.append(text)
    for name, reason in result.skipped_subchecks.items():
        blocks.append(Text(f"  not run: {name} — {reason}", style="dim"))
    hidden = len(result.findings) - len(shown)
    if hidden:
        blocks.append(
            Text(f"  ({hidden} lower-severity finding(s) in the JSON report)", style="dim")
        )
    return Panel(
        Group(*blocks),
        title=f"{result.title or result.id}",
        title_align="left",
        border_style=STATUS_STYLE[result.status].replace("bold ", ""),
        box=box.ROUNDED,
    )


def render_report(
    report: Report, console: Console | None = None, min_severity: str = "low"
) -> None:
    """Print the report: header, per-check summary, then findings at ``min_severity``+."""
    out = console or Console()
    out.print(_header(report))
    out.print(_summary(report))
    for result in report.diagnostics:
        panel = _findings(result, min_severity)
        if panel is not None:
            out.print(panel)
    if any(r.id == "leakage" and r.status != "skipped" for r in report.diagnostics):
        out.print(
            Text(
                "Note: leakage checks are indicators only; pipeline/preprocessing leakage cannot "
                "be ruled out from CSV files.",
                style="dim italic",
            )
        )
    out.print(
        Text(
            "Thresholds are configurable screening heuristics, not significance tests.",
            style="dim italic",
        )
    )
