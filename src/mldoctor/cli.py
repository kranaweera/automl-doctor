"""``mldoctor`` command-line interface.

Exit codes:

* 0 – run completed and no finding meets ``--fail-on``;
* 1 – at least one finding meets or exceeds ``--fail-on``;
* 2 – invalid CLI usage, input data or configuration;
* 3 – internal execution/serialization error, or any diagnostic ended in ``error``
  (unless ``--continue-on-error`` is set; the partial report is still written).
"""

from __future__ import annotations

import sys
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from mldoctor._version import SCHEMA_VERSION, __version__
from mldoctor.api import DATA_ONLY_DIAGNOSTICS, LEAKAGE_DIAGNOSTICS, diagnose
from mldoctor.core.config import ConfigError, build_config
from mldoctor.core.registry import RegistryError, default_registry
from mldoctor.core.schemas import Report
from mldoctor.io.validation import InputValidationError
from mldoctor.recommendations.rules import severity_at_least
from mldoctor.reports.json_report import (
    ReportFormatError,
    load_report,
    report_to_json,
    write_report,
)
from mldoctor.reports.terminal_report import render_report


class ExitCode(IntEnum):
    OK = 0
    FINDINGS = 1
    INVALID_INPUT = 2
    INTERNAL_ERROR = 3


class TaskChoice(StrEnum):
    classification = "classification"
    regression = "regression"


class SeverityChoice(StrEnum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class EvaluationTargetChoice(StrEnum):
    unseen_rows = "unseen_rows"
    unseen_groups = "unseen_groups"
    unspecified = "unspecified"


app = typer.Typer(
    name="mldoctor",
    help="AutoML Doctor: local-first statistical diagnostics for tabular ML.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

err_console = Console(stderr=True)

TrainOpt = Annotated[Path, typer.Option("--train", help="Training CSV.", show_default=False)]
TestOpt = Annotated[Path | None, typer.Option("--test", help="Held-out/test CSV.")]
PredOpt = Annotated[
    Path | None, typer.Option("--predictions", help="Held-out predictions CSV (row ID, y_pred).")
]
TrainPredOpt = Annotated[
    Path | None,
    typer.Option("--train-predictions", help="Training-set predictions CSV (row ID, y_pred)."),
]
TargetOpt = Annotated[str, typer.Option("--target", help="Target column.", show_default=False)]
TaskOpt = Annotated[
    TaskChoice, typer.Option("--task", help="Prediction task.", case_sensitive=False)
]
RowIdOpt = Annotated[
    str | None, typer.Option("--row-id", help="Unique row identifier used to join predictions.")
]
GroupOpt = Annotated[
    str | None, typer.Option("--group", help="Group / subject identifier column.")
]
ConfigOpt = Annotated[Path | None, typer.Option("--config", help="JSON or TOML config file.")]
OutputOpt = Annotated[
    Path | None, typer.Option("--output", "-o", help="Write JSON report here ('-' for stdout).")
]
SeedOpt = Annotated[int | None, typer.Option("--seed", help="Random seed (default 42).")]
EvalTargetOpt = Annotated[
    EvaluationTargetChoice | None,
    typer.Option(
        "--evaluation-target",
        help="What the held-out set should measure: unseen_rows or unseen_groups.",
    ),
]
FailOnOpt = Annotated[
    SeverityChoice | None,
    typer.Option("--fail-on", help="Exit 1 if any finding has at least this severity."),
]
ContinueOpt = Annotated[
    bool | None,
    typer.Option(
        "--continue-on-error/--no-continue-on-error",
        help="Exit 0/1 even if a diagnostic raised an internal error.",
    ),
]
AllowExtraOpt = Annotated[
    bool | None,
    typer.Option(
        "--allow-extra-columns/--strict-columns",
        help="Allow train/test feature columns to differ (shared columns are used).",
    ),
]
QuietOpt = Annotated[bool, typer.Option("--quiet", "-q", help="Do not print the terminal report.")]


def _run(
    *,
    command: str,
    diagnostics: list[str] | None,
    train: Path,
    target: str,
    task: TaskChoice,
    test: Path | None = None,
    predictions: Path | None = None,
    train_predictions: Path | None = None,
    row_id: str | None = None,
    group: str | None = None,
    config: Path | None = None,
    output: Path | None = None,
    seed: int | None = None,
    evaluation_target: EvaluationTargetChoice | None = None,
    fail_on: SeverityChoice | None = None,
    continue_on_error: bool | None = None,
    allow_extra_columns: bool | None = None,
    quiet: bool = False,
) -> int:
    overrides: dict[str, Any] = {
        "seed": seed,
        "evaluation_target": evaluation_target.value if evaluation_target else None,
        "fail_on": fail_on.value if fail_on else None,
        "continue_on_error": continue_on_error,
        "allow_extra_columns": allow_extra_columns,
    }
    try:
        cfg = build_config(config, overrides)
        report = diagnose(
            train=train,
            test=test,
            predictions=predictions,
            train_predictions=train_predictions,
            target=target,
            task=task.value,  # type: ignore[arg-type]
            row_id=row_id,
            group=group,
            config=cfg,
            diagnostics=diagnostics,
            command=command,
        )
    except (InputValidationError, ConfigError, RegistryError) as exc:
        err_console.print(f"[bold red]Invalid input:[/] {exc}")
        return ExitCode.INVALID_INPUT
    except Exception as exc:  # pragma: no cover - defensive
        err_console.print(f"[bold red]Internal error:[/] {type(exc).__name__}: {exc}")
        return ExitCode.INTERNAL_ERROR

    try:
        to_stdout = output is not None and str(output) == "-"
        if to_stdout:
            sys.stdout.write(report_to_json(report))
        elif output is not None:
            path = write_report(report, output)
            if not quiet:
                err_console.print(f"Report written to [bold]{path}[/]")
        if not quiet and not to_stdout:
            render_report(report)
    except (OSError, ValueError) as exc:
        err_console.print(f"[bold red]Could not write report:[/] {exc}")
        return ExitCode.INTERNAL_ERROR

    return exit_code_for(report, cfg.fail_on, cfg.continue_on_error)


def exit_code_for(report: Report, fail_on: str | None, continue_on_error: bool) -> int:
    if report.summary.error and not continue_on_error:
        err_console.print(
            f"[bold red]{report.summary.error} diagnostic(s) ended in error[/] "
            "(partial report emitted; use --continue-on-error to ignore)"
        )
        return ExitCode.INTERNAL_ERROR
    if fail_on is not None:
        hits = [
            f
            for r in report.diagnostics
            for f in r.findings
            if severity_at_least(f.severity, fail_on)  # type: ignore[arg-type]
        ]
        if hits:
            err_console.print(f"{len(hits)} finding(s) at or above --fail-on={fail_on}")
            return ExitCode.FINDINGS
    return ExitCode.OK


@app.command("diagnose")
def diagnose_cmd(
    train: TrainOpt,
    target: TargetOpt,
    task: TaskOpt,
    test: TestOpt = None,
    predictions: PredOpt = None,
    train_predictions: TrainPredOpt = None,
    row_id: RowIdOpt = None,
    group: GroupOpt = None,
    config: ConfigOpt = None,
    output: OutputOpt = None,
    seed: SeedOpt = None,
    evaluation_target: EvalTargetOpt = None,
    fail_on: FailOnOpt = None,
    continue_on_error: ContinueOpt = None,
    allow_extra_columns: AllowExtraOpt = None,
    quiet: QuietOpt = False,
) -> None:
    """Run all applicable diagnostics."""
    raise typer.Exit(
        _run(
            command="diagnose", diagnostics=None, train=train, target=target, task=task,
            test=test, predictions=predictions, train_predictions=train_predictions,
            row_id=row_id, group=group, config=config, output=output, seed=seed,
            evaluation_target=evaluation_target, fail_on=fail_on,
            continue_on_error=continue_on_error, allow_extra_columns=allow_extra_columns,
            quiet=quiet,
        )
    )


@app.command("data")
def data_cmd(
    train: TrainOpt,
    target: TargetOpt,
    task: TaskOpt,
    group: GroupOpt = None,
    config: ConfigOpt = None,
    output: OutputOpt = None,
    seed: SeedOpt = None,
    fail_on: FailOnOpt = None,
    continue_on_error: ContinueOpt = None,
    quiet: QuietOpt = False,
) -> None:
    """Run dataset checks (integrity, imbalance, correlation, outliers) on training data."""
    raise typer.Exit(
        _run(
            command="data", diagnostics=DATA_ONLY_DIAGNOSTICS, train=train, target=target,
            task=task, group=group, config=config, output=output, seed=seed, fail_on=fail_on,
            continue_on_error=continue_on_error, quiet=quiet,
        )
    )


@app.command("leakage")
def leakage_cmd(
    train: TrainOpt,
    target: TargetOpt,
    task: TaskOpt,
    test: TestOpt = None,
    row_id: RowIdOpt = None,
    group: GroupOpt = None,
    config: ConfigOpt = None,
    output: OutputOpt = None,
    seed: SeedOpt = None,
    evaluation_target: EvalTargetOpt = None,
    fail_on: FailOnOpt = None,
    continue_on_error: ContinueOpt = None,
    allow_extra_columns: AllowExtraOpt = None,
    quiet: QuietOpt = False,
) -> None:
    """Run leakage indicators (cross-split overlap, group overlap, target copies)."""
    raise typer.Exit(
        _run(
            command="leakage", diagnostics=LEAKAGE_DIAGNOSTICS, train=train, target=target,
            task=task, test=test, row_id=row_id, group=group, config=config, output=output,
            seed=seed, evaluation_target=evaluation_target, fail_on=fail_on,
            continue_on_error=continue_on_error, allow_extra_columns=allow_extra_columns,
            quiet=quiet,
        )
    )


@app.command("checks")
def checks_cmd() -> None:
    """List installed checks, their task support and required inputs."""
    console = Console()
    table = Table(title="Installed diagnostics", show_lines=False)
    for column in ("id", "title", "tasks", "requires", "version"):
        table.add_column(column)
    for diagnostic in default_registry():
        table.add_row(
            diagnostic.id,
            diagnostic.title,
            ", ".join(sorted(diagnostic.supported_tasks)),
            ", ".join(sorted(diagnostic.requires)) or "train",
            diagnostic.version,
        )
    console.print(table)


@app.command("report")
def report_cmd(
    path: Annotated[Path, typer.Argument(help="Report JSON produced by mldoctor.")],
) -> None:
    """Validate an existing report JSON and render it in the terminal."""
    try:
        report = load_report(path)
    except ReportFormatError as exc:
        err_console.print(f"[bold red]Invalid report:[/] {exc}")
        raise typer.Exit(ExitCode.INVALID_INPUT) from None
    render_report(report)


@app.command("version")
def version_cmd() -> None:
    """Print package and report-schema versions."""
    typer.echo(f"automl-doctor {__version__} (report schema {SCHEMA_VERSION})")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
