from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from mldoctor.cli import app
from mldoctor.reports.json_report import load_report

runner = CliRunner()


def _train(tmp_path: Path) -> Path:
    path = tmp_path / "train.csv"
    pd.DataFrame({"row_id": [1, 2, 3, 4], "x": [1.0, 2, 3, 4], "y": [2.0, 4, 6, 9]}).to_csv(
        path, index=False
    )
    return path


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_diagnose_writes_valid_report(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["diagnose", "--train", str(_train(tmp_path)), "--target", "y", "--task", "regression",
         "--row-id", "row_id", "--output", str(out), "--quiet"],
    )
    assert result.exit_code == 0, result.output
    report = load_report(out)
    assert report.schema_version == "0.1.0"
    assert report.dataset.train_rows == 4
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_invalid_input_exit_code_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["diagnose", "--train", str(_train(tmp_path)), "--target", "missing", "--task",
              "regression"],
    )
    assert result.exit_code == 2


def test_bad_task_is_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["diagnose", "--train", str(_train(tmp_path)), "--target", "y", "--task", "ranking"]
    )
    assert result.exit_code == 2


def test_report_command_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": "0.1.0", "x": NaN}')
    result = runner.invoke(app, ["report", str(bad)])
    assert result.exit_code == 2
