"""End-to-end CLI tests on the bundled examples and contract violations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
from typer.testing import CliRunner

from mldoctor import cli
from mldoctor.core.context import DiagnosticContext
from mldoctor.core.protocols import BaseDiagnostic
from mldoctor.core.registry import DiagnosticRegistry
from mldoctor.core.schemas import DiagnosticResult, Report
from mldoctor.reports.json_report import load_report

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

EXAMPLE_ARGS = {
    "regression": ["--target", "price", "--row-id", "row_id", "--task", "regression"],
    "classification": ["--target", "churned", "--row-id", "customer_id", "--task",
                       "classification"],
    "grouped_data": ["--target", "gait_speed", "--row-id", "visit_id", "--group",
                     "participant_id", "--evaluation-target", "unseen_groups", "--task",
                     "regression"],
}


def _example(name: str, out: Path, *extra: str) -> tuple[int, Report | None]:
    d = EXAMPLES / name
    args = ["diagnose", "--train", str(d / "train.csv"), "--test", str(d / "test.csv"),
            "--predictions", str(d / "test_predictions.csv"),
            "--train-predictions", str(d / "train_predictions.csv"),
            *EXAMPLE_ARGS[name], "--output", str(out), "--quiet", *extra]
    result = runner.invoke(cli.app, args)
    report = load_report(out) if out.exists() else None
    return result.exit_code, report


def _codes(report: Report) -> dict[str, set[str]]:
    return {r.id: {f"{f.code}:{f.severity}" for f in r.findings} for r in report.diagnostics}


def test_regression_example_detects_planted_issues(tmp_path: Path) -> None:
    code, report = _example("regression", tmp_path / "r.json")
    assert code == 0 and report is not None
    codes = _codes(report)
    assert "HIGH_MISSINGNESS:high" in codes["data_integrity"]
    assert "HIGH_FEATURE_CORRELATION:medium" in codes["feature_correlation"]
    assert any(c.startswith("CROSS_SPLIT_DUPLICATE") for c in codes["leakage"])
    assert "GENERALIZATION_GAP:high" in codes["overfitting"]
    assert "SYSTEMATIC_BIAS:medium" in codes["prediction_errors"]
    statuses = {r.id: r.status for r in report.diagnostics}
    assert statuses["class_imbalance"] == "skipped"
    assert report.summary.checks_total == 8


def test_classification_example(tmp_path: Path) -> None:
    code, report = _example("classification", tmp_path / "c.json")
    assert code == 0 and report is not None
    codes = _codes(report)
    assert "POSSIBLE_TARGET_COPY:high" in codes["leakage"]
    assert any(c.startswith("CLASS_IMBALANCE") for c in codes["class_imbalance"])
    assert "UNEVEN_CLASS_PERFORMANCE:low" in codes["prediction_errors"]
    assert report.dataset.class_labels == ["0", "1"]


def test_grouped_example_uses_cluster_bootstrap(tmp_path: Path) -> None:
    code, report = _example("grouped_data", tmp_path / "g.json")
    assert code == 0 and report is not None
    assert "GROUP_OVERLAP:critical" in _codes(report)["leakage"]
    unc = next(r for r in report.diagnostics if r.id == "uncertainty")
    assert unc.metrics["resampling_unit"] == "group"
    assert unc.metrics["n_units"] == 21
    assert "p04" not in (tmp_path / "g.json").read_text()  # no raw participant IDs


def test_fail_on_threshold_exit_codes(tmp_path: Path) -> None:
    assert _example("grouped_data", tmp_path / "a.json", "--fail-on", "critical")[0] == 1
    assert _example("regression", tmp_path / "b.json", "--fail-on", "critical")[0] == 0
    assert _example("regression", tmp_path / "c.json", "--fail-on", "high")[0] == 1


def test_reports_are_deterministic(tmp_path: Path) -> None:
    _, a = _example("grouped_data", tmp_path / "a.json")
    _, b = _example("grouped_data", tmp_path / "b.json")
    assert a is not None and b is not None

    def stable(report: Report) -> list[dict]:
        return [r.model_dump(exclude={"duration_ms"}) for r in report.diagnostics]

    assert stable(a) == stable(b)
    assert a.run.config_fingerprint == b.run.config_fingerprint
    assert a.dataset.inputs == b.dataset.inputs


def test_strict_json_output(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    _example("classification", out)
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert list(data) == ["schema_version", "tool", "run", "dataset", "summary", "diagnostics",
                          "execution"]
    assert [d["id"] for d in data["diagnostics"]] == [
        "data_integrity", "class_imbalance", "feature_correlation", "outliers", "leakage",
        "overfitting", "uncertainty", "prediction_errors"]


def test_stdout_output(tmp_path: Path) -> None:
    d = EXAMPLES / "regression"
    result = runner.invoke(cli.app, ["diagnose", "--train", str(d / "train.csv"), "--target",
                                     "price", "--task", "regression", "--output", "-"])
    assert result.exit_code == 0
    Report.model_validate_json(result.stdout)


def _write(tmp_path: Path, name: str, frame: pd.DataFrame) -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected", "nonfinite"])
def test_bad_prediction_files_exit_2(tmp_path: Path, problem: str) -> None:
    train = pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2, 3], "y": [1.0, 2, 3]})
    test = pd.DataFrame({"id": ["d", "e", "f"], "x": [1.0, 2, 3], "y": [1.0, 2, 3]})
    preds = {
        "missing": pd.DataFrame({"id": ["d", "e"], "y_pred": [1.0, 2]}),
        "duplicate": pd.DataFrame({"id": ["d", "e", "f", "f"], "y_pred": [1.0, 2, 3, 3]}),
        "unexpected": pd.DataFrame({"id": ["d", "e", "f", "z"], "y_pred": [1.0, 2, 3, 4]}),
        "nonfinite": pd.DataFrame({"id": ["d", "e", "f"], "y_pred": [1.0, float("inf"), 3]}),
    }[problem]
    out = tmp_path / "r.json"
    result = runner.invoke(cli.app, [
        "diagnose", "--train", _write(tmp_path, "tr.csv", train),
        "--test", _write(tmp_path, "te.csv", test),
        "--predictions", _write(tmp_path, "p.csv", preds),
        "--target", "y", "--row-id", "id", "--task", "regression", "--output", str(out)])
    assert result.exit_code == 2
    assert not out.exists()  # no "successful" report from invalid inputs


def test_data_only_invocation_never_claims_false_passes(tmp_path: Path) -> None:
    d = EXAMPLES / "regression"
    out = tmp_path / "r.json"
    result = runner.invoke(cli.app, ["diagnose", "--train", str(d / "train.csv"), "--target",
                                     "price", "--task", "regression", "-o", str(out), "-q"])
    assert result.exit_code == 0
    report = load_report(out)
    by_id = {r.id: r for r in report.diagnostics}
    for check in ("overfitting", "uncertainty", "prediction_errors"):
        assert by_id[check].status == "skipped"
        assert by_id[check].skip_reason
    assert {"cross_split_overlap", "group_overlap"} <= set(by_id["leakage"].skipped_subchecks)


def test_data_command_runs_only_dataset_checks(tmp_path: Path) -> None:
    d = EXAMPLES / "classification"
    out = tmp_path / "r.json"
    result = runner.invoke(cli.app, ["data", "--train", str(d / "train.csv"), "--target",
                                     "churned", "--task", "classification", "-o", str(out), "-q"])
    assert result.exit_code == 0
    report = load_report(out)
    assert [r.id for r in report.diagnostics] == [
        "data_integrity", "class_imbalance", "feature_correlation", "outliers"]
    assert report.run.command == "data"


def test_leakage_command(tmp_path: Path) -> None:
    d = EXAMPLES / "grouped_data"
    out = tmp_path / "r.json"
    result = runner.invoke(cli.app, [
        "leakage", "--train", str(d / "train.csv"), "--test", str(d / "test.csv"),
        "--target", "gait_speed", "--group", "participant_id", "--task", "regression",
        "--evaluation-target", "unseen_groups", "-o", str(out), "-q"])
    assert result.exit_code == 0
    report = load_report(out)
    assert [r.id for r in report.diagnostics] == ["leakage"]
    assert report.summary.highest_severity == "critical"


class _Broken(BaseDiagnostic):
    id = "broken"
    title = "Broken plugin"
    description = "raises"
    limitations: ClassVar[tuple[str, ...]] = ()

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        raise ZeroDivisionError("boom")


def test_diagnostic_error_exit_3_and_partial_report(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from mldoctor import api

    def registry(load_plugins: bool = False) -> DiagnosticRegistry:
        reg = DiagnosticRegistry()
        reg.register(_Broken())
        return reg

    monkeypatch.setattr(api, "default_registry", registry)
    d = EXAMPLES / "regression"
    args = ["diagnose", "--train", str(d / "train.csv"), "--target", "price", "--task",
            "regression", "-q"]
    out = tmp_path / "r.json"
    result = runner.invoke(cli.app, [*args, "-o", str(out)])
    assert result.exit_code == 3
    report = load_report(out)
    assert report.summary.error == 1
    assert "ZeroDivisionError" in (report.diagnostics[0].error_message or "")
    result = runner.invoke(cli.app, [*args, "-o", str(out), "--continue-on-error"])
    assert result.exit_code == 0


def test_config_file_and_cli_precedence(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        'seed = 5\nevaluation_target = "unseen_rows"\n[uncertainty]\nn_bootstrap = 100\n'
    )
    code, report = _example("grouped_data", tmp_path / "r.json", "--config", str(cfg),
                            "--seed", "9")
    assert code == 0 and report is not None
    assert report.run.seed == 9  # CLI overrides file
    assert report.run.evaluation_target == "unseen_groups"  # CLI flag in EXAMPLE_ARGS wins
    unc = next(r for r in report.diagnostics if r.id == "uncertainty")
    assert unc.metrics["n_bootstrap"] == 100  # from file


def test_invalid_config_exit_2(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"correlation": {"threshold": 5}}')
    code, _ = _example("regression", tmp_path / "r.json", "--config", str(cfg))
    assert code == 2


def test_checks_command_lists_all_diagnostics() -> None:
    result = runner.invoke(cli.app, ["checks"])
    assert result.exit_code == 0
    for name in ("data_integrity", "class_imbalance", "feature_correlation", "outliers",
                 "leakage", "overfitting", "uncertainty", "prediction_errors"):
        assert name in result.stdout


def test_report_command_renders_sample(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["report", str(EXAMPLES / "grouped_data" /
                                                   "sample_report.json")])
    assert result.exit_code == 0
    assert "Group overlap" in result.stdout
