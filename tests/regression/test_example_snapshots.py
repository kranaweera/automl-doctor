"""Golden snapshots of example reports: statuses and (code, severity) per diagnostic.

If a deliberate change alters these, regenerate with::

    UPDATE_SNAPSHOTS=1 pytest tests/regression
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mldoctor.api import diagnose

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).with_name("example_snapshots.json")

CASES = {
    "regression": dict(target="price", row_id="row_id", task="regression"),
    "classification": dict(target="churned", row_id="customer_id", task="classification"),
    "grouped_data": dict(target="gait_speed", row_id="visit_id", group="participant_id",
                         task="regression", config={"evaluation_target": "unseen_groups"}),
}


def _summarise(name: str) -> dict:
    d = ROOT / "examples" / name
    report = diagnose(train=d / "train.csv", test=d / "test.csv",
                      predictions=d / "test_predictions.csv",
                      train_predictions=d / "train_predictions.csv", **CASES[name])  # type: ignore[arg-type]
    return {
        r.id: {"status": r.status, "findings": [f"{f.code}:{f.severity}" for f in r.findings]}
        for r in report.diagnostics
    }


@pytest.fixture(scope="module")
def snapshots() -> dict:
    current = {name: _summarise(name) for name in CASES}
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    if not SNAPSHOT.exists():
        pytest.fail(f"missing {SNAPSHOT.name}; run with UPDATE_SNAPSHOTS=1 to create it")
    return {"current": current, "golden": json.loads(SNAPSHOT.read_text())}


@pytest.mark.parametrize("name", list(CASES))
def test_example_report_matches_snapshot(snapshots: dict, name: str) -> None:
    assert snapshots["current"][name] == snapshots["golden"][name]
