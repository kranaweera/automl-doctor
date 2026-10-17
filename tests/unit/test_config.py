from __future__ import annotations

import json
from pathlib import Path

import pytest

from mldoctor.core.config import ConfigError, DoctorConfig, build_config


def test_defaults_match_spec() -> None:
    cfg = DoctorConfig()
    assert cfg.seed == 42
    assert cfg.integrity.missing_warning_fraction == 0.05
    assert cfg.integrity.missing_high_fraction == 0.30
    assert cfg.correlation.threshold == 0.90
    assert cfg.correlation.min_pair_n == 10
    assert cfg.outliers.modified_z_threshold == 3.5
    assert cfg.uncertainty.n_bootstrap == 1000
    assert cfg.uncertainty.min_valid_fraction == 0.80
    assert cfg.imbalance.min_minority_count == 20
    assert cfg.imbalance.min_minority_proportion == 0.05


def test_precedence_cli_over_file_over_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"seed": 7, "uncertainty": {"n_bootstrap": 200}}))
    cfg = build_config(path, {"seed": 11, "fail_on": None})
    assert cfg.seed == 11  # CLI wins
    assert cfg.uncertainty.n_bootstrap == 200  # file wins over default
    assert cfg.uncertainty.confidence_level == 0.95  # default kept by deep merge


def test_toml_config(tmp_path: Path) -> None:
    path = tmp_path / "cfg.toml"
    path.write_text('evaluation_target = "unseen_groups"\n[correlation]\nthreshold = 0.8\n')
    cfg = build_config(path)
    assert cfg.evaluation_target == "unseen_groups"
    assert cfg.correlation.threshold == 0.8


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown_key": 1},
        {"integrity": {"missing_warning_fraction": 0.5, "missing_high_fraction": 0.1}},
        {"evaluation_target": "everything"},
        {"uncertainty": {"n_bootstrap": 3}},
    ],
)
def test_invalid_config_rejected(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError):
        build_config(path)


def test_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        build_config(tmp_path / "nope.json")
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 1")
    with pytest.raises(ConfigError, match="unsupported"):
        build_config(path)


def test_fingerprint_is_stable_and_sensitive() -> None:
    assert DoctorConfig().fingerprint() == DoctorConfig().fingerprint()
    assert DoctorConfig(seed=1).fingerprint() != DoctorConfig(seed=2).fingerprint()
