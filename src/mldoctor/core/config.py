"""Configuration models.

Precedence: built-in defaults < config file (JSON or TOML) < CLI flags.

Every threshold below is a *screening heuristic*, not a significance test. Defaults follow
the v0.1 specification where it gives values and are documented in
``docs/statistical_methods.md`` otherwise.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mldoctor.core.schemas import Severity

EvaluationTarget = Literal["unseen_rows", "unseen_groups", "unspecified"]


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class IntegrityConfig(_Cfg):
    missing_warning_fraction: float = Field(0.05, ge=0, le=1)
    missing_high_fraction: float = Field(0.30, ge=0, le=1)
    near_constant_fraction: float = Field(0.99, gt=0, le=1)
    mixed_type_min_fraction: float = Field(0.01, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> IntegrityConfig:
        if self.missing_high_fraction < self.missing_warning_fraction:
            raise ValueError("missing_high_fraction must be >= missing_warning_fraction")
        return self


class ImbalanceConfig(_Cfg):
    min_minority_count: int = Field(20, ge=1)
    min_minority_proportion: float = Field(0.05, gt=0, lt=1)
    severe_minority_count: int = Field(10, ge=1)
    severe_minority_proportion: float = Field(0.02, gt=0, lt=1)


class CorrelationConfig(_Cfg):
    threshold: float = Field(0.90, gt=0, le=1)
    near_perfect_threshold: float = Field(0.99, gt=0, le=1)
    min_pair_n: int = Field(10, ge=3)
    max_features: int = Field(1000, ge=2)
    top_n: int = Field(25, ge=1)


class OutlierConfig(_Cfg):
    modified_z_threshold: float = Field(3.5, gt=0)
    iqr_multiplier: float = Field(1.5, gt=0)
    use_iqr_fallback: bool = True
    min_n: int = Field(5, ge=3)
    top_n: int = Field(25, ge=1)


class LeakageConfig(_Cfg):
    cross_split_high_fraction: float = Field(0.10, ge=0, le=1)
    association_threshold: float = Field(0.98, gt=0, le=1)
    auc_threshold: float = Field(0.99, gt=0.5, le=1)
    purity_threshold: float = Field(0.99, gt=0, le=1)
    min_n: int = Field(10, ge=3)
    suspicious_name_patterns: list[str] = Field(
        default_factory=lambda: [
            r"target",
            r"label",
            r"outcome",
            r"leak",
            r"future",
            r"y_?true",
            r"ground_?truth",
            r"diagnos",
            r"result",
        ]
    )


class OverfittingConfig(_Cfg):
    # Absolute gaps for bounded, higher-is-better metrics.
    score_gap_medium: float = Field(0.05, ge=0)
    score_gap_high: float = Field(0.15, ge=0)
    r2_gap_medium: float = Field(0.10, ge=0)
    r2_gap_high: float = Field(0.25, ge=0)
    # Relative gaps ((test - train) / train) for scale-dependent losses.
    loss_relative_gap_medium: float = Field(0.25, ge=0)
    loss_relative_gap_high: float = Field(0.50, ge=0)


class UncertaintyConfig(_Cfg):
    n_bootstrap: int = Field(1000, ge=50)
    confidence_level: float = Field(0.95, gt=0.5, lt=1)
    min_valid_fraction: float = Field(0.80, gt=0, le=1)
    min_resampling_units: int = Field(30, ge=2)
    # Interval width above which WIDE_METRIC_INTERVAL fires.
    bounded_width_threshold: float = Field(0.10, gt=0)
    relative_width_threshold: float = Field(0.50, gt=0)


class PredictionErrorConfig(_Cfg):
    bias_fraction_threshold: float = Field(0.25, gt=0, le=1)
    prevalence_shift_threshold: float = Field(0.15, gt=0, le=1)
    min_n: int = Field(5, ge=1)
    top_confusions: int = Field(10, ge=1)
    recall_ratio_threshold: float = Field(0.5, gt=0, lt=1)
    min_class_support: int = Field(10, ge=1)


class DoctorConfig(_Cfg):
    """Complete, validated configuration for one run."""

    seed: int = 42
    evaluation_target: EvaluationTarget = "unspecified"
    allow_extra_columns: bool = False
    exclude_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    fail_on: Severity | None = None
    continue_on_error: bool = False
    load_plugins: bool = False
    enabled_diagnostics: list[str] | None = None

    integrity: IntegrityConfig = Field(default_factory=IntegrityConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    outliers: OutlierConfig = Field(default_factory=OutlierConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    overfitting: OverfittingConfig = Field(default_factory=OverfittingConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    prediction_errors: PredictionErrorConfig = Field(default_factory=PredictionErrorConfig)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON form of the effective configuration."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConfigError(ValueError):
    """Raised for unreadable or invalid configuration (CLI exit code 2)."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON or TOML configuration file into a plain dict (no validation yet)."""
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"config file not found: {file}")
    try:
        if file.suffix.lower() == ".toml":
            with file.open("rb") as handle:
                data: Any = tomllib.load(handle)
        elif file.suffix.lower() == ".json":
            data = json.loads(file.read_text(encoding="utf-8"))
        else:
            raise ConfigError(f"unsupported config format '{file.suffix}' (use .json or .toml)")
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not parse config file {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config file must contain a mapping at the top level")
    return data


def build_config(
    config_file: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> DoctorConfig:
    """Merge defaults, an optional config file and CLI overrides into a validated config.

    ``None`` values in ``overrides`` are ignored so that unset CLI flags do not mask
    config-file values.
    """
    data: dict[str, Any] = {}
    if config_file is not None:
        data = load_config_file(config_file)
    if overrides:
        data = _deep_merge(data, {k: v for k, v in overrides.items() if v is not None})
    try:
        return DoctorConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc
