"""Shared, typed context handed to every diagnostic."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mldoctor.core.config import DoctorConfig
from mldoctor.core.schemas import Task


@dataclass
class DiagnosticContext:
    """Validated inputs for one run.

    Invariants (enforced by :func:`mldoctor.api.build_context`):

    * ``train``/``test`` are private copies; diagnostics must not mutate them.
    * For classification the target column holds canonical string labels.
    * ``test_predictions``/``train_predictions`` are indexed exactly like ``test``/``train``
      (joined by row ID), so ``frame[target]`` and the predictions are aligned element-wise.
    * ``feature_columns`` excludes target, row ID, group, time and excluded columns.
    """

    task: Task
    target: str
    train: pd.DataFrame
    feature_columns: list[str]
    numeric_features: list[str]
    config: DoctorConfig
    test: pd.DataFrame | None = None
    row_id: str | None = None
    group: str | None = None
    test_predictions: pd.Series | None = None
    train_predictions: pd.Series | None = None
    class_labels: list[str] | None = None
    test_has_target: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def seed(self) -> int:
        return self.config.seed

    def rng(self, stream: str) -> np.random.Generator:
        """Independent, reproducible generator per named stream.

        Streams are derived from ``(seed, crc32(stream))`` so a diagnostic's random draws do
        not depend on which other diagnostics ran before it.
        """
        return np.random.default_rng([self.config.seed, zlib.crc32(stream.encode("utf-8"))])

    @property
    def has_test(self) -> bool:
        return self.test is not None

    @property
    def has_group(self) -> bool:
        return self.group is not None

    def split(self, name: str) -> pd.DataFrame | None:
        if name == "train":
            return self.train
        if name == "test":
            return self.test
        raise KeyError(name)

    def predictions(self, name: str) -> pd.Series | None:
        if name == "train":
            return self.train_predictions
        if name == "test":
            return self.test_predictions
        raise KeyError(name)

    def splits(self) -> list[tuple[str, pd.DataFrame]]:
        result: list[tuple[str, pd.DataFrame]] = [("train", self.train)]
        if self.test is not None:
            result.append(("test", self.test))
        return result
