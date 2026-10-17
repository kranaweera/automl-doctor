from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[[pd.DataFrame, str], Path]:
    def _write(frame: pd.DataFrame, name: str) -> Path:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    return _write
