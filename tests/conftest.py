"""Shared fixtures. Tests on the real dataset are skipped if it is not downloaded."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REAL_DATASET = REPO / "data" / "raw" / "dataset3"


@pytest.fixture(scope="session")
def real_dataset():
    from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset

    if not (REAL_DATASET / "meta" / "info.json").exists():
        pytest.skip("real dataset not downloaded (run scripts/inspect_dataset.py --download)")
    return LeRobotV3Dataset(REAL_DATASET)
