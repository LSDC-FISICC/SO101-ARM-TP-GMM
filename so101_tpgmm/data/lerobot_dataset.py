"""Minimal, dependency-light reader for LeRobot v3.0 datasets.

We read the parquet/MP4 layout directly instead of importing ``lerobot`` so the
offline pipeline runs on any machine (``lerobot`` pulls torch and is only
needed to drive the real robot).

v3.0 layout::

    meta/info.json                         fps, features, path templates
    meta/episodes/chunk-XXX/file-YYY.parquet   one row per episode (+ per-episode stats)
    data/chunk-XXX/file-YYY.parquet        many episodes per file
    videos/<key>/chunk-XXX/file-YYY.mp4    many episodes per file, offset by from_timestamp
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from so101_tpgmm.data.video import VideoReader

SUPPORTED_CODEBASE_VERSIONS = ("v3.0",)


@dataclass
class Episode:
    """Per-episode tabular data (follower state only; the leader action is not loaded)."""

    index: int
    timestamps: np.ndarray  # (T,) seconds from episode start
    frame_index: np.ndarray  # (T,)
    state: np.ndarray  # (T, 6) stored units (see data/units.py)

    @property
    def length(self) -> int:
        return int(self.state.shape[0])


def download_dataset(
    repo_id: str,
    local_dir: str | Path,
    video_keys: list[str] | None = None,
    revision: str = "main",
) -> Path:
    """Download metadata, tabular data and selected camera videos from the Hub."""
    from huggingface_hub import snapshot_download

    patterns = ["meta/*", "meta/**", "data/**", "README.md"]
    for key in video_keys or []:
        patterns.append(f"videos/{key}/**")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=patterns,
    )
    return Path(path)


class LeRobotV3Dataset:
    """Read-only access to a local LeRobot v3.0 dataset.

    Args:
        root: Local dataset directory (the one containing ``meta/``).
        state_key: Column with the follower joint state.
    """

    def __init__(self, root: str | Path, state_key: str = "observation.state") -> None:
        self.root = Path(root)
        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"{self.root}/meta/info.json not found; download the dataset first.")
        self.state_key = state_key
        with open(self.root / "meta" / "info.json", "r", encoding="utf-8") as f:
            self.info: dict = json.load(f)
        ver = self.info.get("codebase_version")
        if ver not in SUPPORTED_CODEBASE_VERSIONS:
            raise ValueError(f"Unsupported LeRobot codebase_version {ver!r}; expected {SUPPORTED_CODEBASE_VERSIONS}")

    # ------------------------------------------------------------------ metadata
    @property
    def codebase_version(self) -> str:
        return str(self.info["codebase_version"])

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def features(self) -> dict:
        return dict(self.info["features"])

    @property
    def camera_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if v.get("dtype") == "video"]

    @property
    def state_names(self) -> list[str]:
        return list(self.features[self.state_key]["names"])

    @cached_property
    def episodes(self) -> pd.DataFrame:
        """Episode table without the (large) per-episode stats columns."""
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError("No meta/episodes parquet files found.")
        frames = []
        for f in files:
            df = pd.read_parquet(f)
            frames.append(df[[c for c in df.columns if not c.startswith("stats/")]])
        ep = pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
        return ep

    @property
    def num_episodes(self) -> int:
        return int(len(self.episodes))

    # ------------------------------------------------------------------ tabular
    @cached_property
    def frames(self) -> pd.DataFrame:
        """All frames (state, timestamp, indices). The leader ``action`` is dropped."""
        cols = [self.state_key, "timestamp", "frame_index", "episode_index", "index"]
        files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
        return df.sort_values("index").reset_index(drop=True)

    @cached_property
    def _episode_slices(self) -> dict[int, np.ndarray]:
        ep = self.frames["episode_index"].to_numpy()
        order = np.argsort(ep, kind="stable")
        bounds = np.flatnonzero(np.diff(ep[order])) + 1
        groups = np.split(order, bounds)
        return {int(ep[g[0]]): g for g in groups}

    def episode(self, episode_index: int) -> Episode:
        """Return the follower trajectory of one episode."""
        rows = self._episode_slices[int(episode_index)]
        sub = self.frames.iloc[rows]
        return Episode(
            index=int(episode_index),
            timestamps=sub["timestamp"].to_numpy(np.float64),
            frame_index=sub["frame_index"].to_numpy(np.int64),
            state=np.stack(sub[self.state_key].to_numpy()).astype(np.float64),
        )

    def all_states(self) -> np.ndarray:
        """All follower states as an ``(N, 6)`` array."""
        return np.stack(self.frames[self.state_key].to_numpy()).astype(np.float64)

    # ------------------------------------------------------------------ video
    def video_span(self, video_key: str, episode_index: int) -> tuple[Path, float, float]:
        """(mp4 path, from_timestamp, to_timestamp) of an episode inside its video file."""
        row = self.episodes.loc[self.episodes["episode_index"] == episode_index].iloc[0]
        chunk = int(row[f"videos/{video_key}/chunk_index"])
        file = int(row[f"videos/{video_key}/file_index"])
        rel = self.info["video_path"].format(video_key=video_key, chunk_index=chunk, file_index=file)
        return (
            self.root / rel,
            float(row[f"videos/{video_key}/from_timestamp"]),
            float(row[f"videos/{video_key}/to_timestamp"]),
        )

    def read_frames(
        self, video_key: str, episode_index: int, timestamps_s: list[float] | np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Decode RGB frames of an episode at episode-relative timestamps.

        Returns the frames and the decoded *episode-relative* times, which the
        caller can compare with the state timestamps.
        """
        path, t0, _ = self.video_span(video_key, episode_index)
        with VideoReader(path, tolerance_s=0.4 / self.fps) as vr:
            frames, times = vr.frames_at(np.asarray(timestamps_s, dtype=np.float64) + t0)
        return frames, times - t0

    def first_frame(self, video_key: str, episode_index: int) -> np.ndarray:
        """First RGB frame of an episode (arm at home in this dataset)."""
        frames, _ = self.read_frames(video_key, episode_index, [0.0])
        return frames[0]
