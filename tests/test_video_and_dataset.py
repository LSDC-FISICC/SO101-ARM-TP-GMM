"""Silent failure: video timestamp vs state timestamp misalignment.

Builds a tiny synthetic LeRobot v3.0 dataset (2 episodes in one parquet and one
AV1/MPEG-4 file) where every frame encodes its global index as brightness.
"""

from __future__ import annotations

import json

import av
import numpy as np
import pandas as pd
import pytest

from so101_tpgmm.data.lerobot_dataset import LeRobotV3Dataset
from so101_tpgmm.data.video import FrameTimestampError, VideoReader, available_av1_decoders

FPS = 30
LENGTHS = [20, 25]
KEY = "observation.images.top"


def _encode(path, n_frames: int) -> None:
    codec = "libsvtav1" if "libsvtav1" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w") as out:
        st = out.add_stream(codec, rate=FPS)
        st.width, st.height, st.pix_fmt = 64, 48, "yuv420p"
        if codec == "mpeg4":
            st.options = {"qscale": "1"}
        for i in range(n_frames):
            img = np.full((48, 64, 3), (i * 5) % 250, np.uint8)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    vdir = root / "videos" / KEY / "chunk-000"
    vdir.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0", "fps": FPS, "total_frames": sum(LENGTHS), "total_tasks": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "names": [f"j{i}.pos" for i in range(6)]},
            KEY: {"dtype": "video", "shape": [48, 64, 3], "names": ["height", "width", "channels"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    rows, ep_rows, g = [], [], 0
    for e, n in enumerate(LENGTHS):
        for f in range(n):
            rows.append({"observation.state": np.full(6, g, np.float32), "timestamp": f / FPS,
                         "frame_index": f, "episode_index": e, "index": g})
            g += 1
        start = sum(LENGTHS[:e])
        ep_rows.append({"episode_index": e, "tasks": ["t"], "length": n,
                        f"videos/{KEY}/chunk_index": 0, f"videos/{KEY}/file_index": 0,
                        f"videos/{KEY}/from_timestamp": start / FPS, f"videos/{KEY}/to_timestamp": (start + n) / FPS,
                        "stats/dummy/min": [0.0]})
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    _encode(vdir / "file-000.mp4", sum(LENGTHS))
    return LeRobotV3Dataset(root)


def test_metadata_and_episode_slicing(tiny_dataset):
    ds = tiny_dataset
    assert ds.codebase_version == "v3.0" and ds.fps == FPS and ds.camera_keys == [KEY]
    assert "stats/dummy/min" not in ds.episodes.columns
    ep1 = ds.episode(1)
    assert ep1.length == LENGTHS[1]
    np.testing.assert_array_equal(ep1.state[:, 0], np.arange(LENGTHS[0], sum(LENGTHS)))


def test_frames_follow_episode_offset(tiny_dataset):
    """Frame k of episode 1 must be global frame LENGTHS[0] + k (brightness-coded)."""
    frames, t = tiny_dataset.read_frames(KEY, 1, np.array([0, 7, 24]) / FPS)
    np.testing.assert_allclose(t, np.array([0, 7, 24]) / FPS, atol=1e-6)
    for f, k in zip(frames, [0, 7, 24]):
        expected = ((LENGTHS[0] + k) * 5) % 250
        assert abs(float(f.mean()) - expected) < 4.0


def test_out_of_range_timestamp_raises(tiny_dataset):
    path, _, _ = tiny_dataset.video_span(KEY, 0)
    with VideoReader(path) as vr:
        with pytest.raises(FrameTimestampError):
            vr.frame_at(10.0)


def test_av1_decoder_available():
    assert available_av1_decoders(), "No AV1 decoder: install PyAV wheels (bundle libdav1d)"


def test_real_dataset_alignment(real_dataset):
    ep = real_dataset.episode(3)
    q = ep.timestamps[[0, 100, ep.length - 1]]
    _, t = real_dataset.read_frames(real_dataset.camera_keys[-1], 3, q)
    assert np.max(np.abs(t - q)) < 0.5 / real_dataset.fps
