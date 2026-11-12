"""Frame-accurate video decoding for LeRobot v3 MP4 files (AV1 via PyAV/libdav1d).

LeRobot v3 concatenates many episodes into one MP4 and stores, per episode,
``from_timestamp``. The frame of an episode sample is therefore at
``from_timestamp + timestamp``. We decode with PyAV and *verify* that the
returned frame's presentation time matches the query within a tolerance,
so a misalignment fails loudly instead of returning the wrong frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


class FrameTimestampError(RuntimeError):
    """Raised when a decoded frame does not match the requested timestamp."""


@dataclass
class VideoInfo:
    """Basic stream properties."""

    codec: str
    decoder: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float


def available_av1_decoders() -> list[str]:
    """Names of AV1 decoders available in the linked FFmpeg."""
    names = []
    for name in ("libdav1d", "libaom-av1", "av1"):
        try:
            av.codec.Codec(name, "r")
            names.append(name)
        except Exception:  # noqa: BLE001 - codec simply not built in
            pass
    return names


class VideoReader:
    """Random-access RGB frame reader with timestamp verification.

    Args:
        path: MP4 file.
        tolerance_s: Max |decoded_pts - query| accepted (default: 0.4 frame at 30 fps).
    """

    def __init__(self, path: str | Path, tolerance_s: float = 0.4 / 30.0) -> None:
        self.path = Path(path)
        self.tolerance_s = tolerance_s
        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

    def info(self) -> VideoInfo:
        """Return stream properties."""
        s = self._stream
        dur = float(s.duration * s.time_base) if s.duration else float("nan")
        return VideoInfo(
            codec=s.codec_context.name,
            decoder=s.codec_context.codec.name,
            width=s.codec_context.width,
            height=s.codec_context.height,
            fps=float(s.average_rate),
            n_frames=int(s.frames),
            duration_s=dur,
        )

    def frames_at(self, timestamps_s: list[float] | np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Decode the frames closest to sorted query timestamps.

        Returns:
            (frames as HxWx3 uint8 RGB, decoded presentation times in seconds)

        Raises:
            FrameTimestampError: if a decoded frame is farther than ``tolerance_s``.
        """
        ts = np.asarray(timestamps_s, dtype=np.float64)
        if ts.size == 0:
            return [], np.empty(0)
        order = np.argsort(ts)
        out: list[np.ndarray | None] = [None] * ts.size
        times = np.full(ts.size, np.nan)
        tb = self._stream.time_base
        for k in order:
            t = ts[k]
            self._container.seek(int(max(t - 1.0, 0.0) / tb), stream=self._stream, backward=True, any_frame=False)
            best, best_t = None, None
            for frame in self._container.decode(self._stream):
                ft = float(frame.pts * tb)
                if best is None or abs(ft - t) < abs(best_t - t):
                    best, best_t = frame, ft
                if ft >= t + self.tolerance_s:
                    break
            if best is None or abs(best_t - t) > self.tolerance_s:
                raise FrameTimestampError(
                    f"{self.path.name}: requested t={t:.4f}s, closest decoded frame at "
                    f"{best_t}s (tolerance {self.tolerance_s:.4f}s)"
                )
            out[k] = best.to_ndarray(format="rgb24")
            times[k] = best_t
        return [o for o in out if o is not None], times  # type: ignore[misc]

    def frame_at(self, t: float) -> tuple[np.ndarray, float]:
        """Decode a single frame at time ``t`` (seconds)."""
        frames, times = self.frames_at([t])
        return frames[0], float(times[0])

    def count_frames(self) -> tuple[int, float, float]:
        """Decode the whole stream. Returns (n_frames, first_pts_s, last_pts_s)."""
        self._container.seek(0, stream=self._stream)
        n, first, last = 0, None, None
        tb = self._stream.time_base
        for frame in self._container.decode(self._stream):
            t = float(frame.pts * tb)
            first = t if first is None else first
            last = t
            n += 1
        return n, float(first or 0.0), float(last or 0.0)

    def close(self) -> None:
        """Release the container."""
        self._container.close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
