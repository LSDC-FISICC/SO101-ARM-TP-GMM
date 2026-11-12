"""Cut episodes into grasp cycles and each cycle into two movement primitives.

Primitives
----------
* ``approach_grasp`` = [approach_start, close_idx]: from the start of motion (or the
  end of the previous cycle) until the jaws reach the cube.
* ``transport_release`` = [close_idx, release_end]: from the contact until the
  release opening is complete. The return to home after it is discarded.

Physical gates (no stacking in this task)
-----------------------------------------
* A grasp starts at table height: TCP z at contact <= ``grasp_max_z``. A hold-level
  plateau that starts higher is the operator half-closing the gripper in the air
  (16 episodes, Phase 0) -> rejected as ``contact_in_air``.
* A release happens above the table (in the box): TCP z at release > ``release_min_z``.
  A hold released at table height is an aborted grasp / re-grasp -> rejected as
  ``release_at_table`` (the next valid cycle's approach contains it, and the cycle is
  flagged ``aborted_before``, as are cycles whose approach contains a missed grasp).
* "Quiet" releases: the gripper goes from *hold* straight to *closed-empty* with an
  opening below the hysteresis threshold (5 episodes, Phase 0). If this happens above
  the table, the end of the hold plateau is taken as the release instant.

Keypoints (used by the IK-based planner and for diagnostics)
-------------------------------------------------------------
* ``pregrasp_idx``: last frame before contact with TCP z >= z_contact + ``keypoint_height``
  (start of the final descent).
* ``lift_idx``: first frame after contact with TCP z >= z_contact + ``keypoint_height``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from so101_tpgmm.data.units import JointUnits
from so101_tpgmm.kinematics.fk import SO101Kinematics
from so101_tpgmm.segmentation.gripper import GripperParams, detect_gripper_events

PRIMITIVES = ("approach_grasp", "transport_release")


@dataclass
class SegmentationParams:
    """Thresholds of the cycle/primitive segmentation (see module docstring)."""

    grasp_max_z: float = 0.03
    release_min_z: float = 0.03
    onset_joint_speed_deg_s: float = 5.0
    onset_min_s: float = 0.2
    keypoint_height: float = 0.03
    min_primitive_s: float = 0.5
    quiet_release_tail_s: float = 0.3

    @classmethod
    def from_config(cls, cfg: dict) -> "SegmentationParams":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in (cfg or {}).items() if k in keys})


@dataclass
class SegmentedCycle:
    """One valid grasp cycle. All indices are episode frame indices (inclusive)."""

    episode: int
    cycle: int
    approach_start: int
    close_idx: int
    open_idx: int
    release_end: int
    pregrasp_idx: int
    lift_idx: int
    hold_level: float
    quiet_release: bool = False
    aborted_before: bool = False  # a missed or aborted grasp happened inside the approach

    def primitive(self, name: str) -> tuple[int, int]:
        """(start, end) inclusive frame indices of a primitive."""
        if name == "approach_grasp":
            return self.approach_start, self.close_idx
        if name == "transport_release":
            return self.close_idx, self.release_end
        raise KeyError(name)


@dataclass
class EpisodeSegmentation:
    """Valid cycles and rejected candidates of one episode."""

    episode: int
    onset: int
    cycles: list[SegmentedCycle] = field(default_factory=list)
    rejected: list[tuple[int, str]] = field(default_factory=list)  # (contact frame, reason)


def motion_onset(q_deg: np.ndarray, fps: float, start: int, params: SegmentationParams) -> int:
    """First frame >= ``start`` from which the max joint speed exceeds the threshold for
    at least ``onset_min_s``. Returns ``start`` if the arm is already moving."""
    speed = np.abs(np.gradient(q_deg[:, :5], axis=0)).max(axis=1) * fps
    moving = speed > params.onset_joint_speed_deg_s
    need = max(1, int(round(params.onset_min_s * fps)))
    run = 0
    for i in range(start, len(moving)):
        run = run + 1 if moving[i] else 0
        if run >= need:
            return i - need + 1
    return start


def segment_episode(
    state: np.ndarray,
    fps: float,
    kin: SO101Kinematics,
    units: JointUnits | str = JointUnits.DEGREES,
    gripper_params: GripperParams | None = None,
    params: SegmentationParams | None = None,
    episode: int = -1,
) -> EpisodeSegmentation:
    """Segment one episode (``state``: (T, 6) in dataset units)."""
    p = params or SegmentationParams()
    gp = gripper_params or GripperParams()
    g = state[:, 5]
    z = kin.fk_from_dataset(state, units)[:, 2, 3]
    ev = detect_gripper_events(g, fps, gp)

    # Candidate cycles: regular OPEN->HOLD->OPEN, plus quiet releases above the table.
    cands = []
    for c in ev.cycles:
        seg = g[c.open_idx : c.release_end]
        peak = c.open_idx + int(np.argmax(seg)) if seg.size else c.open_idx
        cands.append(dict(close=c.close_idx, open=c.open_idx, end=peak, level=c.hold_level, quiet=False))
    tail = int(round(p.quiet_release_tail_s * fps))
    for s, e, level in ev.unreleased_holds:
        on = np.flatnonzero(np.abs(g[s:e] - level) <= gp.plateau_tol)
        if on.size == 0:
            continue
        close, open_ = s + int(on[0]), s + int(on[-1])
        if z[open_] > p.release_min_z and open_ < e - 1:  # plateau ends (not at episode end) above table
            cands.append(dict(close=close, open=open_, end=min(open_ + tail, len(g) - 1), level=level, quiet=True))
    cands.sort(key=lambda d: d["close"])

    seg_out = EpisodeSegmentation(episode=episode, onset=motion_onset(state, fps, 0, p))
    boundary = seg_out.onset
    aborted = False
    min_len = int(round(p.min_primitive_s * fps))
    for d in cands:
        if z[d["close"]] > p.grasp_max_z:
            seg_out.rejected.append((d["close"], "contact_in_air"))
            continue
        if z[d["open"]] <= p.release_min_z:
            seg_out.rejected.append((d["close"], "release_at_table"))
            aborted = True
            continue
        start = boundary
        if d["close"] - start < min_len or d["end"] - d["close"] < min_len:
            seg_out.rejected.append((d["close"], "too_short"))
            continue
        missed = any(start <= s_ < d["close"] for s_, _, _ in ev.empty_closures)
        zc = z[d["close"]]
        above = np.flatnonzero(z[start : d["close"]] >= zc + p.keypoint_height)
        pre = start + int(above[-1]) if above.size else start
        up = np.flatnonzero(z[d["close"] : d["end"] + 1] >= zc + p.keypoint_height)
        lift = d["close"] + int(up[0]) if up.size else d["open"]
        seg_out.cycles.append(SegmentedCycle(
            episode=episode, cycle=len(seg_out.cycles), approach_start=int(start), close_idx=int(d["close"]),
            open_idx=int(d["open"]), release_end=int(d["end"]), pregrasp_idx=int(pre), lift_idx=int(lift),
            hold_level=float(d["level"]), quiet_release=bool(d["quiet"]), aborted_before=aborted or missed,
        ))
        aborted = False
        boundary = int(d["end"])
    return seg_out


def segment_dataset(ds, kin: SO101Kinematics, cfg: dict) -> tuple[list[EpisodeSegmentation], pd.DataFrame]:
    """Segment every episode; returns the per-episode results and a per-cycle table."""
    gp = GripperParams.from_config(cfg["gripper"])
    sp = SegmentationParams.from_config(cfg.get("segmentation", {}))
    units = JointUnits(cfg["units"]["joints"]) if cfg["units"]["joints"] != "auto" else JointUnits.DEGREES
    results = [segment_episode(ds.episode(k).state, ds.fps, kin, units, gp, sp, episode=k)
               for k in range(ds.num_episodes)]
    rows = [asdict(c) for r in results for c in r.cycles]
    return results, pd.DataFrame(rows)


def load_cycles(path) -> list[SegmentedCycle]:
    """Read the per-cycle table written by ``scripts/segment_dataset.py``."""
    df = pd.read_csv(path)
    return [SegmentedCycle(**{k: (bool(v) if k in ("quiet_release", "aborted_before") else
                                  float(v) if k == "hold_level" else int(v)) for k, v in r.items()})
            for r in df.to_dict("records")]
