"""Grasp verification: did the gripper actually close on the cube?

Two signals, both measured on dataset3 and in MuJoCo (``docs/grasp_check.md``):

1. **Gripper (main signal).** While a 4 cm cube is held, the cube stops the jaws: the measured gripper stays
   near 20 although the command (``inference.gripper_command.hold_command``) is ~2. An empty gripper reaches
   its command. So ``measured − commanded`` is large when holding and ~0 when the grasp was missed. On the
   real follower the jaws need ~0.5 s to reach a closed command (0.17 s after closing empty the difference
   is still up to 41 units), hence ``settle_s``.

   * dataset3 (follower state − leader command): holding median 19.1 (399 cycles); closed empty, 0.5 s after
     closing: median 0.7, max 6.6 (398 cycles); the three demonstrations with a missed grasp: −0.8, 0.7, −0.2.
   * MuJoCo with the hold command: holding 15–17 one second after contact; missed −0.3 (50 grasps).

2. **Wrist camera (optional confirmation).** A held cube fills the lower centre of the wrist image: the
   largest blob of the cube's colour covers ~55 % of a fixed ROI (dataset3: p5 43 %); after the release,
   with the gripper empty, ~0 %. It catches jaws stopped by something else (the box wall, the cube's edge
   under a fingertip) once the arm has lifted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraspCheckParams:
    settle_s: float = 0.5  # after the gripper command reaches its hold value
    min_margin: float = 8.0  # measured − commanded gripper with the cube between the jaws
    hold_range: tuple[float, float] = (12.0, 40.0)  # plausible measured gripper on a 4 cm cube
    max_retries: int = 2  # replanning attempts per cube after a missed grasp
    retreat_height_m: float = 0.06  # straight lift after a miss, before going home
    open_gripper: float = 40.0  # gripper opened to this after a miss (releases whatever is pinched)
    wrist_roi: tuple[float, float, float, float] = (0.33, 0.35, 0.67, 0.90)  # x0, y0, x1, y1 (image fractions)
    wrist_min_blob: float = 0.15  # largest cube-coloured blob / ROI area

    @classmethod
    def from_config(cls, cfg: dict) -> "GraspCheckParams":
        g = dict(cfg.get("grasp_check", {}))
        wrist = g.pop("wrist", {}) or {}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in g.items() if k in cls.__dataclass_fields__}
        if "roi" in wrist:
            kw["wrist_roi"] = tuple(wrist["roi"])
        if "min_blob" in wrist:
            kw["wrist_min_blob"] = float(wrist["min_blob"])
        return cls(**kw)


@dataclass
class GraspVerdict:
    held: bool
    g_meas: float
    g_cmd: float
    wrist_blob: float | None = None
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        w = "" if self.wrist_blob is None else f", wrist blob {self.wrist_blob:.2f}"
        why = "" if self.held else f" ({'; '.join(self.reasons)})"
        return (f"{'HELD' if self.held else 'MISSED'}: gripper measured {self.g_meas:.1f} vs commanded "
                f"{self.g_cmd:.1f}{w}{why}")


def check_index(g_cmd: np.ndarray, k_contact: int, dt: float, settle_s: float) -> int:
    """Sample at which to verify the grasp: ``settle_s`` after the gripper command, closing from the contact
    sample ``k_contact`` on, comes within 1 unit of its lowest value of the next 2 s. Clipped to the last
    sample."""
    g = np.asarray(g_cmd, float)
    k_contact = int(np.clip(k_contact, 0, len(g) - 1))
    w = g[k_contact:k_contact + max(2, int(round(2.0 / dt)))]
    k_hold = k_contact + int(np.flatnonzero(w <= w.min() + 1.0)[0])
    return int(min(len(g) - 1, k_hold + int(round(settle_s / dt))))


def wrist_blob_fraction(rgb: np.ndarray, color: str, detector_params, roi=(0.33, 0.35, 0.67, 0.90)) -> float:
    """Largest blob of ``color`` inside the wrist-image ROI, as a fraction of the ROI area."""
    import cv2

    from so101_tpgmm.vision.detection import hue_mask

    spec = next(c for c in detector_params.colors if c.name == color)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = roi
    sub = np.ascontiguousarray(rgb[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)])
    m = hue_mask(cv2.cvtColor(sub, cv2.COLOR_RGB2HSV), spec).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return float(stats[1:, cv2.CC_STAT_AREA].max() / m.size) if n > 1 else 0.0


def verify_grasp(g_meas: float, g_cmd: float, p: GraspCheckParams, wrist_blob: float | None = None) -> GraspVerdict:
    """Held if the jaws are stopped by an object of cube size and, when the wrist image is given, the cube is
    between them."""
    reasons = []
    if g_meas - g_cmd < p.min_margin:
        reasons.append(f"jaws reached the command (margin {g_meas - g_cmd:.1f} < {p.min_margin})")
    if not p.hold_range[0] <= g_meas <= p.hold_range[1]:
        reasons.append(f"gripper {g_meas:.1f} outside {p.hold_range}")
    if wrist_blob is not None and wrist_blob < p.wrist_min_blob:
        reasons.append(f"cube not between the jaws in the wrist image ({wrist_blob:.2f} < {p.wrist_min_blob})")
    return GraspVerdict(not reasons, float(g_meas), float(g_cmd), wrist_blob, reasons)
