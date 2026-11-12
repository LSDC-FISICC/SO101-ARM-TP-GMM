"""Debug drawings of cube detections."""

from __future__ import annotations

import cv2
import numpy as np

from so101_tpgmm.vision.detection import CubeDetection

COLORS_RGB = {"red": (230, 40, 40), "yellow": (240, 200, 0), "green": (20, 170, 60), "blue": (40, 80, 230)}


def draw_detections(
    rgb: np.ndarray,
    dets: list[CubeDetection],
    labels: list[str] | None = None,
    exclude_boxes: list[tuple[int, int, int, int]] = (),
    arm_roi: tuple[int, int, int, int] | None = None,
    header: str = "",
    highlight: int | None = None,
) -> np.ndarray:
    """Contours, minAreaRect, centre, yaw arrow and flags drawn on a copy of ``rgb``."""
    img = rgb.copy()
    for x0, y0, x1, y1 in exclude_boxes:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (128, 128, 128), 1)
    if arm_roi is not None:
        cv2.rectangle(img, arm_roi[:2], arm_roi[2:], (255, 0, 255), 1)
    for i, d in enumerate(dets):
        col = COLORS_RGB.get(d.color, (255, 255, 255))
        cv2.drawContours(img, [d.contour], -1, col, 1)
        box = cv2.boxPoints(((d.center_px[0], d.center_px[1]), d.size_px, np.degrees(d.yaw_img)))
        cv2.polylines(img, [box.astype(np.int32)], True, (255, 255, 255) if i != highlight else (0, 255, 255), 2)
        c = tuple(int(round(v)) for v in d.center_px)
        tip = (int(c[0] + 22 * np.cos(d.yaw_img)), int(c[1] + 22 * np.sin(d.yaw_img)))
        cv2.arrowedLine(img, c, tip, (0, 0, 0), 2, tipLength=0.3)
        cv2.circle(img, c, 3, (0, 0, 0), -1)
        text = labels[i] if labels else f"{d.color} {np.degrees(d.yaw_img):+.0f}deg"
        if d.flags:
            text += " [" + ",".join(d.flags) + "]"
        cv2.putText(img, text, (c[0] + 12, c[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (c[0] + 12, c[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if header:
        cv2.putText(img, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img
