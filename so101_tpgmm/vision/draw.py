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


def draw_plan_overlay(img: np.ndarray, plan, p2b, kin, tcp_offset: np.ndarray | None = None) -> np.ndarray:
    """Planned TCP path of every cube (vertical projection on the table, in the cube's colour: approach
    solid, transport thin) and the grasp: TCP (filled dot) and, with ``tcp_offset``, where the cube
    centre will be when the jaws close (ring + cross; it should sit on the detected cube)."""
    img = img.copy()
    for k, cp in enumerate(plan.cubes):
        if cp.status != "planned":
            continue
        col = COLORS_RGB.get(cp.color, (255, 255, 255))
        sel = np.flatnonzero((plan.cube_index == cp.order) & np.isfinite(plan.cartesian[:, 0]))
        if not len(sel):
            continue
        phase = np.array(plan.phase)[sel]
        app, tra = sel[phase == "approach_grasp"], sel[phase != "approach_grasp"]
        for part, th in ((app, 2), (tra, 1)):
            if len(part) > 1:
                px = p2b.to_pixel(plan.cartesian[part, :2]).round().astype(np.int32)
                cv2.polylines(img, [px], False, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.polylines(img, [px], False, col, th, cv2.LINE_AA)
        if not len(app):
            continue
        i_g = app[-1]  # grasp: end of the approach primitive
        tcp_px = tuple(int(v) for v in p2b.to_pixel(plan.cartesian[i_g, :2])[0].round())
        cv2.circle(img, tcp_px, 4, (0, 0, 0), -1)
        cv2.circle(img, tcp_px, 3, col, -1)
        if tcp_offset is not None:
            T = kin.fk(np.deg2rad(plan.joints_deg[i_g]))
            c = T[:3, 3] + T[:3, :3] @ np.asarray(tcp_offset, float)
            cpx = tuple(int(v) for v in p2b.to_pixel(c[None, :2])[0].round())
            cv2.circle(img, cpx, 9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(img, cpx, 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(img, cpx, (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
        txt = f"grasp #{cp.order + 1}"
        cv2.putText(img, txt, (tcp_px[0] + 8, tcp_px[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (tcp_px[0] + 8, tcp_px[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    legend = "path: TCP projected on the table | dot: TCP at grasp | ring: cube centre at grasp"
    h = img.shape[0]
    cv2.putText(img, legend, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, legend, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img
