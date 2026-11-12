"""Markdown/LaTeX documentation of the FK and IK actually used, with numbers from the URDF.

Used by ``scripts/check_kinematics.py`` to write the "Forward kinematics used" and
"inverse kinematics" sections of docs/kinematics_report.md. The verification errors
are computed every time the section is generated, so the document cannot drift from
the code.
"""

from __future__ import annotations

import numpy as np

from so101_tpgmm.kinematics.analytic import AnalyticIK, closed_form_fk
from so101_tpgmm.kinematics.ik import SO101IK

EQ_START = "<!-- equations:start -->"
EQ_END = "<!-- equations:end -->"


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _mm(v: float) -> str:
    return f"{v * 1e3:.1f}"


def verification_errors(ik: SO101IK, n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Max errors of the closed-form FK, closed-form IK and the q5 formula vs the URDF chain."""
    kin = ik.kin
    aik = AnalyticIK(kin)
    g = aik.g
    Q = np.random.default_rng(seed).uniform(ik.lower, ik.upper, size=(n, 5))
    e_fk = max(float(np.abs(closed_form_fk(q, g) - kin.fk(q)).max()) for q in Q)
    e_ik = e_q5 = 0.0
    for q in Q[: n // 4]:
        T = kin.fk(q)
        sols = aik.solve_pose(T)
        e_ik = max(e_ik, min(float(np.abs(_wrap(s - q)).max()) for s in sols))
        M = (_rz(-q[0]) @ _ry(q[1] + q[2] + q[3])).T @ T[:3, :3] @ g.tool_rotation.T
        e_q5 = max(e_q5, float(abs(_wrap(-np.arctan2(M[2, 1], M[2, 2]) - q[4]))))
    return {"fk": e_fk, "ik_deg": float(np.degrees(e_ik)), "q5_deg": float(np.degrees(e_q5)), "n": n}


def _rpy(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])


def equations_markdown(ik: SO101IK) -> str:
    """The full equations section (between EQ_START and EQ_END markers)."""
    kin = ik.kin
    g = AnalyticIK(kin).g
    err = verification_errors(ik)
    L: list[str] = [EQ_START, "## Forward kinematics used", ""]
    A = L.append

    A("Joint values are converted explicitly from the dataset (LeRobot `DEGREES` mode): "
      r"$q_i = \frac{\pi}{180}\,v_i$, $i = 1..5$ = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, "
      "wrist_roll. Poses are in the `base_link` frame (z up, origin at the bottom of the base); lengths in mm.")
    A("")
    A("### Reference: product of the URDF transforms (`kinematics/fk.py`)")
    A("")
    A(r"$$ {}^{B}T_{\mathrm{TCP}}(q) = \Big[\prod_{i=1}^{5} T^{\mathrm{o}}_{i}\,\mathrm{Rot}(\mathbf a_i, q_i)\Big]\,"
      r"T^{\mathrm{o}}_{\mathrm{tcp}} $$")
    A("")
    A(r"$T^{\mathrm{o}}_{i}$ is the fixed `<origin xyz rpy>` of joint $i$ (rpy convention $R = R_z(y)\,R_y(p)\,R_x(r)$), "
      r"$\mathrm{Rot}(\mathbf a, q)$ the rotation about its `<axis>` (Rodrigues), and the TCP is `gripper_frame_link`. "
      "Values from `so101_new_calib.urdf`:")
    A("")
    A("| joint | parent → child | origin xyz [mm] | origin rpy [rad] | axis |")
    A("|---|---|---|---|---|")
    for j in kin.chain:
        axis = "—" if j.type == "fixed" else "(" + ", ".join(f"{v:g}" for v in np.round(j.axis, 3)) + ")"
        A(f"| {j.name} | {j.parent} → {j.child} | ({', '.join(_mm(v) for v in j.origin[:3, 3])}) | "
          f"({', '.join(f'{v:.4f}' for v in _rpy(j.origin))}) | {axis} |")
    A("")
    A("### Closed form (`kinematics/analytic.py::closed_form_fk`)")
    A("")
    A("Geometric decomposition of the same chain, joints in the LeRobot/URDF convention "
      r"($c_2 = \cos q_2$, $s_{23} = \sin(q_2 + q_3)$, ...):")
    A("")
    A(r"$$ \alpha = -q_1, \qquad \beta = q_2 + q_3 + q_4 $$")
    A("")
    A(r"$$ r = a_1 + l_5 c_2 + l_4 s_2 + l_6 c_{23} + l_{6b} s_{23} + l_{78} \cos\beta $$")
    A("")
    A(r"$$ z = d_1 + l_4 c_2 - l_5 s_2 - l_6 s_{23} + l_{6b} c_{23} - l_{78} \sin\beta $$")
    A("")
    A(r"$$ \mathbf p_{\mathrm{roll}} = \begin{bmatrix} x_{\mathrm{pan}} + r\cos\alpha - e\sin\alpha \\ "
      r"y_{\mathrm{pan}} + r\sin\alpha + e\cos\alpha \\ z \end{bmatrix} $$")
    A("")
    A(r"$$ R_{\mathrm{TCP}} = R_z(\alpha)\,R_y(\beta)\,R_x(-q_5)\,C, \qquad "
      r"\mathbf p_{\mathrm{TCP}} = \mathbf p_{\mathrm{roll}} + R_{\mathrm{TCP}}\,\mathbf d_{\mathrm{tool}} $$")
    A("")
    A(r"The pan axis points **down** in the URDF, hence $\alpha = -q_1$. $\mathbf p_{\mathrm{roll}}$ is the point of the "
      r"wrist-roll axis at the TCP depth. The approach axis (TCP $z$) is "
      r"$(\cos\beta\cos\alpha,\ \cos\beta\sin\alpha,\ -\sin\beta)$, so the approach pitch used by the IK is "
      r"$\theta = 90^\circ - \beta$.")
    A("")
    A("| parameter | meaning | value |")
    A("|---|---|---|")
    rows = [
        (r"$(x_{\mathrm{pan}}, y_{\mathrm{pan}})$", "pan axis in base_link", f"({_mm(g.pan_xy[0])}, {_mm(g.pan_xy[1])}) mm"),
        (r"$a_1 = l_0 + l_2$", "shoulder-lift axis ahead of the pan axis", f"{_mm(g.a1)} mm"),
        (r"$d_1 = l_1 + l_3$", "shoulder-lift axis height", f"{_mm(g.d1)} mm"),
        (r"$l_5,\ l_4$", r"upper arm (along, vertical) at $q_2 = 0$", f"{_mm(g.l5)}, {_mm(g.l4)} mm"),
        (r"$L_1,\ \varphi$", r"$\sqrt{l_4^2+l_5^2}$, $\mathrm{atan2}(l_4, l_5)$", f"{_mm(g.L1)} mm, {np.degrees(g.phi):.2f}°"),
        (r"$l_6,\ l_{6b}$", "forearm (along, across)", f"{_mm(g.l6)}, {_mm(g.l6b)} mm"),
        (r"$L_2,\ \varphi_2$", r"$\sqrt{l_6^2+l_{6b}^2}$, $\mathrm{atan2}(l_{6b}, l_6)$", f"{_mm(g.L2)} mm, {np.degrees(g.phi2):.2f}°"),
        (r"$l_{78} = l_7 + l_8$", "wrist-flex axis → TCP depth, along the roll axis", f"{_mm(g.l78)} mm"),
        (r"$e$", "lateral offset of the roll axis", f"{_mm(g.lateral)} mm"),
        (r"$\mathbf d_{\mathrm{tool}}$", "TCP − roll-axis point, in the TCP frame", f"({', '.join(_mm(v) for v in g.tcp_offset)}) mm"),
        (r"$C$", r"$R_{\mathrm{TCP}}(q = 0)$, includes a 2.8° mounting twist",
         "[" + "; ".join(" ".join(f"{v:.4f}" for v in row) for row in g.tool_rotation) + "]"),
    ]
    for sym, what, val in rows:
        A(f"| {sym} | {what} | {val} |")
    A("")
    A(f"Verification against the URDF chain on {err['n']} random configurations inside the joint limits: "
      f"max element error of $T$ = {err['fk']:.1e} (residual from constants rounded in the URDF, e.g. π written "
      "as 3.14159).")
    A("")

    A("## Closed-form inverse kinematics (`kinematics/analytic.py::AnalyticIK`)")
    A("")
    A("Kinematic decoupling after Rivas (2026), corrected (`docs/kinematics_paper_review.md`). Input: a full TCP "
      r"pose $(R, \mathbf p)$. Up to 4 solutions (shoulder $\pm$ × elbow $\pm$), filtered by joint limits afterwards.")
    A("")
    A(r"1. Point on the roll axis: $\mathbf p_{\mathrm{roll}} = \mathbf p - R\,\mathbf d_{\mathrm{tool}}$, "
      r"$(P_x, P_y) = \mathbf p_{\mathrm{roll},xy} - (x_{\mathrm{pan}}, y_{\mathrm{pan}})$.")
    A(r"2. Shoulder: $r = \pm\sqrt{P_x^2 + P_y^2 - e^2}$, "
      r"$\alpha = \mathrm{atan2}(P_y, P_x) - \mathrm{atan2}(e, r)$, $q_1 = -\alpha$.")
    A(r"3. Pitch of the roll axis, $\mathbf u = R\,\hat{\mathbf z}$ (no division by $\cos q_1$): "
      r"$\beta = \mathrm{atan2}\big(-u_z,\ u_x\cos\alpha + u_y\sin\alpha\big)$.")
    A(r"4. Wrist-flex axis in the arm plane: $x' = r - a_1 - l_{78}\cos\beta$, "
      r"$z' = p_{\mathrm{roll},z} - d_1 + l_{78}\sin\beta$.")
    A(r"5. Equivalent two-link arm: $\cos\theta_3 = \dfrac{x'^2 + z'^2 - L_1^2 - L_2^2}{2 L_1 L_2}$, "
      r"$\theta_3 = \pm\arccos(\cos\theta_3)$.")
    A(r"6. $q_2 = \varphi - \mathrm{atan2}(z', x') + \mathrm{atan2}\big(L_2\sin\theta_3,\ L_1 + L_2\cos\theta_3\big)$.")
    A(r"7. $q_3 = \varphi_2 - \varphi - \theta_3$.")
    A(r"8. $q_4 = \beta - q_2 - q_3$.")
    A(r"9. Roll: $M = \big(R_z(\alpha)\,R_y(\beta)\big)^{\top} R\,C^{\top} = R_x(-q_5)$, so "
      r"$q_5 = -\mathrm{atan2}(M_{32}, M_{33})$.")
    A("")
    A(f"Verification ({err['n'] // 4} random poses from the URDF FK): the original joints are among the "
      f"solutions with max error {err['ik_deg']:.3f}°; step 9 reproduces $q_5$ within {err['q5_deg']:.3f}°.")
    A("")

    A("## Numerical inverse kinematics (`kinematics/ik.py::SO101IK`)")
    A("")
    A(r"Task variables: TCP position $\mathbf p$; wrist yaw $\psi = \mathrm{atan2}(x_y, x_x)$ of the TCP $x$ axis "
      r"(jaw-closing direction); approach pitch $\theta$ (soft). With 5 joints and 4 hard constraints, the pitch "
      "resolves the redundant DOF.")
    A("")
    A("**Stage 1** (choose the posture), bounded least squares:")
    A("")
    A(r"$$ \min_{q_{\min} \le q \le q_{\max}} \ \|\mathbf p(q) - \mathbf p^*\|^2 + "
      r"w_\psi^2\,\mathrm{wrap}\big(\psi(q) - \psi^*\big)^2 + w_\theta^2\,\|\mathbf z(q) - \mathbf z^*(\theta^*)\|^2 + "
      r"w_r^2\,\|q - q_{\mathrm{ref}}\|^2 $$")
    A("")
    A(r"with $\mathbf z^*(\theta) = (\sin\theta\,\hat{\mathbf r},\ -\cos\theta)$ and $\hat{\mathbf r}$ the horizontal "
      r"radial direction of $\mathbf p^*$ from the pan axis. **Stage 2** (task priority): the same problem without "
      r"the pitch term and with $q_{\mathrm{ref}}$ = stage-1 solution, so position and yaw become exact. Seeds: the "
      r"closed-form solutions first, then pan towards the target with the roll set from $\psi^*$; along a "
      r"trajectory the previous solution is both seed and $q_{\mathrm{ref}}$.")
    A("")
    A(r"Analytic Jacobian ($\mathbf a_i$, $\mathbf o_i$: world axis and origin of joint $i$; $\mathbf x$, $\mathbf z$: "
      "TCP axes):")
    A("")
    A(r"$$ \frac{\partial \mathbf p}{\partial q_i} = \mathbf a_i \times (\mathbf p - \mathbf o_i), \qquad "
      r"\frac{\partial \psi}{\partial q_i} = \frac{x_x\,(\mathbf a_i\times\mathbf x)_y - x_y\,(\mathbf a_i\times\mathbf x)_x}"
      r"{x_x^2 + x_y^2}, \qquad \frac{\partial \mathbf z}{\partial q_i} = \mathbf a_i \times \mathbf z $$")
    A("")
    A(rf"Weights: $w_\psi$ = {ik.yaw_weight} m/rad, $w_\theta$ = {ik.pitch_weight} m/rad, $w_r$ = {ik.reg_weight} m/rad. "
      rf"Success: $\|\Delta\mathbf p\| \le$ {ik.pos_tol * 1e3:.1f} mm, $|\Delta\psi| \le$ {np.degrees(ik.yaw_tol):.1f}°, "
      rf"inside the joint limits, and $\sqrt{{x_x^2 + x_y^2}} \ge$ {ik.min_yaw_observability} (yaw defined).")
    A(EQ_END)
    return "\n".join(L)


def update_report(path, ik: SO101IK, anchor: str = "## Solver\n") -> None:
    """Insert or replace the equations section of an existing report."""
    text = path.read_text(encoding="utf-8")
    eq = equations_markdown(ik)
    if EQ_START in text:
        text = text[: text.index(EQ_START)] + eq + text[text.index(EQ_END) + len(EQ_END):]
    else:
        text = text.replace(anchor, eq + "\n\n" + anchor, 1)
    path.write_text(text, encoding="utf-8")
