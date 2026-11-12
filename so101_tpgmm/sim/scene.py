"""MuJoCo scene for the SO-101 pick-and-place task.

Robot: the official SO-101 MJCF (TheRobotStudio/SO-ARM100, ``Simulation/SO101/so101_new_calib.xml``,
Apache-2.0), generated from the same CAD as ``so101_new_calib.urdf``: its ``gripperframe`` site
matches our FK of ``gripper_frame_link`` to < 4 µm (``tests/test_sim.py``). Fetch it with
``scripts/fetch_so101_mujoco.sh`` (not vendored: ~16 MB of meshes). Its position actuators
(STS3215 gains and torque limit) execute the joint targets; contacts are physical (convex hulls of
the arm meshes, box cubes with friction).

World = robot base frame = the FK frame. The table top is the plane ``z = a + b·x + c·y``
(``table_plane``) on which the fingers rested at the demonstrated grasps, in FK coordinates: the joint
calibration error of the real arm makes the flat table look tilted and ~2.5 cm lower than the
tape-grid estimate (see ``scripts/simulate_mujoco.py --fit-table``). Cubes and box sit on that plane.
The box is an open container centred on the median demonstrated release point; its size was not
measured (``sim.box`` in the config, chosen to hold the release spread without overlapping any
demonstrated grasp).

Integration: 0.5 ms steps. With 2 ms, a finger tip pressing a light cube into the table (full servo
torque on a 20-60 g cube) let it sink several mm and then spin out of the simulation in ~1/3 of the
episodes with noisy RL actions; at 0.5 ms none in the same test (physics stays ~10x cheaper than the
IK per control step). The table is a plane (a tilted box table gave single-point edge contacts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COLORS_RGBA = {"red": "0.85 0.12 0.12 1", "yellow": "0.95 0.8 0.1 1", "green": "0.1 0.65 0.25 1",
               "blue": "0.15 0.3 0.85 1"}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class SimCube:
    color: str
    xyz: np.ndarray  # cube centre [m], base frame
    yaw: float  # [rad]


@dataclass
class SceneSpec:
    robot_xml: Path
    table_plane: tuple[float, float, float] = (-0.03, 0.0, 0.0)  # top surface z = a + b x + c y [m]
    cube_side: float = 0.04
    cube_mass: float = 0.03
    cube_friction: float = 1.2
    finger_friction: float = 1.5
    box_center_xy: tuple[float, float] = (0.057, 0.164)
    box_inner: tuple[float, float] = (0.10, 0.10)
    box_wall_h: float = 0.05
    box_wall_t: float = 0.005
    cubes: list[SimCube] = field(default_factory=list)
    camera: dict | None = None  # {"pos": [x, y, z], "xyaxes": [...]} optional extra camera

    @classmethod
    def from_config(cls, cfg: dict, robot_xml: Path, cubes: list[SimCube]) -> "SceneSpec":
        s = cfg.get("sim", {})
        b = s.get("box", {})
        return cls(robot_xml=Path(robot_xml), table_plane=tuple(float(v) for v in s.get("table_plane", (-0.03, 0, 0))),
                   cube_side=float(cfg["scene"]["cube_side"]), cube_mass=float(s.get("cube_mass", 0.03)),
                   cube_friction=float(s.get("cube_friction", 1.2)), finger_friction=float(s.get("finger_friction", 1.5)),
                   box_center_xy=tuple(b.get("center_xy", (0.057, 0.164))),
                   box_inner=tuple(b.get("inner", (0.10, 0.10))), box_wall_h=float(b.get("wall_h", 0.05)),
                   box_wall_t=float(b.get("wall_t", 0.005)), cubes=cubes)


    def table_height(self, x: float, y: float) -> float:
        a, b, c = self.table_plane
        return a + b * x + c * y

    def table_rotation(self) -> np.ndarray:
        """Rotation taking world z to the table normal (minimal rotation)."""
        _, b, c = self.table_plane
        n = np.array([-b, -c, 1.0]) / np.sqrt(1 + b * b + c * c)
        v, cth = np.cross([0, 0, 1.0], n), n[2]
        V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + V + V @ V / (1 + cth)

    def cube_on_table(self, x: float, y: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """Centre and rotation matrix of a cube resting on the table at (x, y) with the given yaw."""
        R = self.table_rotation() @ _rot_z(yaw)
        n = self.table_rotation()[:, 2]
        base = np.array([x, y, self.table_height(x, y)])
        return base + n * (self.cube_side / 2 + 5e-4), R


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quat(R: np.ndarray) -> str:
    w = np.sqrt(max(1e-12, 1 + np.trace(R))) / 2
    x, y, z = (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


def scene_xml(spec: SceneSpec) -> str:
    """MJCF of the full scene (includes the robot file; written next to it so mesh paths resolve)."""
    h, t = spec.cube_side / 2, spec.box_wall_t
    cx, cy = spec.box_center_xy
    ix, iy = spec.box_inner[0] / 2, spec.box_inner[1] / 2
    wh = spec.box_wall_h / 2
    Rt = spec.table_rotation()
    qt = _quat(Rt)
    tz0 = spec.table_height(0.25, 0.0)
    tz_c = np.array([0.25, 0.0, tz0])  # a plane: robust contacts (a tilted box table made cubes resting on an edge explode)
    walls = [((ix + t / 2, 0, wh), (t / 2, iy + t, wh)), ((-ix - t / 2, 0, wh), (t / 2, iy + t, wh)),
             ((0, iy + t / 2, wh), (ix, t / 2, wh)), ((0, -iy - t / 2, wh), (ix, t / 2, wh))]
    box = "\n".join(f'      <geom type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" '
                    f'material="box" class="env"/>' for p, s in walls)
    cubes = []
    for i, c in enumerate(spec.cubes):
        pos, R = spec.cube_on_table(c.xyz[0], c.xyz[1], c.yaw)  # resting on the table (c.xyz[2] is ignored)
        cubes.append(
            f'    <body name="cube_{i}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{_quat(R)}">\n'
            f'      <freejoint name="cube_{i}"/>\n'
            f'      <geom name="cube_{i}" type="box" size="{h} {h} {h}" mass="{spec.cube_mass}" rgba="{COLORS_RGBA.get(c.color, "0.5 0.5 0.5 1")}" '
            f'friction="{spec.cube_friction} 0.02 0.001" condim="4" contype="1" conaffinity="3" solref="0.004 1"/>\n'
            f'    </body>')
    welds = "\n".join(f'    <weld name="grasp_{i}" body1="gripper" body2="cube_{i}" active="false" solref="0.01 1"/>'
                      for i in range(len(spec.cubes)))
    cam = ""
    if spec.camera:
        cam = f'    <camera name="fixed" pos="{" ".join(map(str, spec.camera["pos"]))}" xyaxes="{" ".join(map(str, spec.camera["xyaxes"]))}"/>'
    return f"""<mujoco model="so101_pick_place">
  <include file="{spec.robot_xml.name}"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <default>
    <default class="env"><geom contype="1" conaffinity="3" friction="0.8 0.02 0.001"/></default>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96" rgb2="0.62 0.68 0.75" width="256" height="1536"/>
    <material name="table" rgba="0.86 0.86 0.84 1"/>
    <material name="box" rgba="0.75 0.8 0.9 0.55"/>
    <material name="tape" rgba="0.15 0.15 0.45 1"/>
  </asset>
  <worldbody>
    <light pos="0.2 0 1.2" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="table" type="plane" pos="{tz_c[0]:.5f} {tz_c[1]:.5f} {tz_c[2]:.5f}" quat="{qt}" size="0.45 0.45 0.01" material="table" class="env"/>
    <body name="box" pos="{cx:.5f} {cy:.5f} {spec.table_height(cx, cy):.5f}" quat="{qt}">
{box}
    </body>
{chr(10).join(cubes)}
{cam}
  </worldbody>
  <equality>
{welds}
  </equality>
</mujoco>
"""


def _mesh_vertices_in_body(model, body: str, mesh: str) -> np.ndarray:
    import mujoco

    b = model.body(body).id
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == b and model.geom_group[g] == 3 and model.geom_dataid[g] >= 0 \
                and model.mesh(model.geom_dataid[g]).name == mesh:
            m = model.geom_dataid[g]
            V = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            return V @ R.reshape(3, 3).T + model.geom_pos[g]
    raise KeyError(f"{body}/{mesh}")


def finger_boxes(V: np.ndarray, R: np.ndarray, z_range: tuple[float, float], step: float = 0.01):
    """Boxes approximating a finger: vertices ``V`` (body frame) sliced every ``step`` along the third axis
    of the frame ``R`` (columns = axes in the body frame); one bounding box per slice, in that frame.

    The meshes' convex hulls cannot be used for the fingers: the hull of the part holding the fixed jaw,
    and that of the moving jaw, fill the gap between the fingers (they overlap at every opening).
    """
    import mujoco

    U = V @ R
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    out = []
    for z0 in np.arange(z_range[0], z_range[1] - 1e-9, step):
        s = U[(U[:, 2] >= z0) & (U[:, 2] < z0 + step)]
        if len(s) < 10:
            continue
        lo, hi = s.min(0), s.max(0)
        out.append((R @ ((lo + hi) / 2), (hi - lo) / 2 + 1e-4, q.copy()))
    return out


FINGER_Z = (-0.106, -0.03)  # finger length along the gripper's −z (gripper body frame, q_jaw = 0) [m]


def build_model(spec: SceneSpec):
    """Compile the scene.

    * Fingers: the convex hulls of ``wrist_roll_follower`` (fixed jaw) and ``moving_jaw`` are replaced by
      boxes fitted to 1 cm slices of their vertices (``finger_boxes``).
    * Arm collision geoms (contype 2, conaffinity 0) collide with cubes, box and table, not with each other.
    """
    import os
    import tempfile

    # A unique file next to the robot model (so its relative mesh paths resolve); parallel workers must
    # never share it.
    fd, name = tempfile.mkstemp(prefix="_scene_", suffix=".xml", dir=spec.robot_xml.parent)
    path = Path(name)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(scene_xml(spec))
    try:
        return _compile(path, spec.finger_friction)
    finally:
        path.unlink(missing_ok=True)


def _compile(path: Path, finger_friction: float):
    import mujoco

    m0 = mujoco.MjModel.from_xml_path(str(path))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    Rg = d0.xmat[m0.body("gripper").id].reshape(3, 3)
    Rj = d0.xmat[m0.body("moving_jaw_so101_v1").id].reshape(3, 3)
    fixed = finger_boxes(_mesh_vertices_in_body(m0, "gripper", "wrist_roll_follower_so101_v1"), np.eye(3), FINGER_Z)
    jaw = finger_boxes(_mesh_vertices_in_body(m0, "moving_jaw_so101_v1", "moving_jaw_so101_v1"), Rj.T @ Rg,
                       FINGER_Z)

    ms = mujoco.MjSpec.from_file(str(path))
    for body, boxes, mesh in (("gripper", fixed, "wrist_roll_follower_so101_v1"),
                              ("moving_jaw_so101_v1", jaw, "moving_jaw_so101_v1")):
        b = ms.body(body)
        for g in b.geoms:
            if g.meshname == mesh and g.group == 3:
                g.contype, g.conaffinity = 0, 0
        for i, (pos, half, quat) in enumerate(boxes):
            b.add_geom(name=f"{body}_pad{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, quat=quat,
                       group=3, contype=2, conaffinity=0, condim=4, friction=[finger_friction, 0.02, 0.001],
                       rgba=[0.2, 0.6, 1.0, 0.5], mass=0)
    model = ms.compile()
    for g in range(model.ngeom):  # remaining arm collision hulls: no arm-arm contacts
        body = model.body(model.geom_bodyid[g]).name
        if model.geom_group[g] == 3 and model.geom_contype[g] == 1 and not body.startswith("cube"):
            model.geom_contype[g], model.geom_conaffinity[g] = 2, 0
    return model
