"""Grasp verification and replanning of a missed cube."""

import cv2
import numpy as np
import pytest

from so101_tpgmm.config import load_config, resolve_path
from so101_tpgmm.grasp import GraspCheckParams, check_index, verify_grasp, wrist_blob_fraction
from so101_tpgmm.vision.detection import DetectorParams

CFG = load_config(None, [])
P = GraspCheckParams.from_config(CFG)


def test_verdict_on_dataset_levels():
    # dataset3: holding, follower ~20 with the hold command 2; closed empty reaches the command (max +6.6)
    assert verify_grasp(20.0, 2.0, P).held
    assert not verify_grasp(8.6, 2.0, P).held          # empty, worst case 0.5 s after closing
    assert not verify_grasp(2.3, 2.0, P).held
    assert not verify_grasp(60.0, 2.0, P).held          # jaws stopped far too wide: not a 4 cm cube
    assert not verify_grasp(20.0, 2.0, P, wrist_blob=0.0).held   # stopped by something that is not the cube
    assert verify_grasp(20.0, 2.0, P, wrist_blob=0.5).held


def test_check_index_waits_for_the_hold_command():
    dt = 1 / 30
    g = np.r_[np.full(30, 40.0), np.linspace(26, 2, 30), np.full(60, 2.0)]  # contact at 30; within 1 unit of 2 at 58
    k = check_index(g, 30, dt, 0.5)
    assert k == 58 + 15
    assert check_index(g[:65], 30, dt, 0.5) == 64  # clipped to the end


def test_wrist_blob_fraction():
    dp = DetectorParams.from_config(CFG["vision"])
    img = np.full((480, 640, 3), 200, np.uint8)
    assert wrist_blob_fraction(img, "green", dp, P.wrist_roi) == 0.0
    img[200:400, 230:410] = (40, 160, 40)  # RGB green, a held cube's place
    assert wrist_blob_fraction(img, "green", dp, P.wrist_roi) > 0.3
    assert wrist_blob_fraction(img, "red", dp, P.wrist_roi) == 0.0


class _Follower:
    """Joints follow the commands; the gripper stops at 20 on a cube when the attempt 'holds'."""

    use_degrees = True

    def __init__(self, holds):
        self.state = np.array([-5.7, -104.3, 91.3, 76.3, 2.8, 1.5])
        self.holds, self.closures, self.was_open = list(holds), 0, False

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_observation(self):
        from so101_tpgmm.robot.so101_runner import JOINT_KEYS

        return {k: float(v) for k, v in zip(JOINT_KEYS, self.state)}

    def send_action(self, action):
        from so101_tpgmm.robot.so101_runner import JOINT_KEYS

        q = np.array([action[k] for k in JOINT_KEYS])
        if q[5] > 26:
            self.was_open = True
        elif q[5] < 12 and self.was_open:
            self.was_open = False
            self.closures += 1
        holding = self.closures and self.holds[min(self.closures, len(self.holds)) - 1] and q[5] < 26
        self.state = q.copy()
        if holding:
            self.state[5] = max(q[5], 20.0)
        return action


class _Camera:
    plane_z = 0.0064

    def __init__(self, rgb, p2b):
        self.rgb, self.p2b = rgb, p2b

    def grab(self):
        return self.rgb

    def view(self, rgb):
        return self.p2b, CFG


@pytest.mark.skipif(not (resolve_path(CFG["tpgmm"]["models_dir"]) / "segmented.json").exists(),
                    reason="trained TP-GMM not available (run scripts/train.py)")
@pytest.mark.parametrize("holds, status, attempts", [([True], "released", 1), ([False, True], "released", 2),
                                                     ([False, False, False], "missed after 3 attempts", 3)])
def test_executor_replans_a_missed_grasp(holds, status, attempts):
    from so101_tpgmm.kinematics.ik import ik_from_config
    from so101_tpgmm.policies import load_policy
    from so101_tpgmm.robot.pick_place import PickPlaceExecutor
    from so101_tpgmm.robot.so101_runner import SO101Runner
    from so101_tpgmm.vision.geometry import PixelToBase

    H = np.array([[-1.1176e-04, 9.6872e-04, 4.3968e-02], [8.8654e-04, 9.5677e-05, -2.8609e-01],
                  [-2.0675e-04, -6.7702e-05, 1.0]])
    img = cv2.cvtColor(cv2.imread(str(resolve_path("configs/reference_top.jpg"))), cv2.COLOR_BGR2RGB)
    cv2.fillPoly(img, [cv2.boxPoints(((200.0, 250.0), (38.0, 38.0), 20.0)).astype(np.int32)], (40, 160, 40))
    follower = _Follower(holds)
    runner = SO101Runner(robot=follower)
    ex = PickPlaceExecutor(CFG, runner, _Camera(img, PixelToBase(H)), load_policy(CFG, "segmented"), ik_from_config(CFG),
                           realtime=False, log=lambda *_: None)
    (r,) = ex.run(["green"])
    assert (r.status, r.attempts) == (status, attempts)
    assert [("HELD" in v) for v in r.verdicts] == holds[:attempts]
    assert ex.run(["red"])[0].status == "not found"
