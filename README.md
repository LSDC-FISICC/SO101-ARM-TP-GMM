# SO101-ARM-TP-GMM

Learning from Demonstration with **Task-Parameterized Gaussian Mixture Models** for the
SO-101 arm (LeRobot, 5 DOF + gripper), trained on the teleoperation dataset
[`SebRivGar/dataset3`](https://huggingface.co/datasets/SebRivGar/dataset3) (LeRobot v3.0,
originally recorded for ACT). Task: pick a coloured cube from the table and drop it in a fixed box.

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Dataset inspection → [`docs/dataset_report.md`](docs/dataset_report.md) | done |
| 1 | FK (done) + numerical IK | in progress |
| 2 | Gripper-based segmentation into primitives | detector done, primitives pending |
| 3 | Fixed-camera vision + pixel→table calibration | pending (preliminary geometry in the report) |
| 4 | TP-GMM (numpy EM + GMR) | pending |
| 5 | Leave-one-out evaluation, inference, robot stub | pending |

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
python scripts/inspect_dataset.py --download   # ~1 GB: tabular data + fixed-camera videos
python -m pytest
```

## Layout

```
configs/default.yaml        all thresholds / paths (YAML, overridable with --set key=value)
so101_tpgmm/
  data/        LeRobot v3 reader, AV1 video reader (timestamp-checked), unit inference
  kinematics/  URDF parser, batched FK (vendored so101_new_calib.urdf)
  segmentation/ gripper event detection (hysteresis, 3-level plateaus)
  vision/      camera <-> robot geometry (homography, decomposition)
  tpgmm/  robot/   (later phases)
scripts/inspect_dataset.py  Phase 0 report generator
docs/                       reports, silent-failure catalogue, figures
tests/                      pytest (real-data tests skip if the dataset is absent)
```

## Design decisions (so far)

* **No `lerobot` import in the offline pipeline.** The v3.0 parquet/MP4 layout is read
  directly, so training/evaluation run without torch; `lerobot` is only needed by the robot
  runner on the machine connected to the arm.
* **Units are inferred, never assumed.** LeRobot's normalization leaves integer-tick
  fingerprints; `infer_joint_units` checks them and the range (|v| > 100 is impossible in
  `RANGE_M100_100`). The dataset is in **degrees**; FK only accepts radians and the conversion
  is explicit (`fk_from_dataset`).
* **Video frames are verified by timestamp**: `VideoReader` raises if the decoded
  presentation time differs from the requested one by more than 0.4 frame.
* **Gripper: three levels + hysteresis**, not one threshold: closed-empty / holding (~20, the
  4 cm cube stops the jaws) / open. A level-only rule creates phantom cycles when the operator
  half-closes the gripper in the air; a table-height gate is added in Phase 2.
* **Cube centre ≠ TCP.** `gripper_frame_link` is offset from the centre of a grasped cube
  (single moving jaw); calibration and the cube frame use `p_tcp + R_tcp · offset`.
* **Camera geometry from the data is only partly identifiable**: pixel→table mapping and
  focal (given the measured height) yes; lens position/tilt only within several cm/degrees.
  See report §6.
