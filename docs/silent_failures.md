# Silent failures — catalogue

Failures that do not crash but corrupt results. Each entry: symptom, how it is detected
(test/assert), and status. Updated phase by phase.

| # | Failure | Why it is silent | Guard | Status |
|---|---|---|---|---|
| 1 | **Joint units misread** (degrees vs `RANGE_M100_100`) | Both are floats of similar magnitude (±100); FK still returns plausible-looking poses | `infer_joint_units` (quantization 360/4095° + range evidence) runs at load; contradictory config raises. Tests: `tests/test_units.py` (both modes, float32, contradictory data), real-data test asserts DEGREES; grasp heights tight within ±1.5 cm (report §4) | guarded (Phase 0) |
| 2 | **Video vs state timestamp misalignment** (v3 packs many episodes per MP4) | Off-by-episode or off-by-frames returns a valid image of the wrong instant | `VideoReader` raises `FrameTimestampError` beyond 0.4 frame; episode offset `from_timestamp` applied in one place. Tests: synthetic brightness-coded video (`test_video_and_dataset.py`), real-data alignment; report §5: 0.00 ms on 8 episodes, span == length for all 400 | guarded (Phase 0) |
| 3 | **Gripper threshold without hysteresis** (phantom cycles) | Plateaus at 20 (holding) and 25–28 (small openings) hover near any single threshold | Hysteresis band 23.5/25.5 + debounce + plateau-level classification + min hold duration. Tests: `test_gripper.py` (noise at threshold, short holds, missed grasps) | guarded (Phase 0) |
| 3b | **Hold-level plateau in the air** (operator half-closes the leader on the way home) | Looks exactly like a grasp by gripper value alone (16 episodes, report §3) | Reported by TCP height in Phase 0; table-height gate to be enforced in Phase 2 | open |
| 3c | **Release with a tiny opening** (< open threshold) | Cycle is lost (5 episodes) instead of mis-segmented | Detected (hold ends over the box); Phase 2 will use the hold-plateau end as release | open |
| 4 | Red hue wrap-around | — | Phase 3 | pending |
| 5 | Arm occluding cubes in the first frame | — | Phase 3 | pending |
| 6 | 90° yaw ambiguity of the cube | — | Phase 3 | pending |
| 7 | **Homography to the wrong plane** (table vs cube centre) / TCP ≠ cube centre | A homography fits either plane with small error; the bias appears only at execution | Phase 0: cube-centre offset estimated jointly (10.6 → 8.4 mm); `homography_is_orientation_reversing` detects a mirrored joint convention. Tests: `test_camera_pose.py`. Plane choice formalized in Phase 3 | partial |
| 8 | Positions outside the reachable workspace / joint limits | IK "solutions" clipped to limits silently | Phase 0 finding: 64 % of recorded frames violate the URDF limits (rest pose `shoulder_lift` −104° < −100°) → IK limits must include the observed range (Phase 1) | open |
| 9 | **Camera pose over-trusted** | Lens position/tilt look precise but are poorly determined from a near-overhead planar fit | Bootstrap CIs (dataset_report §6); metric grid calibration with principal-point/height sensitivity table (camera_calibration.md) | documented |
| 10 | **FK metric scale vs reality** | A uniform scale error keeps trajectories smooth and plausible | Grid ↔ FK similarity fit reports scale 0.953; to be verified on the robot | open |
