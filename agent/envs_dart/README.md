# DART-Based Manipulator Environments

Compiled gym environments for serial manipulators using the [DART](https://github.com/dartsim/dart) physics engine for inverse kinematics, dynamics and simulation.

## Environments

| Environment | Robot | DOF | Class | Description |
|-------------|-------|-----|-------|-------------|
| **iiwa_dart** | Kuka LBR iiwa 14 R820 | 7 | `IiwaDartEnv` | Stand-alone DART simulation of the iiwa manipulator |
| **iiwa_dart_unity** | Kuka LBR iiwa 14 + Gripper | 7 | `IiwaDartUnityEnv` | Integrated DART (IK) + Unity (physics & rendering) |
| **so100_dart** | Standard Open SO-100 | 5 | `SO100DartEnv` | Stand-alone DART simulation of the SO-100 arm |
| **so100_dart_unity** | Standard Open SO-100 | 5 | `SO100DartUnityEnv` | Integrated DART (IK) + Unity (physics & rendering) |

All environments inherit from `BaseDartSimulation`, which provides differential IK/ID for arbitrary serial chains with two IK methods: minimum-norm and SNS (Flacco et al.). Action spaces are configurable — joint-space velocity, Cartesian position, or full SE(3).

## Files

| File | Purpose |
|------|---------|
| `base_dart_simulation.py` | Abstract base class — differential IK/ID, PD control, torque/velocity modes |
| `iiwa_dart.py` | Kuka iiwa 7-DOF pure-DART gym environment |
| `iiwa_dart_unity.py` | Kuka iiwa DART ↔ Unity bridge (image capture, telemetry) |
| `so100_dart.py` | SO-100 5-DOF pure-DART gym environment |
| `so100_dart_unity.py` | SO-100 DART ↔ Unity bridge |
| `LICENSE` | BSD-2-Clause (DART physics engine) |

## Resources (`misc/`)

| Directory | Contents |
|-----------|----------|
| `kuka_iiwa_urdf/` | URDF descriptions for iiwa14 variants (base, 2-finger gripper, 3-finger gripper, calibration pin) and mesh assets |
| `so100_urdf/` | URDF description and meshes for the SO-100 arm |
| `generated_random_targets/` | Pre-generated CSV files of random Cartesian target poses |
| `real_two_joints_targets/` | Recorded real-world target trajectories on sphere paths |
