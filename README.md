
<p align="center">
  <img src="/resources/sim_a-iq-ready_image.png?raw=true" alt="VTPRL Warehouse Environment" width="100%"/>
</p>

<h1 align="center">VTPRL — Virtual Training Platform for Robot Learning</h1>

<p align="center">
  <b>A Unity-based simulation platform for training and evaluating reinforcement-learning agents<br>in robotic manipulation and warehouse AMR navigation tasks.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Unity-6-black?logo=unity" alt="Unity"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/RL-Stable--Baselines3-orange" alt="SB3"/>
  <img src="https://img.shields.io/badge/IK-DART-red" alt="DART"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue" alt="BSD-3-Clause"/></a>
</p>

---

## Overview

VTPRL connects a Unity physics simulator with Python gym environments over gRPC for end-to-end robot learning. The Unity side handles rigid-body physics, rendering, and sensor simulation while the Python side provides gym-compatible environments, RL training via Stable-Baselines3 (e.g., PPO), inverse kinematics via DART, autonomous navigation planning (A\* + DWA/DWB), live task monitoring, full-episode data-trace recording, and cross-language step profiling.

### Key capabilities

- **Multi-robot warehouse navigation** — SAFELOG S2 AMRs with laser scan, NavMesh-based occupancy grids, A\* global planning, and DWA/DWB local trajectory planning
- **Manipulator task-space control** — Kuka IIWA 14 (7-DoF), and SO-100 (5-DoF) with DART inverse kinematics and optional grippers
- **Vectorised training** — parallel environments with Stable-Baselines3 `VecEnv` interface for scalable PPO training
- **Four transport modes** — `GRPC`, `GRPC_NRP`, `GRPC_BIN` (MessagePack), `GRPC_SHM` (shared-memory ring buffer)
- **Data-trace recording** — telemetry channels (laser, images, occupancy, planner paths, item/obstacle poses, etc.) saved to NPZ + JSONL in a dedicated process
- **Offline trace playback** — replay recorded episodes through the task monitor without the simulator
- **Live task monitoring** — real-time web dashboard (FastAPI + WebSocket + Plotly.js) or desktop GUI (PySide2)
- **Cross-language step profiler** — wall-clock timing of every phase from Python IK through gRPC to Unity physics
- **Domain randomisation** — randomise physics, spawn poses, lighting, materials, camera viewpoints

---

## Project structure

```
VTPRL/
├── agent/
│   ├── main.py                                          # Entry point: train / evaluate / model-based
│   ├── main_advanced.py                                 # Advanced training pipeline (planar grasping)
│   ├── config.py                                        # Centralised configuration (all dictionaries)
│   ├── config_advanced.py                               # Extended config for advanced scenarios
│   ├── simulator_vec_env.py                             # Vectorised env wrapper (Unity ↔ Python bridge)
│   ├── env_gym_run.py                                   # Simple gym environment runner
│   ├── env_sim_run.py                                   # Simulator environment runner
│   ├── envs/
│   │   ├── warehouse_unity_env.py                       # Warehouse AMR navigation
│   │   ├── iiwa_sample_env.py                           # IIWA task-space (DART IK)
│   │   ├── iiwa_sample_joint_vel_env.py                 # IIWA joint-velocity control
│   │   └── so100_sample_env.py                          # SO-100 task-space (DART IK)
│   ├── envs_dart/
│   │   ├── base_dart_simulation.so                      # Abstract base for DART envs
│   │   ├── iiwa_dart_unity.so                           # IIWA DART + Unity bridge
│   │   ├── so100_dart_unity.so                          # SO-100 DART + Unity bridge
│   │   ├── iiwa_dart.so                                 # IIWA DART-only (no Unity)
│   │   ├── so100_dart.so                                # SO-100 DART-only
│   │   ├── README.md                                    # DART environments documentation
│   │   ├── LICENSE                                      # DART license
│   │   └── misc/                                        # DART resource files (URDF, meshes)
│   ├── envs_other/
│   │   ├── cartpole.py                                  # CartPole benchmark
│   │   ├── pendulum.py                                  # Pendulum benchmark
│   │   ├── nlinks_box2d.py                              # N-link chain (Box2D physics)
│   ├── envs_advanced/
│   │   ├── iiwa_numerical_planar_grasping_env.py        # Numerical planar grasping
│   │   ├── iiwa_end_to_end_planar_grasping_env.py       # E2E vision-based grasping
│   │   └── iiwa_ruckig_planar_grasping_env.py           # Trajectory generation (Ruckig)
│   ├── utils/
│   │   ├── data_trace_schema.py                         # Channel definitions and RecordingConfig
│   │   ├── data_trace_recorder.py                       # Episode recorder (NPZ + JSONL)
│   │   ├── data_trace_player.py                         # Offline trace replayer (CLI + API)
│   │   ├── data_trace_proxy.py                          # IPC bridge to recorder process
│   │   ├── step_profiler.py                             # Cross-language wall-clock profiler
│   │   ├── task_monitor.so                              # Desktop Qt (PySide2) monitor
│   │   ├── task_monitor_ipc.py                          # IPC message protocol
│   │   ├── task_monitor_process.py                      # Qt monitor subprocess launcher
│   │   ├── task_monitor_proxy.py                        # Qt IPC proxy controller
│   │   ├── task_monitor_web.so                          # Web (FastAPI + WebSocket) monitor
│   │   ├── task_monitor_web_client.html                 # Browser dashboard (Plotly.js)
│   │   ├── task_monitor_web_process.py                  # Web monitor subprocess launcher
│   │   ├── task_monitor_web_proxy.py                    # Web IPC proxy controller
│   │   ├── telemetry.py                                 # MonitorPayload and sensor data structures
│   │   ├── astar_planner.py                             # A* global path planner
│   │   ├── dwa_local_planner.py                         # DWA local trajectory planner
│   │   ├── dwb_local_planner.py                         # DWB (Nav2-style) local planner
│   │   ├── navmesh_occupancy.py                         # NavMesh to occupancy grid rasteriser
│   │   ├── policy_networks.py                           # Custom NN architectures
│   │   ├── shared_memory_ring_buffer.py                 # Lock-free SHM ring buffer
│   │   ├── shared_memory_transport.py                   # SHM transport layer
│   │   ├── simulator_configuration.py                   # XML config reader/writer
│   │   ├── config_utils.py                              # Config validation helpers
│   │   ├── helpers.py                                   # Seed utilities
│   │   ├── dart_guide.py                                # DART usage guide
│   │   ├── service_pb2.py                               # gRPC protobuf (generated)
│   │   ├── service_pb2_grpc.py                          # gRPC service stubs (generated)
│   ├── utils_advanced/
│   │   ├── warehouse_trace_analysis_toolkit.py          # 50+ metrics, figures, Markdown reports
│   │   ├── warehouse_trace_sample_tutorial.py           # Tutorial: load and plot traces
│   │   ├── boxes_generator.py                           # Random box spawner for grasping
│   │   ├── evaluate.py                                  # Model evaluation pipeline
│   │   ├── helpers.py                                   # Env creation utilities
│   │   ├── manual_actions.py                            # Manual teleoperation
│   │   └── monitoring_agent.py                          # Training callbacks
│   └── models/
│       └── ruckig_planar_model.py                       # Ruckig trajectory generator
├── Docker/
│   ├── Dockerfile_Python310                             # CUDA 12.8, Python 3.10 (recommended)
│   ├── Dockerfile_Python38                              # CUDA 11.0, Python 3.8
│   ├── Dockerfile                                       # CUDA 11.0, Python 3.7
│   ├── Dockerfile_NoCUDA                                # CPU-only, Python 3.7
│   ├── Dockerfile_ROS                                   # ROS integration
│   ├── Commands                                         # Build and run examples
│   ├── requirements_Python310.txt                       # Python 3.10 dependencies
│   └── requirements_Python38.txt                        # Python 3.8 dependencies
│   ├── requirements.txt                                 # Python 3.7 dependencies
├── Dart_Additional_Files/                               # Custom DartPy bindings source
├── docs/
│   └── Configuration-Parameters.md                      # Unity XML parameter reference
├── external/
│   └── stable-baselines3/                               # RL algorithms (Git submodule)
└── resources/
    ├── sim_a-iq-ready_image.png                         # Warehouse environment screenshot
    └── sim_ai4di_image.png                              # Manipulator environment screenshot
```

---

## Getting started

### 1. Install Git LFS

This repository uses [Git Large File Storage (LFS)](https://git-lfs.com/) for binary assets. Install and configure it **before** cloning:

```bash
# Install Git LFS (one-time system setup)
git lfs install
```

### 2. Clone the repository

Clone with `--recurse-submodules` to fetch all dependencies in one step:

```bash
git clone --recurse-submodules https://github.com/tum-i6/VTPRL
```

If the repository was already cloned without submodules, initialise them with:

```bash
git submodule update --init --recursive
```

### 3. Obtain the Unity simulator

The latest versions of the pre-built Unity simulator executable is **not** included in this repository. It is available by request. Place the executable in `environment/simulator` — it will be launched separately from the Python agent.

### 4. Start the Unity simulator

Launch the VTPRL simulator executable for your platform. The simulator starts a gRPC server and waits for agent connections on the port configured in `configuration.xml` (default `9092`).

### 5. Install Python dependencies

The recommended setup uses Docker (GPU-accelerated when possible):

```bash
# Build the image (Python 3.10 + CUDA 12.8 — recommended)
docker build . -t vtprl:py310 -f Docker/Dockerfile_Python310

# Run with GPU, shared memory, and display forwarding
docker run --gpus all --shm-size=512m \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  -it vtprl:py310 bash
```

Alternatively, a pre-built Python 3.10 image is available on [Docker Hub](https://hub.docker.com/r/mhmalmir/vtprl) to avoid building from scratch (useful when compute resources are limited):

```bash
# Pull the pre-built image
docker pull mhmalmir/vtprl:py310

# Run with the same flags as above
docker run --gpus all --shm-size=512m \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  -it mhmalmir/vtprl:py310 bash
```

> **Note:** DART is not supported natively on Windows — use Docker or WSL. See `Docker/Commands` for platform-specific examples including WSL2 on Windows 11.

### 6. Run training or evaluation

```bash
cd workspace/agent
python main.py
```

All behaviour is controlled through `agent/config.py`. The default mode is `evaluate_model_based` with the `warehouse_unity_env` environment.

---

## Environments

### Standard environments

| `env_key` | Class | Robot | Description |
|---|---|---|---|
| `warehouse_unity_env` | `WarehouseUnityEnv` | SAFELOG S2 AMR | Multi-robot navigation with laser scan, NavMesh occupancy, A\* + DWA/DWB planning, obstacle avoidance |
| `iiwa_sample_dart_unity_env` | `IiwaSampleEnv` | Kuka IIWA 14 | 7-DoF task-space reaching with DART inverse kinematics, optional gripper |
| `iiwa_joint_vel` | `IiwaJointVelEnv` | Kuka IIWA 14 | 7-DoF joint-velocity control, supports image observations |
| `so100_sample_dart_unity_env` | `SO100SampleEnv` | SO-100 | 5-DoF task-space control with DART IK |

### Additional robots

| Class | Robot | Notes |
|---|---|---|
| `CartPoleEnv` / `PendulumEnv` | Classic benchmarks | Python-only, no Unity required |
| `NLinksBox2DEnv` | N-link chains | Configurable link count (Box2D physics) |

### Advanced manipulation

| Class | Task |
|---|---|
| `IiwaNumericalPlanarGraspingEnv` | Planar grasping with numerical 7D box poses |
| `IiwaEndToEndPlanarGraspingEnv` | End-to-end vision-based grasping |
| `IiwaRuckigPlanarGraspingEnv` | Model-based trajectory generation (Ruckig) |

To select an environment, set `env_key` in `config.py` → `get_gym_environment_dict()`.

---

## Configuration

All configuration is centralised in `agent/config.py` via the `Config` class. The Python config is written to `configuration.xml` which the Unity simulator reads on startup.

| Dictionary | Purpose |
|---|---|
| `agent_dict` | Simulation mode (`train` / `evaluate` / `evaluate_model_based`), log directory, total timesteps |
| `gym_environment_dict` | Environment key, parallel env count, episode length, task monitor, data trace settings |
| `simulation_dict` | Communication type, gRPC address/port, profiling, physics timestep, domain randomisation |
| `manipulator_environment_dict` | Robot model, instance count, end-effector, joint drives, floor, items |
| `warehouse_environment_dict` | AMR model, laser scan config, obstacle manager, transport enable, ground/wall setup |
| `observation_dict` | Camera image capture, resolution, encoding, segmentation, shadows |

### Simulation modes

| Mode | Behaviour |
|---|---|
| `train` | PPO training with Stable-Baselines3 for the configured number of timesteps; saves checkpoint to `log_dir` |
| `evaluate` | Loads a saved PPO checkpoint and runs deterministic inference |
| `evaluate_model_based` | Runs a P-controller or scripted policy — no learned model needed |

### Communication types

| Type | Description |
|---|---|
| `GRPC` | Standard gRPC with JSON serialisation |
| `GRPC_BIN` | gRPC with MessagePack binary payloads |
| `GRPC_SHM` | Shared-memory ring buffer + gRPC control plane (highest throughput) |
| `GRPC_NRP` | gRPC with Neurorobotics platform |

### Manipulator-specific settings

Key options in `get_manipulator_gym_environment_dict()` and `get_dart_dict()`:

- **DART IK** — `use_inverse_kinematics`, `orientation_control` (3D position or 6D pose), `linear_motion_conservation` (SNS IK)
- **Target generation** — `target_mode`: `random`, `random_joint_level`, `import` (CSV), `fixed`
- **Safety** — `joints_safety_limit`, velocity/acceleration caps (`max_joint_vel`, `max_ee_cart_vel`, etc.)
- **End-effectors** — `ROBOTIQ_3F`, `ROBOTIQ_2F85`, `CALIBRATION_PIN`, `DEFAULT_GRIPPER`
- **Robot models** — `IIWA14`, `SO100`

### Warehouse-specific settings

Key options in `get_warehouse_gym_environment_dict()`:

- **Navigation** — `success_distance_threshold`, `success_yaw_threshold`, reward shaping weights
- **Laser** — `enable_laser_scan`, range, angles, measurement count, sensor offset
- **Planning** — local controller type (`DWA` or `DWB`), A\* obstacle clearance, DWA/DWB gain tuning
- **Obstacles** — static and dynamic pools; dynamic motion patterns (`Random`, `Circle`, `Linear`)
- **Transport** — `enable_transport` for trolley transport operations

For the full Unity-side XML parameter reference, see [docs/Configuration-Parameters.md](/docs/Configuration-Parameters.md).

---

## Data-trace recording

The data-trace system records per-step telemetry to disk for offline replay and analysis. Recording runs in a dedicated child process so training throughput is unaffected.

### Enabling

In `config.py` → `get_gym_environment_dict()`:

```python
'data_trace': {
    'enabled': True,
    'trace_root': 'traces',
    'channels': None,                # None = all; or ['AGENT_STATE', 'IMAGES', ...]
    'compress_arrays': True,
    'max_episodes': None,            # None = unlimited
    'flush_interval_steps': 200,
    'record_env_ids': None,          # None = all envs; or [0, 1]
},
```

Console output confirms recording: `[DATA-TRACE] Recorder process started -> traces`.

### Channels

| Channel | Format | Content |
|---|---|---|
| `AGENT_STATE` | NPZ | Observation vectors per robot per step |
| `LASER_SCAN` | NPZ | Raw laser ranges + metadata |
| `LASER_POINTS` | NPZ | Projected 2D laser points (N×2) |
| `OCCUPANCY_GRID` | NPZ | Rasterised NavMesh occupancy |
| `COSTMAP` | NPZ | Planner costmap arrays |
| `NAVMESH` | NPZ | NavMesh vertices + triangle indices |
| `PLANNER_PATHS` | JSONL | Global path + local trajectory per step |
| `IMAGES` | NPZ | Overhead and per-robot camera frames |
| `ITEM_POSES` | JSONL | Movable item positions and orientations |
| `OBSTACLE_POSES` | JSONL | Static/dynamic obstacle poses |
| `ROBOTS_PAYLOAD` | JSONL | Per-robot scalar telemetry dictionaries |

### Storage layout

```
traces/
├── metadata.json
└── env_000/
    └── episode_0000/
        ├── robots_payload.jsonl
        ├── planner_paths.jsonl
        ├── item_poses.jsonl
        ├── obstacle_poses.jsonl
        ├── laser_scans.npz
        ├── laser_points.npz
        ├── occupancy_grids.npz
        ├── costmaps.npz
        ├── navmeshes.npz
        ├── agent_states.npz
        └── images.npz
```

NPZ arrays use per-robot prefixed keys: `r{robot_idx}_{type}_{step:06d}`.

### Supported environments

- **Warehouse** (`warehouse_unity_env`): all 11 channels — laser, occupancy, costmap, navmesh, planner paths, images, item/obstacle poses, and per-robot state.
- **Manipulator** (`iiwa_joint_vel`, `iiwa_sample_dart_unity_env`, `so100_sample_dart_unity_env`): joint angles, velocities, end-effector/target/object poses, gripper state, collision flag, reward, and images.

---

## Data-trace playback

Replay recorded traces through the task monitor without the Unity simulator.

### Command line

```bash
cd agent

# Play all episodes for run_001 at real-time speed
python -m utils.data_trace_player traces/run_001/ --speed 1.0 --backend web

# Play at 2× speed, specific env and episode
python -m utils.data_trace_player traces/run_001/ --speed 2.0 --env 0 --episode 0

# Play using the Qt desktop monitor
python -m utils.data_trace_player traces/run_001/ --backend qt
```

| Argument | Default | Description |
|---|---|---|
| `trace_dir` | *(required)* | Path to the trace root directory |
| `--speed` | `1.0` | Playback multiplier (`0` = as fast as possible) |
| `--backend` | `web` | `web` (browser dashboard) or `qt` (PySide2 desktop window) |
| `--env` | all | Replay only this environment id |
| `--episode` | all | Replay only this episode index |

### Python API

```python
from utils.data_trace_player import DataTracePlayer

with DataTracePlayer("./traces", monitor_backend="web") as player:
    for env_id, ep_idx in player.episodes():
        print(f"  env {env_id}  episode {ep_idx}  ({player.episode_step_count(env_id, ep_idx)} steps)")

    player.play_episode(env_id=0, episode_index=0, speed=1.0)

    # Programmatic access
    payloads = player.load_episode(env_id=0, episode_index=0)
    for p in payloads:
        print(p.reward, p.collision)
```

---

## Data-trace analysis toolkit

For offline experiment evaluation, the analysis toolkit computes 50+ metrics from recorded warehouse traces — single-agent, multi-agent, planner, and sensor metrics — then exports structured result tables and generates summary figures and Markdown reports.

```bash
cd agent
python utils_advanced/warehouse_trace_analysis_toolkit.py --traces ./traces --output ./analysis_results
```

A companion tutorial script shows how to load traces and plot trajectories:

```bash
python utils_advanced/warehouse_trace_sample_tutorial.py
```

---

## Task monitor

The task monitor provides a live dashboard for inspecting robot state, reward curves, sensor data, planner maps, and camera images during training or playback.

### Enabling

In `config.py`:

```python
'task_monitor': True,
'task_monitor_type': 'web',   # 'web' (browser dashboard) or 'qt' (PySide2 desktop window)
```

| Backend | Technology | URL | Notes |
|---|---|---|---|
| `web` | FastAPI + WebSocket + Plotly.js | `http://127.0.0.1:8050` | Headless-friendly; dark/light theme; environment dropdown |
| `qt` | PySide2 | *(desktop window)* | Requires display; embedded Qt charts |

The web dashboard supports real-time streaming via WebSocket with MessagePack-encoded telemetry payloads.

---

## Step profiler

Measures wall-clock timing for every phase in the simulation loop across both Python and Unity to identify bottlenecks.

### Enabling

In `config.py` → `get_simulation_dict()`:

```python
'enable_profiling': True,
'profiling_print_every_n': 1    # print every N steps
```

### Measured sections

**Python side:**

| Section | Description |
|---|---|
| `python_action_conversion` | IK solve / action formatting |
| `python_create_request` | Command dict building + JSON serialisation |
| `python_send_request` | Overall transport round-trip |
| `python_request_encode` | MsgPack + Base64 encode (`GRPC_BIN` only) |
| `python_grpc_call` | Blocking gRPC call |
| `python_response_decode` | Response deserialisation |
| `python_update_envs` | Observation parsing, DART chain update, reward computation |
| `python_total` | Total `step()` wall-clock time |

**Unity side** *(received via `ProfilingData` in observation payload):*

| Section | Description |
|---|---|
| `unity_shm_read` | Shared-memory read (`GRPC_SHM` only) |
| `unity_request_decode` | MsgPack deserialisation |
| `unity_command_parsing` | Applying actions to articulation bodies |
| `unity_physics` | `Physics.Simulate()` calls |
| `unity_observation_collection` | `GetObservationPayload()` per environment |
| `unity_response_serialize` | Response encoding |

**Derived:**

| Metric | Formula |
|---|---|
| `communication_overhead` | `python_grpc_call − unity_total` |
| `step_gap` | Wall-clock time between successive `step()` calls |

---

## Installation

### Docker (recommended)

Docker is the recommended setup on all platforms. Multiple Dockerfile variants are provided:

| Dockerfile | Base | Python | GPU |
|---|---|---|---|
| `Dockerfile_Python310` | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` | 3.10 | CUDA 12.8 |
| `Dockerfile` | `nvidia/cudagl:11.0-devel-ubuntu18.04` | 3.7 | CUDA 11.0 |
| `Dockerfile_Python38` | Ubuntu 18.04 | 3.8 | CUDA 11.0 |
| `Dockerfile_NoCUDA` | Ubuntu 18.04 | 3.7 | None |
| `Dockerfile_ROS` | ROS base | — | — |

Build and run commands are documented in `Docker/Commands`.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux)
- [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) for Windows hosts
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) for GPU acceleration

> **Note:** DART is not natively supported on Windows. Use Docker or WSL for environments that require DART (all task-space manipulator environments).

### Running from Docker on Windows

Set `ip_address` to `host.docker.internal` in `config.py` → `get_simulation_dict()` so the container can reach the simulator running on the Windows host.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **Simulator errors** | Check `Player.log` at the OS-specific [Unity log location](https://docs.unity3d.com/Manual/LogFiles.html) (Company: `TUM-CIT-AIR`, Product: `VTPRL-Simulator`) |
| **Docker build stuck** | Increase RAM/CPU limits in Docker Desktop settings |
| **`No module named 'stable_baselines3'`** | Run `pip install --upgrade stable_baselines3` inside the container |
| **`grpc … UNAVAILABLE`** | Verify the simulator is running and `configuration.xml` port matches `config.py`. From Docker on Windows, use `host.docker.internal` as `ip_address` |
| **Data-trace records nothing** | Ensure `'enabled': True` in `data_trace` config. Check console for `[DATA-TRACE] Recorder process started` |
| **DART import error** | DART requires Linux; use the Docker setup or WSL |

---

## Authors and acknowledgment

Developed at the [Chair of Robotics, Artificial Intelligence and Real-time Systems](https://www.ce.cit.tum.de/air/home/), Technical University of Munich.

This work has been performed in the following projects:

- **[AI4DI](https://ai4di.eu/)** — Artificial Intelligence for Digitizing Industry, under grant agreement No. 826060. Co-funded by grants from Germany, Austria, Finland, France, Norway, Latvia, Belgium, Italy, Switzerland, and the Czech Republic, and by the Electronic Component Systems for European Leadership Joint Undertaking (ECSEL JU).
- **[A-IQ READY](https://www.aiqready.eu/)** — Artificial Intelligence using Quantum Measured Information for Realtime Distributed Systems at the Edge, under grant agreement No. 101096658. Funded within the Chips Joint Undertaking (Chips JU) — the Public-Private Partnership for research, development, and innovation under Horizon Europe — and National Authorities.

<p align="center">
  <img src="/resources/sim_ai4di_image.png?raw=true" alt="VTPRL Manipulator Environment" width="48%"/>
</p>
