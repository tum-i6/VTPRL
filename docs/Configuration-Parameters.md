# VTPRL Simulator Configuration Parameters

This document lists every configuration parameter that can be set via the XML configuration file (`configuration.xml`). You can edit the XML before launching the simulator executable. All parameters shown here map 1:1 to fields in the configuration classes; no code access is required.

Notes
- Units are SI unless specified (meters, seconds, radians, kilograms).
- Colors are RGBA components in 0–1 range unless noted.
- Vectors are comma-separated triples: x,y,z.
- Arrays are comma-separated lists.

## Top-level structure

Root element: `Configuration`
- `Simulation` (SimulationParameters)
- `ManipulatorEnvironment` (ManipulatorEnvironmentParameters)
- `WarehouseEnvironment` (WarehouseEnvironmentParameters)
- `Observation` (ObservationParameters)

- `EnvironmentMode`: Manipulator | Warehouse
  Description: Selects which environment family to initialize and which environment-specific parameter block is applied.
  Default: Manipulator

---

## SimulationParameters (Configuration.Simulation)

- `CommunicationType`: ROS | ZMQ | GRPC | GRPC_NRP | GRPC_SHM | GRPC_BIN
  Description: Communication backend used by the simulator. GRPC_SHM uses shared-memory ring buffers with a lightweight gRPC control plane. GRPC_BIN uses MessagePack binary serialization over gRPC for reduced overhead.
  Default: GRPC

- `IPAddress`: string
  Description: Host/IP to bind/connect (e.g., "localhost").
  Default: localhost

- `PortNumber`: int
  Description: Service port for the simulator.
  Default: 9090

- `SharedMemoryDirectory`: string
  Description: Filesystem directory for shared-memory backing files used by GRPC_SHM communication mode. This path must be accessible from both the Unity host and the Python agent container (e.g., a Docker bind-mount such as C:\shm → /mnt/shm).
  Default: C:\shm

- `SharedMemoryCapacity`: int
  Description: Number of slots in each shared-memory ring buffer. Must be a power of two. Higher values allow more in-flight messages but consume proportionally more memory.
  Default: 4

- `SharedMemorySlotSizeMB`: int
  Description: Maximum payload size per ring-buffer slot in mebibytes (MiB). Must be large enough to hold the biggest single observation or action payload.
  Default: 64

- `SharedMemorySameKernel`: bool
  Description: When true, both the Unity simulator and the Python agent share the same OS kernel (e.g., both run natively on Linux or via WSL2 with a native-Linux Docker container). This skips costly flush/invalidate operations on the memory-mapped files, maximizing throughput. Set to false when the processes run across different kernels (e.g., Windows host ↔ WSL2/Docker via the 9P filesystem).
  Default: false

- `EnableProfiling`: bool
  Description: When true, the simulator collects wall-clock timings for each major phase of a simulation step (request decode, command parsing, physics, observation collection, response serialization) and sends them back to the Python agent inside the first environment's observation payload. The Python side merges these with its own timings and prints a detailed per-step profiling report. Use for performance analysis; disable in production/training to avoid overhead.
  Default: false

- `TimestepDurationInSeconds`: float
  Description: Duration of one simulation step.
  Default: 0.02

- `PhysicsSimulationIncrementInSeconds`: float
  Description: Size of the physics substep. Simulator will run as many substeps as needed to cover one timestep. Recommended <= 0.03.
  Default: 0.02

- `ImprovedPatchFriction`: bool
  Description: Enables PhysX improved patch friction for more accurate results.
  Default: true

- `RandomSeed`: int
  Description: Fixed RNG seed; -1 means random each run.
  Default: -1

- `Evaluation`: bool
  Description: Evaluation vs training mode selector.
  Default: false

- `RandomizeEnvironmentPhysics`: bool
  Description: Randomize environment-level physics properties.
  Default: false

- `RandomizeTorque`: bool
  Description: Randomize joint motor torque limits (where applicable).
  Default: false

- `PersistEpisodeManifests`: bool
  Description: When true, a JSON manifest capturing all randomized parameters (lighting, palettes, camera deltas, physics, etc.) is written for every episode into a newly created run folder under Application.persistentDataPath/manifests (e.g., manifests/run_1700000000000/episode_1.json).
  Default: true

- `ReplayEpisodeManifest`: bool
  Description: Enables deterministic replay; suppresses new randomization and pulls values from manifest(s). ReplayManifestPath should point to a directory, where all episode_*.json files are loaded as a sequence and replay cycles with wrap-around.
  Default: false

- `ReplayManifestPath`: string (directory path)
  Description: Relative (resolved against Application.persistentDataPath/manifests) or absolute path. Supply a run folder containing multiple manifests. Required for meaningful replay; if omitted while ReplayEpisodeManifest is true, no data is loaded.
  Default: ""

---

## ManipulatorEnvironmentParameters (Configuration.ManipulatorEnvironment)

- `Manipulators`: ManipulatorParameters[]
  Description: Manipulator parameter segments. Each segment contributes `Count` robots. Targets are created per robot.

- `Floor`: FloorParameters
  Description: Stand/platform floor the manipulator is mounted on (material and size).

- `Items`: ItemParameters[]
  Description: Item parameter segments. Each segment contributes `Count` items. Items are not tied to robots or targets.

### ManipulatorParameters (ManipulatorEnvironment.Manipulators[n])

- `Count`: int
  Description: Number of manipulators to instantiate with this segment.
  Default: 1

- `ManipulatorModel`: IIWA14 | SO100
  Description: Robot arm model used for this segment.
  Default: IIWA14

- `EnableEndEffector`: bool
  Description: Attach an end effector; when disabled, the terminal link is used as the end-effector pose.
  Default: true

- `EndEffectorModel`: ROBOTIQ_3F | ROBOTIQ_2F85 | CALIBRATION_PIN | DEFAULT_GRIPPER
  Description: End-effector model to mount when `EnableEndEffector` is true.
  Default: CALIBRATION_PIN

- `RobotSegmentationColor`: Color (r,g,b,a)
  Description: Segmentation color used for this manipulator segment when segmentation mode is enabled.
  Default: magenta

- `RandomizeRobotAppearance`: bool
  Description: Allow per-episode appearance randomization for this manipulator segment.
  Default: true

- `JointDriveStiffness`: float
  Description: Stiffness gain used by the manipulator joint drives.
  Default: 10000

- `JointDriveDamping`: float
  Description: Damping gain used by the manipulator joint drives.
  Default: 10000

- `TrajectoryString`: string
  Description: Optional serialized trajectory program consumed by trajectory-following logic.
  Default: ""

- `TargetSize`: Vector3 (x,y,z)
  Description: Dimensions of the visual target marker spawned for this manipulator.
  Default: 0.1,0.001,0.1

- `TargetMaterialColor`: Color (r,g,b,a)
  Description: Color applied to the manipulator target marker material.
  Default: red

### FloorParameters (ManipulatorEnvironment.Floor)

- `FloorType`: CHECKERBOARD | WOOD | MONOCHROMATIC
  Description: Visual floor style for manipulator scenes.
  Default: MONOCHROMATIC

- `FloorSize`: Vector3 (x,y,z)
  Description: Floor dimensions in meters.
  Default: 2.4,0.01,2.4

- `FloorMaterial`: HOMOGENEOUS | HETEROGENEOUS
  Description: Whether floor friction is single-valued or spatially varying.
  Default: HOMOGENEOUS

- `VisualizeFloorMaterial`: bool
  Description: Visualize floor heterogeneity grid/material partitions when enabled.
  Default: false

- `FloorMaterialColor`: Color (r,g,b,a)
  Description: Base color used for monochromatic floor rendering.
  Default: 0.235,0.510,0.941,1

- `FloorMaterialGridX`: float[]
  Description: Normalized partition lengths along floor X used by heterogeneous material layout.
  Default: [1.0]

- `FloorMaterialGridY`: float[]
  Description: Normalized partition lengths along floor Y used by heterogeneous material layout.
  Default: [1.0]

- `FloorMaterialGridZ`: float[]
  Description: Normalized partition lengths along floor Z used by heterogeneous material layout.
  Default: [1.0]

- `FloorMaterialGridDynamicFriction`: float[]
  Description: Dynamic-friction value per floor material cell (ordered by Z-major traversal).
  Default: [1.0]

- `FloorMaterialGridStaticFriction`: float[]
  Description: Static-friction value per floor material cell (ordered by Z-major traversal).
  Default: [1.0]

### ItemParameters (ManipulatorEnvironment.Items[n])

- `Count`: int
  Description: Number of items to instantiate with this segment.
  Default: 1

- `ItemType`: SPHERE | BOX
  Description: Primitive geometry type used for items in this segment.
  Default: BOX

- `ItemSize`: Vector3 (x,y,z meters)
  Description: Item dimensions in meters.
  Default: 0.1,0.1,0.1

- `ItemMass`: float (kg)
  Description: Rigid-body mass of each spawned item.
  Default: 1.0

- `ItemCenterOfMass`: Vector3 (x,y,z meters)
  Description: Local center-of-mass offset relative to the item transform origin.
  Default: 0,0,0

- `ItemLinearDamping`: float
  Description: Linear damping (drag) applied to item rigid bodies.
  Default: 2.0

- `ItemObservability`: bool
  Description: Include this item segment in streamed state observations.
  Default: true

- `ItemMaterial`: HOMOGENEOUS | HETEROGENEOUS
  Description: Whether item friction is single-valued or grid-varying.
  Default: HOMOGENEOUS

- `VisualizeItemMaterial`: bool
  Description: Visualize item heterogeneity layout when enabled.
  Default: false

- `ItemMaterialColor`: Color (r,g,b,a)
  Description: Base item color used when heterogeneity visualization is disabled.
  Default: green

- `ItemMaterialGridX`: float[]
  Description: Normalized partition lengths along item X for heterogeneous materials.
  Default: [1.0]

- `ItemMaterialGridY`: float[]
  Description: Normalized partition lengths along item Y for heterogeneous materials.
  Default: [1.0]

- `ItemMaterialGridZ`: float[]
  Description: Normalized partition lengths along item Z for heterogeneous materials.
  Default: [1.0]

- `ItemMaterialGridDynamicFriction`: float[] (0–1)
  Description: Dynamic-friction value per item material cell (ordered by Z-major traversal).
  Default: [0.6]

- `ItemMaterialGridStaticFriction`: float[] (0–1)
  Description: Static-friction value per item material cell (ordered by Z-major traversal).
  Default: [0.6]

- `RandomizeItemMass`: bool
  Description: Enable per-episode randomization of `ItemMass` using the configured range.
  Default: false

- `ItemMassRandomizationRange`: float
  Description: Uniform ± range applied around `ItemMass` when randomization is enabled.
  Default: 0.1

- `RandomizeItemCenterOfMass`: bool
  Description: Enable per-episode randomization of local center of mass.
  Default: false

- `ItemCenterOfMassRandomizationRange`: Vector3
  Description: Uniform ± range per axis around `ItemCenterOfMass` when randomization is enabled.
  Default: 0.1,0.1,0.1

- `RandomizeItemFriction`: bool
  Description: Enable per-episode randomization of static and dynamic friction.
  Default: false

- `ItemDynamicFrictionRandomizationRange`: float
  Description: Uniform ± range around `ItemMaterialGridDynamicFriction` values.
  Default: 0.1

- `ItemStaticFrictionRandomizationRange`: float
  Description: Uniform ± range around `ItemMaterialGridStaticFriction` values.
  Default: 0.1

---

## WarehouseEnvironmentParameters (Configuration.WarehouseEnvironment)

- `AMRs`: AMRParameters[]
  Description: AMR parameter segments. Each segment contributes `Count` robots. Targets are created per AMR.

- `Ground`: GroundParameters
  Description: Warehouse floor/ground material and walls.

- `Items`: ItemParameters[]
  Description: Item parameter segments. Each segment contributes `Count` items. Items are shared across all robots and are not tied to targets.

- `EnableObstacleManager`: bool
  Description: Master toggle to spawn/manage static and dynamic obstacles.
  Default: true

- `EnableNavMesh`: bool
  Description: Master toggle for warehouse NavMesh build, carving, and navmesh observations.
  Default: true

- `ObstaclePlacementSeparationMultiplier`: float
  Description: Center-to-center spacing multiplier for placement safety (>= 1 recommended).
  Default: 1.0

- `ObstacleSpawnBoundaryMargin`: float (m)
  Description: Margin from ground edges to avoid when spawning.
  Default: 0.25

- `StaticObstacles`: StaticObstacleParameters[]
  Description: Static non-moving obstacle parameter segments. Each segment contributes `Count` obstacles.

- `DynamicObstacles`: DynamicObstacleParameters[]
  Description: Dynamic obstacle parameter segments. Each segment contributes `Count` obstacles.

### AMRParameters (WarehouseEnvironment.AMRs[n])

- `Count`: int
  Description: Number of AMRs to instantiate with this segment.
  Default: 1

- `AMRModel`: SAFELOG_S2
  Description: Mobile robot model used for this AMR segment.
  Default: SAFELOG_S2

- `EnableTransport`: bool
  Description: Enable transport mode and pin-style item interactions.
  Default: false

- `RobotSegmentationColor`: Color (r,g,b,a)
  Description: Segmentation color for this AMR segment.
  Default: magenta

- `RandomizeRobotAppearance`: bool
  Description: Apply appearance randomization to this AMR segment when enabled globally.
  Default: true

- `EnableLaserScan`: bool
  Description: Enable laser scan observations for this AMR segment.
  Default: false

- `MaxChassisLinearSpeed`: float (m/s)
  Description: Upper bound for chassis linear speed commands.
  Default: 0.8

- `MaxChassisAngularSpeed`: float (rad/s)
  Description: Upper bound for chassis angular speed commands.
  Default: 0.5

- `WheelDriveForceLimit`: float
  Description: Force limit used by wheel joint drives.
  Default: 10

- `WheelDriveDamping`: float
  Description: Damping used by wheel joint drives.
  Default: 10

- `TargetSize`: Vector3 (x,y,z)
  Description: Size of the robot's target marker.
  Default: 0.5,0.001,0.8

- `TargetMaterialColor`: Color (r,g,b,a)
  Description: Color of the robot's target marker.
  Default: red

### LaserScanSensorParameters (WarehouseEnvironment.AMRs[n].LaserScan)

- `RangeMetersMin`: float (m)
  Description: Minimum measurable range; rays start at this offset to avoid self-hits.
  Default: 0.12

- `RangeMetersMax`: float (m)
  Description: Maximum measurable range returned by the scan.
  Default: 100

- `ScanAngleStartDegrees`: float (degrees)
  Description: Start angle of the scan fan in sensor-local yaw frame.
  Default: -179

- `ScanAngleEndDegrees`: float (degrees)
  Description: End angle of the scan fan in sensor-local yaw frame.
  Default: 180

- `NumMeasurementsPerScan`: int
  Description: Number of evenly spaced rays cast per scan.
  Default: 360

- `SensorOffsetX`: float (m)
  Description: Local X offset of the laser sensor frame.
  Default: 0.29

- `SensorOffsetY`: float (m)
  Description: Local Y offset of the laser sensor frame.
  Default: 0.0

### GroundParameters (WarehouseEnvironment.Ground)

- `GroundType`: MONOCHROMATIC | TEXTURED | PREFAB
  Description: Visual ground representation mode in warehouse scenes.
  Default: MONOCHROMATIC

- `GroundSize`: Vector3 (x,y,z)
  Description: Ground dimensions in meters.
  Default: 12.5,0.1,7.5

- `WallHeight`: float (m)
  Description: Height of boundary walls spawned at ground edges.
  Default: 0.5

- `GroundMaterial`: HOMOGENEOUS | HETEROGENEOUS
  Description: Whether ground friction is single-valued or spatially varying.
  Default: HOMOGENEOUS

- `VisualizeGroundMaterial`: bool
  Description: Visualize ground heterogeneity layout when enabled.
  Default: false

- `GroundMaterialColor`: Color (r,g,b,a)
  Description: Base color used for monochromatic ground rendering.
  Default: 0.235,0.510,0.941,1

- `GroundMaterialGridX`: float[]
  Description: Normalized partition lengths along ground X for heterogeneous materials.
  Default: [1.0]

- `GroundMaterialGridY`: float[]
  Description: Normalized partition lengths along ground Y for heterogeneous materials.
  Default: [1.0]

- `GroundMaterialGridZ`: float[]
  Description: Normalized partition lengths along ground Z for heterogeneous materials.
  Default: [1.0]

- `GroundMaterialGridDynamicFriction`: float[]
  Description: Dynamic-friction value per ground material cell (ordered by Z-major traversal).
  Default: [1.0]

- `GroundMaterialGridStaticFriction`: float[]
  Description: Static-friction value per ground material cell (ordered by Z-major traversal).
  Default: [1.0]

### ItemParameters (WarehouseEnvironment.Items[n])

- `Count`: int
  Description: Number of items to instantiate with this segment.
  Default: 1

- `ItemType`: SPHERE | BOX
  Description: Primitive geometry type used for warehouse items in this segment.
  Default: BOX

- `ItemSize`: Vector3 (x,y,z meters)
  Description: Item dimensions in meters.
  Default: 0.1,0.1,0.1

- `ItemMass`: float (kg)
  Description: Rigid-body mass of each spawned item.
  Default: 1.0

- `ItemCenterOfMass`: Vector3 (x,y,z meters)
  Description: Local center-of-mass offset relative to the item transform origin.
  Default: 0,0,0

- `ItemLinearDamping`: float
  Description: Linear damping (drag) applied to item rigid bodies.
  Default: 2.0

- `ItemObservability`: bool
  Description: Include this item segment in streamed state observations.
  Default: true

- `ItemMaterial`: HOMOGENEOUS | HETEROGENEOUS
  Description: Whether item friction is single-valued or grid-varying.
  Default: HOMOGENEOUS

- `VisualizeItemMaterial`: bool
  Description: Visualize item heterogeneity layout when enabled.
  Default: false

- `ItemMaterialColor`: Color (r,g,b,a)
  Description: Base item color used when heterogeneity visualization is disabled.
  Default: green

- `ItemMaterialGridX`: float[]
  Description: Normalized partition lengths along item X for heterogeneous materials.
  Default: [1.0]

- `ItemMaterialGridY`: float[]
  Description: Normalized partition lengths along item Y for heterogeneous materials.
  Default: [1.0]

- `ItemMaterialGridZ`: float[]
  Description: Normalized partition lengths along item Z for heterogeneous materials.
  Default: [1.0]

- `ItemMaterialGridDynamicFriction`: float[] (0–1)
  Description: Dynamic-friction value per item material cell (ordered by Z-major traversal).
  Default: [0.6]

- `ItemMaterialGridStaticFriction`: float[] (0–1)
  Description: Static-friction value per item material cell (ordered by Z-major traversal).
  Default: [0.6]

- `RandomizeItemMass`: bool
  Description: Enable per-episode randomization of `ItemMass` using the configured range.
  Default: false

- `ItemMassRandomizationRange`: float
  Description: Uniform ± range applied around `ItemMass` when randomization is enabled.
  Default: 0.1

- `RandomizeItemCenterOfMass`: bool
  Description: Enable per-episode randomization of local center of mass.
  Default: false

- `ItemCenterOfMassRandomizationRange`: Vector3
  Description: Uniform ± range per axis around `ItemCenterOfMass` when randomization is enabled.
  Default: 0.1,0.1,0.1

- `RandomizeItemFriction`: bool
  Description: Enable per-episode randomization of static and dynamic friction.
  Default: false

- `ItemDynamicFrictionRandomizationRange`: float
  Description: Uniform ± range around `ItemMaterialGridDynamicFriction` values.
  Default: 0.1

- `ItemStaticFrictionRandomizationRange`: float
  Description: Uniform ± range around `ItemMaterialGridStaticFriction` values.
  Default: 0.1

### StaticObstacleParameters (WarehouseEnvironment.StaticObstacles[n])

- `Count`: int
  Description: Number of static obstacles to instantiate with this segment.
  Default: 1

- `ObstacleType`: BOX
  Description: Primitive geometry type used for static obstacles in this segment.
  Default: BOX

- `ObstacleSize`: Vector3 (x,y,z meters)
  Description: Static-obstacle dimensions in meters.
  Default: 1.0,1.0,1.0

- `ObstacleObservability`: bool
  Description: Include this static obstacle segment in streamed state observations.
  Default: false

- `ObstacleMinDistanceFromRobot`: float (m)
  Description: Minimum spawn clearance from robot footprints for this segment.
  Default: 1.0

- `ObstacleMaterialColor`: Color (r,g,b,a)
  Description: Base color applied to static obstacles of this segment.
  Default: green

### DynamicObstacleParameters (WarehouseEnvironment.DynamicObstacles[n])

- `Count`: int
  Description: Number of dynamic obstacles to instantiate with this segment.
  Default: 1

- `ObstacleModel`: SAFELOG_S2
  Description: Dynamic obstacle prefab/model used for this segment.
  Default: SAFELOG_S2

- `ObstacleObservability`: bool
  Description: Include this dynamic obstacle segment in streamed state observations.
  Default: false

- `ObstacleMotion`: None | Random | Circle | Linear
  Description: Motion controller mode used by dynamic obstacles in this segment.
  Default: None

- `ObstacleMinDistanceFromRobot`: float (m)
  Description: Minimum spawn clearance from robot footprints for this segment.
  Default: 1.0

- `ObstacleMaxLinearSpeed`: float (m/s)
  Description: Upper bound for dynamic-obstacle linear speed.
  Default: 0.8

- `ObstacleMaxAngularSpeed`: float (rad/s)
  Description: Upper bound for dynamic-obstacle angular speed.
  Default: 0.25

- `RandomMotionChangePeriodSeconds`: float (s)
  Description: Interval between direction/target updates in random-motion mode.
  Default: 2

- `LinearMotionCycleSeconds`: float (s)
  Description: Duration of one leg before direction reversal in linear-motion mode.
  Default: 8

- `LinearMotionTravelSpeed`: float (m/s)
  Description: Translation speed used in linear-motion mode.
  Default: 0.1

- `CircleMotionTravelSpeed`: float (m/s)
  Description: Tangential speed used in circle-motion mode.
  Default: 0.8

- `CircleMotionCurvature`: float (1/m)
  Description: Signed curvature used in circle-motion mode (`omega = v * curvature`).
  Default: -0.1 (negative = clockwise)

---

## ObservationParameters (Configuration.Observation)

These settings control rendered images and randomization.

- `EnableObservationImage`: bool
  Description: Include an image in the observation or not.
  Default: false

- `SaveObservationImageAsFile`: bool
  Description: Save rendered image to disk instead of sending.
  Default: false

- `ObservationImageEncoding`: PNG | JPG
  Description: Image encoding format.
  Default: PNG

- `ObservationImageQuality`: int (1–100, JPG only)
  Description: JPEG quality setting applied when `ObservationImageEncoding` is JPG.
  Default: 75

- `ObservationImageWidth`: int
  Description: Output image width in pixels.
  Default: 640

- `ObservationImageHeight`: int
  Description: Output image height in pixels.
  Default: 480

- `ObservationImageBackgroundColor`: Color (r,g,b,a)
  Description: Background clear color used for camera rendering.
  Default: white

- `EnableSegmentation`: bool
  Description: Replace realistic materials with segmentation-friendly flat colors.
  Default: false

- `EnableShadows`: bool
  Description: Enable shadow casting from the main light in rendered observations.
  Default: true

- `ShadowType`: None | Hard | Soft (Unity LightShadows)
  Description: Shadow rendering mode for the main light.
  Default: Soft

- `ObservationCameras`: CameraParameters[]
  Description: One or more virtual cameras used for image observations. CLI overrides accept an index to address any camera.

- `RandomizeAppearance`: bool
  Description: Randomize scene appearance and lighting.
  Default: false

- `CameraPositionRandomizationRangeInMeters`: float (uniform ± per axis)
  Description: Per-axis uniform randomization range applied to camera local position.
  Default: 0.05

- `CameraRotationRandomizationRangeInDegrees`: float (uniform ± per axis)
  Description: Per-axis uniform randomization range applied to camera Euler rotation.
  Default: 3

### CameraParameters (Observation.ObservationCameras[n])

- `CameraPosition`: float[3] (x,y,z meters)
  Description: Default camera local position.
  Default: 2,2,2

- `CameraRotation`: float[3] (Euler angles x,y,z in radians)
  Description: Default camera local rotation in Euler angles.
  Default: 0,0,0

- `CameraVerticalFOV`: float (degrees)
  Description: Vertical field of view used by the observation camera.
  Default: 45

---

## Editing the XML

- The simulator reads `configuration.xml` from the working directory. If the file is missing or invalid, defaults are used.
- You can set any of the parameters above by matching the XML element names to the field names shown here.
- Arrays (e.g., Items, StaticObstacles, ObservationCameras) are standard XML arrays of their element types.

Example snippets

Manipulator environment mode

```xml
<Configuration>
  <EnvironmentMode>Manipulator</EnvironmentMode>
  <Simulation>
    <CommunicationType>GRPC</CommunicationType>
    <IPAddress>localhost</IPAddress>
    <PortNumber>9090</PortNumber>
  </Simulation>
  <ManipulatorEnvironment>
    <Manipulators>
      <ManipulatorParameters>
        <Count>1</Count>
        <ManipulatorModel>IIWA14</ManipulatorModel>
        <EnableEndEffector>true</EnableEndEffector>
        <EndEffectorModel>CALIBRATION_PIN</EndEffectorModel>
      </ManipulatorParameters>
    </Manipulators>
    <Floor>
      <FloorType>MONOCHROMATIC</FloorType>
      <FloorSize>
        <x>2.4</x><y>0.01</y><z>2.4</z>
      </FloorSize>
    </Floor>
    <Items>
      <ItemParameters>
        <Count>1</Count>
        <ItemType>BOX</ItemType>
        <ItemSize><x>0.1</x><y>0.1</y><z>0.1</z></ItemSize>
      </ItemParameters>
    </Items>
  </ManipulatorEnvironment>
  <Observation>
    <EnableObservationImage>true</EnableObservationImage>
    <ObservationImageWidth>640</ObservationImageWidth>
    <ObservationImageHeight>480</ObservationImageHeight>
    <ObservationCameras>
      <CameraParameters>
        <CameraPosition>
          <float>0.2</float><float>1.25</float><float>1.3</float>
        </CameraPosition>
        <CameraRotation>
          <float>130</float><float>0</float><float>180</float>
        </CameraRotation>
      </CameraParameters>
    </ObservationCameras>
  </Observation>
</Configuration>
```

Warehouse environment mode with laser scan

```xml
<Configuration>
  <EnvironmentMode>Warehouse</EnvironmentMode>
  <Simulation>
    <TimestepDurationInSeconds>0.02</TimestepDurationInSeconds>
  </Simulation>
  <WarehouseEnvironment>
    <AMRs>
      <AMRParameters>
        <Count>1</Count>
        <AMRModel>SAFELOG_S2</AMRModel>
        <EnableLaserScan>true</EnableLaserScan>
        <LaserScan>
          <RangeMetersMin>0.12</RangeMetersMin>
          <RangeMetersMax>20</RangeMetersMax>
          <NumMeasurementsPerScan>720</NumMeasurementsPerScan>
        </LaserScan>
      </AMRParameters>
    </AMRs>
    <EnableObstacleManager>true</EnableObstacleManager>
    <EnableNavMesh>true</EnableNavMesh>
    <Ground>
      <GroundSize><x>12.5</x><y>0.1</y><z>7.5</z></GroundSize>
      <WallHeight>0.5</WallHeight>
    </Ground>
    <StaticObstacles>
      <StaticObstacleParameters>
        <Count>1</Count>
        <ObstacleType>BOX</ObstacleType>
        <ObstacleSize><x>1</x><y>1</y><z>1</z></ObstacleSize>
        <ObstacleMinDistanceFromRobot>1.0</ObstacleMinDistanceFromRobot>
      </StaticObstacleParameters>
    </StaticObstacles>
    <DynamicObstacles>
      <DynamicObstacleParameters>
        <Count>2</Count>
        <ObstacleMotion>Random</ObstacleMotion>
        <ObstacleMaxLinearSpeed>0.6</ObstacleMaxLinearSpeed>
      </DynamicObstacleParameters>
    </DynamicObstacles>
  </WarehouseEnvironment>
  <Observation>
    <EnableObservationImage>true</EnableObservationImage>
  </Observation>
</Configuration>
```

Tips
- XML element names are case-sensitive to match the fields.
- For colors, you can omit alpha; it defaults to 1.
- If arrays are omitted, defaults are used.

---

## Appendix A: CLI-to-XML Mapping

This appendix maps supported command-line flags to the XML paths in `configuration.xml`. Use either the short or long flag. When arrays are involved, CLI requires `<index> <value>` to target the desired element.

General / Simulation
- -ct, --communication-type → Configuration.Simulation.CommunicationType
- -ipa, --ip-address → Configuration.Simulation.IPAddress
- -pn, --port-number → Configuration.Simulation.PortNumber
- -smd, --shared-memory-directory → Configuration.Simulation.SharedMemoryDirectory
- -smc, --shared-memory-capacity → Configuration.Simulation.SharedMemoryCapacity
- -smssmb, --shared-memory-slot-size-mb → Configuration.Simulation.SharedMemorySlotSizeMB
- -smsk, --shared-memory-same-kernel → Configuration.Simulation.SharedMemorySameKernel
- -ep, --enable-profiling → Configuration.Simulation.EnableProfiling
- -tdis, --timestep-duration-in-seconds → Configuration.Simulation.TimestepDurationInSeconds
- -psiis, --physics-simulation-increment-in-seconds → Configuration.Simulation.PhysicsSimulationIncrementInSeconds
- -ipf, --improved-patch-friction → Configuration.Simulation.ImprovedPatchFriction
- -rs, --random-seed → Configuration.Simulation.RandomSeed
- -e, --evaluation → Configuration.Simulation.Evaluation
- -rep, --randomize-environment-physics → Configuration.Simulation.RandomizeEnvironmentPhysics
- -rt, --randomize-torque → Configuration.Simulation.RandomizeTorque
- -pem, --persist-episode-manifests → Configuration.Simulation.PersistEpisodeManifests
- -rem, --replay-episode-manifest → Configuration.Simulation.ReplayEpisodeManifest
- -rmp, --replay-manifest-path → Configuration.Simulation.ReplayManifestPath
- -em, --environment-mode → Configuration.EnvironmentMode

Manipulator Environment
- -mmc, --manipulator-manipulator-count <index> <value> → Configuration.ManipulatorEnvironment.Manipulators[index].Count
- -mmm, --manipulator-manipulator-model <index> <value> → Configuration.ManipulatorEnvironment.Manipulators[index].ManipulatorModel
- -meee, --manipulator-enable-end-effector <index> <value> → Configuration.ManipulatorEnvironment.Manipulators[index].EnableEndEffector
- -meem, --manipulator-end-effector-model <index> <value> → Configuration.ManipulatorEnvironment.Manipulators[index].EndEffectorModel
- -mtrs, --manipulator-trajectory-string <index> <value> → Configuration.ManipulatorEnvironment.Manipulators[index].TrajectoryString
- -mtmc, --manipulator-target-material-color <index> <value> → Configuration.ManipulatorEnvironment.Manipulators[index].TargetMaterialColor
- -mts, --manipulator-target-size <index> <value> → Configuration.ManipulatorEnvironment.Manipulators[index].TargetSize
- -mft, --manipulator-floor-type → Configuration.ManipulatorEnvironment.Floor.FloorType
- -mfs, --manipulator-floor-size → Configuration.ManipulatorEnvironment.Floor.FloorSize
- -mfm, --manipulator-floor-material → Configuration.ManipulatorEnvironment.Floor.FloorMaterial
- -mvfm, --manipulator-visualize-floor-material → Configuration.ManipulatorEnvironment.Floor.VisualizeFloorMaterial
- -mfmc, --manipulator-floor-material-color → Configuration.ManipulatorEnvironment.Floor.FloorMaterialColor
- -mfmgx, --manipulator-floor-material-grid-x → Configuration.ManipulatorEnvironment.Floor.FloorMaterialGridX
- -mfmgy, --manipulator-floor-material-grid-y → Configuration.ManipulatorEnvironment.Floor.FloorMaterialGridY
- -mfmgz, --manipulator-floor-material-grid-z → Configuration.ManipulatorEnvironment.Floor.FloorMaterialGridZ
- -mfmgdf, --manipulator-floor-material-grid-dynamic-friction → Configuration.ManipulatorEnvironment.Floor.FloorMaterialGridDynamicFriction
- -mfmgsf, --manipulator-floor-material-grid-static-friction → Configuration.ManipulatorEnvironment.Floor.FloorMaterialGridStaticFriction
- -mic, --manipulator-item-count <index> <value> → Configuration.ManipulatorEnvironment.Items[index].Count
- -mit, --manipulator-item-type <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemType
- -mis, --manipulator-item-size <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemSize
- -mim, --manipulator-item-mass <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMass
- -micom, --manipulator-item-center-of-mass <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemCenterOfMass
- -mild, --manipulator-item-linear-damping <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemLinearDamping
- -mio, --manipulator-item-observability <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemObservability
- -mimat, --manipulator-item-material <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMaterial
- -mvim, --manipulator-visualize-item-material <index> <value> → Configuration.ManipulatorEnvironment.Items[index].VisualizeItemMaterial
- -mimc, --manipulator-item-material-color <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMaterialColor
- -mimgx, --manipulator-item-material-grid-x <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMaterialGridX
- -mimgy, --manipulator-item-material-grid-y <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMaterialGridY
- -mimgz, --manipulator-item-material-grid-z <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMaterialGridZ
- -mimgdf, --manipulator-item-material-grid-dynamic-friction <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMaterialGridDynamicFriction
- -mimgsf, --manipulator-item-material-grid-static-friction <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMaterialGridStaticFriction
- -mrim, --manipulator-randomize-item-mass <index> <value> → Configuration.ManipulatorEnvironment.Items[index].RandomizeItemMass
- -mimrr, --manipulator-item-mass-randomization-range <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemMassRandomizationRange
- -mricom, --manipulator-randomize-item-center-of-mass <index> <value> → Configuration.ManipulatorEnvironment.Items[index].RandomizeItemCenterOfMass
- -micomrr, --manipulator-item-center-of-mass-randomization-range <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemCenterOfMassRandomizationRange
- -mrif, --manipulator-randomize-item-friction <index> <value> → Configuration.ManipulatorEnvironment.Items[index].RandomizeItemFriction
- -midfrr, --manipulator-item-dynamic-friction-randomization-range <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemDynamicFrictionRandomizationRange
- -misfrr, --manipulator-item-static-friction-randomization-range <index> <value> → Configuration.ManipulatorEnvironment.Items[index].ItemStaticFrictionRandomizationRange

Warehouse Environment
- -wac, --warehouse-amr-count <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].Count
- -wam, --warehouse-amr-model <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].AMRModel
- -wet, --warehouse-enable-transport <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].EnableTransport
- -wrsc, --warehouse-robot-segmentation-color <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].RobotSegmentationColor
- -wrra, --warehouse-randomize-robot-appearance <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].RandomizeRobotAppearance
- -wels, --warehouse-enable-laser-scan <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].EnableLaserScan
- -wlsrmmin, --warehouse-laser-scan-range-meters-min <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].LaserScan.RangeMetersMin
- -wlsrmmax, --warehouse-laser-scan-range-meters-max <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].LaserScan.RangeMetersMax
- -wlssasd, --warehouse-laser-scan-scan-angle-start-degrees <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].LaserScan.ScanAngleStartDegrees
- -wlssaed, --warehouse-laser-scan-scan-angle-end-degrees <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].LaserScan.ScanAngleEndDegrees
- -wlsnmps, --warehouse-laser-scan-num-measurements-per-scan <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].LaserScan.NumMeasurementsPerScan
- -wlssox, --warehouse-laser-scan-sensor-offset-x <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].LaserScan.SensorOffsetX
- -wlssoy, --warehouse-laser-scan-sensor-offset-y <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].LaserScan.SensorOffsetY
- -wmcls, --warehouse-max-chassis-linear-speed <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].MaxChassisLinearSpeed
- -wmcas, --warehouse-max-chassis-angular-speed <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].MaxChassisAngularSpeed
- -wwdfl, --warehouse-wheel-drive-force-limit <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].WheelDriveForceLimit
- -wwdd, --warehouse-wheel-drive-damping <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].WheelDriveDamping
- -wtmc, --warehouse-target-material-color <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].TargetMaterialColor
- -wts, --warehouse-target-size <index> <value> → Configuration.WarehouseEnvironment.AMRs[index].TargetSize
- -wgt, --warehouse-ground-type → Configuration.WarehouseEnvironment.Ground.GroundType
- -wgs, --warehouse-ground-size → Configuration.WarehouseEnvironment.Ground.GroundSize
- -wwh, --warehouse-wall-height → Configuration.WarehouseEnvironment.Ground.WallHeight
- -wgm, --warehouse-ground-material → Configuration.WarehouseEnvironment.Ground.GroundMaterial
- -wvgm, --warehouse-visualize-ground-material → Configuration.WarehouseEnvironment.Ground.VisualizeGroundMaterial
- -wgmc, --warehouse-ground-material-color → Configuration.WarehouseEnvironment.Ground.GroundMaterialColor
- -wgmgx, --warehouse-ground-material-grid-x → Configuration.WarehouseEnvironment.Ground.GroundMaterialGridX
- -wgmgy, --warehouse-ground-material-grid-y → Configuration.WarehouseEnvironment.Ground.GroundMaterialGridY
- -wgmgz, --warehouse-ground-material-grid-z → Configuration.WarehouseEnvironment.Ground.GroundMaterialGridZ
- -wgmgdf, --warehouse-ground-material-grid-dynamic-friction → Configuration.WarehouseEnvironment.Ground.GroundMaterialGridDynamicFriction
- -wgmgsf, --warehouse-ground-material-grid-static-friction → Configuration.WarehouseEnvironment.Ground.GroundMaterialGridStaticFriction
- -wic, --warehouse-item-count <index> <value> → Configuration.WarehouseEnvironment.Items[index].Count
- -wit, --warehouse-item-type <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemType
- -wis, --warehouse-item-size <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemSize
- -wim, --warehouse-item-mass <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMass
- -wicom, --warehouse-item-center-of-mass <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemCenterOfMass
- -wild, --warehouse-item-linear-damping <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemLinearDamping
- -wio, --warehouse-item-observability <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemObservability
- -wimat, --warehouse-item-material <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMaterial
- -wvim, --warehouse-visualize-item-material <index> <value> → Configuration.WarehouseEnvironment.Items[index].VisualizeItemMaterial
- -wimc, --warehouse-item-material-color <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMaterialColor
- -wimgx, --warehouse-item-material-grid-x <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMaterialGridX
- -wimgy, --warehouse-item-material-grid-y <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMaterialGridY
- -wimgz, --warehouse-item-material-grid-z <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMaterialGridZ
- -wimgdf, --warehouse-item-material-grid-dynamic-friction <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMaterialGridDynamicFriction
- -wimgsf, --warehouse-item-material-grid-static-friction <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMaterialGridStaticFriction
- -wrim, --warehouse-randomize-item-mass <index> <value> → Configuration.WarehouseEnvironment.Items[index].RandomizeItemMass
- -wimrr, --warehouse-item-mass-randomization-range <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemMassRandomizationRange
- -wricom, --warehouse-randomize-item-center-of-mass <index> <value> → Configuration.WarehouseEnvironment.Items[index].RandomizeItemCenterOfMass
- -wicomrr, --warehouse-item-center-of-mass-randomization-range <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemCenterOfMassRandomizationRange
- -wrif, --warehouse-randomize-item-friction <index> <value> → Configuration.WarehouseEnvironment.Items[index].RandomizeItemFriction
- -widfrr, --warehouse-item-dynamic-friction-randomization-range <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemDynamicFrictionRandomizationRange
- -wisfrr, --warehouse-item-static-friction-randomization-range <index> <value> → Configuration.WarehouseEnvironment.Items[index].ItemStaticFrictionRandomizationRange
- -weom, --warehouse-enable-obstacle-manager → Configuration.WarehouseEnvironment.EnableObstacleManager
- -wenm, --warehouse-enable-navmesh → Configuration.WarehouseEnvironment.EnableNavMesh
- -wopsm, --warehouse-obstacle-placement-separation-multiplier → Configuration.WarehouseEnvironment.ObstaclePlacementSeparationMultiplier
- -wosbm, --warehouse-obstacle-spawn-boundary-margin → Configuration.WarehouseEnvironment.ObstacleSpawnBoundaryMargin
- -wsoc, --warehouse-static-obstacle-count <index> <value> → Configuration.WarehouseEnvironment.StaticObstacles[index].Count
- -wsot, --warehouse-static-obstacle-type <index> <value> → Configuration.WarehouseEnvironment.StaticObstacles[index].ObstacleType
- -wsos, --warehouse-static-obstacle-size <index> <value> → Configuration.WarehouseEnvironment.StaticObstacles[index].ObstacleSize
- -wsoo, --warehouse-static-obstacle-observability <index> <value> → Configuration.WarehouseEnvironment.StaticObstacles[index].ObstacleObservability
- -wsomdfr, --warehouse-static-obstacle-min-distance-from-robot <index> <value> → Configuration.WarehouseEnvironment.StaticObstacles[index].ObstacleMinDistanceFromRobot
- -wsomc, --warehouse-static-obstacle-material-color <index> <value> → Configuration.WarehouseEnvironment.StaticObstacles[index].ObstacleMaterialColor

Dynamic Obstacles (Warehouse)
- -wdoc, --warehouse-dynamic-obstacle-count <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].Count
- -wdom, --warehouse-dynamic-obstacle-model <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].ObstacleModel
- -wdoo, --warehouse-dynamic-obstacle-observability <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].ObstacleObservability
- -wdomot, --warehouse-dynamic-obstacle-motion <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].ObstacleMotion
- -wdomdfr, --warehouse-dynamic-obstacle-min-distance-from-robot <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].ObstacleMinDistanceFromRobot
- -wdomls, --warehouse-dynamic-obstacle-max-linear-speed <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].ObstacleMaxLinearSpeed
- -wdomas, --warehouse-dynamic-obstacle-max-angular-speed <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].ObstacleMaxAngularSpeed
- -wdormcps, --warehouse-dynamic-obstacle-random-motion-change-period-seconds <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].RandomMotionChangePeriodSeconds
- -wdolmcs, --warehouse-dynamic-obstacle-linear-motion-cycle-seconds <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].LinearMotionCycleSeconds
- -wdocmts, --warehouse-dynamic-obstacle-circle-motion-travel-speed <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].CircleMotionTravelSpeed
- -wdocmc, --warehouse-dynamic-obstacle-circle-motion-curvature <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].CircleMotionCurvature
- -wdolmts, --warehouse-dynamic-obstacle-linear-motion-travel-speed <index> <value> → Configuration.WarehouseEnvironment.DynamicObstacles[index].LinearMotionTravelSpeed

Observation
- -eoi, --enable-observation-image → Configuration.Observation.EnableObservationImage
- -soiaf, --save-observation-image-as-file → Configuration.Observation.SaveObservationImageAsFile
- -oie, --observation-image-encoding → Configuration.Observation.ObservationImageEncoding
- -oiq, --observation-image-quality → Configuration.Observation.ObservationImageQuality
- -oiw, --observation-image-width → Configuration.Observation.ObservationImageWidth
- -oih, --observation-image-height → Configuration.Observation.ObservationImageHeight
- -oibc, --observation-image-background-color → Configuration.Observation.ObservationImageBackgroundColor
- -es, --enable-segmentation → Configuration.Observation.EnableSegmentation
- -rsc, --robot-segmentation-color → Configuration.Observation.RobotSegmentationColor
- -esd, --enable-shadows → Configuration.Observation.EnableShadows
- -st, --shadow-type → Configuration.Observation.ShadowType
- -ra, --randomize-appearance → Configuration.Observation.RandomizeAppearance
- -cprrim, --camera-position-randomization-range-in-meters → Configuration.Observation.CameraPositionRandomizationRangeInMeters
- -crrrid, --camera-rotation-randomization-range-in-degrees → Configuration.Observation.CameraRotationRandomizationRangeInDegrees
- -ocp, --observation-camera-position <index> <value> → Configuration.Observation.ObservationCameras[index].CameraPosition
- -ocr, --observation-camera-rotation <index> <value> → Configuration.Observation.ObservationCameras[index].CameraRotation
- -ocvfov, --observation-camera-vertical-fov <index> <value> → Configuration.Observation.ObservationCameras[index].CameraVerticalFOV

---

## Appendix B: Additional parameter details and usage notes

This section expands on parameters that didn’t have an explicit description above and clarifies how the simulator uses them.

Simulation
- TimestepDurationInSeconds: Controls observation/control cadence; affects perceived dynamics and log sizes.
- PhysicsSimulationIncrementInSeconds: Substep size; stability improves with smaller values at higher CPU cost.
- ImprovedPatchFriction: Prevents unrealistically high friction forces; recommended on.
- SharedMemoryDirectory: The directory where shared-memory backing files (`shm_request.bin`, `shm_response.bin`) are created. On a Windows host running the simulator with a Docker/WSL2 Python agent, this should be a path on the Windows filesystem that is bind-mounted into the container (e.g., `C:\shm` on Windows ↔ `/mnt/shm` inside Docker). Both sides must see the same files for the memory-mapped ring buffers to work.
- SharedMemoryCapacity: The number of ring-buffer slots. Must be a power of two (e.g., 2, 4, 8). A capacity of 4 means up to 4 messages can be queued before the writer must wait for the reader to consume them. Higher values increase memory usage proportionally.
- SharedMemorySlotSizeMB: Each slot reserves this many mebibytes for one payload. Set this to at least the size of the largest single observation or action JSON/binary payload. A 64 MiB default handles stereo images + joint data comfortably.
- SharedMemorySameKernel: When both processes share the same OS kernel they also share the page cache, so `msync`/`FlushViewOfFile` and file-reopen (`invalidate`) are unnecessary. Setting this to `true` eliminates ~2 ms per step. Leave it `false` when the simulator runs on Windows and the agent runs in WSL2/Docker (9P filesystem), because the page caches are separate and explicit flushing is required for cross-OS coherency.
- EnableProfiling: When enabled, each simulation step collects wall-clock timings for every major phase (request decode, command parsing, physics, observation collection, response serialization) on the C# side. These timings are sent to the Python agent in the first environment's `ProfilingData` field. The Python agent merges them with its own timings (action conversion, request creation, gRPC call, response decode, environment update) and prints a detailed per-step profiling table. On the Python side, set `enable_profiling: true` in the simulation config dict and optionally `profiling_print_every_n: N` to only print every N-th step.

Manipulator: Floor and Items
- FloorMaterialGridX/Y/Z: Define the size of the grid partitions along axes; arrays typically sum to 1 per axis.
- FloorMaterialGridDynamic/StaticFriction: Arrays ordered Z-major then X; used by contact solver during friction queries.
- ItemLinearDamping: Increases energy loss; high values slow items quickly.

Warehouse: Ground and Obstacles
- GroundSize: Sets NavMesh bake bounds and spawn area. Ensure sufficient Y thickness to avoid tunneling.
- WallHeight: Creates low walls at edges; also used to clamp laser rays.
- ObstaclePlacementSeparationMultiplier: Higher values reduce spawn interpenetrations.
- EnableTransport: When transport is disabled, items act as static obstacles with an enabled flag to detect collision with them. When transport is enabled, items act as objects for manipulation/transportation. Since they can be moved, they actively carve NavMesh to update occupancy grid maps. Furthermore, collision with them is considered normal and is not recorded.

Dynamic Obstacles
- Motion parameters (MaxLinearSpeed, MaxAngularSpeed) act as caps; controllers may command lower speeds.
- LinearMotionCycleSeconds and LinearMotionTravelSpeed combine to create back-and-forth motion legs.
- CircleMotionTravelSpeed with CircleMotionCurvature sets angular speed ω = v·k; sign of k defines rotation direction.

Observation and Cameras
- ObservationImageEncoding and Quality: PNG is lossless; JPG trades accuracy for smaller bandwidth/storage.
- CameraPositionRandomizationRangeInMeters and CameraRotationRandomizationRangeInDegrees apply per-camera when RandomizeAppearance is true.
- ObservationCameras can be addressed by index in CLI flags; XML can provide multiple cameras.
