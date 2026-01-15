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
- `EnvironmentMode` (Manipulator | Warehouse)
- `ManipulatorEnvironment` (ManipulatorEnvironmentParameters)
- `WarehouseEnvironment` (WarehouseEnvironmentParameters)
- `Observation` (ObservationParameters or WarehouseObservationParameters depending on EnvironmentMode)

When `EnvironmentMode` is Warehouse, `Observation` should be `WarehouseObservationParameters` (contains extra laser-scan settings). Otherwise, the base `ObservationParameters` applies. Switching modes preserves shared observation settings.

---

## SimulationParameters (Configuration.Simulation)

- CommunicationType: ROS | ZMQ | GRPC | GRPC_NRP
  Description: Communication backend used by the simulator.
  Default: GRPC

- IPAddress: string
  Description: Host/IP to bind/connect (e.g., "localhost").
  Default: localhost

- PortNumber: int
  Description: Service port for the simulator.
  Default: 9090

- TimestepDurationInSeconds: float
  Description: Duration of one simulation step.
  Default: 0.02

- PhysicsSimulationIncrementInSeconds: float
  Description: Size of the physics substep. Simulator will run as many substeps as needed to cover one timestep. Recommended <= 0.03.
  Default: 0.02

- ImprovedPatchFriction: bool
  Description: Enables PhysX improved patch friction for more accurate results.
  Default: true

- RandomSeed: int
  Description: Fixed RNG seed; -1 means random each run.
  Default: -1

- Evaluation: bool
  Description: Evaluation vs training mode selector.
  Default: false

- RandomizeEnvironmentPhysics: bool
  Description: Randomize environment-level physics properties.
  Default: false

- RandomizeTorque: bool
  Description: Randomize joint motor torque limits (where applicable).
  Default: false

- PersistEpisodeManifests: bool
  Description: When true, a JSON manifest capturing all randomized parameters (lighting, palettes, camera deltas, physics, etc.) is written for every episode into a newly created run folder under Application.persistentDataPath/manifests (e.g., manifests/run_1700000000000/episode_1.json).
  Default: true

- ReplayEpisodeManifest: bool
  Description: Enables deterministic replay; suppresses new randomization and pulls values from manifest(s). ReplayManifestPath should point to a directory, where all episode_*.json files are loaded as a sequence and replay cycles with wrap-around.
  Default: false

- ReplayManifestPath: string (directory path)
  Description: Relative (resolved against Application.persistentDataPath/manifests) or absolute path. Supply a run folder containing multiple manifests. Required for meaningful replay; if omitted while ReplayEpisodeManifest is true, no data is loaded.
  Default: ""

---

## ManipulatorEnvironmentParameters (Configuration.ManipulatorEnvironment)

- ManipulatorModel: IIWA14 | SO100
  Description: Robot arm model.
  Default: IIWA14

- EnableEndEffector: bool
  Description: Attach default end-effector/gripper. If true, tip becomes the observed EE pose; otherwise last link is used.
  Default: true

- EndEffectorModel: ROBOTIQ_3F | ROBOTIQ_2F85 | CALIBRATION_PIN
  Description: End effector selection.
  Default: CALIBRATION_PIN

- Floor: FloorParameters
  Description: Stand/platform floor the manipulator is mounted on (material and size).

- Items: ItemParameters[]
  Description: Objects to spawn for manipulation. Only the first item is commonly used by the CLI overrides, but the XML can contain more.

- TrajectoryString: string
  Description: A saved trajectory sequence for the manipulator to follow (implementation-specific format).
  Default: ""

### FloorParameters (ManipulatorEnvironment.Floor)

- FloorType: CHECKERBOARD | WOOD | MONOCHROMATIC
  Default: MONOCHROMATIC

- FloorSize: Vector3 (x,y,z)
  Default: 2.4,0.01,2.4

- FloorMaterial: HOMOGENEOUS | HETEROGENEOUS
  Default: HOMOGENEOUS

- VisualizeFloorMaterial: bool
  Default: false

- FloorMaterialColor: Color (r,g,b,a)
  Default: 0.235,0.510,0.941,1

- FloorMaterialGridX: float[]
  Default: [1.0]

- FloorMaterialGridY: float[]
  Default: [1.0]

- FloorMaterialGridZ: float[]
  Default: [1.0]

- FloorMaterialGridDynamicFriction: float[]
  Default: [1.0]

- FloorMaterialGridStaticFriction: float[]
  Default: [1.0]

### ItemParameters (ManipulatorEnvironment.Items[n])

- ItemType: SPHERE | BOX
  Default: BOX

- ItemSize: Vector3 (x,y,z meters)
  Default: 0.1,0.1,0.1

- ItemMass: float (kg)
  Default: 1.0

- ItemCenterOfMass: Vector3 (x,y,z meters)
  Default: 0,0,0

- ItemLinearDamping: float
  Default: 2.0

- ItemObservability: bool
  Default: true

- ItemMaterial: HOMOGENEOUS | HETEROGENEOUS
  Default: HOMOGENEOUS

- VisualizeItemMaterial: bool
  Default: false

- ItemMaterialColor: Color (r,g,b,a)
  Default: green

- TargetMaterialColor: Color (r,g,b,a)
  Default: red

- ItemMaterialGridX: float[]
  Default: [1.0]

- ItemMaterialGridY: float[]
  Default: [1.0]

- ItemMaterialGridZ: float[]
  Default: [1.0]

- ItemMaterialGridDynamicFriction: float[] (0–1)
  Default: [0.6]

- ItemMaterialGridStaticFriction: float[] (0–1)
  Default: [0.6]

- RandomizeItemMass: bool
  Default: false

- ItemMassRandomizationRange: float
  Default: 0.1

- RandomizeItemCenterOfMass: bool
  Default: false

- ItemCenterOfMassRandomizationRange: Vector3
  Default: 0.1,0.1,0.1

- RandomizeItemFriction: bool
  Default: false

- ItemDynamicFrictionRandomizationRange: float
  Default: 0.1

- ItemStaticFrictionRandomizationRange: float
  Default: 0.1

---

## WarehouseEnvironmentParameters (Configuration.WarehouseEnvironment)

- AMRModel: SAFELOG_S2
  Description: Mobile robot model.
  Default: SAFELOG_S2

- EnableTransport: bool
  Description: Enable payload transport/pin actions.
  Default: false

- MaxChassisLinearSpeed: float (m/s)
  Default: 0.8

- MaxChassisAngularSpeed: float (rad/s)
  Default: 0.5

- WheelDriveForceLimit: float
  Default: 10

- WheelDriveDamping: float
  Default: 10

- Ground: GroundParameters
  Description: Warehouse floor/ground material and walls.

- Items: ItemParameters[]
  Description: Transport items used for item/target creation. Only the first item is commonly used by the CLI overrides, but the XML can contain more.

- EnableObstacleManager: bool
  Description: Master toggle to spawn/manage static and dynamic obstacles.
  Default: true

- ObstaclePlacementSeparationMultiplier: float
  Description: Center-to-center spacing multiplier for placement safety (>= 2 recommended).
  Default: 2.8

- ObstacleSpawnBoundaryMargin: float (m)
  Description: Margin from ground edges to avoid when spawning.
  Default: 0.25

- StaticObstacleCount: int
  Description: Samples n non-moving obstacles from StaticObstacles pool array.
  Default: 0

- StaticObstacles: StaticObstacleParameters[]
  Description: Templates for static obstacles.

- DynamicObstacles: DynamicObstaclesParameters
  Description: Dynamic obstacle spawning and motion.

### GroundParameters (WarehouseEnvironment.Ground)

- GroundType: MONOCHROMATIC | TEXTURED | PREFAB
  Default: MONOCHROMATIC

- GroundSize: Vector3 (x,y,z)
  Default: 7.5,0.1,12.5

- WallHeight: float (m)
  Default: 0.5

- GroundMaterial: HOMOGENEOUS | HETEROGENEOUS
  Default: HOMOGENEOUS

- VisualizeGroundMaterial: bool
  Default: false

- GroundMaterialColor: Color (r,g,b,a)
  Default: 0.235,0.510,0.941,1

- GroundMaterialGridX: float[]
  Default: [1.0]

- GroundMaterialGridY: float[]
  Default: [1.0]

- GroundMaterialGridZ: float[]
  Default: [1.0]

- GroundMaterialGridDynamicFriction: float[]
  Default: [1.0]

- GroundMaterialGridStaticFriction: float[]
  Default: [1.0]

### ItemParameters (WarehouseEnvironment.Items[n])

- ItemType: SPHERE | BOX
  Default: BOX

- ItemSize: Vector3 (x,y,z meters)
  Default: 0.1,0.1,0.1

- ItemMass: float (kg)
  Default: 1.0

- ItemCenterOfMass: Vector3 (x,y,z meters)
  Default: 0,0,0

- ItemLinearDamping: float
  Default: 2.0

- ItemObservability: bool
  Default: true

- ItemMaterial: HOMOGENEOUS | HETEROGENEOUS
  Default: HOMOGENEOUS

- VisualizeItemMaterial: bool
  Default: false

- ItemMaterialColor: Color (r,g,b,a)
  Default: green

- TargetMaterialColor: Color (r,g,b,a)
  Default: red

- ItemMaterialGridX: float[]
  Default: [1.0]

- ItemMaterialGridY: float[]
  Default: [1.0]

- ItemMaterialGridZ: float[]
  Default: [1.0]

- ItemMaterialGridDynamicFriction: float[] (0–1)
  Default: [0.6]

- ItemMaterialGridStaticFriction: float[] (0–1)
  Default: [0.6]

- RandomizeItemMass: bool
  Default: false

- ItemMassRandomizationRange: float
  Default: 0.1

- RandomizeItemCenterOfMass: bool
  Default: false

- ItemCenterOfMassRandomizationRange: Vector3
  Default: 0.1,0.1,0.1

- RandomizeItemFriction: bool
  Default: false

- ItemDynamicFrictionRandomizationRange: float
  Default: 0.1

- ItemStaticFrictionRandomizationRange: float
  Default: 0.1

### StaticObstacleParameters (WarehouseEnvironment.StaticObstacles[n])

- ObstacleType: BOX
  Default: BOX

- ObstacleSize: Vector3 (x,y,z meters)
  Default: 1.0,1.0,1.0

- ObstacleObservability: bool
  Default: false

- ObstacleMaterialColor: Color (r,g,b,a)
  Default: green

### DynamicObstaclesParameters (WarehouseEnvironment.DynamicObstacles)

- DynamicObstacleCount: int
  Default: 0

- DynamicObstacleObservability: bool
  Default: false

- DynamicObstacleMotion: None | Random | Circle | Linear
  Default: None

- DynamicObstacleMinDistanceFromRobot: float (m)
  Default: 1.6

- MaxLinearSpeed: float (m/s)
  Default: 0.8

- MaxAngularSpeed: float (rad/s)
  Default: 0.25

- RandomMotionChangePeriodSeconds: float (s)
  Default: 2

- LinearMotionCycleSeconds: float (s)
  Default: 8

- LinearMotionTravelSpeed: float (m/s)
  Default: 0.1

- CircleMotionTravelSpeed: float (m/s)
  Default: 0.8

- CircleMotionCurvature: float (1/m)
  Default: -0.1 (negative = clockwise)

---

## ObservationParameters (Configuration.Observation)

These settings control rendered images and randomization. Warehouse adds laser-scan options via `WarehouseObservationParameters` (see below).

- EnableObservationImage: bool
  Description: Include an image in the observation or not.
  Default: false

- SaveObservationImageAsFile: bool
  Description: Save rendered image to disk instead of sending.
  Default: false

- ObservationImageEncoding: PNG | JPG
  Description: Image encoding format.
  Default: PNG

- ObservationImageQuality: int (1–100, JPG only)
  Default: 75

- ObservationImageWidth: int
  Default: 640

- ObservationImageHeight: int
  Default: 480

- ObservationImageBackgroundColor: Color (r,g,b,a)
  Default: white

- EnableSegmentation: bool
  Default: false

- RobotSegmentationColor: Color (r,g,b,a)
  Default: magenta

- EnableShadows: bool
  Default: true

- ShadowType: None | Hard | Soft (Unity LightShadows)
  Default: Soft

- ObservationCameras: CameraParameters[]
  Description: One or more virtual cameras used for image observations. The first camera is commonly targeted by CLI overrides; XML can provide multiple.

- RandomizeAppearance: bool
  Description: Randomize scene appearance and lighting.
  Default: false

- CameraPositionRandomizationRangeInMeters: float (uniform ± per axis)
  Default: 0.05

- CameraRotationRandomizationRangeInDegrees: float (uniform ± per axis)
  Default: 3

### CameraParameters (Observation.ObservationCameras[n])

- CameraPosition: float[3] (x,y,z meters)
  Default: 2,2,2

- CameraRotation: float[3] (Euler degrees x,y,z)
  Default: 0,0,0

- CameraVerticalFOV: float (degrees)
  Default: 45

---

## WarehouseObservationParameters (Observation when EnvironmentMode = Warehouse)

Inherits all fields from ObservationParameters and adds laser scan sensor settings.

- EnableLaserScan: bool
  Default: false

- LaserScan: LaserScanSensorParameters

### LaserScanSensorParameters (Observation.LaserScan)

- RangeMetersMin: float (m)
  Default: 0.12

- RangeMetersMax: float (m)
  Default: 100

- ScanAngleStartDegrees: float (degrees)
  Default: 180

- ScanAngleEndDegrees: float (degrees)
  Default: -179

- NumMeasurementsPerScan: int
  Default: 360

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
    <ManipulatorModel>IIWA14</ManipulatorModel>
    <EnableEndEffector>true</EnableEndEffector>
    <EndEffectorModel>CALIBRATION_PIN</EndEffectorModel>
    <Floor>
      <FloorType>MONOCHROMATIC</FloorType>
      <FloorSize>
        <x>2.4</x><y>0.01</y><z>2.4</z>
      </FloorSize>
    </Floor>
    <Items>
      <ItemParameters>
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
    <AMRModel>SAFELOG_S2</AMRModel>
    <EnableObstacleManager>true</EnableObstacleManager>
    <Ground>
      <GroundSize><x>7.5</x><y>0.1</y><z>12.5</z></GroundSize>
      <WallHeight>0.5</WallHeight>
    </Ground>
    <StaticObstacleCount>1</StaticObstacleCount>
    <StaticObstacles>
      <StaticObstacleParameters>
        <ObstacleType>BOX</ObstacleType>
        <ObstacleSize><x>1</x><y>1</y><z>1</z></ObstacleSize>
      </StaticObstacleParameters>
    </StaticObstacles>
    <DynamicObstacles>
      <DynamicObstacleCount>2</DynamicObstacleCount>
      <DynamicObstacleMotion>Random</DynamicObstacleMotion>
      <MaxLinearSpeed>0.6</MaxLinearSpeed>
    </DynamicObstacles>
  </WarehouseEnvironment>
  <Observation xsi:type="WarehouseObservationParameters" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <EnableLaserScan>true</EnableLaserScan>
    <LaserScan>
      <RangeMetersMin>0.12</RangeMetersMin>
      <RangeMetersMax>20</RangeMetersMax>
      <NumMeasurementsPerScan>720</NumMeasurementsPerScan>
    </LaserScan>
  </Observation>
</Configuration>
```

Tips
- XML element names are case-sensitive to match the fields.
- For colors, you can omit alpha; it defaults to 1.
- If arrays are omitted, defaults are used.

---

## Appendix A: CLI-to-XML Mapping

This appendix maps supported command-line flags to the XML paths in `configuration.xml`. Use either the short or long flag. When arrays are involved, CLI typically targets only the first element (index 0), while XML supports full arrays.

General / Simulation
- -ct, --communication-type → Configuration.Simulation.CommunicationType
- -ipa, --ip-address → Configuration.Simulation.IPAddress
- -pn, --port-number → Configuration.Simulation.PortNumber
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
- -mmm, --manipulator-manipulator-model → Configuration.ManipulatorEnvironment.ManipulatorModel
- -meee, --manipulator-enable-end-effector → Configuration.ManipulatorEnvironment.EnableEndEffector
- -meem, --manipulator-end-effector-model → Configuration.ManipulatorEnvironment.EndEffectorModel
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
- -mit, --manipulator-item-type → Configuration.ManipulatorEnvironment.Items[0].ItemType
- -mis, --manipulator-item-size → Configuration.ManipulatorEnvironment.Items[0].ItemSize
- -mim, --manipulator-item-mass → Configuration.ManipulatorEnvironment.Items[0].ItemMass
- -micom, --manipulator-item-center-of-mass → Configuration.ManipulatorEnvironment.Items[0].ItemCenterOfMass
- -mild, --manipulator-item-linear-damping → Configuration.ManipulatorEnvironment.Items[0].ItemLinearDamping
- -mio, --manipulator-item-observability → Configuration.ManipulatorEnvironment.Items[0].ItemObservability
- -mimat, --manipulator-item-material → Configuration.ManipulatorEnvironment.Items[0].ItemMaterial
- -mvim, --manipulator-visualize-item-material → Configuration.ManipulatorEnvironment.Items[0].VisualizeItemMaterial
- -mimc, --manipulator-item-material-color → Configuration.ManipulatorEnvironment.Items[0].ItemMaterialColor
- -mtmc, --manipulator-target-material-color → Configuration.ManipulatorEnvironment.Items[0].TargetMaterialColor
- -mimgx, --manipulator-item-material-grid-x → Configuration.ManipulatorEnvironment.Items[0].ItemMaterialGridX
- -mimgy, --manipulator-item-material-grid-y → Configuration.ManipulatorEnvironment.Items[0].ItemMaterialGridY
- -mimgz, --manipulator-item-material-grid-z → Configuration.ManipulatorEnvironment.Items[0].ItemMaterialGridZ
- -mimgdf, --manipulator-item-material-grid-dynamic-friction → Configuration.ManipulatorEnvironment.Items[0].ItemMaterialGridDynamicFriction
- -mimgsf, --manipulator-item-material-grid-static-friction → Configuration.ManipulatorEnvironment.Items[0].ItemMaterialGridStaticFriction
- -mrim, --manipulator-randomize-item-mass → Configuration.ManipulatorEnvironment.Items[0].RandomizeItemMass
- -mimrr, --manipulator-item-mass-randomization-range → Configuration.ManipulatorEnvironment.Items[0].ItemMassRandomizationRange
- -mricom, --manipulator-randomize-item-center-of-mass → Configuration.ManipulatorEnvironment.Items[0].RandomizeItemCenterOfMass
- -micomrr, --manipulator-item-center-of-mass-randomization-range → Configuration.ManipulatorEnvironment.Items[0].ItemCenterOfMassRandomizationRange
- -mrif, --manipulator-randomize-item-friction → Configuration.ManipulatorEnvironment.Items[0].RandomizeItemFriction
- -midfrr, --manipulator-item-dynamic-friction-randomization-range → Configuration.ManipulatorEnvironment.Items[0].ItemDynamicFrictionRandomizationRange
- -misfrr, --manipulator-item-static-friction-randomization-range → Configuration.ManipulatorEnvironment.Items[0].ItemStaticFrictionRandomizationRange
- -mts, --manipulator-trajectory-string → Configuration.ManipulatorEnvironment.TrajectoryString

Warehouse Environment
- -wam, --warehouse-amr-model → Configuration.WarehouseEnvironment.AMRModel
- -wet, --warehouse-enable-transport → Configuration.WarehouseEnvironment.EnableTransport
- -wmcls, --warehouse-max-chassis-linear-speed → Configuration.WarehouseEnvironment.MaxChassisLinearSpeed
- -wmcas, --warehouse-max-chassis-angular-speed → Configuration.WarehouseEnvironment.MaxChassisAngularSpeed
- -wwdfl, --warehouse-wheel-drive-force-limit → Configuration.WarehouseEnvironment.WheelDriveForceLimit
- -wwdd, --warehouse-wheel-drive-damping → Configuration.WarehouseEnvironment.WheelDriveDamping
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
- -weom, --warehouse-enable-obstacle-manager → Configuration.WarehouseEnvironment.EnableObstacleManager
- -wopsm, --warehouse-obstacle-placement-separation-multiplier → Configuration.WarehouseEnvironment.ObstaclePlacementSeparationMultiplier
- -wosbm, --warehouse-obstacle-spawn-boundary-margin → Configuration.WarehouseEnvironment.ObstacleSpawnBoundaryMargin
- -wsoc, --warehouse-static-obstacle-count → Configuration.WarehouseEnvironment.StaticObstacleCount
- -wsot, --warehouse-static-obstacle-type → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleType
- -wsos, --warehouse-static-obstacle-size → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleSize
- -wsom, --warehouse-static-obstacle-mass → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMass
- -wsocom, --warehouse-static-obstacle-center-of-mass → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleCenterOfMass
- -wsold, --warehouse-static-obstacle-linear-damping → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleLinearDamping
- -wsoo, --warehouse-static-obstacle-observability → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleObservability
- -wsomat, --warehouse-static-obstacle-material → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMaterial
- -wvsom, --warehouse-visualize-static-obstacle-material → Configuration.WarehouseEnvironment.StaticObstacles[0].VisualizeObstacleMaterial
- -wsomc, --warehouse-static-obstacle-material-color → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMaterialColor
- -wsomgx, --warehouse-static-obstacle-material-grid-x → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMaterialGridX
- -wsomgy, --warehouse-static-obstacle-material-grid-y → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMaterialGridY
- -wsomgz, --warehouse-static-obstacle-material-grid-z → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMaterialGridZ
- -wsomgdf, --warehouse-static-obstacle-material-grid-dynamic-friction → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMaterialGridDynamicFriction
- -wsomgsf, --warehouse-static-obstacle-material-grid-static-friction → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMaterialGridStaticFriction
- -wrsom, --warehouse-randomize-static-obstacle-mass → Configuration.WarehouseEnvironment.StaticObstacles[0].RandomizeObstacleMass
- -wsomrr, --warehouse-static-obstacle-mass-randomization-range → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleMassRandomizationRange
- -wrsocom, --warehouse-randomize-static-obstacle-center-of-mass → Configuration.WarehouseEnvironment.StaticObstacles[0].RandomizeObstacleCenterOfMass
- -wsocomrr, --warehouse-static-obstacle-center-of-mass-randomization-range → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleCenterOfMassRandomizationRange
- -wrsof, --warehouse-randomize-static-obstacle-friction → Configuration.WarehouseEnvironment.StaticObstacles[0].RandomizeObstacleFriction
- -wsodfrr, --warehouse-static-obstacle-dynamic-friction-randomization-range → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleDynamicFrictionRandomizationRange
- -wsosfrr, --warehouse-static-obstacle-static-friction-randomization-range → Configuration.WarehouseEnvironment.StaticObstacles[0].ObstacleStaticFrictionRandomizationRange

Dynamic Obstacles (Warehouse)
- -wdoc, --warehouse-dynamic-obstacle-count → Configuration.WarehouseEnvironment.DynamicObstacles.DynamicObstacleCount
- -wdoo, --warehouse-dynamic-obstacle-observability → Configuration.WarehouseEnvironment.DynamicObstacles.DynamicObstacleObservability
- -wdom, --warehouse-dynamic-obstacle-motion → Configuration.WarehouseEnvironment.DynamicObstacles.DynamicObstacleMotion
- -wdomdfr, --warehouse-dynamic-obstacle-min-distance-from-robot → Configuration.WarehouseEnvironment.DynamicObstacles.DynamicObstacleMinDistanceFromRobot
- -wdomls, --warehouse-dynamic-obstacle-max-linear-speed → Configuration.WarehouseEnvironment.DynamicObstacles.MaxLinearSpeed
- -wdomas, --warehouse-dynamic-obstacle-max-angular-speed → Configuration.WarehouseEnvironment.DynamicObstacles.MaxAngularSpeed
- -wdormcps, --warehouse-dynamic-obstacle-random-motion-change-period-seconds → Configuration.WarehouseEnvironment.DynamicObstacles.RandomMotionChangePeriodSeconds
- -wdolmcs, --warehouse-dynamic-obstacle-linear-motion-cycle-seconds → Configuration.WarehouseEnvironment.DynamicObstacles.LinearMotionCycleSeconds
- -wdocmts, --warehouse-dynamic-obstacle-circle-motion-travel-speed → Configuration.WarehouseEnvironment.DynamicObstacles.CircleMotionTravelSpeed
- -wdocmc, --warehouse-dynamic-obstacle-circle-motion-curvature → Configuration.WarehouseEnvironment.DynamicObstacles.CircleMotionCurvature
- -wdolmts, --warehouse-dynamic-obstacle-linear-motion-travel-speed → Configuration.WarehouseEnvironment.DynamicObstacles.LinearMotionTravelSpeed

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
- -ocp, --observation-camera-position → Configuration.Observation.ObservationCameras[0].CameraPosition
- -ocr, --observation-camera-rotation → Configuration.Observation.ObservationCameras[0].CameraRotation
- -ocvfov, --observation-camera-vertical-fov → Configuration.Observation.ObservationCameras[0].CameraVerticalFOV

Warehouse Observation (Laser)
- -wels, --warehouse-enable-laser-scan → Configuration.Observation.EnableLaserScan
- -wlsrmmin, --warehouse-laser-scan-range-meters-min → Configuration.Observation.LaserScan.RangeMetersMin
- -wlsrmmax, --warehouse-laser-scan-range-meters-max → Configuration.Observation.LaserScan.RangeMetersMax
- -wlssasd, --warehouse-laser-scan-scan-angle-start-degrees → Configuration.Observation.LaserScan.ScanAngleStartDegrees
- -wlssaed, --warehouse-laser-scan-scan-angle-end-degrees → Configuration.Observation.LaserScan.ScanAngleEndDegrees
- -wlsnmps, --warehouse-laser-scan-num-measurements-per-scan → Configuration.Observation.LaserScan.NumMeasurementsPerScan

---

## Appendix B: Additional parameter details and usage notes

This section expands on parameters that didn’t have an explicit description above and clarifies how the simulator uses them.

Simulation
- TimestepDurationInSeconds: Controls observation/control cadence; affects perceived dynamics and log sizes.
- PhysicsSimulationIncrementInSeconds: Substep size; stability improves with smaller values at higher CPU cost.
- ImprovedPatchFriction: Prevents unrealistically high friction forces; recommended on.

Manipulator: Floor and Items
- FloorMaterialGridX/Y/Z: Define the size of the grid partitions along axes; arrays typically sum to 1 per axis.
- FloorMaterialGridDynamic/StaticFriction: Arrays ordered Z-major then X; used by contact solver during friction queries.
- ItemLinearDamping: Increases energy loss; high values slow objects quickly.

Warehouse: Ground and Obstacles
- GroundSize: Sets NavMesh bake bounds and spawn area. Ensure sufficient Y thickness to avoid tunneling.
- WallHeight: Creates low walls at edges; also used to clamp laser rays.
- ObstaclePlacementSeparationMultiplier: Higher values reduce spawn retries and interpenetrations.
- EnableTransport: When transport is disabled, items act as static obstacles with an enabled flag to detect collision with them. When transport is enabled, items act as objects for manipulation/transportation. Since they can be moved, they actively carve NavMesh to update occupancy grid maps. Furthermore, collision with them is considered normal and is not recorded.

Dynamic Obstacles
- Motion parameters (MaxLinearSpeed, MaxAngularSpeed) act as caps; controllers may command lower speeds.
- LinearMotionCycleSeconds and LinearMotionTravelSpeed combine to create back-and-forth motion legs.
- CircleMotionTravelSpeed with CircleMotionCurvature sets angular speed ω = v·k; sign of k defines rotation direction.

Observation and Cameras
- ObservationImageEncoding and Quality: PNG is lossless; JPG trades accuracy for smaller bandwidth/storage.
- CameraPositionRandomizationRangeInMeters and CameraRotationRandomizationRangeInDegrees apply per-camera when RandomizeAppearance is true.
- ObservationCameras[0] is the default camera targeted by CLI flags; XML can provide multiple cameras.
