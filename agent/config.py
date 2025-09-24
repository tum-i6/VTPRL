"""
Main configuration for:
- Python agents (DART and Gym environments)
- Unity simulator XML (section-aligned configuration)

Best practices:
- Each Unity XML section has a dedicated dictionary and generator method named
    after the XML parent (e.g., <Simulation> -> get_simulation_dict()).
- Pools (e.g., cameras, items, static obstacles) are arrays of dictionaries, one per element.
- Agent-related settings are separated into agent, gym environment, and DART groups.

Run this script to update the Unity configuration.xml, or edit fields manually.
"""
import os
import numpy as np
from utils.simulator_configuration import update_simulator_configuration

class Config:
    def __init__(self):
        """
        Initialize all configuration groups.

        Agent-related groups are not used by the XML writer but are kept for training/runtime configuration.
        Unity XML section groups map directly to XML parents and tags.
        """
        # Agent-related groups
        self.agent_dict = self.get_agent_dict()
        self.gym_environment_dict = self.get_gym_environment_dict()

        # Unity XML groups (section-aligned)
        # XML-related sections consumed by simulator_configuration.get_param()
        # Note: get_param searches these specific dict attributes, so we keep
        # the same attribute names while organizing content by XML sections.
        self.root_dict = self.get_root_dict()                                           # Root-level settings (e.g., EnvironmentMode)
        self.simulation_dict = self.get_simulation_dict()                               # <Simulation>
        self.manipulator_environment_dict = self.get_manipulator_environment_dict()     # <ManipulatorEnvironment>
        self.warehouse_environment_dict = self.get_warehouse_environment_dict()         # <WarehouseEnvironment>
        self.observation_dict = self.get_observation_dict()                             # <Observation>

    # ---------------------------
    # Agent-related groups
    # ---------------------------
    @staticmethod
    def get_agent_dict():
        """
        Agent-level configuration (algorithm, runtime, and rewards).

        Includes algorithm selection and high-level runtime settings. The
        reward configuration is provided via get_reward_dict().
        """
        return {
            'custom_hps': True,                                                         # Whether to use custom hyperparameters or stable-baseline specific parameters
            'model': 'PPO',                                                             # Agent model name

            # options: 'train', 'evaluate', 'evaluate_model_based' -> not for 'iiwa_sample_joint_vel_env' environment (see below). Refer to the main.py for more details
            # 'evaluate':             load an RL saved checkpoint
            # 'evaluate_model_based': use a model-based-controller to solve the task for debugging
            'simulation_mode': 'evaluate_model_based',                                  # Run mode for agents: 'train' | 'evaluate' | 'evaluate_model_based'
            'total_timesteps': 10000,                                                   # Training timesteps for RL algorithms

            # Logging
            'log_dir': './agent/logs/',                                                 # Folder to save checkpoints and tensorboard logs
            'tb_log_name': 'ppo_log_tb',                                                # Tensorboard log subfolder name
        }

    # ---------------------------
    # Agent sub-groups (Gym/DART)
    # ---------------------------
    @staticmethod
    def get_gym_environment_dict():
        """
        Gym environment settings for Python agents. This aggregates common
        parameters and specialized sub-configs for manipulator and warehouse.
        """
        return {
            # Common
            'max_time_step': 4000,                                                      # Episode length in simulation steps

            # available manipulator environments (*) 'env_key' should include 'iiwa' or 'so100' -- relevant to set Unity's manipulator_model
            # 'env_key': 'iiwa_sample_dart_unity_env',                                    # For control in task space with dart
            # 'env_key': 'iiwa_joint_vel',                                                # Without dart (only joint velocity control) -> enable gripper (see below). Sample env for testing images state representation
            # 'env_key': 'so100_sample_dart_unity_env',                                   # For control in task space with dart for the SO-100 arm

            # available warehouse environments (*) 'env_key' should include 'warehouse'
            'env_key': 'warehouse_unity_env',                                           # For simple warehouse environment

            # environment-specific parameter, e.g, number of links, only relevant for 'iiwa_joint_vel' env
            # currently, num_joints should be always 7/iiwa or 5/so100 when the DART based environment is used
            # use 3 joints for warehouse environment (the two wheels and the lift pin)
            'num_joints': 3,                                                            # Number of controllable joints/DoFs used by the gym environment

            # vectorized environments to run in parallel, 8, 16, ...
            # you may need to restart unity
            'num_envs': 1,                                                              # Number of parallel vectorized environments to launch

            # Sub-environments
            'manipulator_gym_environment': Config.get_manipulator_gym_environment_dict(),   # Manipulator-specific gym settings (sub-dict)
            'warehouse_gym_environment': Config.get_warehouse_gym_environment_dict(),       # Warehouse-specific gym settings (sub-dict)
        }

    @staticmethod
    def get_manipulator_gym_environment_dict():
        """Manipulator-specific gym environment settings, including DART sub-config."""
        return {
            # The state of the RL agent in case of numeric values
            'state': 'a',                                                               # Agent observation state type: 'a' (angles) | 'av' (angles+velocities)

            # These functionallities are not supported for the standalone env 'iiwa_sample_joint_vel_env'
            # - see reset() and __init__ to adapt if needed
            # When the manipulator is spawned to a different position than the vertical position,
            # - the parallel envs should terminate at the same timestep due to Unity synchronization behaviour
            # when the observation space is an image, can not sample random_initial_joint_positions, set to False
            'random_initial_joint_positions': False,                                    # If set to True, it overrides the values set in 'initial_positions'.
            'initial_positions': None,                                                  # Example options: [0, 0, 0, 0, 0, 0, 0] same as None, [0, 0, 0, -np.pi/2, 0, np.pi/2, 0]
                                                                                        # For SO-100 initilize the robot from non-zero position, otherwise you'd receive collision at start.
                                                                                        # Example SO-100 initial_positions: [0, np.pi/2, -np.pi/2, 0, 0]

            # DART-specific settings
            'dart': Config.get_dart_dict(),                                             # DART physics/control sub-configuration used by manipulator envs

            # Reward configuration for manipulator training/evaluation
            'reward': {
                'reward_terminal': 0.0                                                  # Given at the end of the episode if it was successful - see simulator_vec_env.py, step() method during reset
            }
        }

    @staticmethod
    def get_warehouse_gym_environment_dict():
        """Warehouse-specific gym environment settings (agent-side only)."""
        return {
            'max_v': 1.0,                                                               # Max forward linear velocity command [m/s]
            'max_omega': 1.0,                                                           # Max yaw rate command [rad/s]
            'use_laser_scan': False,                                                    # Whether to include laser scan in observations
            'laser_count': 0,                                                           # e.g., 180 or 360
            'normalize_obs': False,                                                     # Normalize observations to roughly [-1, 1]
            'pos_norm': 10.0,                                                           # meters; scales x,y,dx,dy
            'yaw_norm': float(np.pi),                                                   # radians; scales yaw, dyaw
            'success_distance_threshold': 0.10,                                         # meters
            'success_yaw_threshold': np.deg2rad(10),                                    # Success threshold for yaw error [rad]
            'step_penalty': 0.01,                                                       # Per-step penalty applied to encourage faster completion
            'success_reward': 1.0,                                                      # Reward granted when reaching success conditions
            'collision_penalty': 1.0,                                                   # Penalty applied when colliding with environment/objects
            'distance_weight': 0.0,                                                     # Weight of distance-based shaping term in reward
            'yaw_weight': 0.0,                                                          # Weight of yaw-based shaping term in reward
            'num_joints': 3,                                                            # Number of controllable joints/DoFs for the AMR model
            'initial_positions': [0.0, 0.0, 0.0],                                       # Initial joint positions of the AMR [rad or m depending on joint]
            'initial_velocities': [0.0, 0.0, 0.0],                                      # Initial joint velocities of the AMR
            'target_pose_unity': [4.0, 0.05, 2.0, 0.0, 0.0, 0.0],                       # Target pose in Unity coords [x,y,z,roll,pitch,yaw]
            'object_pose_unity': [-3.0, 0.505, -2.0, 0.0, 0.0, 0.0],                    # Object pose in Unity coords [x,y,z,roll,pitch,yaw]
            'robot_pose_unity': [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],                         # Robot spawn pose in Unity coords [x,y,z,roll,pitch,yaw]
        }

    @staticmethod
    def get_dart_dict():
        """DART-only configuration for manipulator dynamics and viewer/debugging."""
        return {
            # should control end-effector orientation or not
            'orientation_control': True,                                                # Control end-effector orientation in addition to position

            # when True: actions in task space, when False: actions in joint space
            'use_inverse_kinematics': True,                                             # Use IK (task space control) vs direct joint-space control

            # when True: SNS algorithm is used for inverse kinematics to conserve optimal linear motion in task space
            # note: might conflict with training agents in task space velocities, in such cases set it to False
            'linear_motion_conservation': False,                                        # Preserve linear motion direction via SNS IK (advanced)

            # when True: the task can be also rendered in the DART viewer
            # Important: Use it when evaluating an agent (e.g. checkpoint). Only for debugging when training an RL agent ('simulation_mode': 'train') - set to False in this case
            # Advanced:  with 'weights & biases', you can log videos during training
            'enable_dart_viewer': False,                                                # Render task in DART viewer (debug/visualization)

            # enable task monitor to visualize states, velocities, agent actions, reward of the robot.
            # 'enable_dart_viewer' should be set to True
            'task_monitor': False,                                                      # Show live task monitor overlay in DART viewer

            # whether to load additional objects in the DART simulation and viewer - ground, background, etc.
            'with_objects': False,                                                      # Load ground/background/objects into DART scene for context

            # how to spawn the red targets in the dart simulation
            # Options: 'random', 'random_joint_level', 'import', 'fixed', 'None'
            # import   -> import targets from a .csv file (see 'target_path' below)
            # None     -> default behaviour. Can be adapted: See iiwa_sample_dart_unity_env, create_target() method
            'target_mode': 'random_joint_level',                                        # DART target generation mode

            # when target_mode is 'import': load targets from a .csv file
            'target_path': '/misc/generated_random_targets/cart_pose_7dof.csv',         # CSV path for imported targets when target_mode='import'

            # Velocity/acceleration limits and safety for 'iiwa_sample_env'
            # Note: 'joints_safety_limit' -> set to higher value depending on your task and velocity ranges
            #                                the UNITY behaviour may be unstable when having a 0.0 safety limit with high velocities
            #                                e.g. nan tensor joint error -> the robot is in an invalid configuration - reset the manipulator
            'joints_safety_limit': 0.0,                                                 # [deg]
            'max_joint_vel': 20.0,                                                      # [deg/s] - Joint space
            'max_ee_cart_vel': 10.0,                                                    # [m/s]   - Task space  -- Not optimized values for sim2real transfer 
            'max_ee_cart_acc': 3.0,                                                     # [m/s^2]
            'max_ee_rot_vel': 4.0,                                                      # [rad/s]
            'max_ee_rot_acc': 1.2,                                                      # [rad/s^2]
        }

    # ---------------------------
    # Unity-XML groups
    # ---------------------------
    @staticmethod
    def get_root_dict():
        """Root-level Unity XML configuration."""
        return {
            'environment_mode': 'Warehouse',                                            # (*) Options: 'Manipulator', 'Warehouse'
        }

    @staticmethod
    def get_simulation_dict():
        """Configuration for the <Simulation> XML section only."""
        return {
            'communication_type': 'GRPC',                                               # (*) Options: GRPC, GRPC_NRP, ROS or ZMQ
            'ip_address': 'localhost',                                                  # (*) The ip address of the simulator server -- 'host.docker.internal' (use for windows and docker),  'localhost' (use for linux for connections on local machine)
            'port_number': '9092',                                                      # (*) Port number for communication with the simulator
            'timestep_duration_in_seconds': 0.02,                                       # (*) Agent observation/action control cycle
            'physics_simulation_increment_in_seconds': 0.02,                            # (*) Unity PhysX discrete step update
            'improved_patch_friction': True,                                            # (*) Make PhysX use the friction mode that guarantees static and dynamic friction do not exceed analytical results
            'random_seed': 256,                                                         # (*) The seed used for generating pseudo-random sequences
            'evaluation': False,                                                        # (*) Whether the simulation should be instantiated in the evaluation or training mode
            'randomize_environment_physics': False,                                     # (*) When set to False, it disables all randomization aspects related to simulation dynamics
            'randomize_torque': False,                                                  # (*) Randomize the max torque that is applied by joint's motor in order to reach the desired velocity
            'persist_episode_manifests': False,                                         # (*) Save per-episode manifests (metadata) to disk during simulation
            'replay_episode_manifest': False,                                           # (*) Load and replay a previously saved episode manifest
            'replay_manifest_path': '',                                                 # (*) Filesystem path to a manifest file used for replay
        }

    @staticmethod
    def get_manipulator_environment_dict():
        """Configuration for the <ManipulatorEnvironment> XML section."""
        return {
            # Manipulator core
            'manipulator_model': 'IIWA14',                                              # (*) Options: IIWA14, SO100
            'enable_end_effector': True,                                                # (*) Set to False if no tool is attached. Important: in that case, set 'end_effector_model' to None
            'end_effector_model': 'CALIBRATION_PIN',                                    # (*) Options: ROBOTIQ_3F, ROBOTIQ_2F85, CALIBRATION_PIN, None. Also, 'enable_end_effector' should be set to True.
                                                                                        #               -> For 'iiwa_sample_joint_vel_env' select a gripper
                                                                                        #     Options: 'DEFAULT_GRIPPER' for SO-100 arm. Also, 'enable_end_effector' should be set to True.
            'trajectory_string': '',                                                    # (*) Optional serialized trajectory definition for task playback

            # Floor
            'floor': {
                'floor_type': 'MONOCHROMATIC',                                          # (*) Options: 'CHECKERBOARD', 'WOOD', 'MONOCHROMATIC'
                'floor_size': [2.4, 0.01, 2.4],                                         # (*) [X, Y, Z] [m] - Floor dimensions
                'floor_material': 'HOMOGENEOUS',                                        # (*) Options: 'HOMOGENEOUS' and 'HETEROGENEOUS'
                'visualize_floor_material': False,                                      # (*) Whether to visualize the material heterogeneity of the floor
                'floor_material_color': [0.235294119, 0.509803951, 0.9411765, 1.0],     # (*) [R, G, B, A] -- only used when floor_type is set to 'MONOCHROMATIC'

                # The division of the material homogenity along the X/Y/Z axis
                # ⤷ -- for a homogeneous floor is always { 1.0f }
                # ⤷ -- for each axis an array of floats summing up to 1.0 should be defined
                'floor_material_grid_x': [1.0],                                         # (*) an array with elements between 0.0 and 1.0
                'floor_material_grid_y': [1.0],                                         # (*) an array with elements between 0.0 and 1.0
                'floor_material_grid_z': [1.0],                                         # (*) an array with elements between 0.0 and 1.0

                # The dynamic/static friction coefficient for each division of the material grid (order Z,Y,X)
                # ⤷ -- for a homogeneous floor is always { 1.0f }
                # ⤷ -- Example: If there is a 2*2 grid with divisions in X and Z axes (e.g. floor_material_grid_x=[0.5,0.5], floor_material_grid_y=[1.0], floor_material_grid_z=[0.5,0.5]), 
                # ⤷ -- then the array has the length of 4 and the elements are ordered in a way that the grid cell increment first occurs on the Z axis, then on the X axis.
                'floor_material_grid_dynamic_friction': [1.0],                          # (*) an array with elements between 0.0 and 1.0
                'floor_material_grid_static_friction': [1.0],                           # (*) an array with elements between 0.0 and 1.0
            },

            # Items pool
            'items': [
                {
                    'item_type': 'BOX',                                                 # (*) Options: 'BOX' ( and 'SPHERE' legacy)

                    'item_size': [0.2, 0.2, 0.3],                                       # (*) [X, Y, Z] [m] - Item dimensions
                    'item_mass': 6.0,                                                   # (*) [kg]
                    'item_center_of_mass': [0.0, 0.0, 0.0],                             # (*) [X, Y, Z] [m] - Center of mass position
                    'item_linear_damping': 2.0,                                         # (*) decay rate of linear velocity, to simulate drag, air resistance, or friction
                    'item_observability': True,                                         # (*) Whether to observe the numeric pose of the item

                    'item_material': 'HETEROGENEOUS',                                   # (*) Options: 'HOMOGENEOUS' and 'HETEROGENEOUS'
                    'visualize_item_material': True,                                    # (*) Whether to visualize the material heterogeneity of the item
                    'item_material_color': [0.0, 1.0, 0.0, 1.0],                        # (*) [R, G, B, A] -- only used when visualize_item_material is False
                    'target_material_color': [1.0, 0.0, 0.0, 1.0],                      # (*) [R, G, B, A]

                    # The division of the material homogenity along the X/Y/Z axis
                    # ⤷ -- for a homogeneous item is always { 1.0f }
                    # ⤷ -- for each axis an array of floats summing up to 1.0 should be defined
                    'item_material_grid_x': [1.0],                                      # (*) an array with elements between 0.0 and 1.0
                    'item_material_grid_y': [1.0],                                      # (*) an array with elements between 0.0 and 1.0
                    'item_material_grid_z': [0.5, 0.5],                                 # (*) an array with elements between 0.0 and 1.0

                    # The dynamic/static friction coefficient for each division of the material grid (order Z,Y,X)
                    # ⤷ -- for a homogeneous item is always { 1.0f }
                    # ⤷ -- Example: If there is a 2*2 grid with divisions in X and Z axes (e.g. item_material_grid_x=[0.5,0.5], item_material_grid_y=[1.0], item_material_grid_z=[0.5,0.5]), 
                    # ⤷ -- then the array has the length of 4 and the elements are ordered in a way that the grid cell increment first occurs on the Z axis, then on the X axis.
                    'item_material_grid_dynamic_friction': [0.01, 0.01],                # (*) an array with elements between 0.0 and 1.0
                    'item_material_grid_static_friction': [0.01, 0.01],                 # (*) an array with elements between 0.0 and 1.0

                    # Randomization (per item)
                    'randomize_item_mass': False,                                       # (*) Randomize item mass
                    'item_mass_randomization_range': 0.1,                               # (*) [kg] - The range to use for uniform sampling when randomizing mass around its default value
                    'randomize_item_center_of_mass': False,                             # (*) Randomize item center of mass
                    'item_center_of_mass_randomization_range': [0.0, 0.0, 0.075],       # (*) [m] - The range to use for uniform sampling along x,y,z axes when randomizing center of mass around its default value
                    'randomize_item_friction': False,                                   # (*) Randomize item friction coefficients
                    'item_dynamic_friction_randomization_range': 0.1,                   # (*) The range to use for uniform sampling when randomizing dynamic friction around its default value
                    'item_static_friction_randomization_range': 0.1,                    # (*) The range to use for uniform sampling when randomizing static friction around its default value
                }
            ],
        }

    @staticmethod
    def get_warehouse_environment_dict():
        """Configuration for the <WarehouseEnvironment> XML section."""
        warehouse_dict = {
            # Top-level
            'amr_model': 'SAFELOG_S2',                                                  # (*) Autonomous Mobile Robot model to spawn
            'enable_transport': False,                                                  # (*) Enable pallet transport lift mechanism on the AMR
            'max_chassis_linear_speed': 0.8,                                            # (*) Max linear speed of the chassis [m/s]
            'max_chassis_angular_speed': 0.5,                                           # (*) Max angular speed of the chassis [rad/s]
            'wheel_drive_force_limit': 10.0,                                            # (*) Drive force limit applied to wheel motors
            'wheel_drive_damping': 10.0,                                                # (*) Damping applied to wheel drive to stabilize motion

            # Ground
            'ground': {
                'ground_type': 'TEXTURED',                                              # MONOCHROMATIC, TEXTURED, PREFAB
                'ground_size': [12.5, 0.1, 7.5],                                        # (*) [X, Y, Z] [m] - Ground plane dimensions
                'wall_height': 0.5,                                                     # (*) Height of boundary walls surrounding the ground [m]
                'ground_material': 'HOMOGENEOUS',                                       # (*) Options: 'HOMOGENEOUS' and 'HETEROGENEOUS'
                'visualize_ground_material': False,                                     # (*) Whether to visualize heterogeneity of ground material
                'ground_material_color': [0.235294119, 0.509803951, 0.9411765, 1.0],    # (*) [R, G, B, A] -- used when ground is MONOCHROMATIC
                'ground_material_grid_x': [1.0],                                        # (*) Material homogeneity divisions along X axis
                'ground_material_grid_y': [1.0],                                        # (*) Material homogeneity divisions along Y axis
                'ground_material_grid_z': [1.0],                                        # (*) Material homogeneity divisions along Z axis
                'ground_material_grid_dynamic_friction': [1.0],                         # (*) Dynamic friction per grid cell (order Z,Y,X)
                'ground_material_grid_static_friction': [1.0],                          # (*) Static friction per grid cell (order Z,Y,X)
            },

            # Obstacle manager
            'enable_obstacle_manager': False,                                           # (*) Enable automatic placement/spawn of obstacles in the map
            'obstacle_placement_separation_multiplier': 2.8,                            # (*) Multiplier ensuring minimum separation between obstacles
            'obstacle_spawn_boundary_margin': 0.25,                                     # (*) Margin [m] from boundaries where obstacles cannot spawn
            'static_obstacle_count': 0,                                                 # (*) Number of static obstacles to spawn (drawn from pool below)

            # Static obstacles pool
            'static_obstacles': [                                                       # (*) Pool of static obstacle parameter variants
                # Example element (duplicate to define more varieties in the pool):
                {
                    'obstacle_type': 'BOX',                                             # (*) Options: 'BOX'
                    'obstacle_size': [1.0, 1.0, 1.0],                                   # (*) [X, Y, Z] [m] - Obstacle dimensions
                    'obstacle_mass': 10.0,                                              # (*) [kg]
                    'obstacle_center_of_mass': [0.0, 0.0, 0.0],                         # (*) [X, Y, Z] [m] - Center of mass position
                    'obstacle_linear_damping': 2.0,                                     # (*) Decay rate of linear velocity for obstacle body
                    'obstacle_observability': False,                                    # (*) Whether the obstacle pose is observable numerically
                    'obstacle_material': 'HOMOGENEOUS',                                 # (*) Options: 'HOMOGENEOUS' and 'HETEROGENEOUS'
                    'visualize_obstacle_material': False,                               # (*) Whether to visualize material heterogeneity of obstacle
                    'obstacle_material_color': [0.0, 1.0, 0.0, 1.0],                    # (*) [R, G, B, A] -- used when visualize_obstacle_material is False
                    'obstacle_material_grid_x': [1.0],                                  # (*) Material homogeneity divisions along X axis
                    'obstacle_material_grid_y': [1.0],                                  # (*) Material homogeneity divisions along Y axis
                    'obstacle_material_grid_z': [1.0],                                  # (*) Material homogeneity divisions along Z axis
                    'obstacle_material_grid_dynamic_friction': [0.6],                   # (*) Dynamic friction per grid cell (order Z,Y,X)
                    'obstacle_material_grid_static_friction': [0.6],                    # (*) Static friction per grid cell (order Z,Y,X)
                    'randomize_obstacle_mass': False,                                   # (*) Randomize obstacle mass
                    'obstacle_mass_randomization_range': 1.0,                           # (*) [kg] - Uniform range around default mass
                    'randomize_obstacle_center_of_mass': False,                         # (*) Randomize obstacle center of mass
                    'obstacle_center_of_mass_randomization_range': [0.5, 0.5, 0.5],     # (*) [m] - Range along x,y,z axes
                    'randomize_obstacle_friction': False,                               # (*) Randomize obstacle friction coefficients
                    'obstacle_dynamic_friction_randomization_range': 0.1,               # (*) Uniform range around default dynamic friction
                    'obstacle_static_friction_randomization_range': 0.1,                # (*) Uniform range around default static friction
                }
            ],

            # Dynamic obstacles
            'dynamic_obstacles': {
                'dynamic_obstacle_count': 0,                                            # (*) Number of dynamic obstacles to spawn
                'dynamic_obstacle_motion': 'Random',                                    # (*) Motion pattern: None | Random | Circle | Linear
                'dynamic_obstacle_min_distance_from_robot': 1.6,                        # (*) Minimum spawn distance from the robot [m]
                'max_linear_speed': 0.2,                                                # (*) Max linear speed for dynamic obstacle [m/s]
                'max_angular_speed': 0.2,                                               # (*) Max angular speed for dynamic obstacle [rad/s]
                'random_motion_change_period_seconds': 2.0,                             # (*) Period [s] to re-sample random motion commands
                'linear_motion_cycle_seconds': 8.0,                                     # (*) Cycle period [s] for linear back-and-forth motion
                'linear_motion_travel_speed': 0.1,                                      # (*) Translation speed [m/s] for linear motion
                'circle_motion_travel_speed': 0.1,                                      # (*) Tangential speed [m/s] for circular motion
                'circle_motion_curvature': -1.2,                                        # (*) Curvature of circle path; sign chooses rotation direction
            },
        }
        return warehouse_dict

    @staticmethod
    def get_observation_dict():
        """Configuration for the <Observation> XML section."""
        return {
            # Image settings
            'enable_observation_image': False,                                          # (*) Use images as state representation. Example code at 'iiwa_joint_vel', _retrieve_image() and 'iiwa_sample_dart_unity_env', update() functions
            'save_observation_image_as_file': False,                                    # (*) A boolean defining whether the observation image is sent as observation or stored to the hard drive
            'observation_image_encoding': 'JPG',                                        # (*) Encoding of Unity observation image (losseles 'PNG' or lossy 'JPG')
            'observation_image_quality': 50,                                            # (*) Compression of the lossy JPG image -- from 1 (worst) to 100 (best)
            'observation_image_width': 128,                                             # (*) Observation image width dimension in pixels. See 'iiwa_joint_vel', __init__()
            'observation_image_height': 128,                                            # (*) Observation image height dimension in pixels. See 'iiwa_joint_vel', __init__()
            'observation_image_background_color': [1.0, 1.0, 1.0, 1.0],                 # (*) [R, G, B, A]
            'enable_grayscale': False,                                                  # (*) Unity observation image as Gray-scale instead of RGB
            'enable_segmentation': False,                                               # (*) Unity observation of robot as one color useful for segmentation
            'robot_segmentation_color': [1.0, 0.0, 1.0, 1.0],                           # (*) [R, G, B, A]

            # Shadows
            'enable_shadows': True,                                                     # (*) Note: For sim2real transfer (images) try also False
            'shadow_type': 'Soft',                                                      # (*) Options: 'Soft', 'Hard', 'None'

            # Cameras pool
            'observation_cameras': [
                {
                    'camera_position': [0.0, 1.25, 1.3],                                # (*) [X, Y, Z] [m] in Unity coordinates
                    'camera_rotation': [130.0, 0.0, 180.0],                             # (*) [RX, RY, RZ] [deg] in Unity coordinates
                    'camera_vertical_fov': 45.0,                                        # (*) Camera's vertical field of view in degrees
                }
            ],

            # Appearance/camera randomization
            'randomize_appearance': False,                                              # (*) Whether to randomize the lighting, the appearance of the environment (colors/viewpoint/background)
            'camera_position_randomization_range_in_meters': 0.05,                      # (*) [m] - The range to use for uniform sampling when randomizing camera position around its default position
            'camera_rotation_randomization_range_in_degrees': 3.0,                      # (*) [deg] - The range to use for uniform sampling when randomizing camera rotation around its default rotation along each axis

            # Laser scan (Warehouse)
            'enable_laser_scan': False,                                                 # (*) Enable 2D laser scanner in the warehouse environment
            'laser_scan': {
                'range_meters_min': 0.12,                                               # (*) Minimum measurable laser range in meters
                'range_meters_max': 100.0,                                              # (*) Maximum measurable laser range in meters
                'scan_angle_start_degrees': 180.0,                                      # (*) Start angle of the scan sector in degrees
                'scan_angle_end_degrees': -179.0,                                       # (*) End angle of the scan sector in degrees (clockwise)
                'num_measurements_per_scan': 360.0,                                     # (*) Number of laser range samples per full scan (clockwise)
            }
        }


if __name__ == "__main__":
    """
        If you have updated the Config class, you can run this script to update the configuration.xml of the
        UNITY simulator instead of updating the fields manually

        Important: In this case the simulator folder should be located inside the vtprl/ folder
                   with name e.g., simulator/Linux/VTPRL-Simulator.x86_64 (or .exe),
                   or update the path of the 'xml_file' variable below

        Note:      You many need to restart the UNITY simulator after updating the .xml file
    """
    config = Config()

    # Change the path if needed
    simulator_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/environment/simulator/'
    simulator_version = 'v1.0.0'
    simulator_platform = 'Windows'  # 'Linux', 'Mac'
    xml_file = simulator_path + simulator_version + '/' + simulator_platform + '/configuration.xml'

    # Update the .xml based on the new changes in the Config class #
    update_simulator_configuration(config, xml_file)
    print('Successfully updated the simulator configuration file:')
    print(xml_file)
