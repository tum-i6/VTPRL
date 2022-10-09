class Config:

    def __init__(self):
        pass

    @staticmethod
    def get_dart_env_dict():
        dart_env_dict = {
            # episode length in no. of simulation steps
            'max_time_step': 200,

            # should control end-effector orientation or not
            'orientation_control': False,

            # when True: actions in task space, when False: actions in joint space
            'use_inverse_kinematics': True,

            # when True: SNS algorithm is used for inverse kinematics to conserve optimal linear motion in task space
            # note: might conflict with training agents in task space velocities, in such cases set it to False
            'linear_motion_conservation': False,

            # when True: the reaching task can be also rendered in the DART viewer
            # note: only for visualization, not appropriate for training a model on the pixels
            'enable_dart_viewer': False,
        }

        return dart_env_dict

    @staticmethod
    def get_config_dict():
        config_dict = {

            ###################################### GENERAL PARAMETERS ##################################################

            # whether the RL algorithm uses customized or stable-baseline specific parameters
            'custom_hps': True,

            # use 'iiwa_joint_vel' for iiwa_environment without dart (only joint velocity control), or
            # 'iiwa_sample_dart_unity_env' for control in task space with dart
            'env_key': 'iiwa_sample_dart_unity_env',

            # environment-specific parameter, e.g, number of links, only relevant for iiwa_joint_vel env
            # currently, num_joints should be always 7 when the DART based environment is used
            'num_joints': 7,

            # whether to use images as state representation, an example code is in iiwa_joint_vel env
            'use_images': False,

            # GRPC, ROS or ZMQ
            'communication_type': 'GRPC',

            # the ip address of the server, for windows and docker use 'host.docker.internal',
            # for linux use 'localhost' for connections on local machine
            'ip_address': 'localhost',  # 'host.docker.internal',  # 'localhost'

            # port number for communication with the simulator
            'port_number': '9092',

            # the seed used for generating pseudo-random sequences
            'seed': None,

            # the state of the RL agent in case of numeric values for manipulators 'a' for angles only
            # or 'av' for angles and velocities
            'state': 'a',

            # number of environments to run in parallel
            'num_envs': 16,
        }

        return config_dict
