"""
Manual actions functionality used in simulator_vec_env.py - see step() method

Note: you can also implement your own manual function - in that case create a new function below and call it in the simulator_vec_env.py
"""

import ast
import numpy as np

def configure_manual_settings_and_get_manual_function(vec_env, manual_actions_dict):
    """
        Depending of the input manual_behaviour (see config_advanced.py) setting, set the class variables of a SimulatorVecEnv class related to the manual actions,
        and return a corresponding manual function to excecute after the end of the episode - step() method during reseting in simulator_vec_env.py

        :param vec_env:             SimulatorVecEnv object
        :param manual_actions_dict: manual actions dictionary settings

        :return: manual_func: returns a function to excecute at the end of the episode - step() method during reseting - SimulatorVecEnv class
                              format: manual_func(vec_env, rews, infos, enable_rewards=True)
                                      where:
                                        vec_env        -> SimulatorVecEnv object (contains the envs objects, etc)
                                        rews           -> rewards list of all envs
                                        infos          -> infos list of dictionaries of all envs
                                        enable_rewards -> whether to give reward/penalties during manual actions
    """

    manual_func = _send_no_manual # Do nothing by default

    # Planar envs: Down -> close -> up #
    if(vec_env.manual_behaviour == "planar_grasping"):
        # Clip gripper action to this limit #
        if(vec_env.robotic_tool.find("3_gripper") != -1):
            vec_env.gripper_clip = 90
        elif(vec_env.robotic_tool.find("2_gripper") != -1):
            vec_env.gripper_clip = 250
        else:
            vec_env.gripper_clip = 90

        vec_env.manual_rewards = manual_actions_dict["manual_rewards"]                   # For enable_rewards
        vec_env.manual_kp = manual_actions_dict["manual_kp"]                             # Gain for P-controller, position error
        vec_env.manual_kpr = manual_actions_dict["manual_kpr"]
        vec_env.manual_tol = manual_actions_dict["manual_tol"]                           # Tolerance of P-controller
        vec_env.manual_down_height = manual_actions_dict["manual_down_height"]
        vec_env.manual_up_height = manual_actions_dict["manual_up_height"]
        vec_env.manual_close_steps = manual_actions_dict["manual_close_steps"]           # How many steps to close the gripper (with 'manual_close_vel' per each step)
        vec_env.manual_close_vel = manual_actions_dict["manual_close_vel"]               # Velocity value to close the gripper per each step
        vec_env.manual_steps_same_state = manual_actions_dict["manual_steps_same_state"] # How many steps the ee is allowed to not moved during the manual actions down/up 
        vec_env.manual_tol_same_state = manual_actions_dict["manual_tol_same_state"]     # If the gripper has not moved more than this distance threshold value within 'steps_same_state' -> it means collision 

        # Controll all DoF with a P-controller during the manual actions #
        vec_env.config_p_controller = {"kpr_x": 1,
                                       "kpr_y": 1,
                                       "kpr_z": 1,
                                       "kp_x": 1,
                                       "kp_y": 1,
                                       "kp_z": 1
                                      }


        #####################################################################################################
        # prev_dist: needed for 'manual_steps_same_state'. Save the previous distance of the ee to the goal #
        #            and compare it to the current distance. See _go_down_up_manual function below          #
        ####################################################################################################################################
        # pid_step: counter to reset the p-controllers target pose for each env. During the down/up manual actions,                        #
        #           keep in the other DoF the same pose as the pose that the agent had at the end of the RL episode                        #
        #           e.g. rotation of the gripper - it means the P-controller does not correct the rotation when approaching the target box #
        ####################################################################################################################################
        vec_env.prev_dist = []
        vec_env.pid_step = []
        for env_idx in range(len(vec_env.envs)):
            vec_env.pid_step.append(0)
            vec_env.prev_dist.append(0)

        # Set grasping manual function to call at the end of the episode #
        manual_func = _send_manual_actions_planar_grasping

    # Close the gripper only #
    elif(vec_env.manual_behaviour == "close_gripper"): 
        # Clip gripper action to this limit #
        if(vec_env.robotic_tool.find("3_gripper") != -1):
            vec_env.gripper_clip = 90
        elif(vec_env.robotic_tool.find("2_gripper") != -1):
            vec_env.gripper_clip = 250
        else:
            vec_env.gripper_clip = 90

        vec_env.manual_close_steps = manual_actions_dict["manual_close_steps"]
        vec_env.manual_close_vel = manual_actions_dict["manual_close_vel"]

        # Set close manual function to call at the end of the episode #
        manual_func = _send_manual_actions_close_gripper

    # Invalid option do nothing #
    else:
        manual_func = _send_no_manual

    return manual_func

##################################
# Manual actions implementations #
##################################

######################
# 0. Nothing happens #
######################
def _send_no_manual(vec_env, rews=None, infos=None, enable_rewards=False):
    pass

######################
# 1. Planar grasping #
######################
def _send_manual_actions_planar_grasping(vec_env, rews, infos, enable_rewards=True):
    """
        Planar grasping manual actions:
        1. down
        2. close
        3. up

        :param rews:           rewards of envs
        :param infos:          infos dicts of envs
        :param enable_rewards: whether to give rewards/penalties during manual actions
    """
    envs_active = []
    for env in vec_env.envs: # Send manual actions commands only to valid envs
        if env.collided_env != 1:
            envs_active.append(env)

    # 1.Down #
    envs_active = _go_down_up_manual(vec_env, envs_active, rews, vec_env.manual_down_height, gripper=0.0, enable_rewards=enable_rewards)

    # 2.Close #
    envs_active = _close_manual(vec_env, envs_active, rews, steps=vec_env.manual_close_steps, vel=vec_env.manual_close_vel, enable_rewards=enable_rewards)

    # 3.Up #
    gripper_pos = np.clip(vec_env.manual_close_steps * vec_env.manual_close_vel, 0, vec_env.gripper_clip) # Gripper is closed - during up movement
    envs_active = _go_down_up_manual(vec_env, envs_active, rews, vec_env.manual_up_height, gripper=gripper_pos, enable_rewards=enable_rewards)

########################
# 2. Close the gripper #
########################
def _send_manual_actions_close_gripper(vec_env, rews, infos, enable_rewards=True):
    """
        Close the gripper at the end of the episode

        :param rews:           rewards of envs
        :param infos:          infos dicts of envs
        :param enable_rewards: whether to give rewards/penalties during manual actions
    """
    envs_active = []
    for env in vec_env.envs: # Send manual actions commands only to valid envs 
        if env.collided_env != 1:
            envs_active.append(env)

    # 1. Close #
    envs_active = _close_manual(vec_env, envs_active, rews, steps=vec_env.manual_close_steps, vel=vec_env.manual_close_vel, enable_rewards=enable_rewards)

###########
# Helpers #
###########
def _update_envs_manual_and_check_collisions(vec_env, envs_active, rews, observations, enable_rewards=True):
    """
        update the active envs (dart chain, etc) during the manual actions and give rewards/collision penalties if enabled

        :param observations:     by unity
        :param time_step_update: whether to update the timestep of the gym agents
                                 - it is always set to False, as the RL-episode has been finished - else adapt to your task

        :return: terminated_environments, rews, infos, obs_converted, dones - see simulator_vec_env.py step() method for more details
    """
    envs_active_new = []

    for env in envs_active:
        observation = observations[env.id]
        ob_coll = ast.literal_eval(observation)["Observation"][33]

        ##########################################################################
        # During the manual actions, the RL agents do not generate any velocites #
        # This line of code is related to the task monitor visualization         #
        ##########################################################################
        env.action_state = np.zeros(env.action_space_dimension)

        # Update only valid envs #
        if (float(ob_coll) != 1.0 and env.joints_limits_violation() == False):
            env.update(ast.literal_eval(observation), time_step_update=False)
            envs_active_new.append(env)

        # Collision - add penalty #
        else:
            if (enable_rewards):
                rews[env.id] += vec_env.reward_dict["reward_collision"]
            env.collided_env = 1
            env.collision_flag = True # Give penality only

    # Render: Not advised to set 'enable_dart_viewer': True, during RL-training. Use it only for debugging #
    vec_env.render()

    return envs_active_new

def _go_down_up_manual(vec_env, envs_active, rews, height_target, gripper=0.0, enable_rewards=True):
    """
        go down or up by using a P-controller

        :param envs_active:     active envs
        :param rews:            rewards of envs
        :param height_target:   height to move the ee
        :param gripper: gipper  position to keep during up/down (e.g. close/open)
        :param enable_rewards:  whether to give penalties during manual actions

        :return: envs_active - which envs did not collided during manual actions
    """

    for env in envs_active:
        env.dart_sim.ik_by_sns = True # Use sns during manual actions - adapt if needed

        # Set viewer and the change target during the up movement #
        if(gripper != 0.0):

            ############################################################################################################################
            # Note: in the dart viewer the ee at the goal is more up as we assume that we have a gripper (urdf) but the gripper is not #
            #       yet visualized in the viewer. The vec_env.dart_sim.get_pos_distance() returns 0 correctly at the goal              #
            ############################################################################################################################
            target_object_X, _, target_object_Z = env.init_object_pose_unity[0], env.init_object_pose_unity[1], env.init_object_pose_unity[2]
            tool_length = 0.0 # Adapt if the height of the box changes
            target_object_RX, target_object_RY, target_object_RZ = env.get_box_rotation_in_target_dart_coords_angle_axis()

            target = [target_object_RX, target_object_RY, target_object_RZ, target_object_Z, -target_object_X, height_target + tool_length]
            env.set_target(target) # Set the dart target

    # Unity expects velocities in joint space for each env #
    # Send zero velocities to collided envs                #
    actions = gripper * np.ones((len(vec_env.envs), 8))

    ###################################################################
    # Save the last pose of the ee before the manual actions and keep #
    # the DoF that we do not control with the manual actions fixed    #
    # e.g. do not rotate the gripper during the up/down movement      #
    ###################################################################
    target_pos_quat = [None] * len(vec_env.envs)                      # Target orientation in quat
    target_pos_x = np.zeros(len(vec_env.envs))
    target_pos_z = np.zeros(len(vec_env.envs))

    # Set the target height #
    target_pos_y = height_target

    # Wait all envs to finish #
    while True:
        actions[:, :7] = np.zeros((len(vec_env.envs), 7))
        flag_active = False # At least one envs has not finished with its manual actions when set to True

        # Active envs (non-collided) during this loop #
        envs_active_new = [] 
        for env in envs_active:

            # Position of the ee in Unity #
            ee_x, ee_y, ee_z, _, _, _ = env.get_ee_pose_unity()

            # Current orientation of ee in quaternion in Dart #
            curr_quat = env.get_rot_ee_quat()

            # Set target pose - 0 timestep #
            if (vec_env.pid_step[env.id] == 0):
                target_pos_quat[env.id] = curr_quat # Keep fixed orientation
                target_pos_x[env.id] = ee_x         # Only move down/up keep. Keep fixed x, z position (dart)
                target_pos_z[env.id] = ee_z

            # Current positional errors in dart #
            x_diff = target_pos_z[env.id] - ee_z
            y_diff = -target_pos_x[env.id] + ee_x
            z_diff = target_pos_y - ee_y

            # Current orientation error in dart #
            rx_diff, ry_diff, rz_diff = env.get_rot_error_from_quaternions(target_pos_quat[env.id], curr_quat)

            #############################################################################################
            # Distance from the goal pose                                                               #
            # If we moved in some directions other than up/down make corrections using the P-controller #
            #############################################################################################
            dist = np.linalg.norm(np.array([x_diff, y_diff, z_diff, rx_diff, ry_diff, rz_diff]))

            # Goal reached - zero vel should be send to the Unity simulator #
            if (dist < vec_env.manual_tol):
                envs_active_new.append(env)
                continue

            ###########################################################################################
            # Check if we have moved at all within "vec_env.manual_steps_same_state" time             #
            # e.g. case: ee fingers bounce at the top surface of the box when we go down -> collision #
            ###########################################################################################
            if (vec_env.pid_step[env.id] != 0 and vec_env.pid_step[env.id] % vec_env.manual_steps_same_state == 0):

                # The ee has not moved a lot from the previous time -> collision #
                if (abs(vec_env.prev_dist[env.id] - dist) <= vec_env.manual_tol_same_state): 
                    if (enable_rewards):
                        rews[env.id] += vec_env.reward_dict["reward_collision"]
                    else:
                        pass

                    env.collided_env = 1
                    env.collision_flag = True # Give penality only once
                    continue

                ######################################################
                # No collision yet and the goal has not been reached #
                # Save current distance to the goal from the ee      #
                ######################################################
                vec_env.prev_dist[env.id] = dist

            ##################################################################################
            # Calculate joint velocities. Call the custom P-controller that first calculates #
            # task-space velocities and then returns, using IK, joint-space velocities       #
            ##################################################################################
            vel = env.action_by_p_controller_custom(rx_diff, ry_diff, rz_diff, x_diff, y_diff, z_diff, vec_env.manual_kpr, vec_env.manual_kp, config_p_controller=vec_env.config_p_controller)

            actions[env.id][0:7] = vel # Set action for this env 

            # Update P-controller step #
            vec_env.pid_step[env.id] += 1
            flag_active = True            # At lease one env is active - continue with manual actions
            envs_active_new.append(env)

        # No more actions to take: all envs have reached their goal, and/or no more active envs #
        if (flag_active == False):
            break
        else:
            ############################
            # Send JSON action command #
            ############################
            request = vec_env._create_request("ACTION", vec_env.envs, actions)
            observations = vec_env._send_request(request)

            #####################################################
            # Update envs (dart chain) and check for collisions #
            #####################################################
            envs_active = _update_envs_manual_and_check_collisions(vec_env, envs_active_new, rews, observations, enable_rewards)

    ############################################################
    # Manual action down/up finished. Set envs_active with     #
    # the envs that did not collided during this manual action #
    ############################################################
    envs_active = envs_active_new

    # Deactivate sns and reset P-controller #
    for env in vec_env.envs:
        vec_env.pid_step[env.id] = 0
        vec_env.prev_dist[env.id] = 0
        env.dart_sim.ik_by_sns = False

    return envs_active

def _close_manual(vec_env, envs_active, rews, steps=4, vel=15, enable_rewards=True):
    """
        close the gripper

        :param envs_active:    active envs
        :param rews:           rewards of envs
        :param steps:          how many gripper commands to send
        :param vel:            velocity per command
        :param enable_rewards: whether to give reward/penalties during manual actions

        :return: envs_active - which envs did not collided during manual actions
    """

    actions_close = np.zeros((len(vec_env.envs), 8))  # Do not move the robot: joints - zeros, and gripper position

    for j in range(steps):  # Send gripper position for this steps 

        for env in envs_active: # For each active env 
            actions_close[env.id][7] = np.clip((actions_close[env.id][7] + vel), 0.0, vec_env.gripper_clip)

        ############################
        # Send JSON action command #
        ############################
        request = vec_env._create_request("ACTION", vec_env.envs, actions_close)
        observations = vec_env._send_request(request)

        #####################################################
        # Update envs (dart chain) and check for collisions #
        #####################################################
        _update_envs_manual_and_check_collisions(vec_env, envs_active, rews, observations, enable_rewards=enable_rewards)

    return envs_active
