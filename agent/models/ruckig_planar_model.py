"""
A time-optimal trajectory generation model - RUCKIG

Notes:
           - The DoF are optimized to reach at the same time the goal
           - Support for 2 or 3 DoF control - it is used for planar envs - adapt to your task
           - The accelerations and velocities are approximated by 'first-order' approximation - e.g. (dx1 - dx2) / control_cycle
           - Docs: https://docs.ruckig.com/index.html
           - ruckig allows for both task and joint space control. It is used for task-space control and the agent with IK then maps the velocities to the joint space
Important:
           -All the calculations are in the DART coordinate system
"""

import numpy as np

from ruckig import InputParameter, OutputParameter, Result, Ruckig

class RuckigPlanarModel():

    def __init__(self, env, control_cycle, hyper_dict):
        self.envs = env                                      # Gym env: 'iiwa_ruckig_planar_grasping_dart_unity_env'
        self.control_cycle = control_cycle                   # In sec. Set the same value in the Unity simulator
        self.dof = np.asarray(hyper_dict["dof"])
        self.target_velocity = hyper_dict["target_vel"]      # Velocity in the target pose 
        self.target_acceleration = hyper_dict["target_acc"]
        self.max_vel = hyper_dict["max_vel"]
        self.max_acc = hyper_dict["max_acc"]
        self.max_jerk = hyper_dict["max_jerk"]

        ###########
        # Set DoF #
        ###########
        self.dof = np.count_nonzero(self.dof)
        if(self.dof != 3 and self.dof != 2):
            raise Exception("Only 2 or 3 DoF are supported for Ruckig Planar Model")

        self.target_velocity = self.target_velocity[3-self.dof:]
        self.target_acceleration = self.target_velocity[3-self.dof:]
        self.max_vel = self.max_vel[3-self.dof:]
        self.max_acc = self.max_acc[3-self.dof:]
        self.max_jerk = self.max_jerk[3-self.dof:]

        ###############################
        # DoF we control              #
        # Dart 0 -> rotation Zd = Yu  #
        # Dart 1 -> position Xd = Zu  #
        # Dart 2 -> position Yd = -Xu #
        ###############################

        ##############################################################################
        # Set ruckig objects for each env. Supports many envs but the processing     #
        # is performed in a sequential manner -  no GPU support or parallelization   #
        ##############################################################################
        self.ruckig_models = {"otg": [], "res": [], "inp": [], "out": [],
                              "prev_pose": [], "prev_vels": [], "prev_accs": [], "time_step": []
                             }

        # See ruckig documentation for better understanding #
        for i in range(self.envs.num_envs):
            self.ruckig_models["otg"].append(Ruckig(self.dof, self.control_cycle))
            self.ruckig_models["res"].append(Result.Working)                                # Flag
            self.ruckig_models["out"].append(OutputParameter(self.dof))                     # Velocities result of optimization

            self.ruckig_models["inp"].append(InputParameter(self.dof))                      # Current pose
            self.ruckig_models["inp"][i].target_velocity = self.target_velocity
            self.ruckig_models["inp"][i].target_acceleration = self.target_acceleration
            self.ruckig_models["inp"][i].max_velocity = self.max_vel
            self.ruckig_models["inp"][i].max_acceleration = self.max_acc
            self.ruckig_models["inp"][i].max_jerk = self.max_jerk

            # For velocity and acceleration 'first-order' approximation #
            self.ruckig_models["prev_pose"].append(np.zeros(self.dof))
            self.ruckig_models["prev_vels"].append(np.zeros(self.dof))
            self.ruckig_models["prev_accs"].append(np.zeros(self.dof))
            self.ruckig_models["time_step"].append(1)                                       # Reset - at the beginning of each episode

    def predict(self, obs, deterministic=True):
        """
           given the observations from 'iiwa_ruckig_planar_grasping_dart_unity_env' type envs, generate time-optimal task-space velocities using the ruckig algorithm
                 Note: the agents will receive ruckig generated velocities, and then using IK - update_action() method - joint velocities will be sent to the unity simulator

           :param obs:           environments observation shape(num_envs, 5 or 6) - depending on the active DoF (rotation control is active)
                                    - Format: [reset, rz_ee_d, x_ee_d, y_ee_d, rz_box_d, x_box_d, y_box_d]
                                              - Should reset, rotation of ee in dart, x position of the ee in dart, etc
                                              - box pose is used only once (at reset_ruckig_model()) as the box does not move during the planar episode
                                              - Not normalized in [-1, 1] as in most gym agents
           :param deterministic: unused

           :return: action - shape(num_envs, 2 or 3). Important: the actions are not normalized in [-1, 1] as in the case of the PPO
        """
        action = np.zeros((self.envs.num_envs, self.dof)) # For each env

        # For each env generate the next action given the gym env observations - dart coords #
        for i in range(self.envs.num_envs): 

            # Reset the target ruckig pose - a new agent episode has started #
            # First "inp" is also set - refer to the function for more       #                         
            if(obs[i][0] == 1):
                self.reset_ruckig_model(obs[i], i)

            if self.ruckig_models["res"][i] == Result.Working:

                # Current pose of the ee # 
                curr_pose = np.asarray(obs[i][1:self.dof+1])

                curr_vels = self.ruckig_models["prev_vels"][i]
                curr_accs = self.ruckig_models["prev_accs"][i]

                # Velocity approx #
                if(self.ruckig_models["time_step"][i] >= 2):
                    curr_vels = (curr_pose - self.ruckig_models["prev_pose"][i]) / self.control_cycle

                # Acceleration approx #
                if(self.ruckig_models["time_step"][i] >= 3):
                    curr_accs = (curr_vels - self.ruckig_models["prev_vels"][i]) / self.control_cycle

                    ##################################
                    # Update - linear approximation  #
                    # Vel and acc are both available #
                    ##################################
                    self.ruckig_models["inp"][i].current_velocity = curr_vels
                    self.ruckig_models["inp"][i].current_acceleration = curr_accs

                # Save current state #
                self.ruckig_models["prev_pose"][i] = curr_pose
                self.ruckig_models["prev_vels"][i] = curr_vels
                self.ruckig_models["prev_accs"][i] = curr_accs

                # Get the next result given the current "inp". If "inp" is not expected, re-calculate a new trajectory #
                self.ruckig_models["res"][i] = self.ruckig_models["otg"][i].update(self.ruckig_models["inp"][i], self.ruckig_models["out"][i])

                # Get the new generated velocity #
                ruckig_vel = self.ruckig_models["out"][i].new_velocity

                #####################################################
                # Ip == out                                         #
                # Move to the next step of the generated trajectory #
                #####################################################
                self.ruckig_models["out"][i].pass_to_input(self.ruckig_models["inp"][i])

                action[i] = ruckig_vel # Save to action array for this env

            self.ruckig_models["time_step"][i] += 1

        return action, None

    def reset_ruckig_model(self, obs, index_model):
        """
           reset target task-space pose for ruckig

           :affects:   "inp" - current_position, target_position, target_velocity, target_acceleration
                       "res" - Working
                       "prev_pos", "prev_vels", "prev_accs", "time_step"

           :param obs: environment (single) observation shape(, 5 or 6) - depending on the active DoF 
                                   - Format: [reset, rz_ee_d, x_ee_d, y_ee_d, rz_box_d, x_box_d, y_box_d]
                                             - Should reset, rotation of ee in dart, x position of the ee in dart, etc
                                             - box pose is used to set the target planar pose (above the box)

           :param index_mode: which ruckig object to reset
        """
        ###############################
        # 3, or 2 DoF are controlled  #
        # Dart 0 -> rotation Zd = Yu  #
        # Dart 1 -> position Xd = Zu  #
        # Dart 2 -> position Yd = -Xu ########4444444444##################################
        # obs for 2 DoF -> [reset, rz_ee_d, x_ee_d, y_ee_d, x_box_d, y_box_d]            #
        # obs for 3 DoF- > [reset, rz_ee_d, x_ee_d, y_ee_d, rz_box_d, x_box_d, y_box_d]  #
        ################################################4444444444########################
        self.ruckig_models["inp"][index_model].current_position = obs[1:self.dof+1] 

        self.ruckig_models["inp"][index_model].target_position = obs[self.dof + 1:]
        self.ruckig_models["inp"][index_model].target_velocity = self.target_velocity
        self.ruckig_models["inp"][index_model].target_acceleration = self.target_acceleration
 
        self.ruckig_models["res"][index_model] = Result.Working

        self.ruckig_models["prev_pose"][index_model] = np.zeros(self.dof)
        self.ruckig_models["prev_vels"][index_model] = np.zeros(self.dof)
        self.ruckig_models["prev_accs"][index_model] = np.zeros(self.dof)
        self.ruckig_models["time_step"][index_model] = 1
