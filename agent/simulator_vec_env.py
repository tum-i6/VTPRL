"""
Class to enable vectorized environment which can contain multiple iiwa envs and communicate with Unity through gRPC.
Ideally you do not need to modify anything here and you should always use this environment as a wrapper of a single
iiwa environment in order to make it possible to communicate with unity and train RL algorithms which expect openAI gym
interface in SB3
"""
import ast
import json
import time
import roslibpy
import zmq
import grpc

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Optional, List

from utils import service_pb2_grpc
from utils.service_pb2 import StepRequest
from utils.task_monitor import (
    build_monitor_spec,
    collect_monitor_data,
    MonitorSpec,
)
from utils.task_monitor_proxy import TaskMonitorController

try:
    from utils_advanced.manual_actions import configure_manual_settings_and_get_manual_function
except:
    pass

class CommandModel:

    """Lightweight command envelope for gRPC/ZMQ/ROS requests."""

    def __init__(self, id, command, env_key, value):
        """Create a command payload.

        Args:
            id (int): Environment identifier.
            command (str): Command verb, e.g., "ACTION" or "RESET".
            env_key (str): Environment label understood by Unity.
            value (Any): Command payload (action list or reset string).
        """
        self.id = id
        self.environment = env_key
        self.command = command
        self.value = value

class SimulatorVecEnv(DummyVecEnv):
    """Vectorized Unity bridge used by SB3-compatible agents."""
    _client = None

    def __init__(
        self,
        env_fns,
        agent_dict,
        root_dict,
        simulation_dict,
        gym_environment_dict,
        manipulator_environment_dict,
        reward_dict,
        manual_actions_dict=None,
        observation_dict=None,
        spaces=None,
    ):
        """Instantiate the vectorized Unity environment wrapper.

        Args:
            env_fns (List[callable]): Factories producing single-env instances.
            agent_dict (dict): Agent-level configuration.
            root_dict (dict): Root configuration (global/environment mode).
            simulation_dict (dict): Communication/simulator configuration.
            gym_environment_dict (dict): Gym-specific configuration.
            manipulator_environment_dict (dict): Manipulator-specific config.
            reward_dict (dict): Reward configuration.
            manual_actions_dict (dict, optional): Manual action configuration.
            observation_dict (dict, optional): Observation configuration.
            spaces (gym.Space, optional): Optional space override.
        """
        DummyVecEnv.__init__(self, env_fns)
        # self.env_process = subprocess.Popen(
        #     'J:/NoSync/Data/Code/prototype2/BUilds/Windows/Mono/ManipulatorEnvironment_v0_6/Unity3D.exe '
        #     + config['command_line_params'] + " -pn "+ str(config["port_number"]),
        #     stdout=PIPE, stderr=PIPE, stdin=PIPE,
        #     cwd='J:/NoSync/Data/Code/prototype2/BUilds/Windows/Mono/ManipulatorEnvironment_v0_6',
        #     shell=False)

        self.current_step = 0
        self.agent = agent_dict
        self.root = root_dict
        self.sim = simulation_dict
        self.gym = gym_environment_dict
        self.manip_env = manipulator_environment_dict
        self.observation_cfg = observation_dict or {}
        self.reward_dict = reward_dict
        self.communication_type = self.sim['communication_type']
        self.port_number = self.sim['port_number']
        print("Port number: " + str(self.port_number))
        self.ip_address = self.sim['ip_address']
        print("Ip address: " + str(self.ip_address))
        self.start = 0
        self.nenvs = len(env_fns)
        self.train_envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        # self.validation_envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        self.envs = self.train_envs
        print("Number of envs: " + str(len(self.envs)))
        #self.envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]

        # Centralized task monitor controller (single Qt window)
        self.task_monitor: Optional[TaskMonitorController] = None
        self._task_monitor_specs: Dict[int, MonitorSpec] = {}
        self._task_monitor_enabled = bool(self.gym.get('task_monitor', False))

        # Initial position flag for the manipulator/robot after reseting. 1 means different than the default vertical position #
        manip_gym = self.gym.get('manipulator_gym_environment', {})
        if (manip_gym.get("initial_positions") is None or np.count_nonzero(manip_gym.get("initial_positions", [])) == 0) and manip_gym.get("random_initial_joint_positions", False) == False:
            self.flag_zero_initial_positions = 0
        else:
            self.flag_zero_initial_positions = 1
        if self.root.get("environment_mode") == "Warehouse":
            self.flag_zero_initial_positions = 1

        if self.communication_type == 'ROS':
            # Connect to ROS server
            if SimulatorVecEnv._client is None:
                SimulatorVecEnv._client = roslibpy.Ros(host=self.ip_address, port=int(self.port_number))
                SimulatorVecEnv._client.run()
            self.service = roslibpy.Service(SimulatorVecEnv._client, '/step', 'rosapi/GetParam')
            self.request = roslibpy.ServiceRequest([['name', 'none'], ['default', 'none']])
        elif self.communication_type == 'ZMQ':
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://127.0.0.1:" + str(self.port_number))
        elif self.communication_type == 'GRPC':
            # Create gRPC channel and stub; defer connectivity checks to first call with timeout
            self.channel = grpc.insecure_channel(self.ip_address + ":" + str(self.port_number))
            self.stub = service_pb2_grpc.CommunicationServiceStub(self.channel)
        else:
            print("Please specify a supported communication mode: 'ROS', 'ZMQ', or 'GRPC'.")

        ###################################################################
        # Manual/hard-coded actions to command at the end of the episode  #
        # Controlled by config_advanced: only enabled if provided.        #
        ###################################################################
        self.manual = False
        self.manual_behaviour = None
        self.manual_rewards = False

        # If not passed explicitly, try to discover from Config (advanced)
        candidate_manual = manual_actions_dict
        try:
            if candidate_manual is None and isinstance(self.agent, dict):
                candidate_manual = self.agent.get('manual_actions_dict', None)
        except Exception:
            candidate_manual = manual_actions_dict

        if isinstance(candidate_manual, dict) and candidate_manual.get('manual', False):
            self.manual = True
            self.manual_behaviour = candidate_manual.get("manual_behaviour")        # Behaviour to excecute, 'planar_grasping' (down/close/up) or 'close_gripper' (close)
            self.manual_rewards = candidate_manual.get("manual_rewards", False)     # True/False -> whether to add rewards/penalties during manual actions
            # Set-up manual actions and return a function which will be called after the end of the agent episode
            self.manual_func = configure_manual_settings_and_get_manual_function(self, candidate_manual)

        # End-effector info
        self.ee_enabled = bool(self.manip_env.get('enable_end_effector', False))
        self.ee_model = self.manip_env.get('end_effector_model', None)
        if not self.ee_enabled or self.ee_model in (None, 'None'):
            print("The robot has no tool attached to the end-effector")
        else:
            print(f"End-effector enabled: {self.ee_model}")
        

    def switch_to_training(self):
        """Route operations to the training environment list.

        Returns:
            None
        """
        self.envs = self.train_envs

    def switch_to_validation(self):
        """Route operations to the validation environment list.

        Returns:
            None
        """
        self.envs = self.validation_envs

    def step(self, actions, dart_convert=True):
        """
        Advance all Unity environments one step.

        Args:
            actions (List[List[float]]): Per-env actions. Each sublist entails actions that will be transfered to each corresponding env. 
                These actions originate either from a model-based controller or from the output of a neural network of an RL model.
            dart_convert (bool): If True, convert task-space actions to joint space via IK. The UNITY simulator always expects velocities in joint space.

        Returns:
            tuple: observations, rewards, dones, infos stacked across envs.
        """

        self.current_step += 1

        if dart_convert and 'dart' in str(self.gym.get('env_key', '')):
            actions_converted = []
            for env, action in zip(self.envs, actions):
                act = env.update_action(action)         # Convert agent action to UNITY format - joint space and tool action
                actions_converted.append(act)
            actions = actions_converted

        ####################################################################################################
        # Send actions to UNITY (JSON message) and update the envs with the new observation                #
        # after executing these actions (update the dart kinematic (robotic) chain, agents time-step, etc) #
        ####################################################################################################
        terminated_environments, rews, infos, observations_converted, dones = self._send_actions_and_update(actions)

        # Render: Not advised to set 'enable_dart_viewer': True, during RL-training. Use it only for debugging #
        if(self.agent.get("simulation_mode") == 'train'):
            self.render()

        ##########################################################################################################
        # TODO: Make sure the last observation for terminated environment is the correct one:                    #
        #       https://github.com/hill-a/stable-baselines/issues/400 talks about reset observation being wrong, #
        #       use terminated observation instead                                                               #
        # TODO: make sure this does not cause problems, when the when images are transfered it might be slow     #
        #       to get all environment observations again                                                        #   
        ##########################################################################################################

        # Reset all the terminated environments #
        if len(terminated_environments) > 0:

            ####################################################################################
            # Manual actions at the end of the agent episode.                                  #
            # Disabled by default. Refer to utils_advanced/                                    #
            # Note: the envs should terminate at the same timestep due to UNITY sychronization #
            #       for collided envs -> send zero velocities for the remaining steps          #
            ####################################################################################
            if(self.manual == True): 
                self.manual_func(self, rews, infos, self.manual_rewards)

            # Successful episode: give terminal reward # 
            for env in terminated_environments:
                if env.get_terminal_reward():
                    rews[env.id] += self.reward_dict["reward_terminal"]
                    infos[env.id]["success"] = True

            ###########################################
            # Reset gym envs and UNITY simulator envs #
            ###########################################
            [env.reset() for env in terminated_environments]
            observations_converted = self._send_reset_and_update(terminated_environments, time_step_update=True)

            # Render: Not advised to set 'enable_dart_viewer': True, during RL-training. Use it only for debugging #
            if(self.agent.get("simulation_mode") == 'train'):
                self.render()

            ########################################################################################
            # The manipulator resets to non-zero initial joint positions.                          #
            # Correct the UNITY observation and update the dart chain due to UNITY synchronization #
            # Important: Make sure the agent episodes terminate at the same time-step in this case #           
            ########################################################################################
            if(self.flag_zero_initial_positions == 1):
                observations_converted = self._send_zero_vel_and_update(self.envs, True)

        # Cache rewards for monitor panels and push latest telemetry
        for env, reward in zip(self.envs, rews):
            try:
                env._last_reward = float(reward)
            except Exception:
                env._last_reward = reward

        self._refresh_task_monitor()

        return np.stack(observations_converted), np.stack(rews), np.stack(dones), infos

    def step_wait(self):
        """Execute deferred actions (SB3 compatibility).

        Returns:
            tuple: observations, rewards, dones, infos stacked across envs.
        """
        # only because VecFrameStack uses step_async to provide the actions, then step_wait to execute a step
        return self.step(self.actions)

    def _create_request(self, command, environments, actions=None):
        """Create a serialized request for Unity.

        Args:
            command (str): "ACTION" or "RESET".
            environments (List): Target environments.
            actions (List, optional): Per-env actions for ACTION requests.

        Returns:
            str: Serialized JSON array payload for Unity.
        """
        content = ''
        env_label = "manipulator_environment"
        try:
            mode = str(self.root.get('environment_mode', '')).lower()
            key = str(self.gym.get('env_key', '')).lower()
            if 'warehouse' in mode or 'warehouse' in key:
                env_label = "warehouse_environment"
        except Exception:
            pass

        if command == "ACTION":
            for act, env in zip(actions, environments):
                # print("id: {}\t step: {}\t action = {}".format(env.id, env.ts, str(np.around(act, decimals=3))))

                act_json = json.dumps(CommandModel(env.id, "ACTION", env_label, str(act.tolist())), default=lambda x: x.__dict__)
                content += (act_json + ",")

        elif command == "RESET":
            #print("Time: " + str(time.time() - self.start))
            self.start = time.time()
            for env in environments:
                reset_string = str(env.reset_state)
                act_json = json.dumps(CommandModel(env.id, "RESET", env_label, reset_string), default=lambda x: x.__dict__)
                content += (act_json + ",")

        return '[' + content + ']'

    def _send_request(self, content):
        """Dispatch the serialized request over the configured transport.

        Args:
            content (str): Serialized JSON payload destined for Unity.

        Returns:
            Any: Parsed Unity response (list/dict) depending on transport.
        """

        # "{\"Environment\":\"manipulator\",\"Action\":\"" + translated_action + "\"}"
        if self.communication_type == 'ROS':
            self.request['name'] = content
            return self._parse_result(self.service.call(self.request))
        elif self.communication_type == 'ZMQ':
            self.socket.send_string(content)
            response = self.socket.recv()
            return self._parse_result(response)
        else:
            timeout_seconds = self.sim.get('grpc_timeout_seconds', None)
            call_kwargs = {}
            if timeout_seconds not in (None, 0):
                call_kwargs['timeout'] = timeout_seconds
            try:
                reply = self.stub.step(StepRequest(data=content), **call_kwargs)
            except grpc.RpcError as e:
                code = e.code() if hasattr(e, 'code') else None
                details = e.details() if hasattr(e, 'details') else str(e)
                if timeout_seconds not in (None, 0) and code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise RuntimeError(
                        f"gRPC step timed out after {timeout_seconds} seconds. "
                        f"Verify that the Unity server at {self.ip_address}:{self.port_number} is running and unpaused."
                    ) from e
                raise RuntimeError(
                    f"gRPC step failed (code={code}): {details}. "
                    f"Check that the Unity server is running at {self.ip_address}:{self.port_number} and reachable."
                ) from e
            return self._parse_result(reply.data)

    def _parse_result(self, result):
        """Parse Unity response (supports JSON or Python-literal formats).

        Args:
            result (Any): Raw response from transport (string/bytes/object).

        Returns:
            Any: Parsed Python object (usually list of per-env observations).
        """
        if self.communication_type == 'ROS':
            data = result['value']
            try:
                return ast.literal_eval(data)
            except Exception:
                return json.loads(data)
        elif self.communication_type == 'ZMQ':
            data = result.decode("utf-8")
            try:
                return ast.literal_eval(data)
            except Exception:
                return json.loads(data)
        else:
            data = result
            try:
                return ast.literal_eval(data)
            except Exception:
                return json.loads(data)

    def reset(self, should_reset=True):
        """Reset all environments locally and in Unity.

        Args:
            should_reset (bool): Whether to call the local env reset before Unity reset.

        Returns:
            np.ndarray: Initial observations after reset.
        """
        if should_reset:
            [env.reset() for env in self.envs]

        # Reset UNITY environments and update the agent envs (dart chain) #
        self._send_reset_and_update(self.envs, time_step_update=False)

        # Correct dart chain #
        observations_converted = self._send_zero_vel_and_update(self.envs, True)

        for env in self.envs:
            env._last_reward = 0.0

        self._refresh_task_monitor()

        return np.array(observations_converted)

    ###########
    # Helpers #
    ###########
    def _send_reset_and_update(self, envs, time_step_update=True):
        """Send RESET to Unity and update local envs.

        Args:
            envs (List): Environments to reset.
            time_step_update (bool): Whether to advance local timesteps.

        Returns:
            list: Converted observations for the provided envs.
        """

        # UNITY #
        request = self._create_request("RESET", envs)
        observations = self._send_request(request)

        # Agents #
        observations_converted, _, _, _ = self._update_envs(observations, time_step_update=time_step_update)

        return observations_converted

    def _send_actions_and_update(self, actions):
        """Send ACTIONs to Unity and update local envs with returned observations.
        Note: zero velocities are sent for collided envs.

        Args:
            actions (List[List[float]]): Per-env actions in Unity format.

        Returns:
            tuple: (terminated_envs, rewards, infos, observations, dones).
                terminated_envs (list): Envs that finished this step.
                rewards (list): Per-env reward scalars.
                infos (list): Per-env info dictionaries.
                observations (list): Per-env observations converted to agent format.
                dones (list): Per-env done flags.
        """

        rews = [0] * len(self.envs)

        action_dim = self._action_dim()

        # For collided envs send zero velocities #
        for env in self.envs:
            if env.collided_env == 1:
                actions[env.id] = np.zeros(action_dim)

        for env, action in zip(self.envs, actions):
            try:
                env._monitor_last_action = np.asarray(action, dtype=float)
            except Exception:
                env._monitor_last_action = action

        #print("current step:" + str(self.current_step))
        # create request containing all environments with the actions to be executed

        #####################################################################################################
        # Execute a UNITY simulation step for all environments and parse the returned observations (result) #
        #####################################################################################################
        request = self._create_request("ACTION", self.envs, actions)
        observations = self._send_request(request)

        ##########################################################################################
        # Update the envs using the obs returned from UNITY (dart chain of the manipulator, etc) #
        ##########################################################################################
        observations_converted = []
        terminated_environments = []                                                             # Assume done in the same timestep else move it outside the for
        dones = []
        infos = []

        observations_converted, rews, dones, infos = self._update_envs(observations, time_step_update=True)
        for env, done in zip(self.envs, dones): # Scan for terminated envs
            if done is True:
                terminated_environments.append(env)

        return terminated_environments, rews, infos, observations_converted, dones

    def _send_zero_vel_and_update(self, envs, time_step_update=True):
        """Send zero velocities to Unity, then update local envs.
        This corrects the articulation chain when robots reset to non-default poses.

        Args:
            envs (List): Target environments.
            time_step_update (bool): Whether to advance local timesteps.

        Returns:
            list: Converted observations.
        """
        # Send zero velocities to the UNITY envs  #
        action_dim = self._action_dim()

        actions = np.zeros((len(envs), action_dim))
        request = self._create_request("ACTION", envs, actions)
        observations = self._send_request(request)

        # Update the agents envs #
        observations_converted, _, _, _ = self._update_envs(observations, time_step_update=time_step_update)

        return observations_converted

    def _update_envs(self, observations, time_step_update):
        """Update local envs using Unity observations.

        Args:
            observations (Iterable): Raw Unity observations (may be strings or dicts).
            time_step_update (bool): Whether to advance local timesteps.

        Returns:
            list: Transposed env stack [observations, rewards, dones, infos].
        """
        env_stack = []
        for obs, env in zip(observations, self.envs):
            parsed = obs
            if isinstance(obs, (str, bytes)):
                # Expect legacy Python-literal string from Unity; fall back to JSON if needed
                try:
                    parsed = ast.literal_eval(obs)
                except Exception:
                    try:
                        parsed = json.loads(obs)
                    except Exception:
                        raise ValueError(f"Failed to parse observation for env id={env.id}: {obs!r}")
            env_stack.append(env.update(parsed, time_step_update))

        return [list(param) for param in zip(*env_stack)]

    def _ensure_task_monitor_initialized(self):
        """Initialize the task monitor controller on first use.

        Returns:
            None
        """
        if not self._task_monitor_enabled or self.task_monitor is not None:
            return

        controller: Optional[TaskMonitorController] = None
        try:
            controller = TaskMonitorController()
            specs: Dict[int, MonitorSpec] = {}
            for env in self.envs:
                spec = build_monitor_spec(env, self.observation_cfg)
                if spec is None:
                    continue
                if controller.register_environment(spec):
                    specs[env.id] = spec

            if not specs:
                controller.close()
                self._task_monitor_enabled = False
                return

            self.task_monitor = controller
            self._task_monitor_specs = specs
        except Exception as exc:
            print(f"Task monitor initialization failed: {exc}")
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    pass
            self.task_monitor = None
            self._task_monitor_specs = {}
            self._task_monitor_enabled = False

    def _refresh_task_monitor(self):
        """Push latest telemetry to the task monitor UI.

        Returns:
            None
        """
        if not self._task_monitor_enabled:
            return

        self._ensure_task_monitor_initialized()

        if not self.task_monitor:
            return

        for env in self.envs:
            if env.id not in self._task_monitor_specs:
                continue
            data = collect_monitor_data(env)
            if data is None:
                continue
            try:
                self.task_monitor.update_environment(env.id, data)
            except Exception as exc:
                print(f"Task monitor update failed for env {env.id}: {exc}")
                try:
                    self.task_monitor.close()
                except Exception:
                    pass
                self.task_monitor = None
                self._task_monitor_specs.clear()
                self._task_monitor_enabled = False
                break

    ###############
    # End helpers #
    ###############

    def reset_task(self):
        """Placeholder to satisfy VecEnv API; no-op here.

        Returns:
            None
        """
        pass

    def close(self):
        """Close transports and task monitor resources.

        Returns:
            None
        """
        # Cleanly close the underlying transport if applicable
        try:
            if self.communication_type == 'ROS' and SimulatorVecEnv._client is not None:
                SimulatorVecEnv._client.terminate()
            elif self.communication_type == 'ZMQ':
                try:
                    self.socket.close(0)
                finally:
                    self.context.term()
            elif self.communication_type == 'GRPC':
                if hasattr(self, 'channel') and self.channel is not None:
                    self.channel.close()
            if self.task_monitor:
                try:
                    self.task_monitor.close()
                finally:
                    self.task_monitor = None
                    self._task_monitor_specs.clear()
                    self._task_monitor_enabled = False
        except Exception:
            # Swallow close-time exceptions to avoid masking teardown
            pass

    def __len__(self):
        """Number of vectorized environments.

        Returns:
            int: Count of managed environments.
        """
        return self.nenvs

    def render(self, mode=None):
        """
            Override default vectorized render behaviour.
            SB3 renders all envs into a single window; instead render each env into separate windows.

            Args:
                mode (Any): Unused; kept for API compatibility.

            Returns:
                None
        """
        if(self.gym.get("env_key") != 'iiwa_joint_vel'):
            for env in self.envs:
                if hasattr(env, 'dart_sim') and getattr(env.dart_sim, 'enable_viewer', False): # Render is active
                    env.render()

    def _action_dim(self) -> int:
        """Compute action dimension based on env key and EE config.

        Returns:
            int: Dimension of the action vector expected by Unity.
        """
        env_key = str(self.gym.get('env_key', ''))
        if "iiwa" in env_key:
            has_gripper = self.ee_enabled and self.ee_model in ('ROBOTIQ_2F85', 'ROBOTIQ_3F', 'DEFAULT_GRIPPER')
            if env_key == 'iiwa_joint_vel':
                return int(self.gym.get('num_joints', 7)) + (1 if has_gripper else 0)
            return 8 if has_gripper else 7
        if "so100" in env_key:
            return 6
        if "warehouse" in env_key:
            return 2
        return 0

    # Calling destructor
    # def __del__(self):
    #     print("Destructor called")
    #     self.env_process.terminate()
