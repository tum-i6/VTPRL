"""
Class to enable vectorized environment which can contain multiple iiwa envs and communicate with Unity through gRPC.
Ideally you do not need to modify anything here and you should always use this environment as a wrapper of a single
iiwa environment in order to make it possible to communicate with unity and train RL algorithms which expect openAI gym
interface in SB3
"""
import json
import time
import roslibpy
import zmq
import grpc

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Optional, List, Union

from utils import service_pb2_grpc
from utils.service_pb2 import StepRequest
from utils.task_monitor import (
    build_monitor_spec,
    collect_monitor_data,
    MonitorSpec,
)
from utils.task_monitor_proxy import TaskMonitorController
from utils.task_monitor_web_proxy import TaskMonitorWebController
from utils.config_utils import first_manipulator_instance
from utils.step_profiler import StepProfiler
from utils.data_trace_schema import RecordingConfig, Channel
from utils.data_trace_proxy import DataTraceController

try:
    from utils_advanced.manual_actions import configure_manual_settings_and_get_manual_function
except:
    pass

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
        self.validation_envs = list(self.train_envs)
        self.envs = self.train_envs
        print("Number of envs: " + str(len(self.envs)))
        #self.envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]

        # Centralized task monitor controller (Qt window or Plotly web app)
        self.task_monitor: Optional[Union[TaskMonitorController, TaskMonitorWebController]] = None
        self._task_monitor_specs: Dict[int, MonitorSpec] = {}
        self._task_monitor_enabled = bool(self.gym.get('task_monitor', False))
        self._task_monitor_type = self.gym.get('task_monitor_type', 'qt').lower()

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
        elif self.communication_type == 'GRPC_SHM':
            # Hybrid gRPC + shared-memory transport
            self.channel = grpc.insecure_channel(self.ip_address + ":" + str(self.port_number))
            self.stub = service_pb2_grpc.CommunicationServiceStub(self.channel)
            from utils.shared_memory_transport import SharedMemoryTransport
            shm_dir = self.sim.get('shared_memory_directory', '/mnt/shm')
            shm_capacity = int(self.sim.get('shared_memory_capacity', 4))
            shm_slot_mb = int(self.sim.get('shared_memory_slot_size_mb', 64))
            grpc_timeout = self.sim.get('grpc_timeout_seconds', None)
            self.shm_transport = SharedMemoryTransport(
                stub=self.stub,
                shm_dir=shm_dir,
                capacity=shm_capacity,
                slot_size_mb=shm_slot_mb,
                same_kernel=bool(self.sim.get('shared_memory_same_kernel', False)),
                grpc_timeout=grpc_timeout,
            )
            print(f"[SHM] Shared-memory transport initialised (dir={shm_dir}, same_kernel={self.sim.get('shared_memory_same_kernel', False)})")
        elif self.communication_type == 'GRPC_BIN':
            # Binary (MessagePack) serialization over gRPC
            self.channel = grpc.insecure_channel(
                self.ip_address + ":" + str(self.port_number),
                options=[
                    ('grpc.max_receive_message_length', 64 * 1024 * 1024),  # 64 MB
                    ('grpc.max_send_message_length', 64 * 1024 * 1024),
                ],
            )
            self.stub = service_pb2_grpc.CommunicationServiceStub(self.channel)
            try:
                import msgpack
                self._msgpack = msgpack
            except ImportError:
                raise ImportError(
                    "GRPC_BIN mode requires the 'msgpack' package.  "
                    "Install it with: pip install msgpack"
                )
            print(f"[BIN-GRPC] Binary gRPC transport initialised")
        else:
            print("Please specify a supported communication mode: 'ROS', 'ZMQ', 'GRPC', 'GRPC_SHM', or 'GRPC_BIN'.")

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

        # ── Data-trace recorder (optional, runs in a child process) ────
        self._data_trace_recorder: Optional[DataTraceController] = None
        self._data_trace_enabled = False
        self._data_trace_skip_envs: set = set()
        trace_cfg_raw = self.gym.get('data_trace', None)
        if isinstance(trace_cfg_raw, dict) and trace_cfg_raw.get('enabled', False):
            try:
                # Map channel name strings from config to Channel enum members
                channel_names = trace_cfg_raw.get('channels', None)
                if channel_names is not None:
                    channels = {Channel[name.upper()] for name in channel_names}
                else:
                    channels = set(Channel)
                rec_cfg = RecordingConfig(
                    enabled_channels=channels,
                    trace_root=str(trace_cfg_raw.get('trace_root', 'traces')),
                    compress_arrays=bool(trace_cfg_raw.get('compress_arrays', True)),
                    max_episodes=trace_cfg_raw.get('max_episodes', None),
                    flush_interval_steps=int(trace_cfg_raw.get('flush_interval_steps', 200)),
                    record_env_ids=(
                        set(trace_cfg_raw['record_env_ids'])
                        if trace_cfg_raw.get('record_env_ids') is not None
                        else None
                    ),
                )
                self._data_trace_recorder = DataTraceController(rec_cfg)
                self._data_trace_enabled = True
                print(f"[DATA-TRACE] Recorder process started → {rec_cfg.trace_root}")
            except Exception as exc:
                print(f"[DATA-TRACE] Failed to initialise recorder: {exc}")

        # Step profiler (cross-language wall-clock tracing)
        self._profiler = StepProfiler(
            enabled=bool(self.sim.get('enable_profiling', False))
        )
        self._profiling_print_every_n = int(self.sim.get('profiling_print_every_n', 1))
        if self._profiler.enabled:
            print("[PROFILER] Step profiler enabled"
                  + (f" (printing every {self._profiling_print_every_n} steps)"
                     if self._profiling_print_every_n > 1 else ""))

        # End-effector info
        manip_cfg = first_manipulator_instance(self.manip_env)
        self.ee_enabled = bool(manip_cfg.get('enable_end_effector', False))
        self.ee_model = manip_cfg.get('end_effector_model', None)
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
        self._profiler.begin_step()

        if dart_convert:
            self._profiler.begin("python_action_conversion")
            actions_converted = []
            for env, action in zip(self.envs, actions):
                if hasattr(env, 'update_action') and callable(getattr(env, 'update_action')):
                    act = env.update_action(action)         # Convert agent action to UNITY format - joint space and tool action
                else:
                    act = action
                actions_converted.append(act)
            actions = actions_converted
            self._profiler.end("python_action_conversion")

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

            # ── Data-trace: record the final frame then close episodes ──
            if self._data_trace_enabled and self._data_trace_recorder is not None:
                for env in terminated_environments:
                    get_payload = getattr(env, 'get_monitor_payload', None)
                    if callable(get_payload):
                        try:
                            payload = get_payload()
                        except Exception:
                            payload = None
                        if payload is not None:
                            step_idx = getattr(env, 'time_step', self.current_step)
                            self._data_trace_recorder.record_step(env.id, payload, step_idx)
                    self._data_trace_recorder.end_episode(env.id)

            ###########################################
            # Reset gym envs and UNITY simulator envs #
            ###########################################
            [env.reset() for env in terminated_environments]
            observations_converted = self._send_reset_and_update(terminated_environments, time_step_update=True)

            # Render: Not advised to set 'enable_dart_viewer': True, during RL-training. Use it only for debugging #
            if(self.agent.get("simulation_mode") == 'train'):
                self.render()

            # ── Data-trace: begin fresh episodes and record first observation ──
            if self._data_trace_enabled and self._data_trace_recorder is not None:
                for env in terminated_environments:
                    self._data_trace_recorder.begin_episode(env.id)
                    get_payload = getattr(env, 'get_monitor_payload', None)
                    if callable(get_payload):
                        try:
                            payload = get_payload()
                        except Exception:
                            payload = None
                        if payload is not None:
                            step_idx = getattr(env, 'time_step', self.current_step)
                            self._data_trace_recorder.record_step(env.id, payload, step_idx)
                    self._data_trace_skip_envs.add(env.id)

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
        self._refresh_data_trace()

        self._profiler.end_step()
        self._profiler.print_report(every_n=self._profiling_print_every_n)

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

        Internally builds a list of command dictionaries and caches it so
        that binary transports (GRPC_BIN) can access the structured data
        directly via ``self._last_request_dicts`` without re-parsing JSON.

        Args:
            command (str): "ACTION" or "RESET".
            environments (List): Target environments.
            actions (List, optional): Per-env actions for ACTION requests.

        Returns:
            str: Serialized JSON array payload for Unity.
        """
        env_label = "manipulator_environment"
        try:
            mode = str(self.root.get('environment_mode', '')).lower()
            key = str(self.gym.get('env_key', '')).lower()
            if 'warehouse' in mode or 'warehouse' in key:
                env_label = "warehouse_environment"
        except Exception:
            pass

        command_dicts = []

        if command == "ACTION":
            for idx, (act, env) in enumerate(zip(actions, environments)):
                act_array = np.asarray(act, dtype=float)
                robot_count = int(getattr(env, 'robot_count', 1))
                if act_array.ndim == 1:
                    act_array = np.reshape(act_array, (1, -1))
                if act_array.ndim != 2:
                    raise ValueError(f"Action must be 1D or 2D array; got shape {act_array.shape}.")
                if act_array.shape[0] != robot_count:
                    raise ValueError(
                        f"Action batch must have {robot_count} rows; got {act_array.shape[0]}."
                    )

                robots = []
                for r_idx, row in enumerate(act_array):
                    robots.append({
                        'robot_index': r_idx,
                        'values': np.asarray(row, dtype=float).tolist()
                    })

                payload_value = json.dumps({
                    'env_id': int(env.id),
                    'robots': robots
                })
                command_dicts.append({
                    'id': int(env.id),
                    'environment': env_label,
                    'command': "ACTION",
                    'value': payload_value,
                })

        elif command == "RESET":
            #print("Time: " + str(time.time() - self.start))
            self.start = time.time()
            for idx, env in enumerate(environments):
                reset_value = self._normalize_reset_payload(env, env.reset_state, env_label)
                reset_string = json.dumps(reset_value)
                command_dicts.append({
                    'id': int(env.id),
                    'environment': env_label,
                    'command': "RESET",
                    'value': reset_string,
                })

        # Cache for binary transports (avoids JSON round-trip).
        self._last_request_dicts = command_dicts
        return json.dumps(command_dicts)

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
        elif self.communication_type == 'GRPC_SHM':
            # Shared-memory data plane with gRPC control plane
            self._profiler.begin("python_grpc_call")
            try:
                response_json = self.shm_transport.step(content)
            except Exception as e:
                self._profiler.end("python_grpc_call")
                raise RuntimeError(
                    f"SHM step failed: {e}. "
                    f"Check that the Unity server is running at {self.ip_address}:{self.port_number}."
                ) from e
            self._profiler.end("python_grpc_call")
            self._profiler.begin("python_response_decode")
            result = self._parse_result(response_json)
            self._profiler.end("python_response_decode")
            return result
        elif self.communication_type == 'GRPC_BIN':
            # Binary serialization with MessagePack over gRPC.
            # Use the pre-built command dicts cached by _create_request
            # to avoid a redundant JSON parse → msgpack re-encode round-trip.
            import base64
            commands = getattr(self, '_last_request_dicts', None)
            if commands is None:
                # Fallback if called without _create_request (shouldn't happen)
                _c = content.strip()
                if _c.endswith(',]'):
                    _c = _c[:-2] + ']'
                commands = json.loads(_c)

            self._profiler.begin("python_request_encode")
            msgpack_bytes = self._msgpack.packb(commands, use_bin_type=True)
            b64_data = base64.b64encode(msgpack_bytes).decode('ascii')
            self._profiler.end("python_request_encode")

            timeout_seconds = self.sim.get('grpc_timeout_seconds', None)
            call_kwargs = {}
            if timeout_seconds not in (None, 0):
                call_kwargs['timeout'] = timeout_seconds

            self._profiler.begin("python_grpc_call")
            try:
                reply = self.stub.step(StepRequest(data=b64_data), **call_kwargs)
            except grpc.RpcError as e:
                self._profiler.end("python_grpc_call")
                code = e.code() if hasattr(e, 'code') else None
                details = e.details() if hasattr(e, 'details') else str(e)
                raise RuntimeError(
                    f"GRPC_BIN step failed (code={code}): {details}. "
                    f"Check that the Unity server is running at {self.ip_address}:{self.port_number}."
                ) from e
            self._profiler.end("python_grpc_call")

            # Response is Base64-encoded MessagePack
            self._profiler.begin("python_response_decode")
            resp_bytes = base64.b64decode(reply.data)
            result = self._msgpack.unpackb(resp_bytes, raw=False)
            self._profiler.end("python_response_decode")
            return result
        else:
            timeout_seconds = self.sim.get('grpc_timeout_seconds', None)
            call_kwargs = {}
            if timeout_seconds not in (None, 0):
                call_kwargs['timeout'] = timeout_seconds
            self._profiler.begin("python_grpc_call")
            try:
                reply = self.stub.step(StepRequest(data=content), **call_kwargs)
            except grpc.RpcError as e:
                self._profiler.end("python_grpc_call")
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
            self._profiler.end("python_grpc_call")
            self._profiler.begin("python_response_decode")
            result = self._parse_result(reply.data)
            self._profiler.end("python_response_decode")
            return result

    def _parse_result(self, result):
        """Parse Unity response (JSON only).

        Args:
            result (Any): Raw response from transport (string/bytes/object).

        Returns:
            Any: Parsed Python object (usually list of per-env observations).
        """
        if self.communication_type == 'ROS':
            data = result['value']
            return json.loads(data)
        elif self.communication_type == 'ZMQ':
            data = result.decode("utf-8")
            return json.loads(data)
        else:
            data = result
            return json.loads(data)

    def _extract_action_and_robot_index(self, env, action):
        """Normalize an action payload to a per-robot batch.

        Args:
            env: Environment instance to pull default robot count from.
            action: Incoming action payload from the agent.
        Returns:
            Tuple[List[List[float]], bool]: (per-robot actions, is_multi_robot)
        """
        robot_count = int(getattr(env, 'robot_count', 1))
        arr = np.asarray(action, dtype=float)
        if arr.ndim == 1:
            arr = np.reshape(arr, (1, -1))
        if arr.ndim != 2:
            raise ValueError(
                f"Action must be 1D or 2D; got shape {arr.shape} from {type(action).__name__}."
            )
        if arr.shape[0] != robot_count:
            raise ValueError(
                f"Action batch must have {robot_count} rows; got {arr.shape[0]} from {type(action).__name__}."
            )
        return arr.tolist(), arr.shape[0] > 1

    def _normalize_observation_payload(self, parsed, env):
        """Normalize Unity observation payloads to a legacy-compatible dict.

        The simulator now returns EnvironmentObservationModel objects with
        per-robot observations. This helper selects a robot payload and flattens
        the structure to mimic the legacy fields expected by env.update().

        Args:
            parsed: Parsed observation object (dict/list) from Unity.
            env: Target environment instance.
        Returns:
            dict or original parsed object suitable for env.update().
        """
        env_key = str(self.gym.get('env_key', '')).lower()
        if "warehouse" in env_key:
            if not isinstance(parsed, dict) or 'Robots' not in parsed:
                raise ValueError("Warehouse observations must include 'Robots' payloads for multi-robot mode.")
        return parsed

    def _normalize_reset_payload(self, env, reset_value, env_label):
        """Normalize reset payloads to the JSON protocol expected by Unity.

        Warehouse environments already emit structured dict payloads. Legacy
        manipulator/gripper environments may provide a flat list; convert that
        list into the multi-robot JSON schema used by StateModel.

        Args:
            env: Environment instance being reset.
            reset_value: Raw reset payload produced by ``env.reset()``.
            env_label (str): Unity environment label (e.g.
                ``"manipulator_environment"`` or ``"warehouse_environment"``).

        Returns:
            dict: Normalized reset payload dictionary ready for JSON
            serialization.

        Raises:
            ValueError: If ``reset_value`` is not a dictionary.
        """
        if isinstance(reset_value, dict):
            return reset_value

        raise ValueError(
            f"Reset payload must be a JSON dict for env id={getattr(env, 'id', '?')}; "
            f"got {type(reset_value).__name__}."
        )

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

        # ── Data-trace: begin episodes on full reset ─────────────
        if self._data_trace_enabled and self._data_trace_recorder is not None:
            for env in self.envs:
                self._data_trace_recorder.begin_episode(env.id)

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

        # Normalize actions for multi-robot batch control
        prepared_actions = []
        for env, action in zip(self.envs, actions):
            act_values, _ = self._extract_action_and_robot_index(env, action)

            if env.collided_env == 1:
                try:
                    act_values = np.zeros((int(getattr(env, 'robot_count', 1)), self._action_dim()))
                except Exception:
                    act_values = np.zeros((int(getattr(env, 'robot_count', 1)), action_dim))

            prepared_actions.append(act_values)

            try:
                env._monitor_last_action = np.asarray(act_values, dtype=float)
            except Exception:
                env._monitor_last_action = act_values

        #print("current step:" + str(self.current_step))
        # create request containing all environments with the actions to be executed

        #####################################################################################################
        # Execute a UNITY simulation step for all environments and parse the returned observations (result) #
        #####################################################################################################
        self._profiler.begin("python_create_request")
        request = self._create_request("ACTION", self.envs, prepared_actions)
        self._profiler.end("python_create_request")

        self._profiler.begin("python_send_request")
        observations = self._send_request(request)
        self._profiler.end("python_send_request")

        # Extract Unity-side profiling data from the first environment payload
        self._extract_unity_profiling(observations)

        ##########################################################################################
        # Update the envs using the obs returned from UNITY (dart chain of the manipulator, etc) #
        ##########################################################################################
        observations_converted = []
        terminated_environments = []                                                             # Assume done in the same timestep else move it outside the for
        dones = []
        infos = []

        self._profiler.begin("python_update_envs")
        observations_converted, rews, dones, infos = self._update_envs(observations, time_step_update=True)
        self._profiler.end("python_update_envs")
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

        actions = []
        for env in envs:
            robot_count = int(getattr(env, 'robot_count', 1))
            actions.append(np.zeros((robot_count, action_dim), dtype=float))
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
            list: Transposed env stack in the form
            ``[observations, rewards, dones, infos]`` where each element is a
            per-environment list.
        """
        env_stack = []
        for obs, env in zip(observations, self.envs):
            parsed = obs
            if isinstance(obs, (str, bytes)):
                try:
                    parsed = json.loads(obs)
                except Exception:
                    raise ValueError(f"Failed to parse observation for env id={env.id}: {obs!r}")
            parsed = self._normalize_observation_payload(parsed, env)
            env_stack.append(env.update(parsed, time_step_update))

        return [list(param) for param in zip(*env_stack)]

    def _extract_unity_profiling(self, observations) -> None:
        """Extract C# profiling data from the first environment's payload.

        The Unity simulator attaches ``ProfilingData`` to the first
        environment observation when profiling is enabled.  The data is
        a dictionary mapping section labels to elapsed milliseconds.

        For GRPC_BIN the observations are already dicts (MsgPack-decoded).
        For GRPC / GRPC_SHM, each observation is a JSON string that
        ``InjectProfilingJson`` has enriched with a ``ProfilingData`` key.

        Args:
            observations (list): Raw parsed observations — a list of dicts
                (GRPC_BIN) or JSON strings (GRPC / GRPC_SHM).

        Returns:
            None
        """
        if not self._profiler.enabled or not observations:
            return
        try:
            first = observations[0]
            if isinstance(first, (str, bytes)):
                first = json.loads(first)

            profiling = None
            # GRPC_BIN returns EnvironmentObservationModel dicts directly.
            # ProfilingData may be at the top level or nested under
            # 'Observations' → first robot payload.
            if isinstance(first, dict):
                profiling = first.get('ProfilingData') or first.get('profilingData')
                if profiling is None:
                    # Try nested under first robot observation
                    obs_list = first.get('Observations') or first.get('observations') or []
                    if obs_list and isinstance(obs_list, list) and isinstance(obs_list[0], dict):
                        profiling = obs_list[0].get('ProfilingData') or obs_list[0].get('profilingData')

            if profiling and isinstance(profiling, dict):
                self._profiler.merge_unity_timings(profiling)
        except Exception as exc:
            if not getattr(self, '_profiling_extraction_warned', False):
                print(f"[PROFILER] Warning: failed to extract Unity profiling data: {exc}")
                self._profiling_extraction_warned = True

    def _ensure_task_monitor_initialized(self):
        """Initialize the task monitor controller on first use.

        Returns:
            None
        """
        if not self._task_monitor_enabled or self.task_monitor is not None:
            return

        controller = None
        try:
            if self._task_monitor_type == 'web':
                controller = TaskMonitorWebController()
            else:
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

            # Forward spec configs to data-trace recorder so the player
            # can reconstruct full monitor panels during offline replay.
            if self._data_trace_enabled and self._data_trace_recorder is not None:
                for env_id, spec in specs.items():
                    self._data_trace_recorder.register_spec(env_id, {
                        "name": spec.name,
                        "type": spec.type,
                        "config": spec.config,
                    })
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

    def _refresh_data_trace(self):
        """Record one step of telemetry for each environment into the data trace.

        Obtains the raw :class:`~utils.telemetry.MonitorPayload` from every
        environment that exposes ``get_monitor_payload()`` and feeds it to the
        :class:`DataTraceRecorder`. Environments disabled by recorder config,
        environments without ``get_monitor_payload()``, and payload failures are
        silently skipped.

        Terminated environments that already had their final frame recorded
        and their episode closed inside :meth:`step` are skipped via the
        ``_data_trace_skip_envs`` set.

        Returns:
            None
        """
        if not self._data_trace_enabled or self._data_trace_recorder is None:
            return

        skip = self._data_trace_skip_envs
        for env in self.envs:
            if env.id in skip:
                continue
            get_payload = getattr(env, 'get_monitor_payload', None)
            if not callable(get_payload):
                continue
            try:
                payload = get_payload()
            except Exception:
                continue
            if payload is None:
                continue
            step_idx = getattr(env, 'time_step', self.current_step)
            self._data_trace_recorder.record_step(env.id, payload, step_idx)
        skip.clear()

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
            elif self.communication_type == 'GRPC_SHM':
                if hasattr(self, 'shm_transport') and self.shm_transport is not None:
                    self.shm_transport.close()
                if hasattr(self, 'channel') and self.channel is not None:
                    self.channel.close()
            elif self.communication_type == 'GRPC_BIN':
                if hasattr(self, 'channel') and self.channel is not None:
                    self.channel.close()
            if self.task_monitor:
                try:
                    self.task_monitor.close()
                finally:
                    self.task_monitor = None
                    self._task_monitor_specs.clear()
                    self._task_monitor_enabled = False
            if self._data_trace_recorder is not None:
                try:
                    for env in self.envs:
                        self._data_trace_recorder.end_episode(env.id)
                    self._data_trace_recorder.close()
                finally:
                    self._data_trace_recorder = None
                    self._data_trace_enabled = False
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
        if self.envs:
            sample_env = self.envs[0]
            try:
                if hasattr(sample_env, 'n_links'):
                    base_dim = int(getattr(sample_env, 'n_links', 0))
                    has_gripper = bool(getattr(sample_env, 'with_gripper', False))
                    if base_dim > 0:
                        return base_dim + (1 if has_gripper else 0)
            except Exception:
                pass
        raise ValueError(f"Unable to infer action dimension for env_key='{env_key}'.")

    # Calling destructor
    # def __del__(self):
    #     print("Destructor called")
    #     self.env_process.terminate()
