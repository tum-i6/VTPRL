"""
Class to enable vectorized environment which can contain multiple iiwa envs and communicate with Unity through gRPC.
Ideally you do not need to modify anything here and you should always use this environment as a wrapper of a single
iiwa environment in order to make it possible to communicate with unity and train RL algorithms which expect openAI gym
interface in SB3
"""
import ast
import base64
import json
import time

import numpy as np
import roslibpy
#import torch
from stable_baselines3.common.vec_env import DummyVecEnv
import subprocess
from subprocess import PIPE
import zmq
import grpc
from misc import service_pb2_grpc
from misc.service_pb2 import StepRequest


class CommandModel:

    def __init__(self, id, command, env_key, value):
        self.id = id
        self.environment = env_key
        self.command = command
        self.value = value


class SimulatorVecEnv(DummyVecEnv):
    _client = None

    def __init__(self, env_fns, config, spaces=None ):
        """
        envs: list of environments to create
        """
        DummyVecEnv.__init__(self, env_fns)
        # self.env_process = subprocess.Popen(
        #     'J:/NoSync/Data/Code/prototype2/BUilds/Windows/Mono/ManipulatorEnvironment_v0_6/Unity3D.exe '
        #     + config['command_line_params'] + " -pn "+ str(config["port_number"]),
        #     stdout=PIPE, stderr=PIPE, stdin=PIPE,
        #     cwd='J:/NoSync/Data/Code/prototype2/BUilds/Windows/Mono/ManipulatorEnvironment_v0_6',
        #     shell=False)

        self.current_step = 0
        self.config = config
        self.communication_type = config['communication_type']
        self.port_number = config['port_number']
        print(config["port_number"])
        self.ip_address = config['ip_address']
        self.start = 0
        self.nenvs = len(env_fns)
        self.train_envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        # self.validation_envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        self.envs = self.train_envs
        #self.envs = [env_fn(id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]

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
            self.channel = grpc.insecure_channel(self.ip_address + ":" + str(self.port_number))
            self.stub = service_pb2_grpc.CommunicationServiceStub(self.channel)
        else:
            print("Please specify either ROS or ZMQ communication mode for this environment")

    def switch_to_training(self):
        self.envs = self.train_envs

    def switch_to_validation(self):
        self.envs = self.validation_envs

    def step(self, actions, dart_convert=True):
        self.current_step += 1

        if dart_convert and 'dart' in self.config['env_key']:
            actions_converted = []
            for env, action in zip(self.envs, actions):
                act = env.update_action(action)
                actions_converted.append(act)
            actions = actions_converted

        #print("current step:" + str(self.current_step))
        # create request containing all environments with the actions to be executed
        request = self._create_request("ACTION", self.envs, actions)
        # execute the simulation for all environments and get observations
        observations = self._send_request(request)
        observations_converted = []
        terminated_environments = []
        rews = []
        dones = []
        infos = []
        for env, observation in zip(self.envs, observations):
            obs, rew, done, info = env.update(ast.literal_eval(observation))
            observations_converted.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
            if done:
                terminated_environments.append(env)
        # TODO: Make sure the last observation for terminated environment is the correct one:
        #  https://github.com/hill-a/stable-baselines/issues/400 talks about reset observation being wrong,
        #  use terminated observation instead
        # TODO: make sure this does not cause problems, when the when images are transfered it might be slow
        #  to get all environment observations again
        # reset all the terminated environments
        [env.reset() for env in terminated_environments]
        if len(terminated_environments) > 0:
            request = self._create_request("RESET", terminated_environments)
            # currently, the simulator returns array of all the environments, not just the terminated ones
            observations = self._send_request(request)
            for env in terminated_environments:
                obs, _, _, _ = self.envs[env.id].update(ast.literal_eval(observations[env.id]))
                observations_converted[env.id] = obs
        return np.stack(observations_converted), np.stack(rews), np.stack(dones), infos

    def step_wait(self):
        # only because VecFrameStack uses step_async to provide the actions, then step_wait to execute a step
        return self.step(self.actions)


    def _create_request(self, command, environments, actions=None):
        content = ''
        if command == "ACTION":
            for act, env in zip(actions, environments):
                # print("id: {}\t step: {}\t action = {}".format(env.id, env.ts, str(np.around(act, decimals=3))))

                act_json = json.dumps(CommandModel(env.id, "ACTION", "manipulator_environment", str(act.tolist())), default=lambda x: x.__dict__)
                content += (act_json + ",")

        elif command == "RESET":
            #print("Time: " + str(time.time() - self.start))
            self.start = time.time()
            for env in environments:
                reset_string = str(env.reset_state)
                act_json = json.dumps(CommandModel(env.id, "RESET", "manipulator_environment", reset_string), default=lambda x: x.__dict__)
                content += (act_json + ",")

        return '[' + content + ']'

    def _send_request(self, content):

        # "{\"Environment\":\"manipulator\",\"Action\":\"" + translated_action + "\"}"
        if self.communication_type == 'ROS':
            self.request['name'] = content
            return self._parse_result(self.service.call(self.request))
        elif self.communication_type == 'ZMQ':
            self.socket.send_string(content)
            response = self.socket.recv()
            return self._parse_result(response)
        else:
            reply = self.stub.step(StepRequest(data=content))
            return self._parse_result(reply.data)


    def _parse_result(self, result):
        if self.communication_type == 'ROS':
            return ast.literal_eval(result['value'])
        elif self.communication_type == 'ZMQ':
            return ast.literal_eval(result.decode("utf-8"))
        else:
            return ast.literal_eval(result)

    def reset(self, should_reset=True):
        if should_reset:
            [env.reset() for env in self.envs]
        request = self._create_request("RESET", self.envs)
        observations = self._send_request(request)
        observations_converted = []
        for env, observation in zip(self.envs, observations):
            obs, _, _, _ = env.update(ast.literal_eval(observation))
            observations_converted.append(obs)

        return np.array(observations_converted)

    def reset_task(self):
        pass

    def close(self):
        SimulatorVecEnv._client.terminate()

    def __len__(self):
        return self.nenvs


    # Calling destructor
    # def __del__(self):
    #     print("Destructor called")
    #     self.env_process.terminate()