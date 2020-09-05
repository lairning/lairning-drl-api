import os

from datetime import datetime

from multiprocessing import Process, ProcessError, Queue

import gym
from gym.spaces import Space, Discrete, Tuple, Box, Dict, flatten

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import ray
import sys

from ray.rllib.utils.framework import try_import_tf
tf = try_import_tf()

import numpy as np

SERVER_ADDRESS = "localhost"
PORT = 5010
AUTHKEY = b'moontedrl!'

def flatten_space(space):
    """Flatten a space into a single ``Box``.
    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    Example::
        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that flattens a discrete space::
        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that recursively flattens a dict::
        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, MultiBinary):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, MultiDiscrete):
        return Box(
            low=np.zeros_like(space.nvec),
            high=space.nvec,
        )
    raise NotImplementedError

class MKTWorld(gym.Env):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        # self.observation_space = observation_space
        #'''
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(action_space.n, )),
            "cart": observation_space,
        })
#'''
#TODO: Probably to be Removed
'''
class ParametricMKTWorld(gym.Env):
    def __init__(self, action_space, observation_space, action_mask):
        self.action_space = action_space
        self.mktworld = MKTWorld(action_space, observation_space)
        self.action_mask = action_mask
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=( self.action_space.n, )),
            "cart": self.mktworld.observation_space,
        })

    def reset(self):
        return {
            "action_mask": self.action_mask[0],
            "cart": self.mktworld.reset()
        }

    def step(self, action):
        orig_obs, rew, done, info = self.mktworld.step(action)
        obs = {

        }
'''

class FlattenObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = flatten_space(env.observation_space)

    def observation(self, observation):
        return flatten(self.env.observation_space, observation)

observation_space_flatten = None

class ParametricActionsModel(DistributionalQTFModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):

        print("{} : [INFO] ParametricActionsModel {}, {}, {}, {}"
              .format(datetime.now(),action_space, obs_space, num_outputs, name))

        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, 6, model_config, name, **kw)
            # observation_space, action_space, num_outputs, model_config, name, **kw)

        # print("{} : [INFO] ParametricActionsModel Super Done!"
        #      .format(datetime.now()))

        self.action_param_model = FullyConnectedNetwork(
            flatten_space(Tuple((Discrete(17), Discrete(3), Discrete(2), Discrete(5)))), action_space, 6,
            # obs_space, action_space, num_outputs,
            model_config, name + "_action_param")
        self.register_variables(self.action_param_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.

        print("{} : [INFO] Forward Input Dict {}"
              .format(datetime.now(), input_dict['obs']['cart']))

        action_mask = input_dict["obs"]["action_mask"]

        global observation_space_flatten

        # Compute the predicted action embedding
        action_param, _ = self.action_param_model({
            "obs": input_dict["obs"]["cart"]
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_param + inf_mask, state

    def value_function(self):
        return self.action_param_model.value_function()


def drl_trainer(
        log_file: str,
        input_port: int,
        action_space: Space, # A Discrete Space with Max Available Actions across all the Touch Points
        # action_mask: dict,   # Dict containing a Mask for each Touch Point
        observation_space: Space,
        dqn_config: dict,
        q: Queue):
    # Replace file descriptors for stdin, stdout, and stderr
    stdin = '/dev/null'
    stdout = log_file
    stderr = log_file

    try:
        with open(stdin, 'rb', 0) as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open(stdout, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
        with open(stderr, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stderr.fileno())

        ray.init()

        global observation_space_flatten

        observation_space_flatten = FlattenObservation(MKTWorld(action_space, observation_space))

        register_env("srv", lambda _: MKTWorld(action_space, flatten_space(observation_space)))

        print("{} : [INFO] DRL Trainer MKTWorld Env Registered {},{}"
              .format(datetime.now(),action_space, observation_space))

        ModelCatalog.register_custom_model("ParametricActionsModel", ParametricActionsModel)

        #dqn_config.update(
        dqn_config = {"input": (lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, input_port)),
             "model": {"custom_model": "ParametricActionsModel"},
             "num_workers": 0,
             "input_evaluation": []}

        print("{} : [INFO] DRL Trainer ParametricActionsModel Registered {}"
              .format(datetime.now(),dqn_config))

        dqn = DQNTrainer(
            env="srv",
            config=dqn_config
        )
        print("{} : [INFO] DRL Trainer Configured at {}:{}"
              .format(datetime.now(), SERVER_ADDRESS, input_port))
    except Exception as err:
        q.put(False)
        print("{} : [ERROR] DRL Trainer {}"
              .format(datetime.now(), err))
        raise err

    q.put(True)

    checkpoint_file = "last_checkpoint_DQN{}.out".format(input_port)

    '''
    # Attempt to restore from checkpoint if possible.
    if os.path.exists(checkpoint_file):
        checkpoint_path = open(checkpoint_file).read()
        print("Restoring from checkpoint path", checkpoint_path)
        dqn.restore(checkpoint_path)
    '''

    # Serving and training loop
    i = 1
    while True:
        dqn.train()
        checkpoint_path = dqn.save()
        try:
            with open(checkpoint_file, "w") as f:
                f.write(checkpoint_path)
            print("Last checkpoint", checkpoint_path)
        except Exception as e:
            print(e)
        i += 1


def myspace2gymspace(space: dict):
    if space['type'] == 'Discrete':
        return Discrete(space['value'])
    if space['type'] == 'Tuple':
        return Tuple(tuple(myspace2gymspace(s) for s in space['value']))
    raise "Invalid Space Type = {}".format(space['type'])


class DRLServer:
    def __init__(self):
        self.trainer_id = 1
        self.drl_trainers = dict()
        self.queue = Queue()

    def start_trainer(self, payload: dict) -> dict:
        # msg format: {'operation': 'start', 'action_space':..., 'observation_space':..., 'dqn_config':... }
        # Log file fo the new DRL Trainer
        trainer_log_file = '/tmp/drltrainer_{}_{:%Y-%m-%d_%H:%M:%S%f}.log'.format(self.trainer_id, datetime.now())
        try:
            action_space = myspace2gymspace(payload['action_space'])
            observation_space = myspace2gymspace(payload['observation_space'])
            # action_mask = payload['action_mask'],
            model_config = payload['model_config']
        except Exception as err:
            print("{} : [ERROR CREATING GYM SPACES] {}}"
                  .format(datetime.now(), err))
            return {'status': False, 'error': err}

        drl_trainer_args = (
            trainer_log_file,
            PORT + self.trainer_id,
            action_space,
            # action_mask,
            observation_space,
            model_config,
            self.queue
        )
        print("{} : [INFO] DRL Server is about to start a new DRL Trainer"
              .format(datetime.now()))

        try:
            trainer = Process(target=drl_trainer, args=drl_trainer_args)
            trainer.start()
            print("{} : [INFO] DRL Server started DRL Trainer {}"
                  .format(datetime.now(), self.trainer_id))
            trainer_ok = self.queue.get(block=True, timeout=None)
            if trainer_ok:
                self.drl_trainers[self.trainer_id] = trainer
                print("{} : [INFO] Live DRL Trainers {}"
                      .format(datetime.now(), self.drl_trainers))
                return_msg = {'status': True, 'id': self.trainer_id,
                              'address': "http://{}:{}".format(SERVER_ADDRESS, PORT + self.trainer_id)}
                self.trainer_id += 1
                return return_msg
            else:
                print("{} : [PROCESS ERROR] DRL Server failed to create DRL Trainer {}."
                      .format(datetime.now(), self.trainer_id))
                return {'status': False, 'error': 'Failed to Create Trainer'}
        except ProcessError:
            print("{} : [PROCESS ERROR] DRL Server failed to create DRL Trainer {}."
                  .format(datetime.now(), self.trainer_id))
            return {'status': False, 'error': 'Failed to Create Trainer'}
        except Exception as err:
            print("{} : [UNEXPECTED ERROR] DRL Server failed {}"
                  .format(datetime.now(), err))
            return {'status': False, 'error': 'DRL Server Critical error'}

    def stop_trainer(self, payload: dict) -> dict:
        print("{} : [INFO] DRL Server is about to stop Trainer {}"
              .format(datetime.now(), payload['id']))
        try:
            print("{} : [INFO] DRL Trainers {}"
                  .format(datetime.now(), self.drl_trainers))
            trainer = self.drl_trainers.pop(payload['id'])
        except KeyError:
            print("{} : [ERROR] {} is an invalid trainer ID"
                  .format(datetime.now(), payload['id']))
            return {'status': False, 'error': 'Invalid ID'}

        try:
            trainer.terminate()
            trainer.join()
            print("{} : [INFO] DRL Trainer {} terminated with exit exit code {}"
                  .format(datetime.now(), payload['id'], trainer.exitcode))
            return {'status': True}

        except Exception as err:
            print("{} : [UNEXPECTED ERROR] DRL Server failed {}"
                  .format(datetime.now(), err))
        return {'status': False, 'error': 'DRL Server Critical error'}

    def __del__(self):
        trainers_alive = {trainer_id: trainer for trainer_id, trainer in self.drl_trainers.items() if
                          trainer.is_alive()}
        print("{} : [INFO] DRL Trainers still running {}."
              .format(datetime.now(), list(trainers_alive.keys())))
        try:
            for trainer_id, trainer in trainers_alive.items():
                trainer.terminate()
                trainer.join()
                print("{} : [INFO] DRL Trainer {} terminated."
                      .format(datetime.now(), trainer_id))
        except Exception as err:
            print("{} : [PROCESS ERROR] {} when trying terminate Trainers"
                  .format(datetime.now(), err))
            print("{} : [INFO] Process PIDs still alive = {}"
                  .format(datetime.now(), [trainer.pid for trainer in trainers_alive.values()]))
            raise err

        print("{} : [INFO] DRL Server is exiting"
              .format(datetime.now()))
        raise SystemExit
