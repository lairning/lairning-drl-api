import os
import sys

import numpy as np
from datetime import datetime

from multiprocessing import Process, ProcessError, Queue

import gym
from gym.spaces import Discrete, Tuple, Box, Dict, Space

from ray.rllib.agents.dqn import DQNTrainer, ApexTrainer, SimpleQTrainer
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer


from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.registry import register_env
import ray

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.utils.framework import try_import_tf

SERVER_ADDRESS = "localhost"
PORT = 5010
AUTHKEY = b'moontedrl!'

tf1, tf, tfv = try_import_tf()

class MKTWorld(gym.Env):
    def __init__(self, action_space: Discrete, observation_space: Tuple):
        self.action_space = action_space
        self.observation_space = observation_space

class ParametricMKTWorld(gym.Env):
    def __init__(self, action_space: Discrete, observation_space: Box):
        self.action_space = action_space
        self.observation_space = Dict({
            "state": observation_space,
            "action_mask": Box(low=0, high=1, shape=(action_space.n,))
        })

        # print("{} : [DEBUG] KTWorld ActS={}, ObsS={}"
        #     .format(datetime.now(),action_space, observation_space))

class ParametricActionsModel(DistributionalQTFModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):

        # print("{} : [DEBUG] ParametricActionsModel ActS={}, ObsS={}, NOut={}, Name={}, Model Config={}"
        #     .format(datetime.now(), action_space, obs_space, num_outputs, name, model_config))

        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        model_observation_space = Box(low=0, high=1, shape=(obs_space.shape[0]-action_space.n,))

        # print("{} : [DEBUG] ParametricActionsModel model_observation_space = {}"
        #    .format(datetime.now(), model_observation_space))

        self.action_param_model = FullyConnectedNetwork(
            model_observation_space, action_space, num_outputs,
            model_config, name + "_action_param")

        self.register_variables(self.action_param_model.variables())

    def forward(self, input_dict, state, seq_lens):

        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_param, _ = self.action_param_model({
            "obs": input_dict["obs"]["state"]
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_param + inf_mask, state

    def value_function(self):
        return self.action_param_model.value_function()

parametric_model_config = {
    'DQN' : {
        "input_evaluation": [],
        "hiddens": [],
        "dueling": False
     },
    'PPO' : {
    },
    'APPO' : {
    },
    'Impala' : {
    },
    'Apex': {
        "input_evaluation": [],
        "hiddens"         : [],
        "dueling"         : False
    },
    'SimpleQ': {
    }
}

trainers = {'DQN': DQNTrainer, 'PPO':PPOTrainer, 'APPO':APPOTrainer, 'Apex': ApexTrainer, 'SimpleQ':SimpleQTrainer,
            'Impala': ImpalaTrainer}

def drl_trainer(
        log_file: str,
        input_port: int,
        action_space: Discrete,
        observation_space: Space,
        model_type: str,
        model_parametric: bool,
        drl_config: dict,
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

        assert model_type in parametric_model_config.keys(), "Model Type {} not in Configs {}".format(model_type,                                                                                                  parametric_model_config.keys())
        assert model_type in trainers.keys(), "Model Type {} not in Trainers {}".format(model_type, trainers.keys())

        # print("{} : [DEBUG] drl_trainer ActS={}, ObsS={}, MType={}, Parametric={}, Model Config={}"
        #       .format(datetime.now(), action_space, observation_space, model_type, model_parametric, drl_config))

        ray.init()

        drl_config.update(
            {"input": (lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, input_port)),
             "num_workers": 0,
             "input_evaluation": []})

        if model_parametric:
            register_env("env", lambda _: ParametricMKTWorld(action_space, observation_space))
            ModelCatalog.register_custom_model("ParametricActionsModel", ParametricActionsModel)
            drl_config.update(parametric_model_config[model_type])
            drl_config.update(
                {"model": {"custom_model": "ParametricActionsModel"}
            })
        else:
            register_env("env", lambda _: MKTWorld(action_space, observation_space))

        drl = trainers[model_type](
            env="env",
            config=drl_config
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
        drl.train()
        checkpoint_path = drl.save()
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
    if space['type'] == 'Box':
        return Box(low=space['low'], high=space['high'], shape=(space['size'],), dtype=np.int64)
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
        except Exception as err:
            print("{} : [ERROR CREATING GYM SPACES] {}"
                  .format(datetime.now(), err))
            return {'status': False, 'error': err}

        print("{} : DEBUG DRLServer observation_space before {}"
               .format(datetime.now(), payload['observation_space']))
        print("{} : DEBUG DRLServer observation_space after {}"
               .format(datetime.now(), observation_space))

        drl_trainer_args = (
            trainer_log_file,
            PORT + self.trainer_id,
            action_space,
            observation_space,
            payload['model_type'],
            payload['model_parametric'],
            payload['model_config'],
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
