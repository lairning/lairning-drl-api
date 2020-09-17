import numpy as np
import itertools
from datetime import datetime
from gym.spaces import Discrete, Tuple, Dict, Box, flatten
import gym

import requests

import pickle
import json

MKT_TEMPLATES1 = {'eMail'      : ['mail1', 'mail2', 'mail3', 'mail4'],
                  'webDiscount': ['discount1', 'discount2', 'discount3', 'discount4'],
                  'webPremium' : ['premium1', 'premium2', 'premium3', 'premium4'],
                  'callCenter' : ['script1', 'script2', 'script3', 'script4']}

MKT_TEMPLATES2 = {'eMail'      : ['mail1', 'mail2', 'mail3', 'mail4'],
                  'webDiscount': ['discount1', 'discount2', 'discount3'],
                  'webPremium' : ['premium1', 'premium2', 'premium3', 'premium4', 'premium5'],
                  'callCenter' : ['script1', 'script2', 'script3', 'script4']}

MKT_REWARDS = {'email do nothing'      : -0.5,
               'call do nothing'       : -5,
               'call discount purchase': 75,
               'call premium purchase' : 130,
               'web discount purchase' : 80,
               'web premium purchase'  : 135}

CUSTOMER_BEHAVIOR = {'eMail'      : ['email do nothing', 'callCenter', 'webDiscount', 'webPremium'],
                     'webDiscount': ['email do nothing', 'webPremium', 'web discount purchase'],
                     'webPremium' : ['email do nothing', 'webDiscount', 'web premium purchase', 'callCenter'],
                     'callCenter' : ['call do nothing', 'call discount purchase', 'call premium purchase']
                     }

CUSTOMER_ATTRIBUTES = {'age'   : ['<25', '25-45', '>45'],
                       'sex'   : ['Men', 'Women'],
                       'region': ['Lisbon', 'Oporto', 'North', 'Center', 'South']}

CONTEXT_ATTRIBUTES = {}


class Space:
    def discrete(n: int):
        return {'type': 'Discrete', 'value': n}

    def tuple(t: tuple):
        return {'type': 'Tuple', 'value': t}

    def box(low: int, high: int, size: int):
        return {'type': 'Box', 'low': low, 'high': high, 'size': size}


class MKTWorld:
    def __init__(self, config):
        self.probab = dict()
        self.rewards = config["mkt_rewards"]
        self.journeys = config["customer_journeys"]
        self.touch_points = list(config["customer_journeys"].keys()) + list(config["mkt_rewards"].keys())

        #self.customer_features = config["customer_attributes"].keys()
        #self.customer_values = list(config["customer_attributes"].values())
        #self.customer_segments = list(itertools.product(*self.customer_values))
        #self.customer_segment = self.customer_segments[0]

        self.features = list(config["customer_attributes"].keys()) + list(config["context_attributes"].keys())
        self.values = list(config["customer_attributes"].values()) + list(config["context_attributes"].values())
        self.segments = list(itertools.product(*self.values))
        self.segment = self.segments[0]

        self.mkt_offers = config["mkt_offers"]
        self.observation = list((len(self.features) + 1) * [0])
        for cs in self.segments:
            dt = dict()
            for t in self.mkt_offers.keys():
                dt[t] = {mo: np.random.dirichlet(np.ones(len(self.journeys[t])), size=1)[0] for mo in
                         self.mkt_offers[t]}
            self.probab[cs] = dt

    def random_customer_context(self):
        segments = self.segments[np.random.randint(len(self.segments))]
        return dict(zip(self.features, segments))

    def reset(self):
        segments = self.random_customer_context()
        self.segment = tuple(segments.values())
        self.observation[0] = 0
        for i, _ in enumerate(self.features):
            self.observation[i + 1] = self.values[i].index(segments[self.features[i]])
        return self.observation

    def step(self, action: int):
        touch_point = self.touch_points[self.observation[0]]
        assert action < len(self.mkt_offers[touch_point]), \
            "Action={}, TP={}, OFFERS={}".format(action, touch_point, self.mkt_offers[touch_point])
        mkt_offer = self.mkt_offers[touch_point][action]
        new_touch_point = np.random.choice(
            self.journeys[touch_point],
            p=self.probab[self.segment][touch_point][mkt_offer]
        )
        self.observation[0] = self.touch_points.index(new_touch_point)
        done = new_touch_point in self.rewards.keys()
        reward = self.rewards[new_touch_point] if done else 0
        return self.observation, reward, done, {}


def flatten_space(space):
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n,))
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
    raise NotImplementedError


class FlattenObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = flatten_space(env.observation_space['state'])

    def observation(self, observation):
        return flatten(self.env.observation_space['state'], observation)


class MKTEnv(gym.Env):
    def __init__(self, real_observation_space, max_action_size):
        self.action_space = Discrete(max_action_size)
        self.observation_space = Dict({
            "state"      : real_observation_space,
            "action_mask": Box(low=0, high=1, shape=(max_action_size,))
        })


def _get_action_mask(actions: list, max_actions: int):
    action_mask = [0] * max_actions
    action_len = len(actions)
    action_mask[:action_len] = [1] * action_len
    return action_mask


class MKTWorldParametric(MKTWorld):
    def __init__(self, config):
        super(MKTWorldParametric, self).__init__(config)
        self.max_action_size = max([len(options) for options in self.mkt_offers.values()])
        self.action_mask = {tp_id: _get_action_mask(self.mkt_offers[tp], max_action_size) for tp_id, tp
                            in enumerate(self.mkt_offers.keys())}
        real_obs_tuple = (Discrete(len(self.mkt_offers.keys()) + len(self.rewards.keys())),)
        real_obs_tuple += tuple((Discrete(len(l)) for l in self.values))
        print("MKTWorldParametric",real_obs_tuple)
        self.flat = FlattenObservation(MKTEnv(real_observation_space=Tuple(real_obs_tuple),
                                              max_action_size=self.max_action_size))

    def reset(self):
        observation = super().reset()
        print("RESET:",observation)
        return {'action_mask': self.action_mask[0], 'state': self.flat.observation(observation)}

    def step(self, action: int):
        observation, reward, done, info = super().step(action)
        msg = {'action_mask': self.action_mask[observation[0]] if not done else [1] * self.max_action_size,
                'state'      : self.flat.observation(observation)}, reward, done, info
        print("STEP:", observation,msg)
        return msg

env_config = {
    "mkt_rewards"        : MKT_REWARDS,
    "customer_journeys"  : CUSTOMER_BEHAVIOR,
    "customer_attributes": CUSTOMER_ATTRIBUTES,
    "context_attributes" : CONTEXT_ATTRIBUTES,
}

model_config = {
    'DQN'    : {
        "v_min"                  : -5.0,
        "v_max"                  : 135.0,
        "hiddens"                : [128],
        "exploration_config"     : {
            "epsilon_timesteps": 5000,
        },
        'lr'                     : 5e-5,
        "num_atoms"              : 2,
        "learning_starts"        : 100,
        "timesteps_per_iteration": 500
    },
    'Apex'   : {
        "v_min"                  : -5.0,
        "v_max"                  : 135.0,
        "hiddens"                : [128],
        "exploration_config"     : {
            "epsilon_timesteps": 4000,
        },
        'lr'                     : 5e-5,
        "num_atoms"              : 2,
        "learning_starts"        : 100,
        "timesteps_per_iteration": 500
    },
    'PPO'    : {"vf_clip_param": 140.0},
    'APPO'   : {},
    'Impala' : {},
    'SAC'    : {
        "Q_model"     : {
            "fcnet_hiddens": [64, 64],
        },
        "policy_model": {
            "fcnet_hiddens": [64, 64],
        },
    },
    'SimpleQ': {
        "exploration_config": {
            "epsilon_timesteps": 5000
        },
        "learning_starts"   : 100,
    }
}

# Commands for remote inference mode.
START_EPISODE = "START_EPISODE"
GET_ACTION = "GET_ACTION"
LOG_ACTION = "LOG_ACTION"
LOG_RETURNS = "LOG_RETURNS"
END_EPISODE = "END_EPISODE"


class DRLTrainer:
    def __init__(self, trainer_id: int, trainer_address: str):
        self.address = trainer_address

    def _send(self, data):
        payload = pickle.dumps(data)
        try:
            response = requests.post(self.address, data=payload)
        except Exception as err:
            print("{} : Failed to send {}".format(datetime.now(), data))
            raise err
        if response.status_code != 200:
            print("{} : Request failed {}: {}"
                  .format(datetime.now(), response.text, data))
        response.raise_for_status()
        parsed = pickle.loads(response.content)
        return parsed

    def start_episode(self, episode_id: str = None, training_enabled: bool = True):
        return self._send({
            "episode_id"      : episode_id,
            "command"         : START_EPISODE,
            "training_enabled": training_enabled,
        })["episode_id"]

    def get_action(self, episode_id: str, observation: object):
        return self._send({
            "command"    : GET_ACTION,
            "observation": observation,
            "episode_id" : episode_id,
        })["action"]

    def log_action(self, episode_id: str, observation: object, action: object):
        self._send({
            "command"    : LOG_ACTION,
            "observation": observation,
            "action"     : action,
            "episode_id" : episode_id,
        })

    def log_returns(self, episode_id: str, reward: float, info=None):
        self._send({
            "command"   : LOG_RETURNS,
            "reward"    : reward,
            "info"      : info,
            "episode_id": episode_id,
            "done"      : None
        })

    def end_episode(self, episode_id: str, observation: object):
        self._send({
            "command"    : END_EPISODE,
            "observation": observation,
            "episode_id" : episode_id,
        })


class DRLTrainerDebug:
    def __init__(self):
        self.episode_id = 0

    def _print(self, data):
        pass
        # print("TrainerDEBUG {}:{}".format(self.episode_id,data))

    def start_episode(self, episode_id: str = None, training_enabled: bool = True):
        self.episode_id += 1
        self._print({
            "command"         : START_EPISODE,
        })
        return self.episode_id

    def get_action(self, episode_id: str, observation: object):
        action = np.argmax([np.random.rand()*mask for mask in observation['action_mask']])
        self._print({
            "command"    : GET_ACTION,
            "observation": observation,
            "action" : action,
        })
        return action

    def log_action(self, episode_id: str, observation: object, action: object):
        self._send({
            "command"    : LOG_ACTION,
            "observation": observation,
            "action"     : action,
        })

    def log_returns(self, episode_id: str, reward: float, info=None):
        self._print({
            "command"   : LOG_RETURNS,
            "reward"    : reward
        })

    def end_episode(self, episode_id: str, observation: object):
        self._print({
            "command"    : END_EPISODE,
            "observation": observation,
        })

if __name__ == "__main__":

    def parametric(mkt_templates: dict):
        l = [len(l) for l in mkt_templates.values()]
        n = l[0]
        for v in l[1:]:
            if v != n:
                return 1
        return 0


    base_config = {
        "learning_starts": 100,
        "v_min"          : min(MKT_REWARDS.values()),
        "v_max"          : max(MKT_REWARDS.values()),

    }

    START_TRAINER_URL = 'http://localhost:5002/v1/drl/server/start'
    STOP_TRAINER_URL = 'http://localhost:5002/v1/drl/server/stop'

    for mkt_template in [MKT_TEMPLATES2]:

        model_config = base_config.copy()

        max_action_size = max([len(options) for options in mkt_template.values()])
        flat_observation_space_size = len(mkt_template.keys()) + len(MKT_REWARDS.keys()) + \
                                      sum([len(l) for l in CUSTOMER_ATTRIBUTES.values()]) + \
                                      sum([len(l) for l in CONTEXT_ATTRIBUTES.values()])

        model_parametric = parametric(mkt_template)

        env_config.update({"mkt_offers": mkt_template})

        if model_parametric:
            observation_space = Space.box(0, 1, flat_observation_space_size)
            world = MKTWorldParametric(env_config)
        else:
            model_config.update({
                "hiddens"           : [128],
                "exploration_config": {
                    "epsilon_timesteps": 4000,
                },
                'lr'                : 5e-5,
                "num_atoms"         : 2,
            })
            real_obs_tuple = (Space.discrete(len(mkt_template.keys()) + len(MKT_REWARDS.keys())),)
            real_obs_tuple += tuple((Space.discrete(len(l)) for l in CUSTOMER_ATTRIBUTES.values()))
            real_obs_tuple += tuple((Space.discrete(len(l)) for l in CONTEXT_ATTRIBUTES.values()))
            observation_space = Space.tuple(real_obs_tuple)
            world = MKTWorld(env_config)

        start_msg = {'action_space'     : json.dumps(Space.discrete(max_action_size)),
                     'observation_space': json.dumps(observation_space),
                     'model_type'       : 'DQN',
                     'model_parametric' : model_parametric,
                     'model_config'     : json.dumps(model_config)
                     }

        print("{} : Start Message = {}".
              format(datetime.now(), start_msg))

        '''
        msg = requests.post(START_TRAINER_URL, data=start_msg)

        if msg.status_code != 200:
            print("{} : Trainer Creation failed with ERR={}".
                  format(datetime.now(), msg.status_code))
            raise SystemExit

        msg = msg.json()

        if not msg['status']:
            print("{} : Trainer Creation failed with ERR={}".
                  format(datetime.now(), msg['error']))
            raise SystemExit

        trainer_id = msg['id']
        trainer_address = msg['address']
        print("{} : Trainer Created with ID={}, and ADDRESS={}".
              format(datetime.now(), trainer_id, trainer_address))

        '''
        # drl_trainer = DRLTrainer(trainer_id=trainer_id, trainer_address=trainer_address)
        drl_trainer = DRLTrainerDebug()

        for i in range(1):  # 20
            count = 0
            total = 0
            for _ in range(5):  # 500
                eid = drl_trainer.start_episode(training_enabled=True)
                obs = world.reset()
                done = False
                reward = 0
                while not done:
                    action = drl_trainer.get_action(eid, obs)
                    obs, reward, done, info = world.step(action)
                    drl_trainer.log_returns(eid, reward, info=info)
                drl_trainer.end_episode(eid, obs)
                count += 1
                total += reward
            print("{} : Iteration {} - Mean Reward = {}"
                  .format(datetime.now(), i, total / count))

        print("{} : Stop Trainer ID={}".format(datetime.now(), trainer_id))

        stop_msg = {'id': trainer_id}
        stop_result = requests.post(STOP_TRAINER_URL, data=stop_msg).json()

        print("{} : Stop Trainer ID={} message received {}"
              .format(datetime.now(), trainer_id, stop_result))
