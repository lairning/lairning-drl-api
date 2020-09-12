import numpy as np
import itertools
from datetime import datetime
import requests
import pickle
import json

import gym
from gym.spaces import Discrete, Tuple, Dict, Box, flatten

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
    raise NotImplementedError

MKT_TEMPLATES = {'eMail':['mail1','mail2','mail3','mail4'],
                 'webDiscount':['discount1','discount2','discount3'],
                 'webPremium':['premium1','premium2','premium3','premium4','premium5'],
                'callCenter':['script1','script2','script3','script4','script5','script6']}

MKT_REWARDS = { 'email do nothing':-0.5,
                'call do nothing':-5,
                'call discount purchase':75,
                'call premium purchase':130,
                'web discount purchase':80,
                'web premium purchase':135}

CUSTOMER_BEHAVIOR = {'eMail':['email do nothing', 'callCenter', 'webDiscount','webPremium'],
                     'webDiscount':['email do nothing','webPremium','web discount purchase'],
                     'webPremium':['email do nothing','webDiscount','web premium purchase','callCenter'],
                     'callCenter':['call do nothing','call discount purchase','call premium purchase']
                     }

CUSTOMER_ATTRIBUTES = {'age': ['<25', '25-45', '>45'],
                       'sex': ['Men', 'Women'],
                       'region': ['Lisbon', 'Oporto', 'North', 'Center', 'South']}

CONTEXT_ATTRIBUTES = {}

def _get_action_mask(actions: list, max_actions: int):
    action_mask = [0] * max_actions
    action_len = len(actions)
    action_mask[:action_len] = [1] * action_len
    return action_mask

tp_actions = MKT_TEMPLATES
max_action_size = max([len(options) for options in MKT_TEMPLATES.values()])
action_mask = {tp_id: _get_action_mask(tp_actions[tp], max_action_size) for tp_id, tp
               in enumerate(tp_actions.keys())}

flat_observation_space_size = len(MKT_TEMPLATES.keys()) + len(MKT_REWARDS.keys()) + \
                              sum([len(l) for l in CUSTOMER_ATTRIBUTES.values()]) + \
                              sum([len(l) for l in CONTEXT_ATTRIBUTES.values()])

FLAT_OBSERVATION_SPACE = Box(low=0, high=1, shape=(flat_observation_space_size,), dtype=np.int64)
real_obs_tuple = (Discrete(len(MKT_TEMPLATES.keys()) + len(MKT_REWARDS.keys())),)
real_obs_tuple += tuple((Discrete(len(l)) for l in CUSTOMER_ATTRIBUTES.values()))
real_obs_tuple += tuple((Discrete(len(l)) for l in CONTEXT_ATTRIBUTES.values()))

REAL_OBSERVATION_SPACE = Tuple(real_obs_tuple)

class FlattenObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = flatten_space(env.observation_space['state'])

    def observation(self, observation):
        return flatten(self.env.observation_space['state'], observation)

class MKTEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(max_action_size)
        self.observation_space = Dict({
            "state": REAL_OBSERVATION_SPACE,
            "action_mask": Box(low=0, high=1, shape=(max_action_size,))
        })

flat = FlattenObservation(MKTEnv())

class MKTWorld(MKTEnv):
    def __init__(self, config):
        super(MKTWorld, self).__init__()
        self.probab = dict()
        self.rewards = config["mkt_rewards"]
        self.journeys = config["customer_journeys"]
        self.touch_points = list(config["customer_journeys"].keys()) + list(config["mkt_rewards"].keys())
        self.customer_features = config["customer_attributes"].keys()
        self.customer_values = list(config["customer_attributes"].values())
        self.customer_segments = list(itertools.product(*self.customer_values))
        self.customer_segment = self.customer_segments[0]
        self.mkt_offers = config["mkt_offers"]
        self.observation = list(4 * [0])
        for cs in self.customer_segments:
            dt = dict()
            for t in self.mkt_offers.keys():
                dt[t] = {mo: np.random.dirichlet(np.ones(len(self.journeys[t])), size=1)[0] for mo in
                         self.mkt_offers[t]}
            self.probab[cs] = dt
        self.observation_space = Dict({
            "state": FLAT_OBSERVATION_SPACE,
            "action_mask": Box(low=0, high=1, shape=(max_action_size,))
        })

    def random_customer(self):
        cs = self.customer_segments[np.random.randint(len(self.customer_segments))]
        return dict(zip(self.customer_features, cs))

    def reset(self):
        cs = self.random_customer()
        self.customer_segment = tuple(cs.values())
        customer_feature = list(self.customer_features)
        self.observation[0] = 0
        for i,_ in enumerate(CUSTOMER_ATTRIBUTES.keys()):
            self.observation[i+1] = self.customer_values[i].index(cs[customer_feature[i]])

        return {'action_mask': action_mask[0], 'state': flat.observation(self.observation)}

    def step(self, action: int):
        touch_point = self.touch_points[self.observation[0]]
        assert action < len(self.mkt_offers[touch_point]), \
            "Action={}, TP={}, OFFERS={}".format(action, touch_point, self.mkt_offers[touch_point])
        mkt_offer = self.mkt_offers[touch_point][action]
        new_touch_point = np.random.choice(
            self.journeys[touch_point],
            p=self.probab[self.customer_segment][touch_point][mkt_offer]
        )
        self.observation[0] = self.touch_points.index(new_touch_point)
        done = new_touch_point in self.rewards.keys()
        reward = self.rewards[new_touch_point] if done else 0
        return {'action_mask': action_mask[self.observation[0]] if not done else [1]*max_action_size, 'state': flat.observation(
            self.observation)}, reward, done, {}

env_config = {
    "mkt_rewards": MKT_REWARDS,
    "customer_journeys": CUSTOMER_BEHAVIOR,
    "customer_attributes": CUSTOMER_ATTRIBUTES,
    "mkt_offers": MKT_TEMPLATES
}

'''
dqn_config = {
    "v_min": -1.0,
    "v_max": 5.0,
    "hiddens": [128],
    "exploration_config": {
        "epsilon_timesteps": 4000,
    },
    'lr': 5e-5,
    "num_atoms": 2,
    "learning_starts": 100,
    "timesteps_per_iteration": 500
}
'''

# "num_atoms"  Cannot be set with parametric
dqn_config = {
    "v_min": -5,
    "v_max": 135.0,
    "hiddens": [128],
    'lr'                     : 5e-5,
    "learning_starts"        : 100,
    "timesteps_per_iteration": 500
}

# Commands for remote inference mode.
START_EPISODE = "START_EPISODE"
GET_ACTION = "GET_ACTION"
LOG_ACTION = "LOG_ACTION"
LOG_RETURNS = "LOG_RETURNS"
END_EPISODE = "END_EPISODE"


class DRLTrainer:
    def __init__(self, trainer_id: int, trainer_address: str):
        self.id = trainer_id
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
            "episode_id": episode_id,
            "command": START_EPISODE,
            "training_enabled": training_enabled,
        })["episode_id"]

    def get_action(self, episode_id: str, observation: object):
        return self._send({
            "command": GET_ACTION,
            "observation": observation,
            "episode_id": episode_id,
        })["action"]

    def log_action(self, episode_id: str, observation: object, action: object):
        self._send({
            "command": LOG_ACTION,
            "observation": observation,
            "action": action,
            "episode_id": episode_id,
        })

    def log_returns(self, episode_id: str, reward: float, info=None):
        self._send({
            "command": LOG_RETURNS,
            "reward": reward,
            "info": info,
            "episode_id": episode_id,
            "done": None
        })

    def end_episode(self, episode_id: str, observation: object):
        self._send({
            "command": END_EPISODE,
            "observation": observation,
            "episode_id": episode_id,
        })


if __name__ == "__main__":

    START_TRAINER_URL = 'http://localhost:5002/v1/drl/server/start'
    STOP_TRAINER_URL = 'http://localhost:5002/v1/drl/server/stop'

    # Run DQN and PPO Models

    for model in ['ppo']: # ['dqn', 'ppo']:

        start_msg = {'action_space_size': max_action_size,
                     'observation_space_size': flat_observation_space_size,
                     'model_type': model,
                     'model_config': json.dumps(dqn_config)
                     }

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
        print("{} : {} Trainer Created with ID={}, and ADDRESS={}".
              format(datetime.now(), model, trainer_id, trainer_address))

        world = MKTWorld(env_config)
        drl_trainer = DRLTrainer(trainer_id=trainer_id, trainer_address=trainer_address)

        for i in range(20):  # 20
            count = 0
            total = 0
            for _ in range(500):  # 500
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
            print("{} : Model {} Iteration {} - Mean Reward = {}"
                  .format(datetime.now(), model, i, total / count))

        print("{} : Stop Trainer ID={}".format(datetime.now(), trainer_id))

        stop_msg = {'id': trainer_id}
        stop_result = requests.post(STOP_TRAINER_URL, data=stop_msg).json()

        print("{} : Stop Trainer ID={} message received {}"
              .format(datetime.now(), trainer_id, stop_result))
