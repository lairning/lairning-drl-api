import numpy as np
import itertools
from datetime import datetime

import requests

import pickle
import json

MKT_TEMPLATES = {'eMail': ['mail1', 'mail2', 'mail3', 'mail4'],
                 'webDiscount': ['discount1', 'discount2', 'discount3', 'discount4'],
                 'webInfo': ['info1', 'info2', 'info3', 'info4'],
                 'webPremium': ['premium1', 'premium2', 'premium3', 'premium4']}

CUSTOMER_ATTRIBUTES = {'age': ['<25', '25-45', '>45'],
                       'sex': ['Men', 'Women'],
                       'region': ['Lisbon', 'Oporto', 'North', 'Center', 'South']}

MKT_REWARDS = {'do_nothing': -1,
               'discount_purchase': 2,
               'premium_purchase': 5}

CUSTOMER_BEHAVIOR = {'eMail': ['do_nothing', 'webDiscount', 'webInfo'],
                     'webDiscount': ['do_nothing', 'webInfo', 'discount_purchase'],
                     'webInfo': ['do_nothing', 'webDiscount', 'webPremium'],
                     'webPremium': ['do_nothing', 'webDiscount', 'premium_purchase']
                     }


# ACTION_SPACE = Discrete(4)
# OBSERVATION_SPACE = Tuple((Discrete(7), Discrete(3), Discrete(2), Discrete(6)))

class Space:
    def discrete(n: int):
        return {'type': 'Discrete', 'value': n}

    def tuple(t: tuple):
        return {'type': 'Tuple', 'value': t}


ACTION_SPACE = Space.discrete(4)
OBSERVATION_SPACE = Space.tuple((Space.discrete(7), Space.discrete(3), Space.discrete(2), Space.discrete(6)))


class MKTWorld:

    def __init__(self, config):
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
        # self.action_space = ACTION_SPACE
        # self.observation_space = OBSERVATION_SPACE

    def random_customer(self):
        cs = self.customer_segments[np.random.randint(len(self.customer_segments))]
        return dict(zip(self.customer_features, cs))

    def reset(self):
        cs = self.random_customer()
        self.customer_segment = tuple(cs.values())
        customer_feature = list(self.customer_features)
        self.observation[0] = 0
        self.observation[1] = self.customer_values[0].index(cs[customer_feature[0]])
        self.observation[2] = self.customer_values[1].index(cs[customer_feature[1]])
        self.observation[3] = self.customer_values[2].index(cs[customer_feature[2]])
        return self.observation

    def step(self, action):
        assert action in [0, 1, 2, 3], action
        touch_point = self.touch_points[self.observation[0]]
        mkt_offer = self.mkt_offers[touch_point][action]
        new_touch_point = np.random.choice(
            self.journeys[touch_point],
            p=self.probab[self.customer_segment][touch_point][mkt_offer]
        )
        self.observation[0] = self.touch_points.index(new_touch_point)
        done = new_touch_point in self.rewards.keys()
        reward = self.rewards[new_touch_point] if done else 0
        return self.observation, reward, done, {}


env_config = {
    "mkt_rewards": MKT_REWARDS,
    "customer_journeys": CUSTOMER_BEHAVIOR,
    "customer_attributes": CUSTOMER_ATTRIBUTES,
    "mkt_offers": MKT_TEMPLATES
}

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

# Commands for remote inference mode.
START_EPISODE = "START_EPISODE"
GET_ACTION = "GET_ACTION"
LOG_ACTION = "LOG_ACTION"
LOG_RETURNS = "LOG_RETURNS"
END_EPISODE = "END_EPISODE"


class DRLTrainer:
    def __init__(self, trainer_id: int, trainer_address: tuple):
        self.id = trainer_id
        self.address = "http://{}:{}".format(trainer_address[0], trainer_address[1])
        print("{} : Trainer full ADDRESS={}".
              format(datetime.now(), self.address))

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
            # logger.error("Request failed {}: {}".format(response.text, data))
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
        })

    def end_episode(self, episode_id: str, observation: object):
        self._send({
            "command": END_EPISODE,
            "observation": observation,
            "episode_id": episode_id,
        })


if __name__ == "__main__":

    START_TRAINER_URL = 'http://localhost:5002/v1/drl/server/start'

    start_msg = {'action_space': json.dumps(ACTION_SPACE),
                 'observation_space': json.dumps(OBSERVATION_SPACE),
                 'model_config': json.dumps(dqn_config)
                 }

    msg = requests.post(START_TRAINER_URL, data=start_msg)

    print(msg.status_code)
    if msg.status_code != 200:
        print("{} : Trainer Creation failed with ERR={}".
              format(datetime.now(), msg['error']))
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

    world = MKTWorld(env_config)
    drl_trainer = DRLTrainer(trainer_id=trainer_id, trainer_address=trainer_address)

    for i in range(2):  # 20
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
        print("{} : Iteration {} - Mean Reward = {}"
              .format(datetime.now(), i, total / count))

    STOP_TRAINER_URL = 'http://localhost:5002/v1/drl/server/stop'

    print("{} : Stop Trainer ID={}".format(datetime.now(), trainer_id))

    stop_msg = {'id': trainer_id}
    stop_result = requests.post(STOP_TRAINER_URL, data=stop_msg).json()

    print("{} : Stop Trainer ID={} message received {}"
          .format(datetime.now(), trainer_id, stop_result))
