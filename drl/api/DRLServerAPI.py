from flask_restful import Resource, reqparse
#from lairning_core import DRLServer
from flask_injector import inject
from drl.api.drl_server import DRLServer
from datetime import datetime
import os

stdout = 'DRLServerAPI_{:%Y-%m-%d_%H:%M:%S%f}.log'.format(datetime.now())
stderr = 'DRLServerAPI_{:%Y-%m-%d_%H:%M:%S%f}.log'.format(datetime.now())


class DRLServerStart(Resource):
    DECORATORS = []
    ENDPOINT = "/drl/server/start"

    @inject
    def __init__(self, drl_server: DRLServer):
        self.args = reqparse.RequestParser()
        self.args.add_argument("action_space", type=dict)
        self.args.add_argument("observation_space", type=dict)
        self.args.add_argument("model_config", type=dict)
        self.drl_server = drl_server
        with open(stdout, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
        with open(stderr, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stderr.fileno())

    def post(self):

        args = self.args.parse_args()

        payload = {
            "action_space": args.action_space,
            "observation_space": args.observation_space,
            "model_config": args.model_config
        }
        print("{} : [DRL Server API] Payload {}"
              .format(datetime.now(), payload))
        return self.drl_server.start_trainer(payload=payload)


class DRLServerStop(Resource):
    DECORATORS = []
    ENDPOINT = "/drl/server/stop"

    @inject
    def __init__(self, drl_server: DRLServer):
        self.args = reqparse.RequestParser()
        self.args.add_argument("id", type=int, required=True)
        self.drl_server = drl_server

    def post(self):
        args = self.args.parse_args()
        payload = {"id": args.id}
        return self.drl_server.stop_trainer(payload=payload)

