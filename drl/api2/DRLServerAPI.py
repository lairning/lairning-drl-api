from flask_restful import Resource, reqparse
from flask_injector import inject
# from lairning_core.drl import DRLServer
from drl.drl_server2 import DRLServer

import json
from datetime import datetime

class DRLServerStart(Resource):
    DECORATORS = []
    ENDPOINT = "/drl/server/start"

    @inject
    def __init__(self, drl_server: DRLServer):
        self.args = reqparse.RequestParser()
        self.args.add_argument("action_space_size", type=int)
        self.args.add_argument("observation_space_size", type = int)
        self.args.add_argument("model_config")
        # self.args.add_argument("action_space", type=dict)
        # self.args.add_argument("observation_space", type=dict)
        # self.args.add_argument("model_config", type=dict)
        self.drl_server = drl_server

    def post(self):

        try:
            args = self.args.parse_args()
            payload = {
                "action_space_size": args.action_space_size,
                "observation_space_size": args.observation_space_size,
                "model_config": json.loads(args.model_config)
            }
        except Exception as err:
            print("{} : DRLServerStart failed. ERR={}".
                  format(datetime.now(), err))
            raise err

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

        try:
            args = self.args.parse_args()
            payload = {"id": args.id}
        except Exception as err:
            print("{} : DRLServerStart failed. ERR={}".
                  format(datetime.now(), err))
            raise err

        return self.drl_server.stop_trainer(payload=payload)

