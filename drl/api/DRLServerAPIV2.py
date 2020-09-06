from flask_restful import Resource, reqparse
from flask_injector import inject
# from lairning_core.drl import DRLServer
from drl.drl_serverV2 import DRLServer

import json
from datetime import datetime

class DRLServerStart(Resource):
    DECORATORS = []
    ENDPOINT = "/drl/server/start"

    @inject
    def __init__(self, drl_server: DRLServer):
        self.args = reqparse.RequestParser()
        self.args.add_argument("action_space")
        #self.args.add_argument("action_mask")
        self.args.add_argument("observation_space")
        self.args.add_argument("model_config")
        self.drl_server = drl_server

    def post(self):

        try:
            args = self.args.parse_args()
            payload = {
                "action_space": json.loads(args.action_space),
                #"action_mask": json.loads(args.action_mask),
                "observation_space": json.loads(args.observation_space),
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

