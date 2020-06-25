from flask_restful import Resource, reqparse
#from lairning_core import DRLServer
from flask_injector import inject
from drl.api.drl_server import DRLServer
import json

class DRLServerStart(Resource):
    DECORATORS = []
    ENDPOINT = "/drl/server/start"

    @inject
    def __init__(self, drl_server: DRLServer):
        self.args = reqparse.RequestParser()
        self.args.add_argument("action_space")
        self.args.add_argument("observation_space")
        self.args.add_argument("model_config")
        # self.args.add_argument("action_space", type=dict)
        # self.args.add_argument("observation_space", type=dict)
        # self.args.add_argument("model_config", type=dict)

        self.drl_server = drl_server

    def post(self):

        try:
            args = self.args.parse_args()
        except Exception as err:
            print(err)
            raise err

        payload = {
            "action_space": json.loads(args.action_space),
            "observation_space": json.loads(args.observation_space),
            "model_config": json.loads(args.model_config)
        }
        print(payload)
        return None #self.drl_server.start_trainer(payload=payload)


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

