from flask_restful import Resource, reqparse
from lairning_core import DRLServer
from flask_injector import inject


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

    def post(self):

        args = self.args.parse_args()

        payload = {
            "action_space": args.action_space,
            "observation_space": args.observation_space,
            "model_config": args.model_config
        }

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

