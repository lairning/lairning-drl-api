from flask_restful import Resource


class Version(Resource):
    DECORATORS = []
    ENDPOINT = "/version"

    def get(self):
        return {
            "name": "Lairning DRL API",
            "version": "1.0.0",
            "project": "Lairning"
        }
