import pymongo
import datetime
import socket
import config
import sys

class ParallelHelper:
    mongoConnection = "mongodb://simon:asdf1asdf@ds155473.mlab.com:55473/jobs"
    def __init__(self):
        self.client = pymongo.MongoClient(self.mongoConnection)
        self.db = self.client.jobs
        self.parameters = self.db.parameters
        self.conf = config.Config()

    def getNewParameters(self):
        one = self.parameters.find_one({'status': "new"})
        if one:
            id = one["_id"]
            self.parameters.update({'_id':id},{"$set":{'status':'running'}},upsert=False)
        return one

    def writeNewParameters(self, params, experimentName):
        now = datetime.datetime.now()
        fullP = self.conf.getFullConfig(diffDict = params)
        parameter = {
            "experimentName": experimentName,
            "paramsDiff": params,
            "paramsFull": fullP,
            "status": "new",
            "date": now.strftime("%Y_%m_%d_%H_M"),
            "dateFeedback": "",
            "results": "",
            "host": socket.gethostname(),
            "errors": ""
        }
        id = self.parameters.insert_one(parameter).inserted_id
        pass

    def writeError(self, parameterJob, msg):
        id = parameterJob["_id"]
        now = datetime.datetime.now()
        self.parameters.update({'_id':id},{"$set":{"status": "failed", "errors": msg, "dateFeedback": now.strftime("%Y_%m_%d_%H_M")}}, upsert=False)

    def writeResult(self, parameterJob, results):
        id = parameterJob["_id"]
        now = datetime.datetime.now()
        self.parameters.update({'_id':id},{"$set":{'results':results, "status": "finished", "dateFeedback": now.strftime("%Y_%m_%d_%H_M")}},upsert=False)
