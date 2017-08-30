import keras
import time
from ParallelHelper import ParallelHelper
class EpochResultRetrieve(keras.callbacks.Callback):

    def __init__(self, inputV, outputV, parameterJob):
        self.inputV = inputV
        self.outputV = outputV
        self.parameterJob = parameterJob
        self.EpochResults = {}

    def on_epoch_end(self, epoch, logs=None):
        self.EpochResults[str(epoch)] = logs
        logs["updateTime"] = time.time()
        ph = ParallelHelper()
        if self.parameterJob:
            ph.writeTempResult(self.parameterJob, self.EpochResults)
            id = self.parameterJob['_id']
        else:
            id = 1
        self.model.save_weights('cache/model_'+str(id)+'.h5')

    def returnEpochLogs(self):
        return self.EpochResults
