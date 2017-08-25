import keras
from ParallelHelper import ParallelHelper
class EpochResultRetrieve(keras.callbacks.Callback):

    def __init__(self, inputV, outputV, parameterJob):
        self.inputV = inputV
        self.outputV = outputV
        self.parameterJob = parameterJob
        self.EpochResults = {}

    def on_epoch_end(self, epoch, logs=None):
        self.EpochResults[str(epoch)] = logs
        ph = ParallelHelper()
        ph.writeTempResult(self.parameterJob, self.EpochResults)
        self.model.save_weights('cache/model_'+str(self.parameterJob['_id'])+'.h5')

    def returnEpochLogs(self):
        return self.EpochResults
