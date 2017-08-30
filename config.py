import collections
import copy


class Config():
    standardConf = collections.OrderedDict((
        ("rows", 0),
        ("maxClasses", 8),
        ("textLength", 200),
        ("embeddedSize", 128),
        ("hiddenNodes", 600),
        ("hiddenLayers" , 2),
        ("learningRate", 0.001),
        ("optimizer", "adam"),
        ("textType", "text"),
        ("classBalancing", "own"),
        ("upperDictPercentage", 0.95),
        ("validationPercent", 0.1),
        ("testPercent", 0.02),
        ("batchSize", 100),
        ("epochs", 7),
        ("id", 1),
        ("dropoutPercentA", 0.0),
        ("dropoutPercentB", 0.0)
    ))

    parameters = collections.OrderedDict((
      #  ("learningRate", [0.001, 0.01, 0.05]),
      #  ("optimizer", ["adam", "sgd"]),
    # netType
        ("rows", [50000,100000,200000]),
        ("textType", ["abstract", "full"]),
        ("embeddedSize", [64, 128]),
        ("textLength", [100, 200, 300]),
       # ("hiddenNodes", [200,400,600]),
        ("hiddenLayers" , [1, 2]),
        ("batchSize", [100,200,500])
    ))

    def getFullConfig(self, diffDict = {}):
        mergedDict = copy.deepcopy(self.standardConf)
        for k in diffDict:
            mergedDict[k] = diffDict[k]
        return mergedDict
