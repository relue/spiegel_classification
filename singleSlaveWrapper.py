# coding=UTF-8
from loadTransformClass import LoadTransform

import numpy as np
from tempfile import TemporaryFile
import cPickle as pickle
import sys
import json
import config
import traceback
from KerasModel import KerasModel
from ParallelHelper import ParallelHelper
from random import randint
import time

ph = ParallelHelper()
loop = True
c = config.Config()
time.sleep(randint(0, 60))

parameterJob = ph.getNewParameters()
if not parameterJob:
    print "No jobs left"
    exit()

diffDict = parameterJob['paramsFull']
'''
diffDict = {
    "rows": 1000,
    "hiddenNodes": 400,
    "hiddenLayers": 1,
    "epochs": 2,
    "id": 1
}
'''
try:
    while loop:
        configDict = c.getFullConfig(diffDict = diffDict)
        print "execute Diff:"+str(parameterJob['paramsDiff'])
        print "execute Full:"+str(dict(configDict))
        transformer = LoadTransform(configDict)
        kerasModel = KerasModel(transformer, configDict)
        kerasModel.defineModelStructure(transformer.getDictSize(), transformer.getTargetLength(), transformer)
        kerasModel.fitModel(transformer, parameterJob = parameterJob)
        results = kerasModel.getMeasures(kerasModel.inputV, kerasModel.inputVReal, kerasModel.outputV)
        ph.writeResult(parameterJob, results)
        parameterJob = ph.getNewParameters()
        if not parameterJob:
            break
        diffDict = parameterJob['paramsFull']
        if 1 > 2:
            loop = False

except Exception as e:
    tb = traceback.format_exc()
    ph.writeError(parameterJob, tb)
    print tb
    raise



