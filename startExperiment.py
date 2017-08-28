# coding=UTF-8
import collections
import itertools
import random
import json
import config
from KerasModel import KerasModel

from ParallelHelper import ParallelHelper
ph = ParallelHelper()

c = config.Config()
diffConfs = []
standardConf = c.standardConf

'''
sensiDef = {
    "maxClasses": range(5,30,5),
    "hiddenNodes": range(100,200,500),
    "hiddenLayers": [1,2,4],
    "classBalancing": ["auto", "own"],
    "textType": ["abstract", "text"],
    "rows": [100000,200000],
    "epochs": [1,3,6]
}

for parameter in sensiDef:
    for value in sensiDef[parameter]:
        conf = {"id": 1}
        conf[parameter] = value
        diffConfs.append(conf)

'''

conf = collections.OrderedDict((
    ("classBalancing", "auto"),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("textLength", 200),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("textLength", 300),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("textLength", 400),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("embeddedSize", 64),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("hiddenNodes", 600),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("dropoutPercentA", 0.1),
    ("id", 1)
))
diffConfs.append(conf)
conf = collections.OrderedDict((
    ("dropoutPercentA", 0.6),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("dropoutPercentB", 0.1),
    ("id", 1)
))
diffConfs.append(conf)
conf = collections.OrderedDict((
    ("dropoutPercentB", 0.6),
    ("id", 1)
))
diffConfs.append(conf)


configList = diffConfs
i = 1
for c in configList:
    c["id"] = i
    diffDict = {}

    ph.writeNewParameters(c, "experiment1")
    i += 1

'''
permMatrix = list(itertools.product(*parameters.values()))
random.shuffle(permMatrix)
iters = len(permMatrix)
'
print "Anzahl der Permutationen:"+str(iters)
'''
