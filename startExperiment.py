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
standardConf = c.standardConf

diffConfs = []
conf = collections.OrderedDict((
    ("maxClasses", 8),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("hiddenLayers", 1),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("classBalancing", "auto"),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("textType", "abstract"),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("hiddenNodes", "600"),
    ("id", 1)
))
diffConfs.append(conf)

conf = collections.OrderedDict((
    ("rows", "300000"),
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
