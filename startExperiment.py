# coding=UTF-8
from loadTransformClass import LoadTransform
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from tempfile import TemporaryFile
import cPickle as pickle
nltk.data.path.append("/home/dev/PycharmProjects/spiegelMining/nltk_data")
#nltk.download()
import sys
import collections
import itertools
import random
import json
import subprocess
import config

from KerasModel import KerasModel
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



'''
permMatrix = list(itertools.product(*parameters.values()))
random.shuffle(permMatrix)
iters = len(permMatrix)
'
print "Anzahl der Permutationen:"+str(iters)
'''
configList = diffConfs
i = 1
log = open("logs/parallel.log", "w")
for c in configList:
    c["id"] = i
    jsonParams = json.dumps(c)
    execString = "python execModel.py '" + str(jsonParams) + "'"
    print execString
    subprocess.Popen(execString, stdout=log, stderr=log, shell=True)
    i += 1
