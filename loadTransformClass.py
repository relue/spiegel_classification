# -*- coding: utf-8 -*-
#from pyspark.sql import SparkSession
import pandas as pd
import cPickle as pickle
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from collections import OrderedDict
import numpy as np
import scipy.special
from bokeh.plotting import figure, show, output_file, vplot
from bokeh.charts import Histogram
from keras.preprocessing.sequence import pad_sequences
import os.path
import time
import h5py
import math
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.utils import class_weight
import copy

'''
spark = SparkSession\
    .builder\
    .getOrCreate()
'''

class LoadTransform():
    df = {}
    config = {}
    #hdfsFolder = "hdfs:///tmp/spiegelei/"

    initialLoadCSV = True
    initialLoadTokens = False
    tokenizedArr = []
    repTokenizedArr = []
    paddedRepArr = []

    dictSize = 0
    wordDict = {}
    wordDictCount = {}
    wordDictLabels = {}
    wordDictCountLabels = {}
    labelClassWeights = {}

    tokenizedArrLabels = []
    repTokenizedArrLabels = []
    paddedBinaryLabelArr = []
    nextNumber = 0

    minOccurrence = 20
    minOccurrenceLabels = 1



    timeStart = time.time()

    def __init__(self, config):
        self.config = config

        self.textWordLimit = self.config["textLength"]
        self.maxClasses =  self.config["maxClasses"]
        self.topxDict = self.config["upperDictPercentage"]
        self.batchSize = self.config["rows"]
        self.textType = self.config["textType"]
        self.cachePath = "cache/npArr"+str(self.batchSize)+".h5"

        self.initialLoadSpiegelSet()
        self.getSkipWords()

        cacheParts = len(self.df.index) // self.batchSize

        if (self.initialLoadTokens is True or os.path.isfile(self.cachePath) is not True):
            self.tokenizeDf(self.batchSize)
            print "tokenized df"+self.getD()
            self.createDictionary()
            print "Createed dict"+self.getD()
            self.repTokenizedArr = self.replaceTokensWithDictNumbers(self.tokenizedArr, self.wordDict)
            print "Tokens replaced"+self.getD()
            self.createLabelDict()
            self.wordDictLabels, self.wordDictCountLabels = self.cutDict(self.maxClasses, self.wordDictLabels, self.wordDictCountLabels)
            self.repTokenizedArrLabels = self.replaceTokensWithDictNumbers(self.tokenizedArrLabels, self.wordDictLabels)
            self.createBinaryVectorFromDict(self.wordDictLabels, self.repTokenizedArrLabels)
            self.padSequences()
            self.writeMains()
        else:
            self.loadMains()

        self.calculateClassWeights()

    def writeMains(self):

        with open(self.cachePath+"dict", 'wb') as output:
            pickle.dump(self.wordDict, output, -1)
            pickle.dump(self.wordDictCount, output, -1)
            pickle.dump(self.wordDictLabels, output, -1)
            pickle.dump(self.wordDictCountLabels, output, -1)
            pickle.dump(self.tokenizedArr, output, -1)

        '''
        with open(self.cachePath + "targets", 'wb') as output:
            pickle.dump(self.paddedBinaryLabelArr, output, -1)
            pickle.dump(self.dictSize, output, -1)
        '''

        h5f = h5py.File(self.cachePath, 'w')
        h5f.create_dataset('dataset_1', data=self.paddedRepArr)
        h5f.create_dataset('dataset_2', data=self.paddedBinaryLabelArr)
        h5f.create_dataset('dataset_3', data=self.dictSize)
        #h5f.create_dataset('dataset_4', data=self.tokenizedArr)
        #h5f.create_dataset('dataset_4', data=self.wordDict)
        #h5f.create_dataset('dataset_4', data=self.wordDictLabels)
        h5f.close()

    def loadMains(self):
        with open(self.cachePath+"dict", 'rb') as input:
            self.wordDict = pickle.load(input)
            self.wordDictCount = pickle.load(input)
            self.wordDictLabels = pickle.load(input)
            self.wordDictCountLabels = pickle.load(input)
            self.tokenizedArr = pickle.load(input)

        '''
        with open(self.cachePath+"targets", 'rb') as input:
            self.paddedBinaryLabelArr = pickle.load(input)
            self.dictSize = pickle.load(input)
        '''

        h5f = h5py.File(self.cachePath, 'r')
        self.paddedRepArr = h5f['dataset_1'][:]
        self.paddedBinaryLabelArr = h5f['dataset_2'][:]
        self.dictSize = h5f['dataset_3'][()]
        #self.tokenizedArr = h5f['dataset_4'][:]
        #self.wordDict = h5f['dataset_4'][:]
        #self.wordDictLabels = h5f['dataset_4'][:]
        h5f.close()

    def createInputVecFromText(self, inputText):
        wordList = word_tokenize(inputText, language="german")
        newList = []
        wrapperList = []
        for w in range(0, len(wordList)):
            wordText = wordList[w]
            decWord = self.replaceDecimalFromDict(wordText, self.wordDict)
            newList.append(decWord)
        maxLength = self.getSequenceLength()

        for i in range(len(wordList)+1l, maxLength+1):
            newList.append(0)
        wrapperList.append(newList)
        wrapperArr = np.array(wrapperList)
        return wrapperArr

    def tokenizeDf(self, batchSize=100):
        df = self.df.ix[0:batchSize, :]
        textList = []
        keywordList = []

        for index, row in df.iterrows():
            text = row[self.textType].decode('utf8').lower()
            soup = BeautifulSoup(text)
            text = soup.get_text()

            text = re.sub("(^|\W)\d+($|\W)", " ", text)
            wordList = word_tokenize(text, language="german")
            wordsFiltered = [w.lower() for w in wordList if w.isalpha()]
            textList.append(wordsFiltered[:self.textWordLimit])
            keyWords = row['keywords'].split(", ", 3)
            keyWords = keyWords[0:3]
            keywordList.append(keyWords)
        self.tokenizedArr = np.asarray(textList)
        self.tokenizedArrLabels = np.asarray(keywordList)

    def showSampleFrame(self):
        dataExplore2.showDF(self.df, False)

    def initialLoadSpiegelSet(self):
        if self.initialLoadCSV:
            nameList = ["headline","keywords", "abstract","text","date", "bol1", "bol2"]
            self.df = pd.read_csv('data/spiegel_new.csv', names=nameList, header=None, sep='|')
            #self.df = spark.read.csv("/tmp/spiegelei/spiegel_new.csv",sep="|",header=False,inferSchema=True).toPandas()
            self.df = self.df[['keywords', self.textType]]
            self.df.dropna(axis=0, how='any', inplace=True)
            self.df.to_pickle("spiegel.p")
        else:
            self.loadSpiegelSet()

    def loadSpiegelSet(self):
        self.df = pd.read_pickle("spiegel.p")

    def createDictionary(self):
        lines = self.tokenizedArr.shape[0]
        self.nextNumber = 0
        self.wordDict["$UNC$"] = self.findFreeNumber(self.wordDict)
        for line in range(0, lines):
            for word in range(0, len(self.tokenizedArr[line])):
                self.checkAndUpdateDict(self.tokenizedArr[line][word],self.wordDictCount, self.wordDict, self.minOccurrence)

        self.dictSize = len(self.wordDict)
        self.wordDictCount = OrderedDict(sorted(self.wordDictCount.items(), key=lambda t: t[1], reverse=True))
        self.wordDict, self.wordDictCount = self.cutDict(0, self.wordDict, self.wordDictCount, topX = self.topxDict)
        self.reorganizeWordDict()
        pass

    def reorganizeWordDict(self):
        i = 1
        for word in self.wordDict:
            self.wordDict[word] = i
            i += 1

    def getSkipWords(self):
        stopList = stopwords.words('german')
        self.skipWords = stopList

    def createLabelDict(self):
        lines = self.tokenizedArrLabels.shape[0]
        self.nextNumber = 0
        self.wordDictLabels["$UNC$"] = self.findFreeNumber(self.wordDictLabels)

        for line in range(0, lines):
            self.checkAndUpdateDict(self.tokenizedArrLabels[line][0], self.wordDictCountLabels, self.wordDictLabels, self.minOccurrenceLabels)

        ##add UNC token to dict and sort

        self.wordDictCountLabels = OrderedDict(sorted(self.wordDictCountLabels.items(), key=lambda t: t[1], reverse=True))
        self.wordDictLabels = OrderedDict(sorted(self.wordDictLabels.items(), key=lambda t: t[1], reverse=False))

    def cutDict(self, maxClasses, wordDict, wordDictCount, topX = 0):
        sum = len(wordDict)
        if topX > 0:
            maxClasses = sum*topX
        newDict = {}
        newDictCount = {}
        iterations = 0
        for i in wordDictCount:
            newDict[i] = wordDict[i]
            newDictCount[i] = wordDictCount[i]
            iterations += 1
            if iterations == int(maxClasses):
                break
        newDict["$UNC$"] = wordDict["$UNC$"]
        newDictCount["$UNC$"] = 1
        wordDict = OrderedDict(sorted(newDict.items(), key=lambda t: t[1], reverse=False))
        wordDictCount = OrderedDict(sorted(newDictCount.items(), key=lambda t: t[1], reverse=True))
        return wordDict, wordDictCount

    def calculateClassWeights(self):
        #class_weight1 = class_weight.compute_class_weight('balanced', np.unique(self.paddedBinaryLabelArr), self.paddedBinaryLabelArr)

        all = 0
        self.labelClassWeightsNon = {}
        for id in self.wordDictCountLabels:
            all += self.wordDictCountLabels[id]
        i = 0
        for id in self.wordDictCountLabels:
            self.labelClassWeightsNon[i] = 1 / (self.wordDictCountLabels[id]*1.0 / all)
            partial = (1 / (self.wordDictCountLabels[id]*1.0 / all)**3)
            self.labelClassWeights[i] = ((math.log1p(partial)**1.5) / 10)+1
            i += 1

        pass

    def getClassWeights(self):
        return self.labelClassWeights

    def checkAndUpdateDict(self, word, wordCount, wordDict, minOccurence, skipNumbering=False):
        if word in self.skipWords:
            return

        if word in wordCount:
            wordCount[word] += 1
        else:
            wordCount[word] = 1

        if (wordCount[word] > minOccurence):
            if not word in wordDict:
                if not skipNumbering:
                    wordDict[word] = self.findFreeNumber(wordDict)
                else:
                    wordDict[word] = 1

    def findFreeNumber(self, wordDict):
        self.nextNumber += 1
        return self.nextNumber

    def replaceTokensWithDictNumbers(self, tokenizedArr, wordDict):
        repTokenizedArr = copy.deepcopy(tokenizedArr)
        lines = repTokenizedArr.shape[0]
        countUnc = 0
        for line in range(0, lines):
            for word in range(0, len(repTokenizedArr[line])):
                wordText = repTokenizedArr[line][word]
                repTokenizedArr[line][word] = self.replaceDecimalFromDict(wordText, wordDict)
                if repTokenizedArr[line][word] == wordDict["$UNC$"]:
                    countUnc += 1
        self.wordDictCount["$UNC$"] = countUnc
        return repTokenizedArr

    def replaceDecimalFromDict(self, wordText, wordDict):
        if wordText in wordDict:
            return wordDict[wordText]
        else:
            return wordDict["$UNC$"]

    def plotCounts(self, wordDict, name):
        occurences = [s for s in wordDict.values() if s > self.minOccurrence]
        output_file('plots/histogram'+name+'.html')

        p = Histogram(occurences, title="MPG Distribution", bins=7)
        show(p)

    #UNC Token berücksichtigen!
    def createBinaryVectorFromDict(self, wordDict, arrRepLabels):
        self.binaryLabelArr = []
        lines = arrRepLabels.shape[0]
        countUnc = 0
        for line in range(0, lines):
            labelValues = []
            for dictEntry in wordDict.values():
                exists = False
                if dictEntry == arrRepLabels[line][0]:
                    labelValues.append(1)
                else:
                    labelValues.append(0)
            if wordDict["$UNC$"] == arrRepLabels[line][0]:
                countUnc += 1
            self.binaryLabelArr.append(labelValues)
        self.wordDictCountLabels["$UNC$"] = countUnc
        self.binaryLabelArr = np.array(self.binaryLabelArr)

    def padSequences(self):
        self.paddedRepArr = pad_sequences(self.repTokenizedArr, maxlen=None, dtype='int32',
                                                   padding='post', truncating='pre', value=0.)
        self.paddedBinaryLabelArr = pad_sequences(self.binaryLabelArr, maxlen=None, dtype='int32',
                                                  padding='post', truncating='pre', value=0.)

    def getSequenceLength(self):
        return self.paddedRepArr.shape[1]

    def getTargetLength(self):
        return self.paddedBinaryLabelArr.shape[1]

    def getInputMatrix(self):
        return self.paddedRepArr

    def getTargetMatrix(self):
        return self.paddedBinaryLabelArr

    def getDictSize(self):
        return self.dictSize

    def getD(self):
        end = time.time() - self.timeStart
        return str(end)
