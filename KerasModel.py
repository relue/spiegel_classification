from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Masking

from collections import OrderedDict
import math
from random import shuffle
import numpy as np
import json
import datetime
import time
start_time = time.time()



class KerasModel():
    model = None
    modelSavePath = "cache/model.h5"
    inputT = []
    outputT = []
    inputV = []
    outputV = []
    config = {}
    inputTReal = []
    inputVReal = []

    transformer = None

    def __init__(self, transformer, config):
        self.config = config
        self.transformer = transformer
        inputVec = transformer.getInputMatrix()
        targetVec = transformer.getTargetMatrix()
        self.getTestSets(inputVec, targetVec, self.transformer.tokenizedArr, self.config["testPercent"])

    def getTestSets(self, xInput, xOutput, origArr, percentage, shuffle = True):
        if shuffle:
             xInput, xOutput = self.shuffleData(xInput, xOutput)
        fromT = 0
        last = xOutput.shape[0]
        toT = int(math.floor(last * (1 - percentage)))
        self.inputT = xInput[fromT:toT, :]
        self.outputT = xOutput[fromT:toT, :]
        self.inputV = xInput[toT + 1:last, :]
        self.outputV = xOutput[toT + 1:last, :]

        self.inputTReal = origArr[fromT:toT]
        self.inputVReal = origArr[toT + 1:last]
        return self.inputT, self.outputT, self.inputV, self.outputV, self.inputTReal, self.inputVReal

    def shuffleData(self, xInput, xOutput):
        xInputShuf = []
        xOutputShuf = []
        index_shuf = range(len(xInput))
        shuffle(index_shuf)
        for i in index_shuf:
            xInputShuf.append(xInput[i])
            xOutputShuf.append(xOutput[i])
        return np.asarray(xInputShuf), np.asarray(xOutputShuf)

    def defineModelStructure(self, maxFeatures, maxTargets, transformer):
        self.model = Sequential()
        inputMatrix = transformer.getInputMatrix()
        shape = inputMatrix.shape
        #self.model.add(Masking(mask_value=0., input_shape=(shape[1], 1)))
        self.model.add(Embedding(maxFeatures + 1, output_dim=self.config["embeddedSize"])) #, mask_zero=True

        returnSequence = True if self.config["hiddenLayers"] > 1 else False
        self.model.add(LSTM(self.config["hiddenNodes"], return_sequences=returnSequence))
        for hdI in range(2,self.config["hiddenLayers"]+1):
            if hdI == self.config["hiddenLayers"]:
                returnSequence = False

            self.model.add(LSTM(self.config["hiddenNodes"], return_sequences=returnSequence))

        self.model.add(Dense(maxTargets, activation='softmax'))
        self.model.compile(optimizer=self.config["optimizer"],
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])


    def fitModel(self, transformer, saveModel = False):
        #self.model.fit(self.inputT, self.outputT, nb_epoch=1, batch_size=100, validation_split=0.3, class_weight = transformer.getClassWeights())
        if self.config["classBalancing"] == "own":
            classWeightType = transformer.getClassWeights()
        elif self.config["classBalancing"] == "self":
            classWeightType = "auto"

        self.model.fit(self.inputT, self.outputT, nb_epoch=self.config["epochs"], batch_size=self.config["batchSize"], validation_split=self.config["validationPercent"], class_weight = classWeightType)

    def predictModel(self, inputVec):
        return self.model.predict(inputVec, batch_size=1, verbose=2)

    def getMeasures(self, inputVec, inputVecReal, outputVec):
        classifiedList = []
        lengthInput = inputVec.shape[0]
        predictions = self.model.predict(inputVec, batch_size=lengthInput)
        i = 0
        subList = []

        for prediction in predictions:
            topList = self.getKeywordsFromPrediction(prediction, self.transformer.wordDictLabels)
            classEstimate = topList[0]
            '''
            if classEstimate[1] == "$UNC$":
                classEstimate = topList[1]
            '''
            subList.append([inputVecReal[i], classEstimate, self.getLabelByLabelVec(outputVec[i], self.transformer.wordDictLabels)])
            classifiedList.append(subList)
            i += 1

        #test = self.getLabelByLabelVec(outputVec[i-1], self.transformer.wordDictLabels)
        #missclassifications summary
        summaryList = []
        categoryScores = {}

        matches = 0

        for p in subList:
            inputVecReal, classEstimate, outputVecLabel = p
            if not categoryScores.has_key(outputVecLabel):
                categoryScores[outputVecLabel] = {}
                categoryScores[outputVecLabel]['trueClassified'] = 0
                categoryScores[outputVecLabel]['falseClassified'] = 0
                categoryScores[outputVecLabel]['labelOccurence'] = 0
                categoryScores[outputVecLabel]['estimateOccurence'] = 0

            if not categoryScores.has_key(classEstimate[1]):
                categoryScores[classEstimate[1]] = {}
                categoryScores[classEstimate[1]]['trueClassified'] = 0
                categoryScores[classEstimate[1]]['falseClassified'] = 0
                categoryScores[classEstimate[1]]['labelOccurence'] = 0
                categoryScores[classEstimate[1]]['estimateOccurence'] = 0

                #categoryScores[outputVecLabel]['estimateOccurence'] = 0

            if not categoryScores.has_key('all'):
                categoryScores['all'] = {}
                categoryScores['all']['trueClassified'] = 0
                categoryScores['all']['falseClassified'] = 0

            if classEstimate[1] == outputVecLabel:
                match = 1
                categoryScores[outputVecLabel]['trueClassified'] += 1
                categoryScores['all']['trueClassified'] += 1
                matches += 1
            else:
                categoryScores[outputVecLabel]['falseClassified'] += 1
                categoryScores['all']['falseClassified'] += 1
                match = 0

            categoryScores[outputVecLabel]['labelOccurence'] += 1
            categoryScores[classEstimate[1]]['estimateOccurence'] += 1

            p.append(match)
            summaryList.append(p)

        categoryScores['all']['labelOccurence'] = len(subList)
        accuracy = float(matches)/len(subList)
        #output = "acc:"+str(accuracy)+"\n"+"parameters:"+str(self.config) + "\n scores:"+str(categoryScores)

        results = OrderedDict((
            ("acc", accuracy),
            ("config", self.config),
            ("scores", categoryScores),
            ("id", self.config["id"])
        ))
        results["exec_time"] = (time.time() - start_time)
        return results

    def writeToLog(self, results):
        now = datetime.datetime.now()
        filenamePrefix = now.strftime("%Y_%m_%d_%H_"+str(results["id"])+".rst")
        with open("results/"+filenamePrefix,'w') as o:
            json.dump(results, o)
        pass

    def getLabelByLabelVec(self, vec, wordDict):
        i = 0
        for v in vec:
            if v == 1:
                return wordDict.items()[i][0]
            i += 1
        return

    def getKeywordsFromPrediction(self, prediction, wordDict):
        outputDict = {}
        topListDict = []
        for i in range(0, len(prediction)):
            outputDict[i] = prediction[i]
        outputDict = OrderedDict(sorted(outputDict.items(), key=lambda t: t[1], reverse=True))
        count = 1

        #wordDict = dict((v, k) for k, v in wordDict.iteritems())
        for t in outputDict:
            topListDict.append([outputDict[t],wordDict.items()[t][0]])
            count += 1
            if count == 4:
                break
        return topListDict

    def loadModel(self):
        self.model = load_model(self.modelSavePath)

    def saveModel(self):
        self.model.save(self.modelSavePath)
