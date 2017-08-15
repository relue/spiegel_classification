from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Masking
from collections import OrderedDict
import math

class KerasModel():
    model = None
    modelSavePath = "cache/model.h5"
    inputT = []
    outputT = []
    inputV = []
    outputV = []

    inputTReal = []
    inputVReal = []

    transformer = None

    def __init__(self, transformer):
        self.transformer = transformer
        inputVec = transformer.getInputMatrix()
        targetVec = transformer.getTargetMatrix()
        self.getTestSets(inputVec, targetVec, self.transformer.tokenizedArr, 0.1)

    def getTestSets(self, xInput, xOutput, origArr, percentage):
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

    def defineModelStructure(self, maxFeatures, maxTargets, transformer):
        self.model = Sequential()
        inputMatrix = transformer.getInputMatrix()
        shape = inputMatrix.shape
        #self.model.add(Masking(mask_value=0., input_shape=(shape[1], 1)))
        self.model.add(Embedding(maxFeatures + 1, output_dim=64, mask_zero=True)) #, mask_zero=True
       # self.model.add(LSTM(600, return_sequences=True))
        self.model.add(LSTM(200))
        self.model.add(Dense(maxTargets, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])

    def fitModel(self, transformer, saveModel = False):
        self.model.fit(self.inputT, self.outputT, nb_epoch=2, batch_size=100, validation_split=0.1, class_weight = transformer.getClassWeights())

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

        test = self.getLabelByLabelVec(outputVec[i-1], self.transformer.wordDictLabels)
        #missclassifications summary
        summaryList = []
        for p in subList:
            inputVecReal, classEstimate, outputVecLabel = p
            if classEstimate[1] == outputVecLabel:
                match = 1
            else:
                match = 0
            p.append(match)

            summaryList.append(p)
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
