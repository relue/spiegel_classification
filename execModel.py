# coding=UTF-8
from loadTransformClass import LoadTransform
#execfile("loadTransformClass.py")
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from tempfile import TemporaryFile
import cPickle as pickle

nltk.data.path.append("/home/dev/PycharmProjects/spiegelMining/nltk_data")

#nltk.download()
import sys

from KerasModel import KerasModel

'''
todos:
1. Längen der Texte analysieren
2. Tokenizer anwenden
2. Listen iterieren und Dictionary mit Worthäufigkeiten erstellen
3. Unc Ersetzungen vornehmen
4. Pad Sequencing (gleich lange Listen erstellen) Keras Sequence PreProcessing (padSequence truncate)
5. Embedding Matrix einspeißen
'''
inputTexts = []
inputTexts.append("Nach den Kämpfen in Mossul nahmen irakischen Sicherheitskräfte 20 ausländische Dschihadistinnen fest. Jetzt steht fest: Neben der Schülerin Linda W. stehen drei weitere deutsche Frauen unter IS-Verdacht.")
inputTexts.append("Im neuen Podcast Netzteil wirft das Netzwelt-Ressort ab jetzt wöchentlich einen Blick in die Zukunft. In der ersten Folge geht es um Sprachsteuerung - und Siri und Alexa reden mit.")
#inputTexts.append("Wegschließen oder was. Das wissen wir nicht. In diesem Polizeiruf um Sexualstraftäter gibt es keine gedämpften Talkshow-Abwägungen. Harter Stoff am Neujahrsabend.")
inputTexts.append("Die Pflegereform stellt Demenzkranke finanziell besser und stärkt die häusliche Versorgung. Wie bei jeder Reform gibt es allerdings auch Verlierer.")
newModel = True

transformer = LoadTransform()
kerasModel = KerasModel(transformer)

if newModel:
    kerasModel.defineModelStructure(transformer.getDictSize(), transformer.getTargetLength(), transformer)
    kerasModel.fitModel(transformer)
    kerasModel.saveModel()
else:
    #modelOut = transformer.showSampleFrame()
    kerasModel.loadModel()
    kerasModel.getMeasures(kerasModel.inputV, kerasModel.inputVReal, kerasModel.outputV)
    '''
    for inputText in inputTexts:
        inputVec = transformer.createInputVecFromText(inputText)
        prediction = kerasModel.predictModel(inputVec)
        print prediction
        predictedLabels = kerasModel.getKeywordsFromPrediction(prediction, transformer.wordDictLabels)
        print predictedLabels
    testData = transformer.getInputMatrix()
    print kerasModel.predictModel(testData[0:100,])
    '''

df = transformer.df
