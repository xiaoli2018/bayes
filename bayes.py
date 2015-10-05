#-*-coding:utf-8-*-
__author__ = 'liheng'
from numpy import *
#world list to vector function
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', \
'problems', 'help', 'please'],
['maybe', 'not', 'take', 'him', \
'to', 'dog', 'park', 'stupid'],
['my', 'dalmation', 'is', 'so', 'cute', \
'I', 'love', 'him'],
['stop', 'posting', 'stupid', 'worthless', 'garbage'],
['mr', 'licks', 'ate', 'my', 'steak', 'how',\
'to', 'stop', 'him'],
['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  #1 is abusive , 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # merge the two set
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word :%s is not in my Vocabulary!" %word
    return returnVec

#native bayes classifier training function

def tarinNBO(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])


















