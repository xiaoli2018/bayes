#-*- coding:utf-8 -*-
__author__ = 'liheng'
from numpy import *
#向算法提供训练集
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
    return postingList,classVec   #返回一个训练集的文档列表和列表所对应的标签

#传入的是训练集的文档列表，然后将文档列表中的字符进行去重处理，得到一个训练集的字母表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # merge the two set
    return list(vocabSet)   #返回去重之后的字母表

#传入的是训练集的字母列表，和输入的一个文档列表，检测如何的文档中的字符是否出现在字母表中，
# 返回一个只包含0,1元素的列表，其中的1代表这个输入的文档中的字符在字母表中对应的位置出现
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word :%s is not in my Vocabulary!" %word
    return returnVec

#native bayes classifier training function
def trainNB0(trainMatrix,trainCategory):  #ftainMatrix 为输入的文档矩阵，trainCategory为每篇文档类别标签所构成的向量
    '''
函数的功能是将三种概率计算出来： p1Vect 代表的是每个单词在侮辱性的字符列表中出现的概率  p0Vect 代表的是每个单词在非侮辱性的字符列表中出现的概率
                            pAbusive 代表的是侮辱性的列表在总的列表中的概率
传入的参数：      trinMatrix 是一个由0,1组成的列表矩阵，包含了每个文档列表中的单词在总的字母表中出现的情况
                trainCategory是开始提供的标记好了的文档的标签
在此函数中，通过找出在p1Vect 中的出现的概率最高的字母就能确定此单词是带有侮辱性的单词
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)   #代表的非侮辱性的概率的分子
    p1Num = ones(numWords)  #代表的是侮辱性的概率的分子
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num += trainMatrix[i]         #都是相加的处理，这里体现出了利用的是朴素贝叶斯
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)   #这里用到了自然对数log 是为了防止出现下溢的出现，防止出现的数太小导致四舍五入取值不准确
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#native bayes classify function
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''

    :param vec2Classify:   将待测试的文档转化为0,1向量列表的参数
    :param p0Vec:     上面训练算法得到的非侮辱性的概率
    :param p1Vec:     上面训练算法得到的侮辱性的概率
    :param pClass1:   上面训练得到的总体的侮辱性在列表中所占的概率
    :return:      返回函数的预测值   1代表输入的文档是侮辱性的言论，0代表的是非侮辱性的言论
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#一个封装好了的分类器
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))  #将给定的文档列表转换成一个0,1的向量
    print testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))    #将一个给定的文档转换成一个0,1的向量
    print testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb)

#朴素贝叶斯词袋模型
def bagOfWord2VecMN(vocabList,inputSet):
    '''
    这里要实现的是朴素贝叶斯的第二种方法：利用多项式模型
    解决的是一个词可能出现多次的情况
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]   #去掉了少于两个字符的字符串

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt'%i).read()) #spam 表示的是垃圾邮件
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)   #构建字符表
    trainingSet = range(50)
    testSet = []
    '''
    下面的两个for循环实现的 “留存交叉验证”
    随机选出10个测试集，剩余的用作训练集

    '''
    #随机选取测试集集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))  #随机返回在两个参数之间的数
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])   #将测试集对应的索引剔除
    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #测试错误率
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is :',float(errorCount)/len(testSet)

#计算出项的频率
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedFreq[:30]
#以Rss源作为输入
def localWords(feed1,feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount +=1
    print 'the error rate is :',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

#显示地域相关的用词
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key = lambda pair:pair[1],reverse = True)
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY,key = lambda pair:pair[1],reverse = True)
    for item in sortedNY:
        print item[0]










