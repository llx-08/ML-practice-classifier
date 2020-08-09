import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    # labels: 每一个字列表为句子分词后组成的，classVec相当于该list的labels：0：非侮辱性，1：侮辱性

    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # 创建空集

    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 加入新元素

    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

        else:
            print("the word: " + word + "is not in my vocabulary")

    return returnVec


def trainNB0_CountWordsCalculateProbability(trainMatrix, trainCatgory):
    numTrainDocs = len(trainMatrix)

    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCatgory) / float(numTrainDocs)  # 计算侮辱类文档占总文档的多少

    p0Num = np.ones(numWords)  # 每个文档对应的词矩阵
    p1Num = np.ones(numWords)

    p0Denom = 2.0  # 词总数量,避免下溢
    p1Denom = 2.0

    for i in range(numTrainDocs):

        if trainCatgory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num / p1Denom)  #
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0


def testingDB():  # test
    listOPosts, listClasses = loadDataSet()

    myVocabList = createVocabList(listOPosts)

    trainMatrix = []

    for postinDoc in listOPosts:
        trainMatrix.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0_CountWordsCalculateProbability(np.array(trainMatrix), np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry, end="")
    print("classified as: ", end="")
    print(classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ["stupid", "garbage"]

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry, end="")
    print("classified as: ", end="")
    print(classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList, inputSet):  # 计数 instead of 置1

    returnVec = [0] * len(vocabList)

    for words in inputSet:
        if words in vocabList:
            returnVec[vocabList.index(words)] += 1

    return returnVec


