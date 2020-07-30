import numpy as np
import operator
import treePloter
import pickle
from math import log

#  test
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 0, 'no']
    ]

    labels = ['label1', 'label2']

    return dataSet, labels


def calcShannonEnt(dataSet):  # 计算香农熵

    numEntries = len(dataSet)

    labelCounts = {}

    for featuresVector in dataSet:

        currentLabel = featuresVector[-1]

        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries

        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, axis, value):  # 划分数据集
    # 带划分的数据集， 划分数据集的特征， 返回的特征的值:
    # eg 特征a,b,c 按照a划分即为axis=0,value:按照a=value分出对应的其他特征集合
    returnDataSet = []

    for featureVector in dataSet:

        if featureVector[axis] == value:

            reducedFeatVc = featureVector[:axis]
            reducedFeatVc.extend(featureVector[axis + 1:])
            returnDataSet.append(reducedFeatVc)

    return returnDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy

        if infoGain > bestInfoGain:  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i

    return bestFeature  # returns an integer


def majorityCnt(classList):  # 递归创建决策树

    classCount = {}

    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 与kNN排序代码类似，见kNN.py

        return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别完全相同：停止划分

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 遍历完所有特征,仍然无法将剩余数据集划分为唯一一类：返回出现次数最多的类别

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}}

    del (labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]

    uniqueVals = set(featValues)

    for value in uniqueVals:  # 遍历当前所有属性值，递归调用，返回嵌套字典
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


def classify(inputTree, featLabels, testVec):  # 递归调用，每次都判断

    firstStr = list(inputTree)[0]  # 当前的第一个特征
    secondDict = inputTree[firstStr]  # 第一个特征划分出的子树

    featIndex = featLabels.index(firstStr)  # 找到特征的索引

    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    if isinstance(valueOfFeat, dict):  # 是字典：还有子树
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:  # 不是字典，叶节点
        classLabel = valueOfFeat

    return classLabel


def storeTree(inputTree, filename):

    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):

    fr = open(filename, 'rb')
    return pickle.load(fr)


# test area

# myData, myLabs = createDataSet()
# print(myData)
# print(myLabs)
# myTree = createTree(myData, myLabs)
# print(myTree)
# print(splitDataSet(myData, 0, 0))
# print(splitDataSet(myData, 0, 1))
# print(splitDataSet(myData, 1, 0))
# print(splitDataSet(myData, 1, 1))
def test():
    mytree = treePloter.retrieveTree(0)
    testTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    testTree['no surfacing'][3] = 'maybe'

    print(testTree)
    treePloter.createPlot(testTree)

    fileName = 'storedClassifier.txt'
    storeTree(mytree, fileName)
    grabTree(fileName)
