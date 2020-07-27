import numpy as np
import operator

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

        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCounts:

        prob = float(labelCounts[key])/numEntries

        shannonEnt -= prob * np.log2(prob)

    return shannonEnt


def splitDataSet(dataSet, axis, value):  # 划分数据集
    # 带划分的数据集， 划分数据集的特征， 返回的特征的值:
    # eg 特征a,b,c 按照a划分即为axis=0,value:按照a=value分出对应的其他特征集合
    returnDataSet = []

    for featureVector in dataSet:
        if featureVector[axis] == value:

            reducedFeatVc = featureVector[:axis]

            reducedFeatVc.extend(featureVector[axis+1:])

            returnDataSet.append(reducedFeatVc)

    return reducedFeatVc


def chooseBestFeatureToSplit(dataSet):  # 选择最好的划分方式

    numFeatures = len(dataSet[0]) - 1

    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):  # double loop,axis & value

        featList = [example[i] for example in dataSet]

        uniqueVals = set(featList)

        newEntropy = 0.0

        for value in uniqueVals:

            subDataSet = splitDataSet(dataSet, i, value)

            prob = len(subDataSet)/float(len(dataSet))

            newEntropy += prob * calcShannonEnt(subDataSet)  # shannon ent

        infoGain = baseEntropy - newEntropy

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):  # 递归创建决策树

    classCount = {}

    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 与kNN排序代码类似，见kNN.py

        return  sortedClassCount[0][0]


def createTree(dataSet, labels):

    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别完全相同：停止划分

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 遍历完所有特征：返回出现次数最多的类别

    best

# test area
myData, myLabs = createDataSet()
print(myData)
print(myLabs)

print(splitDataSet(myData, 0, 0))
print(splitDataSet(myData, 0, 1))
print(splitDataSet(myData, 1, 0))
print(splitDataSet(myData, 1, 1))