import numpy as np

import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])

    labels = ['A', 'A', 'B', 'B']

    return group, labels


def classify0(inX, dataSet, labels, k):
    # inX: to be classified
    # dataSet: data which has been classified
    # labels: meaning
    # k: the number of min dist neighbour to select

    # calculate dist
    dataSetSize = dataSet.shape[0]

    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 扩展成数据集中每一个数据对应一个inX

    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()  # return the index of sorted list

    classCount = {}

    # select k points - min dist
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # sort
    sortedClassCount = sorted(classCount.items(),  # return tuple
                              key=operator.itemgetter(1),  # sort with the value of key-value
                              reverse=True)  # big - small

    return sortedClassCount[0][0]  # select the class of the max classCount


# main

group, labels = createDataSet()
# print(classify0([1, 0], group, labels, 3))

