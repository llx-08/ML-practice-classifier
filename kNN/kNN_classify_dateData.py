import numpy as np
import matplotlib.pyplot as plt
from kNN import *


def file2Matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}

    fr = open(filename)

    arrayLines = fr.readlines()

    numberOfLines = len(arrayLines)

    returnMat = np.zeros((numberOfLines, 3))

    trainLabelList = []

    index = 0

    for line in arrayLines:
        lineList = line.strip().split("\t")

        returnMat[index, :] = lineList[0:3]

        if lineList[-1].isdigit():
            trainLabelList.append(int(lineList[-1]))
        else:
            trainLabelList.append(love_dictionary.get(lineList[-1]))

        index += 1

    return returnMat, trainLabelList


def drawDataPoints(data, column1, column2):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    if column1 == 0 and column2 == 0:
        pass
    else:
        ax.scatter(data[:, column1], data[:, column2])

    plt.show()


def autoNorm(dataSet):  # 归一化至0~1

    minVal = dataSet.min(0)  # 0:从列中选择最小值，默认选择当前行最小值
    maxVal = dataSet.max(0)

    ranges = maxVal - minVal

    m = dataSet.shape[0]

    normedDataSet = dataSet - np.tile(minVal, (m, 1))

    normedDataSet = normedDataSet/np.tile(ranges, (m, 1))

    return normedDataSet, ranges, minVal


def datingDataTest():
    hoRatio = 0.10

    datingDataMat, datingLabels = file2Matrix('datingTestSet.txt')

    normedMatrix, ranges, minVal = autoNorm(datingDataMat)

    m = normedMatrix.shape[0]

    numTestVecs = int(m*hoRatio)

    errorCount = 0

    for i in range(numTestVecs):
        classifiedResult = classify0(normedMatrix[i, :], normedMatrix[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)

        print("After classifier:" + str(classifiedResult) + ", the real answer is " + str(datingLabels[i]))

        if classifiedResult != datingLabels[i]:
            errorCount += 1

    print("Total loss rate:", end=" ")
    print(errorCount/float(numTestVecs))


def classifyPerson():  # 输入参数预测分类
    resultList = ['not at all', 'in small doses', 'in large doses']

    chara1 = float(input(r"percentage of characteristic 1:"))

    chara2 = float(input(r"percentage of characteristic 2:"))

    chara3 = float(input(r"percentage of characteristic 3:"))

    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')

    normedMatrix, ranges, minVal = autoNorm(datingDataMat)

    testArray = [chara1, chara2, chara3]

    classifiedResult = classify0((testArray - minVal)/ranges, normedMatrix, datingLabels, 3)

    print("The result is:", end=" ")
    print(resultList[classifiedResult - 1])

    return classifiedResult


if __name__ == '__main__':

    classifyPerson()


