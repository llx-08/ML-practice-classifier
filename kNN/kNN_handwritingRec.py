from os import listdir
from kNN import *
import numpy as np


def img2Vec(fileName):

    returnVec = np.zeros((1, 1024))

    fr = open(fileName)

    for i in range(32):  # 循环读取32行
        lineStr = fr.readline()

        for j in range(32):  # 妙啊
            returnVec[0, 32*i + j] = int(lineStr[j])

    return returnVec


def handWritingClassTest_dataProcessing():  # 文件名 n_m 代表数字n的第m个例子

    hwLabels = []

    trainingFileList = listdir('trainingDigits')  # 列表文件做成列表

    m = len(trainingFileList)  # 获取文件个数

    trainingFileMatrix = np.zeros((m, 1024))  # 每一行存一个文件

    for i in range(m):
        fileNameStr = trainingFileList[i]

        fileStr = fileNameStr.split('.')[0]

        classNumStr = int(fileStr.split('_')[0])

        hwLabels.append(classNumStr)

        trainingFileMatrix[i, :] = img2Vec('trainingDigits/%s' % fileNameStr)

    return trainingFileMatrix, hwLabels


def usingFileTestHandWriting(trainingFileMatrix, hwLabels):

    testFileList = listdir('testDigits')

    errorCount = 0.0

    mTest = len(testFileList)

    for i in range(mTest):  # test

        fileNameStr = testFileList[i]

        fileStr = fileNameStr.split('.')[0]

        classNumStr = int(fileStr.split('_')[0])

        vectorUnderTest = img2Vec('testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingFileMatrix, hwLabels, 3)

        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))

        if (classifierResult != classNumStr):
            errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


def usingMyHandWritingPredict(testData, trainingFileMatrix, hwLabels):

    classifierResult = classify0(testData, trainingFileMatrix, hwLabels, 3)

    print(classifierResult)


filePath = "mytest.txt"
testVec = img2Vec(filePath)

mytest = np.random.randint(0, 2, (1, 1024))

for i in range(len(testVec)):
    for j in range(len(testVec[i])):
        print(int(mytest[i, j]), end="")

        if j%32 == 0 and j > 0:
            print()

print()


tFM, hwL = handWritingClassTest_dataProcessing()
myTestData = []

usingMyHandWritingPredict(mytest, tFM, hwL)