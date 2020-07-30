import decisionTree
import treePloter

fr = open('lenses.txt')

lenses = []

for line in fr.readlines():  # turn to list
    lenses.append(line.strip().split('\t'))

print(lenses)

lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

lensesTree = decisionTree.createTree(lenses, lensesLabels)

print(lensesTree)

treePloter.createPlot(lensesTree)

