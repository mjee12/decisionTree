import sys
import csv
import numpy as np
import copy

def openFile(data):
  with open(data, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    data = list(reader)
    return data

def uniqueVal(data, col):
  return sorted(list(set([row[col] for row in data[1:]])))

def countLabel(data, labelList): 
  result = {}
  for lab in labelList:
    result[lab] = 0
  for row in data:
    label = row[-1]
    if label in result:
      result[label] += 1 
  return result

def splitData(data, valList, splitIndex): #splits current data into left and right subset (nondestructive)
  leftSplit, rightSplit = [], []
  for row in range(len(data)):
    if data[row][splitIndex] == valList[0]:
      leftSplit.append(copy.deepcopy(data[row]))
    else:
      rightSplit.append(copy.deepcopy(data[row]))
  return leftSplit, rightSplit

def calculateGini(data, labelList): #calculat GI of entire dataset
  counts = countLabel(data, labelList)
  leftCnt, rightCnt = counts[list(counts)[0]], counts[list(counts)[1]]
  result = 1 - (float(leftCnt/len(data))**2 + float(rightCnt/len(data))**2)
  return result

def findMaxGG(data, labelList, valList, attList): #outputs the index of the best attribute to split and the GG value
  GI = calculateGini(data, labelList)
  allGG = {}
  for i in range(len(data[0])-1):
    leftSplit, rightSplit = splitData(data, valList, i)
    if len(leftSplit) == 0:
      leftGI = 0
    else:
      leftGI = calculateGini(leftSplit, labelList) * float(len(leftSplit)/len(data))
    if len(rightSplit) == 0:
      rightGI = 0
    else:
      rightGI = calculateGini(rightSplit, labelList) * float(len(rightSplit)/len(data))

    allGG[i] = GI - (leftGI + rightGI)
      
  bestAtt, bestGG = max(allGG, key = allGG.get), allGG[max(allGG, key = allGG.get)]
  
  #if there's a tie in the maximum GG
  maxList = []
  tieList = []
  for i in allGG:
    if allGG[i] == bestGG:
      maxList.append(i)
  if len(maxList) > 1:
    for i in maxList:
      tieList.append(attList[i])
    tieList = sorted(tieList, reverse = True)
    bestAtt, bestGG = attList.index(tieList[0]), allGG[bestAtt]
      
  return bestAtt, bestGG

class Leaf:
    
  def __init__(self, data, labelList):
    self.data = data
    self.counts = countLabel(data, labelList)

class Node:

  def __init__(self, att, left, right):
    self.att = att
    self.left = left
    self.right = right

def findMajority(leaf, labelList):
  if leaf.counts[labelList[0]] > leaf.counts[labelList[1]]:
    return labelList[0]
  if leaf.counts[labelList[0]] < leaf.counts[labelList[1]]:
    return labelList[1]
  if leaf.counts[labelList[0]] == leaf.counts[labelList[1]]:
    return labelList[0]

def decisionTree(data, attList, labelList, valList, currDepth, maxDepth):
  bestAtt, bestGG = findMaxGG(data, labelList, valList, attList)
  if currDepth == maxDepth:
    return Leaf(data, labelList)
  if currDepth == len(attList) - 1:
    return Leaf(data, labelList)
  if bestGG <= 0:
    return Leaf(data, labelList)
  else:
    currDepth += 1
    leftSplit, rightSplit = splitData(data, valList, bestAtt)
    leftDict = countLabel(leftSplit, labelList) #counts of label in left data
    rightDict = countLabel(rightSplit, labelList) #counts of label in right data
    
    print("|"*currDepth, attList[bestAtt], "=", leftSplit[0][bestAtt], ":", \
          "[", leftDict[labelList[0]], labelList[0], "/", \
          leftDict[labelList[1]], labelList[1], "]")
    
    left = decisionTree(leftSplit, attList, labelList, valList, currDepth, maxDepth)
    
    print("|"*currDepth, attList[bestAtt], "=", rightSplit[0][bestAtt], ":", \
          "[", rightDict[labelList[0]], labelList[0], "/", \
          rightDict[labelList[1]], labelList[1], "]")

    right = decisionTree(rightSplit, attList, labelList, valList, currDepth, maxDepth)
    return Node(bestAtt, left, right)

def findOutput(row, valList, labelList, tree):
  if isinstance(tree, Leaf):
    return findMajority(tree, labelList)

  if row[tree.att] == valList[0]:
    return findOutput(row, valList, labelList, tree.left)
  else:
    return findOutput(row, valList, labelList, tree.right)
    
def findAllOutputs(data, valList, labelList, tree):
  result = []
  for i in data:
    result.append(findOutput(i, valList, labelList, tree))
  return result

def calculateError(data1, data2):
  with open(data1, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    predictData = np.array([row for row in reader])
  with open(data2, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    actualData = np.array([row for row in reader])
  count = 0
  rowCount = np.shape(predictData)[0]
  attInd = len(actualData[0]) - 1
  for i in range(1, rowCount):
    if predictData[i-1] != actualData[i][attInd]:
        count += 1
  return (float(count) / float(rowCount))

def main():
  trainInput = sys.argv[1]
  testInput = sys.argv[2]
  maxDepth = int(sys.argv[3])
  trainOut = sys.argv[4]
  testOut = sys.argv[5]
  metricsOut = sys.argv[6]

  data = openFile(trainInput)
  dataNA = data[1:]
  attList = data[0]
  valList = uniqueVal(data, 0)
  labelList = sorted(set(dataNA[row][-1] for row in range(len(dataNA))))
  overallDict = countLabel(dataNA, labelList)
  print("[", overallDict[list(overallDict)[0]], list(overallDict)[0], "/", 
             overallDict[list(overallDict)[1]], list(overallDict)[1], "]")
  tree = decisionTree(dataNA, attList, labelList, valList, 0, maxDepth)
  with open(trainOut, "w") as file:
    outcome = findAllOutputs(dataNA, valList, labelList, tree)
    length = len(outcome)
    for i in range(length):
        file.write("%s\n" % (str(outcome[i])))
  file.close()

  data2 = openFile(testInput)
  dataNA2 = data2[1:]
  attList2 = data2[0]
  valList2 = uniqueVal(data2, 0)
  labelList2 = sorted(set(dataNA2[row][-1] for row in range(len(dataNA2))))
  overallDict2 = countLabel(dataNA2, labelList2)
  print("[", overallDict2[list(overallDict2)[0]], list(overallDict2)[0], "/", 
            overallDict2[list(overallDict2)[1]], list(overallDict2)[1], "]")
  #tree2 = decisionTree(dataNA2, attList2, labelList2, valList2, 0, maxDepth)
  with open(testOut, "w") as file:
    outcome = findAllOutputs(dataNA2, valList2, labelList2, tree)
    length = len(outcome)
    for i in range(length):
        file.write("%s\n" % (str(outcome[i])))
  file.close()

  with open(metricsOut, "w") as file:
    file.write("error(train): %f\n" % (calculateError(trainOut, trainInput)))
    file.write("error(test): %f" % (calculateError(testOut, testInput)))
  file.close()

if __name__ == "__main__":
	main()