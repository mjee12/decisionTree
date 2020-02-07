import sys
import csv
import numpy as np

def countLabel(data, labelList): 
  result = {}
  for lab in labelList:
      result[lab] = 0
  for row in data:
    label = row[-1]
    if label in result:
      result[label] += 1 
  return result

def calculateGI(data, labelList): #calculat GI of entire dataset
  counts = countLabel(data, labelList)
  leftCnt, rightCnt = counts[list(counts)[0]], counts[list(counts)[1]]
  result = 1 - (float(leftCnt/len(data))**2 + float(rightCnt/len(data))**2)
  return result

def calculateError(data, labelList):
  counts = countLabel(data, labelList)
  incorrectLabel = min(counts, key = counts.get)
  return float(counts[incorrectLabel]/len(data))


def openFile(data):
  with open(data, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
    data = np.array([row for row in reader])
    return data

def main():
  inputFile = sys.argv[1]
  outputFile = sys.argv[2]

  inputData = openFile(inputFile)
  inputDataNA = inputData[1:]
  labelList = sorted(set(inputDataNA[row][-1] for row in range(len(inputDataNA))))

  with open(outputFile, "w") as file:
    file.write("gini_impurity: %f\n" % (calculateGI(inputDataNA, labelList)))
    file.write("error: %f" % (calculateError(inputDataNA, labelList)))
  file.close()

if __name__ == "__main__":
	main()