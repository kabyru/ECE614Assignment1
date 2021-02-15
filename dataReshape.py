import csv
import numpy as np

#Import and convert the CSV to Tensor format###############################################
with open('Real-estate-valuation-data-set.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

extractDataInputs = []
for i in range(1,len(data)):
    extractDataInputs.append(data[i][1:6])

print(extractDataInputs)

extractDataOutputs = []
for i in range(1,len(data)):
    extractDataOutputs.append(data[i][7])

print(extractDataOutputs)

################################################

#Extract 80% of the data for training, keep 20% of the data for testing.#################

#First, the inputs#######################################################
lenData = len(extractDataInputs)
trainingLen = int(0.8*lenData)
testLen = lenData - trainingLen

trainDataInputs = []
for i in range(0,trainingLen):
    trainDataInputs.append(extractDataInputs[i])

print(trainDataInputs)

testDataInputs = []
for i in range(trainingLen,lenData):
    testDataInputs.append(extractDataInputs[i])

print(testDataInputs)

#Next, the outputs########################################################
lenData = len(extractDataOutputs)
trainingLen = int(0.8*lenData)
testLen = lenData - trainingLen


trainDataOutputs = []
for i in range(0,trainingLen):
    trainDataOutputs.append(extractDataOutputs[i])

print(trainDataOutputs)

testDataOutputs = []
for i in range(trainingLen,lenData):
    testDataOutputs.append(extractDataOutputs[i])

print(testDataOutputs)
#########################################################################





#https://stackoverflow.com/questions/26850355/converting-list-to-numpy-array
#To-Do: convert these lists into numpy arrays for Keras input. See: Lab 1 Code Data Reshaping
