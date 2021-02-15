import csv

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

#https://stackoverflow.com/questions/26850355/converting-list-to-numpy-array
#To-Do: convert these lists into numpy arrays for Keras input. See: Lab 1 Code Data Reshaping
