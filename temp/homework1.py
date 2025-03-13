import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import csv
from sklearn.utils import Bunch

def loadData():
    with open(r'myData.csv') as csv_file:
        data_reader = csv.reader(csv_file)
        coordinates = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            features = row[:-1]
            label = row[-1]
            data.append([float(num) for num in features])
            target.append(label)
        
        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=coordinates)

dataset = loadData()
#print(dataset['data'][0])

x = [8, 9, 111]
distances = {}

index=1
for point in dataset['data']:
    distances[index] = float(np.linalg.norm(point-x))
    index+=1
print(distances)

distances(sorted(x.items(), key=lambda item: item[1]))



knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform', metric = 'euclidean')