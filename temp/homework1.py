import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import csv
from sklearn.utils import Bunch

# 주어진 데이터 bunch로 불러오기
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

# 새로운 샘플 정의
x = [8, 9, 111]

# L2 거리 계산해서 순번대로 저장하기.
# 데이터 1~5: Class A, 데이터 6~10: Class B, etc.
distances = {}
index=1
for point in dataset['data']:
    distances[index] = float(np.linalg.norm(point-x))
    index+=1

# 계산한 거리가 적은 순부터 정렬렬
keys = list(distances.keys())
values = list(distances.values())
sorted_value_index = np.argsort(values)
sorted_distances = {}
for i in sorted_value_index :
    sorted_distances[keys[i]] = values[i]





knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform', metric = 'euclidean')