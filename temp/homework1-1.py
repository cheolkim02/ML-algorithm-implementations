import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.utils import Bunch

# 주어진 데이터 bunch로 불러오기
def loadData():
    with open(r'myData_1.csv') as csv_file:
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

# 새로운 샘플 정의
x = [8, 9, 111]

# L2 거리 계산해서 순번대로 저장
# 데이터 1~5: Class A, 데이터 6~10: Class B, etc.
distances = {}
index=1
for point in dataset['data']:
    distances[index] = float(np.linalg.norm(point-x))
    index+=1

# 계산한 거리가 적은 순부터 정렬
keys = list(distances.keys())
values = list(distances.values())
sorted_value_index = np.argsort(values)
sorted_distances = {}
for i in sorted_value_index :
    sorted_distances[keys[i]] = values[i]





















'''
# a) 가장 가까운 점 3개 출력
print()
print("3 nearest neighbors: ")
print(list(sorted_distances.items())[:3])
'''


'''
# b) 가장 가까운 점 5개 출력
print()
print("5 nearest neighbors: ")
print(list(sorted_distances.items())[:5])
'''


'''
# c) exp(-1*distance) 가중치 이용한 knn-3
# X와 각 데이터 간의 가중치 계산
for i in sorted_distances :
    sorted_distances[i] = float(np.exp(-1*sorted_distances[i]))

# 가장 가까운 데이터 3개 출력
print()
print("3 nearest neighbors: ")
print(list(sorted_distances.items())[:3])

# 가장 가까운 데이터 3개 간의 가중치를 백분율로 계산
sum=0
for i in list(sorted_distances.values())[:3] :
    sum += i
for i in sorted_distances :
    sorted_distances[i] = (sorted_distances[i]/sum)*100

print()
print("3 nearest neighbors, weights shown in percentage: ")
print(list(sorted_distances.items())[:3])
'''