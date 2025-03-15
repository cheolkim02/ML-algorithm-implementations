import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.utils import Bunch

# 주어진 데이터 bunch로 불러오기
def loadData():
    with open(r'myData_2.csv') as csv_file:
        data_reader = csv.reader(csv_file)
        coordinates = next(data_reader)[:-1]
        data = []
        price = []
        for row in data_reader:
            features = row[:-1]
            label = row[-1]
            data.append([float(num) for num in features])
            price.append(float(label))
        
        data = np.array(data)
        price = np.array(price)
    return Bunch(data=data, price=price, coordinates=coordinates)
dataset = loadData()

# 새로운 샘플 정의
x = [6, 200, 5, 30]

# L2 거리 계산해서 순번대로 저장
distances = {}
price = {}
index=1
for car in dataset['data']:
    distances[index] = float(np.linalg.norm(car-x))
    price[index] = float(dataset['price'][index-1])
    index+=1

# 계산한 거리가 적은 순부터 정렬
keys = list(distances.keys())
values = list(distances.values())
price_values = list(price.values())
sorted_value_index = np.argsort(values)
sorted_distances = {}
sorted_price = {}
for i in sorted_value_index :
    sorted_distances[keys[i]] = values[i]
    sorted_price[keys[i]] = price_values[i]

# 가장 가까운 car 5개 출력.
print("\n5 cars nearest to sample")
print(list(sorted_distances.items())[:5])

# w(x)의 합 구하기. w(x) = exp(-dist(x,x')). k=5
sum_of_weights=0
sum_of_weighted_prices=0
for i in list(sorted_distances)[:5] :
    sum_of_weights += np.exp(-1*distances[i])
    sum_of_weighted_prices += np.exp(-1*distances[i]) * price[i]

# 가장 가까운 car 5개의 가중 평균 가격 출력
print("\naverage price of k=5, no weights")
print(sum_of_weighted_prices/sum_of_weights)