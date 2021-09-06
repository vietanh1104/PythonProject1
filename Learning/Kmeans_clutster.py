import numpy as np
import csv


arr=[]
filename="Iris.csv"
file_reader = csv.reader(open(filename, 'r'), delimiter=',')
for row in file_reader:
  arr.append(row[1:])


class_0 = np.array([arr[i][0:4] for i in range(len(arr)) if arr[i][-1]=='Iris-setosa']).astype(np.float)
class_1 = np.array([arr[i][0:4]for i in range(len(arr)) if arr[i][-1]=='Iris-versicolor']).astype(np.float)
class_2 = np.array([arr[i][0:4] for i in range(len(arr)) if arr[i][-1]=='Iris-virginica']).astype(np.float)


def InitKmeans(class_0=None):
  amax = np.max(class_0, 1)
  amin = np.min(class_0, 1)
  means = np.zeros((1, class_0.shape[-1]))
  for i in range(class_0.shape[-1]):
      means[0][i] = np.random.uniform(amin[i], amax[i])
  return means


def Distance(vectorA, vectorB):
  return np.sum(np.sqrt((vectorA - vectorB) ** 2))


def Clusting(X, means):
  store =  []
  for i in range(len(X)):
    min = 0x3f3f3f3f3f3f
    index = -1
    for j in range(len(means)):
      d = Distance(X[i], means[j])
      if d < min:
        min = d
        index = j
        store[index].append(X[i])
  return np.array(store)


def Update(store):
  newMeans = [[] for i in range(len(store))]

  for i in range(len(store)):
    newMeans[i] = np.average(store[i], 0)
  return newMeans


def fit(n_clus, X, epouch):
  means = InitKmeans(n_clus, X)

  for i in range(epouch):
    store = Clusting(X, means)
    means = Update(store)

  return means


def Distance(vecto1,vecto2):
  return np.sum(np.sqrt((vecto1-vecto2)**2))


def Clusting(X, means):
  store = []
  min = 0x3f3f3f3f3f
  for i in range(len(X)):

    d = Distance(X[i], means)
    if d < min:
      min = d
      store.append(X[i])

  return store


def Update(store):
    newMeans = np.average(np.array(store), 0)
    return newMeans


def fit( X, epouch):
  means = InitKmeans(X)
  for i in range(epouch):
    store = Clusting(X, means)
    means = Update(store)

  return means

result=[]
new_kmeans1=fit(class_0,20)
result.append(new_kmeans1)
new_kmeans2=fit(class_1,20)
result.append(new_kmeans2)
new_kmeans3=fit(class_2,20)
result.append(new_kmeans3)
result=np.array(result)
print result

a=[]
print("Nhap lan luot gia tri 4 features:")
for i in range(4):
  x=float(input(arr[0][i]+" = "))
  a.append(x)
a=np.array(a)

type_of_flower=(["Iris-setosa","Iris-versicolor","Iris-virginica"])
pos=0
Distance_min = 0x3f3f3f3f3f

for i in range(3):
  d = Distance(result[i],a)
  if d < Distance_min:
    Distance_min = d
    pos=i

print("Name of flower is: "+type_of_flower[pos])
