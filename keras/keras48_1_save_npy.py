from sklearn. datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset)

#dict-> key와 value가 쌍으로 되어있는것
#앞에있는게 key -> x=dataset.data


print(dataset.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

x_data = dataset.data
y_data = dataset.target
#x_data = dataset['data'] 
#y_data = dataset['target']

#print(x_data)
#print(y_data)

print(dataset.frame)
print(dataset.target_names)  #['setosa' 'versicolor' 'virginica']
print(dataset["DESCR"])
print(dataset["feature_names"]) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(dataset.filename) #C:\Users\choijunho\anaconda3\lib\site-packages\sklearn\datasets\data\iris.csv

print(type(x_data),type(y_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('C:/data/npy/iris_x.npy', arr=x_data)
np.save('C:/data/npy/iris_y.npy', arr=y_data)
