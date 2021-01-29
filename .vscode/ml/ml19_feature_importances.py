from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data,dataset.target,train_size=0.8,random_state=66)

model = DecisionTreeRegressor(max_depth=4)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print(model.feature_importances_)
print("acc:",acc)

#[0.00625261 0.         0.01606923 0.97767816]
#acc: 0.8993288590604027