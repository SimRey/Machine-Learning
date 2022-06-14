from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Use sklearn to train the model using logistic regression.
iris = load_iris()
print(dir(iris))

print(iris.data[0])
print(iris.target[0])
print(iris.data.shape)
print(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.5)

m = LogisticRegression(solver="newton-cg")
m.fit(X_train, y_train)

R_2 = m.score(X_test, y_test)
print(R_2)

y_predict = m.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_predict)
print(confusion_mat)
