#                                               Decision Trees
# The usage and analysis of regression trees is very similar to that of classification trees. There is one particular
# property of using tree-based models for regression that we want to point out, DecisionTreeRegressor (and all other
# tree-based regression models) is not able to extrapolate, or make predictions outside of the range of the
# training data.


import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create a random dataset
x = np.linspace(0, 5, 100)
fun = lambda a: np.sin(a)

# Fit regression model
m = DecisionTreeRegressor()
X = x[:, np.newaxis]
X_train = X[:75]
y_train = fun(X_train)

# Predict values based on split
m.fit(X_train, y_train)

X_test = X[75:]
y_test = fun(X_test)


# Predict all values
pred_vals = m.predict(X)

# Plot the results
plt.figure()
plt.scatter(x, fun(x), s=20, edgecolor="black", c="darkorange", label="data")
plt.scatter(X_test, y_test, color="cornflowerblue", label="Test data")
plt.plot(X, pred_vals, color="yellowgreen", label="Tree prediction", linewidth=1)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

