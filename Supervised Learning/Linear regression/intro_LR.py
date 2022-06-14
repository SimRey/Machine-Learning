import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#                                           Linear models for regression
# For regression, the general prediction formula for a linear model looks as follows:
#                           ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
# Here, x[0] to x[p] denotes the features (in this example, the number of features is p) of a single data point,
# w and b are parameters of the model that are learned, and ŷ is the prediction the model makes. For a dataset with a
# single feature, this is:
#                                               ŷ = w[0] * x[0] + b
# Here, w[0] is the slope and b is the y-axis offset. For more features, w contains the slopes along each feature axis.
# Alternatively, you can think of the predicted response as being a weighted sum of the input features, with weights
# (which can be negative) given by the entries of w.

# --------------------------------------------------------------------------------------------------------------------
#                                            Building a linear regression model

# a) Arranging the data into a matrix and target vector
rng = np.random.RandomState(42)
x = 10 * rng.randn(50)
X = x[:, np.newaxis]  # it can also be used x.reshape((-1,1))
print(X.shape)  # matrix len(y), m

y = 2 * x - 1 + rng.randn(50)
print(y.shape)  # vector

# b) Fitting the data into the model
model = LinearRegression(fit_intercept=True)  # if working with datasets they can be normalized by entering the command
# normalize=True

model.fit(X, y)
print(model.coef_, model.intercept_)

# c) Predict labels for unknown data

x_fit = np.linspace(min(x), max(x))
X_fit = x_fit.reshape((-1, 1))
y_fit = model.predict(X_fit)

# Single value prediction
x_pred_val = np.array([[2]])
y_pred_val = model.predict(x_pred_val)
print(f"The predicted value for an x value of 2 is: {y_pred_val}")

# d) Plotting and evaluating the model
print(f"The R^2 of the model is {model.score(X, y)}")

plt.scatter(x, y, marker="*")
plt.plot(x_fit, y_fit, color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regression")
plt.show()

print()
print()
# ----------------------------------------------------------------------------------------------------------------
#                                     Other methods to perform linear regression

# a) Regression using scipy
from scipy.stats import *
slope, intercept, r_value, p_value, std_error = linregress(x, y)
print(f"""Slope: {slope}
Y-Intercept: {intercept}
R2: {r_value ** 2}""")
