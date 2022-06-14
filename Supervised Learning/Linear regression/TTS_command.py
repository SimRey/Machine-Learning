import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ds = datasets.load_diabetes()
data_vals = ds["data"]
y = ds["target"]
data_names = ds['feature_names']

# Regression works very good with Pandas df
df = pd.DataFrame(data=data_vals, columns=data_names)
df["target"] = y

print(df.head())
print()
print()

# Choosing one feature to compare with target --> in this case bmi is chosen
x = df["s6"]
X = x[:, np.newaxis]

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# In the train_test_split instance, 4 values are required, the first two is the data for the model, test_size is the
# fraction of all the data frame destined to be tested, 1-test_size is used to train the model; and finally,
# random_state is used to declare a random pick of values.


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print(f'Coefficients: {regr.coef_}, Interception: {regr.intercept_}')
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % regr.score(X, y))

# Plot outputs
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)

plt.xlabel("X")
plt.ylabel("y")

plt.show()
