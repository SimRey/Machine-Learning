import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Import data
df = pd.read_csv("Age_height.csv")

X = df[["Age"]]
y = df["Height"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

# Fitting the model to a simple linear regression
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)
m = LinReg.coef_
b = LinReg.intercept_
x = df["Age"]
R2 = LinReg.score(X_test, y_test)
print(f"""Presented model:
Equation: y = {m[0]:.4f}*x + {b:.4f}
R2: {100 * R2:.4f}%""")

plt.scatter(x, y, color="red", label="data")
plt.plot(x, LinReg.predict(X), label="Linear regression")
plt.title("Linear regression")
plt.xlabel("Age")
plt.ylabel("Height [cm]")
plt.legend(loc="best")
plt.show()

# Polynomial regression
poly_mod = PolynomialFeatures(degree=3)  # by default the degree is set to 2
X_polymod = poly_mod.fit_transform(X)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_polymod, y, train_size=0.7, random_state=0)
PolyReg = LinearRegression()
PolyReg.fit(X_train2, y_train2)
exponents = PolyReg.coef_[::-1]  # In this case the values of the list are inverted to have the highest power at
# the beginning

intercept = PolyReg.intercept_
R2 = PolyReg.score(X_test2, y_test2)

print(f"""Presented model:
Equation: y = {exponents[0]:.4f}*x3 + {exponents[1]:.4f}*x2 + {exponents[2]:.4f}*x + {intercept:.4f}
R2: {100 * R2:.4f}%""")

plt.scatter(x, y, color="red", label="data")
plt.plot(x, LinReg.predict(X), label="Linear regression")
plt.plot(x, PolyReg.predict(X_polymod), color="green", label="Polynomial regression")
plt.title("Regression")
plt.xlabel("Age")
plt.ylabel("Height [cm]")
plt.legend(loc="best")
plt.show()
