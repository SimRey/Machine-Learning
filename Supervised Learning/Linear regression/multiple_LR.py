import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

ds = datasets.load_boston()
print(ds.keys())
print(ds['DESCR'])

obj_var = ds["target"]
ind_var = ds["data"]
ind_var_names = ds['feature_names']

m = LinearRegression()

m.fit(X=ind_var, y=obj_var)

print(f"""Coefficients: {m.coef_}
Intercept: {m.intercept_}
R^2: {100*m.score(X=ind_var, y=obj_var):.5f}%""")

# Value prediction
x_pred = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
y_pred = m.predict(x_pred)
print(y_pred)











