#                                               Ordinal encoding

# In this type of encoding, the categorical columns are selected and the order of encoding can be defined

# --------------------------------------------------------------------------------------------------------------------
# Example

import pandas as pd
from sklearn.linear_model import LinearRegression
import category_encoders as ce

df = pd.read_csv("car_data.csv")
print(df)

data = df.drop(columns="price")
target = df.price

print(data.brand.unique())

names = [{"col": "brand", "mapping": {"Audi_A5": 1, "BMW_X5": 2, "Mercedez_Benz_C_class": 3}}]

encoder = ce.OrdinalEncoder(cols=["brand"], mapping=names)

X = encoder.fit_transform(df)  # encoded data

print()
print(X)

m = LinearRegression()

m.fit(X, target)
pred_BMW_brand_new = m.predict([[2, 0, 0]])
print(pred_BMW_brand_new)
