import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Preprocessing/Categorical features/Encoding/homeprices.csv')

# Find the categorical values (non numerical variables) and creating the dummy dataframe
dummy_df = pd.get_dummies(df.town)

# Concatenating and removing the extra columns
df1 = pd.concat([df, dummy_df], axis=1)
df1.drop(columns=["town", "west_windsor"], inplace=True)
print(df1)

m = LinearRegression()
X = df1.drop(columns="price")
y = df1["price"]

m.fit(X, y)
pred_home_robbin = m.predict([[2800, 0, 1]])  # In this case, the first values is area, the second is monroe and third
# robbins
print(pred_home_robbin)
pred_home_ww = m.predict([[2800, 0, 0]])  # For west_windsor, column doesn't appear so in this case 2 and 3 variable
# are 0
print(pred_home_ww)
