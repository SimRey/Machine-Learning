#                                               Label encoding

# In this type of encoding, the categorical columns are selected and the encoder distributes in alphabetical order the
# numeric values. After the encoding, to avoid any confusion, the categorical columns must be erased.

# --------------------------------------------------------------------------------------------------------------------
# Example

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("Preprocessing\Categorical features\Encoding\salaries.csv")
print(df)

data = df.drop(columns="salary_more_then_100k")
target = df["salary_more_then_100k"]

for col in data.columns:
    if isinstance(data[col].dtype, object):
        le = LabelEncoder()
        name = f"{col}_n"
        data[name] = le.fit_transform(data[col])
        data.drop(columns=col, inplace=True)
    else:
        pass

m = tree.DecisionTreeClassifier(max_depth=4)

m.fit(data, target)

R_2 = m.score(data, target)
print(R_2)

y_predict = m.predict(np.array([[2, 0, 1]]))
print(y_predict)
