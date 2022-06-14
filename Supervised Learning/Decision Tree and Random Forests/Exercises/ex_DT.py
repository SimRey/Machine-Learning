import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv("titanic.csv")

used_columns = ["Pclass", 'Sex', 'Age', 'Fare']

data = df.filter(used_columns)
print(data)

target = df["Survived"]
data["Age"].fillna(data["Age"].mean(), inplace=True)

le_sex = LabelEncoder()
data["sex_n"] = le_sex.fit_transform(data["Sex"])

data.drop(columns=["Sex"], inplace=True)
print(data)

m = tree.DecisionTreeClassifier(max_depth=4)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

m.fit(X_train, y_train)

R_2 = m.score(X_test, y_test)
print(f"R_2: {100 * R_2:.4f}")

tree.plot_tree(m,
               feature_names=X_train.columns,
               class_names=str(target.unique()),
               filled=True)
plt.show()


