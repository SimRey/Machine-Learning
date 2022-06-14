import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math

df = pd.read_csv("Machine Learning/Supervised Learning/Logistic regression/insurance.csv")

plt.scatter(x=df.age, y=df.bought_insurance, marker='+', color='red')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[["age"]], df["bought_insurance"], train_size=0.9)

m = LogisticRegression()
m.fit(X_train, y_train)

print(X_test)
y_pred = m.predict(X_test)
print(y_pred)
y_proba_pred = m.predict_proba(X_test)
print(y_proba_pred)

# Now, doing the prediction using the sigmoid function

sigmoid = lambda x: 1 / (1 + math.exp(-x))


def pred_function(age):
    z = m.coef_ * age + m.intercept_
    y = sigmoid(z)
    return y


age = 35
print(f"Predicted probability using the built Linear model {m.predict_proba([[age]])[0][1]}"
      f" Predicted probability using the sigmoid function {pred_function(age)}"
      f" Difference: {m.predict_proba([[age]])[0][1] - pred_function(age)}")
# As we can see the approach with the sigmoid function is the same as with the linear regression, so it can be safely
# said that this type of problems can be solved using logistic regression
