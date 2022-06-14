#                                               Exercise
# 1. Now do some exploratory data analysis to figure out which variables have direct and clear impact on employee
# retention (i.e. whether they leave the company or continue to work)
# 2. Plot bar charts showing impact of employee salaries on retention
# 3. Plot bar charts showing correlation between department and employee retention
# 4. Now build logistic regression model using variables that were narrowed down in step 1
# 5. Measure the accuracy of the model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

df = pd.read_csv("employee_db.csv")
print(df.head())
print(df.shape)
print(df.info())


def inverse_fun(x):
    if x == 1:
        return 0
    else:
        return 1


df["retention"] = df["left"].apply(inverse_fun)

correlations = df.corr(method="pearson")
reten_corr = correlations.loc["retention", :]

cor_cols = ["retention"]
for i in range(len(reten_corr) - 1):
    if float(reten_corr[i]) < 0:
        continue
    else:
        cor_cols.append(reten_corr.index[i])
        print(f"Positive correlation between retention and {reten_corr.index[i]}")


cros_tab = pd.crosstab(df["salary"], df["retention"], margins=True)
print(cros_tab)

N = 3
left = (cros_tab.iloc[0, 0], cros_tab.iloc[1, 0], cros_tab.iloc[2, 0])
stayed = (cros_tab.iloc[0, 1], cros_tab.iloc[1, 1], cros_tab.iloc[2, 1])

ind = np.arange(N)
width = 0.35
plt.bar(ind, left, width, label='Left')
plt.bar(ind + width, stayed, width, label='Stayed')

plt.ylabel('Salary')
plt.title('Retention')

plt.xticks(ind + width / 2, ("high", "low", "medium"))
plt.legend(loc='best')

print()
print()

cros_tab = pd.crosstab(df["Department"], df["retention"], margins=True)
print(cros_tab)

N = len(cros_tab.index) - 1

left = []
stayed = []
for i in range(N):
    left.append(cros_tab.iloc[i, 0])
    stayed.append(cros_tab.iloc[i, 1])

ind = np.arange(N)
width = 0.35
plt.barh(ind, left, width, label='Left')
plt.barh(ind + width, stayed, width, label='Stayed')

plt.xlabel('Department')
plt.title('Retention')

plt.yticks(ind + width / 2, (cros_tab.index[i] for i in range(N)))
plt.legend(loc='best')


df2 = df.filter(cor_cols)

ind_vars = cor_cols[1:]
dep_var = cor_cols[0]

X_train, X_test, y_train, y_test = train_test_split(df2[ind_vars], df2[dep_var], train_size=0.7)

m = LogisticRegression()
m.fit(X_train, y_train)
print(m.coef_ )
print(m.intercept_)
y_pred = m.predict(X_test)

R_2 = m.score(X_test, y_test)
print((R_2)*100)