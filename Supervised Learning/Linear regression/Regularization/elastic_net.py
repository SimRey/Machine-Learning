from sklearn.model_selection import train_test_split
import mglearn
from sklearn.linear_model import LinearRegression
import numpy as np

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix
# of both Ridge and Lasso’s regularization terms, and you can control the mix ratio r. When r = 0, Elastic Net is
# equivalent to Ridge Regression, and when r = 1, it is equivalent to Lasso Regression


from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)
print("Training set score: {:.2f}".format(elastic_net.score(X_train, y_train)))
print("Test set score: {:.2f}".format(elastic_net.score(X_test, y_test)))



