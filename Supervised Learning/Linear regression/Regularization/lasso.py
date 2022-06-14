from sklearn.model_selection import train_test_split
import mglearn
from sklearn.linear_model import LinearRegression
import numpy as np

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# An alternative to Ridge for regularizing linear regression is Lasso. As with ridge regression, using the lasso also
# restricts coefficients to be close to zero, but in a slightly different way, called L1 regularization. The consequence
# of L1 regularization is that when using the lasso, some coefficients are exactly zero. This means some features are
# entirely ignored by the model. This can be seen as a form of automatic feature selection. Having some coefficients be
# exactly zero often makes a model easier to interpret, and can reveal the most important features of your model.


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1, max_iter=1000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# As you can see, Lasso does quite badly, both on the training and the test set. This indicates that we are
# underfitting, and we find that it used only 4 of the 105 features. Similarly to Ridge, the Lasso also has a
# regularization parameter, alpha, that controls how strongly coefficients are pushed toward zero. To reduce
# underfitting, let’s try decreasing alpha. When we do this we also need to increase the default setting of max_iter

lasso = Lasso(alpha=0.1, max_iter=10000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

