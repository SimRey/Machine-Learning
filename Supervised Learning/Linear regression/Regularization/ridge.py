from sklearn.model_selection import train_test_split
import mglearn
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# This discrepancy between performance on the training set and the test set is a clear sign of overfitting, and
# therefore we should try to find a model that allows us to control complexity. One of the most commonly used
# alternatives to standard linear regression is ridge regression

# ------------------------------------------------------------------------------------------------
# Ridge regression is also a linear model for regression,  so the formula it uses to make predictions is the same one
# used for ordinary least squares. In ridge regression, though, the coefficients (w) are chosen not only so that they
# predict well on the training data, but also to fit an additional constraint. We also want the magnitude of
# coefficients to be as small as possible; in other words, all entries of w should be close to zero. Intuitively, this
# means each feature should have as little effect on the outcome as possible (which translates to having a small slope),
# while still predicting well. This constraint is an example of what is called regularization. Regularization means
# explicitly restricting a model to avoid overfitting. The particular kind used by ridge regression is known as L2
# regularization.

# Ridge regression minimizes the objective function:
#                                     ||yi - Xi*wi||^2 + alpha * ||wi||^2

# This model solves a regression model where the loss function is the linear least squares function and regularization
# is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization.

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# The Ridge model makes a trade-off between the simplicity of the model (near-zero coefficients) and its performance on
# the training set. How much importance the model places on simplicity versus training set performance can be specified
# by the user, using the alpha parameter. Increasing alpha forces coefficients to move more toward zero, which
# decreases training set performance but might help generalization. For very small values of alpha, coefficients are
# barely restricted at all, and we end up with a model that resembles LinearRegression


