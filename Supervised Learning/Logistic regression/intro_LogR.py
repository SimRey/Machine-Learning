#                                               Logistic regression

# For linear models for classification,the decision boundary is a linear function of the input. In other words, a
# (binary) linear classifier is a classifier that separates two classes using a line, a plane, or a hyperplane. The two
# most common linear classification algorithms are logistic regression, and linear support vector machines. By default,
# both models apply an L2 regularization, in the same way that Ridge does for regression. For LogisticRegression and
# LinearSVC the trade-off parameter that determines the strength of the regularization is called C, and higher values
# of C correspond to less regularization. In other words, when you use a high value for the parameter C,
# LogisticRegression and LinearSVC try to fit the training set as best as possible, while with low values of the
# parameter C, the models put more emphasis on finding a coefficient vector (w) that is close to zero.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42)

logreg = LogisticRegression(solver="saga").fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100, solver="saga").fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01, solver="saga").fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

# If we desire a more interpretable model, using L1 regularization might help, as it limits the model to using only
# a few features.

for C in [0.001, 1, 100]:
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver="saga").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))
