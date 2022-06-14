#                                   Linear models for multiclass classification

# Many linear classification models are for binary classification only, and don’t extend naturally to the multiclass
# case (with the exception of logistic regression). A common technique to extend a binary classification algorithm to a
# multiclass classification algorithm is the one-vs.-rest approach. In the one-vs.-rest approach, a binary model is
# learned for each class that tries to separate that class from all of the other classes, resulting in as many binary
# models as there are classes. To make a prediction, all binary classifiers are run on a test point. The classifier that
# has the highest score on its single class “wins,” and this class label is returned as the prediction.


from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

digits = load_digits()
print(dir(digits))

# Printing out the first element
first_digit = digits.data[0]
print(first_digit)
print(len(first_digit))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.9)

m = LogisticRegression(solver="newton-cg")
m.fit(X_train, y_train)

R_2 = m.score(X_test, y_test)
print(R_2)

y_predict = m.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_predict)
print(confusion_mat)
print()
print()

# -------------------------------------------------------------------------------------------------------------------
#                                           Softmax Regression


softmax_reg = LogisticRegression(multi_class="multinomial", solver="newton-cg")
softmax_reg.fit(X_train, y_train)

R_2w = softmax_reg.score(X_test, y_test)
print(R_2w)

y_predict2 = softmax_reg.predict(X_test)

confusion_mat2 = confusion_matrix(y_test, y_predict2)
print(confusion_mat2)
