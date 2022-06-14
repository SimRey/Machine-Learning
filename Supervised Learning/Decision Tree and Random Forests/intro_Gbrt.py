#                       Gradient boosted regression trees (gradient boosting machines)

# The gradient boosted regression tree is another ensemble method that combines multiple decision trees to create a more
# powerful model. Despite the “regression” in the name, these models can be used for regression and classification. In
# contrast to the random forest approach, gradient boosting works by building trees in a serial manner, where each tree
# tries to correct the mistakes of the previous one. By default, there is no randomization in gradient boosted
# regression trees; instead, strong pre-pruning is used. Gradient boosted trees often use very shallow trees, of depth
# one to five, which makes the model smaller in terms of memory and makes predictions faster.

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# As the training set accuracy is 100%, we are likely to be overfitting. To reduce overfitting, we could either apply
# stronger pre-pruning by limiting the maximum depth or lower the learning rate

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# Both methods of decreasing the model complexity reduced the training set accuracy as expected. In this case, lowering
# the maximum depth of the trees provided a significant improvement of the model, while lowering the learning rate only
# increased the generalization performance slightly.

# ----------------------------------------------------------------------------------------------------------------------
#                                           Predicting probability
# The output of predict_proba is a probability for each class, the output of predict_proba has the same shape,
# (n_samples, n_classes), for binary cases is (n_samples, 2)

# show the first few entries of predict_proba
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:10]))

print("Argmax of predicted probabilities:\n{}".format(
    np.argmax(gbrt.predict_proba(X_test),axis=1)))  # You can recover the prediction when there are n_classes many
                                                    # columns by computing the argmax across columns.

print("Predictions:\n{}".format(gbrt.predict(X_test)))
