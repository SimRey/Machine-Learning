#                                               Decision Trees

# Decision trees are widely used models for classification and regression tasks. Essentially, they learn a hierarchy of
# if/else questions, leading to a decision.
# Typically, building a tree as described here and continuing until all leaves are pure leads to models that are very
# complex and highly overfit to the training data. The presence of pure leaves mean that a tree is 100% accurate on the
# training set; each data point in the training set is in a leaf that has the correct majority class.


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# As expected, the accuracy on the training set is 100%—because the leaves are pure, the tree was grown deep enough that
# it could perfectly memorize all the labels on the training data. The test set accuracy is slightly worse than for the
# linear models we looked at previously, which had around 95% accuracy.
# If we don’t restrict the depth of a decision tree, the tree can become arbitrarily deep and complex. Unpruned trees
# are therefore prone to overfitting and not generalizing well to new data.

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# Limiting the depth of the tree decreases overfitting. This leads to a lower accuracy on the training set, but an
# improvement on the test set.

#                                             Feature importance in trees
# Instead of looking at the whole tree, which can be taxing, there are some useful properties that we can derive to
# summarize the workings of the tree. The most commonly used summary is feature importance, which rates how important
# each feature is for the decision a tree makes. It is a number between 0 and 1 for each feature, where 0 means “not
# used at all” and 1 means “perfectly predicts the target.”

print("Feature importances:\n{}".format(tree.feature_importances_))
