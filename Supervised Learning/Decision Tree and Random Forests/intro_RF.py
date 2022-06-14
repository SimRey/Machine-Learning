#                                             Random Forest

# A random forest is essentially a collection of decision trees, where each tree is slightly different from the others.
# The idea behind random forests is that each tree might do a relatively good job of predicting, but will likely overfit
# on part of the data. If we build many trees, all of which work well and overfit in different ways, we can reduce the
# amount of overfitting by averaging their results. This reduction in overfitting, while retaining the predictive power
# of the trees, can be shown using rigorous mathematics.

# max_features determines how random each tree is, and a smaller max_features reduces overfitting. In general, it’s a
# good rule of thumb to use the default values: max_features=sqrt(n_features) for classification and
# max_features=log2(n_features) for regression. Adding max_features or max_leaf_nodes might sometimes improve
# performance. It can also drastically reduce space and time requirements for training and prediction.

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
                                alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()

# --------------------------------------------------------------------------------------------------------------------
# Another example

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

y_pred = forest.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
