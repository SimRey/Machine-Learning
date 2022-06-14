#                                           k-Nearest Neighbors

# The k-NN algorithm is arguably the simplest machine learning algorithm. Building the model consists only of storing
# the training dataset. To make a prediction for a new data point, the algorithm finds the closest data points in the
# training dataset—its “nearest neighbors.” In its simplest version, the k-NN algorithm only considers exactly one
# nearest neighbor, which is the closest training data point to the point we want to make  a prediction for.
import mglearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
fitted_model = clf.fit(X_train, y_train)

print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

axes[0].legend(loc="best")

plt.show()

# As you can see on the left in the figure, using a single neighbor results in a decision boundary that follows the
# training data closely. Considering more and more neighbors leads to a smoother decision boundary. A smoother boundary
# corresponds to a simpler model. In other words, using few neighbors corresponds to high model complexity and using
# many neighbors corresponds to low model complexity.

# --------------------------------------------------------------------------------------------------------------------

#                                   Choosing the best number of neighbors

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# Test and train accuracy: with this method both accuracies are measured, the k-point at which both are the closest is
# chosen to be the best number of neighbors


training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
N = 11
neighbors_settings = range(1, N)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Considering a single nearest neighbor, the prediction on the training set is perfect. But when more neighbors are
# considered, the model becomes simpler and the training accuracy drops. The test set accuracy for using a single
# neighbor is lower than when using more neighbors, indicating that using the single nearest neighbor leads to a model
# that is too complex. On the other hand, when considering 10 neighbors, the model is too simple and performance is
# even worse. The best performance is somewhere in the middle, using around six neighbors.

