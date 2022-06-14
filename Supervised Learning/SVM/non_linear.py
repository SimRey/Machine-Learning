#                                       Nonlinear SVM Classification
# Although linear SVM classifiers are efficient and work surprisingly well in many cases, many datasets are not even
# close to being linearly separable. One approach to handling nonlinear datasets is to add more features, such as
# polynomial features; in some cases this can result in a linearly separable dataset.

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

# Adding polynomial features is simple to implement and can work great with all sorts of Machine Learning algorithms
# (not just SVMs), but at a low polynomial degree it cannot deal with very complex datasets, and with a high polynomial
# degree it creates a huge number of features, making the model too slow. Fortunately, when using SVMs you can apply an
# almost miraculous mathematical technique called the kernel trick (it is explained in a moment). It makes it possible
# to get the same result as if you added many polynomial features, even with very highdegree polynomials, without
# actually having to add them. So there is no combinatorial explosion of the number of features since you don’t actually
# add any features. This trick is implemented by the SVC class


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=10, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X, y)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


plot_predictions(poly_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=10, r=1, C=5$", fontsize=18)

plt.show()
