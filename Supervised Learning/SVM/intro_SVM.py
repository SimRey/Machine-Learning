#                                   Support vector machines (SVM)

# Kernelized support vector machines are powerful models and perform well on a variety of datasets. SVMs allow for
# complex decision boundaries, even if the data has only a few features. They work well on low-dimensional and
# high-dimensional data (i.e., few and many features), but don’t scale very well with the number of samples.

# The important parameters in kernel SVMs are the regularization parameter C, the choice of the kernel, and the
# kernel-specific parameters. Although we primarily focused on the RBF kernel, other choices are available in
# scikit-learn. The RBF kernel has only one parameter, gamma, which is the inverse of the width of the Gaussian kernel.
# gamma and C both control the complexity of the model, with large values in either resulting in a more complex model.
# Therefore, good settings for the two parameters are usually strongly correlated, and C and gamma should be adjusted
# together.

# ---------------------------------------------------------------------------------------------------------------------
#                                                       Example

from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

# While SVMs often perform quite well, they are very sensitive to the settings of the parameters and to the scaling of
# the data. In particular, they require all the features to vary on a similar scale.

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")
plt.show()

# From this plot we can determine that features in the Breast Cancer dataset are of completely different orders of
# magnitude. This can be somewhat of a problem for other models (like linear models), but it has devastating effects
# for the kernel SVM.

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
X = scalar.fit_transform(cancer.data)
y = cancer.target
X_train_s, X_test_s, y_train, y_test = train_test_split(X, y, random_state=42)

svc = SVC()
svc.fit(X_train_s, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train_s, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test_s, y_test)))


svc = SVC(C=0.1, kernel="rbf", gamma=5)
svc.fit(X_train_s, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train_s, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test_s, y_test)))





