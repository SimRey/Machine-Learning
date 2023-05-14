import numpy as np
from collections import Counter

def eucledian_distance(x1, x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist

class KNN:

    def __init__(self, k=3):
        self.k = k 
    
    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        # compute distances (x of all the points)
        distances = [eucledian_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples
        k_indices = np.argsort(distances)[:self.k] # only goes till self.k
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1) # Returns a tuple with (value, number of times)
        return most_common[0][0]
