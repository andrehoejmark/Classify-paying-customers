import numpy as np
from collections import Counter
import time


# Takes all the feature vectors asnp.suming normalized numerical form and then just takes the dist
# note it is possible to do the x1-x2 becuase it's a numpy array
def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self, k=6):
        self.start_time = 0
        self.end_time = 0
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        

    def predict(self, X):
        self.start_time = time.time()
        # predicts all rows given
        predictions = [self._predict(x) for x in X]
        self.end_time = time.time()
        
        return predictions

    def _predict(self, x):
        #compute euclidean distances
        dists = []
        for x_train n self.X_train:
            dists.append(euclidean_dist(x, x_train))

        # compute nearest neighbors
        k_indices = np.argsort(dists)[0:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

































