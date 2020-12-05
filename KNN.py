from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import time


# Takes all the feature vectors asnp.suming normalized numerical form and then just takes the dist
# note it is possible to do the x1-x2 becuase it's a numpy array
def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self, k=2):
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
        dists = [euclidean_dist(x, x_train) for x_train in self.X_train]

        # get k-nearest neighbors
        k_indices = np.argsort(dists)[0:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote of most common class 
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common



'''
iris = datasets.load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55, random_state=1)
print(X_train)

model = KNN(k=3)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)



acc = np.sum(y_predicted==y_test)/len(y_predicted)

print(acc)


'''
































