
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier

class CustomMLPClassifier(MLPClassifier):
    def resample(self, X_train, y_train, sample_weight):

        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)
        X_train_new = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_new = np.zeros((len(y_train)), dtype=int)  
        for i in range(len(X_train)):
            x = np.random.choice(np.arange(len(X_train)), p=sample_weight)
            X_train_new[i] = X_train[x]
            y_train_new[i] = y_train[x]

        return X_train_new, y_train_new

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample(X, y, sample_weight)
        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))
