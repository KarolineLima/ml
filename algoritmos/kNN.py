from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def runKNN(X_train, y_train, X_test, y_test, kValue,  folds=10):
    results = []

    for i in range(folds):
        model = KNeighborsClassifier(n_neighbors=kValue, metric='euclidean', algorithm='brute')
        model = model.fit(X_train[i], y_train[i])

        result = model.predict(X_test[i])
        acc = metrics.accuracy_score(result, y_test[i])
        results.append(acc)

    show = round(np.mean(results) * 100)

    return [ results, show ]