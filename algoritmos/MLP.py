from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import numpy as np


def runMLP(X_train, y_train, X_test, y_test, arqX, arqY, activation,  folds=10):
    results = []

    for i in range(folds):
        model = MLPClassifier(hidden_layer_sizes=(arqX,arqY), activation=activation,max_iter=3000, random_state=1)
        model = model.fit(X_train[i], y_train[i])

        result = model.predict(X_test[i])

        acc = metrics.accuracy_score(result, y_test[i])

        results.append(acc)

    show = round(np.mean(results) * 100)

    return [ results, show ]
