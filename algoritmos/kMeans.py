from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
import numpy as np


def runKMeans(X_train, y_train, X_test, y_test, folds=10):
    results = []

    for i in range(folds):
      ## Set up inicial

      myset = set(y_train[i]) # Cria um conjunto. Em conjuntos, dados não se repetem. Assim, esse conjunto conterá apenas um valor de cada, ou seja: [1,2,3]
      clusters = len(myset) # Quantos clusters teremos no KMeans

      model = KMeans(n_clusters = clusters)
      model = model.fit(X_train[i])

      # Pegar os labels dos padrões de Treinamento
      labels = model.labels_

      map_labels = []

      for i in range(clusters):
        map_labels.append([])

      new_y_train = y_train[i]

      for i in range(len(y_train[i])):
        for c in range(clusters):
          if labels[i] == c:
            map_labels[c].append(new_y_train[i])

      # Criar dicionário com os labells a serem mapeados

      mapping = {}

      for i in range(clusters):
        final = Counter(map_labels[i]) # contar a classe que mais aparece
        value = final.most_common(1)[0][0] # retorna a classe com maior frequência
        mapping[i] = value

      result = model.predict(X_test[i])
      result = [mapping[i] for i in result]

      acc = metrics.accuracy_score(result, y_test[i])

      results.append(acc)

    show = round(np.mean(results) * 100)

    return [ results, show ]