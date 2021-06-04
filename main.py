from sklearn import metrics
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from algoritmos.MLP import runMLP
from algoritmos.searchTree import runSearchTree
from algoritmos.kNN import runKNN
from algoritmos.kMeans import runKMeans

url = "https://raw.githubusercontent.com/tmoura/machinelearning/master/iris.data"

# Carregar base de dados
dataset = pd.read_csv(url, header=None)

columns = len(dataset.columns)

y = dataset[0] # extrai a primeira coluna, que é o label
X = dataset.loc[:,1:columns-1]

# Transforma para Array NumPy
X = np.array(X)
y = np.array(y)

### FOLDS

folds = 10

kf = StratifiedKFold(n_splits = folds)

## 10 conjuntos de dados
X_train = []
y_train = []

X_test = []
y_test = []

for train_index, test_index in kf.split(X,y):
  X_train.append(X[train_index])
  X_test.append(X[test_index])
  
  y_train.append(y[train_index])
  y_test.append(y[test_index])

#########
## kMeans
#########
print("###    kMeans   ###") 
results, show = runKMeans(X_train, y_train, X_test, y_test)
print("RESULTADOS")
print(results)
print("MÉDIA")
print("{}%".format(show))
print("")

#########
## kNN ##
#########
print("###    kNN 5   ###") 
results, show = runKNN(X_train, y_train, X_test, y_test, 5)
print("RESULTADOS")
print(results)
print("MÉDIA")
print("{}%".format(show))
print("")

print("###    kNN 10   ###") 
results, show = runKNN(X_train, y_train, X_test, y_test, 10)
print("RESULTADOS")
print(results)
print("MÉDIA")
print("{}%".format(show))
print("")
#########
## MLP###
#########
print("###    MLP TAHN 3,2   ###") 
results, show = runMLP(X_train, y_train, X_test, y_test, 3, 2, 'tanh')
print("RESULTADOS")
print(results)
print("MÉDIA")
print("{}%".format(show))
print("")

print("###    MLP TAHN 4,3   ###")  
results, show = runMLP(X_train, y_train, X_test, y_test, 4, 3, 'tanh')
print("RESULTADOS")
print(results)
print("MÉDIA")
print("{}%".format(show))
print("")

print("###    MLP RELU 3,2   ###")  
results, show = runMLP(X_train, y_train, X_test, y_test, 3, 2, 'relu')
print("RESULTADOS")
print(results)
print("MÉDIA")
print("{}%".format(show))
print("")

print("###    MLP RELU 4,3   ###")  
results, show = runMLP(X_train, y_train, X_test, y_test, 4, 3, 'relu')
print("RESULTADOS")
print(results)
print("MÉDIA")
print("{}%".format(show))
print("")

##############
##searchTree##
##############
print("###    SEARCH TREE GINI  ###")
results, show = runSearchTree(X_train, y_train, X_test, y_test, "gini")
print("RESULTADOS GINI")
print(results)
print("MÉDIA GINI")
print("{}%".format(show))
print("")

print("###    SEARCH TREE ENTROPY  ###")
results, show = runSearchTree(X_train, y_train, X_test, y_test, "entropy")
print("RESULTADOS ENTROPY")
print(results)
print("MÉDIA ENTROPY")
print("{}%".format(show))
print("")
