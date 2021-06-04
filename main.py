from sklearn import metrics
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from algoritmos.MLP import runMLP
from algoritmos.searchTree import runSearchTree
from algoritmos.kNN import runKNN
from algoritmos.kMeans import runKMeans


## Carregar base de dados: haberman
# url = "https://raw.githubusercontent.com/fricaro/ml/development/bases/haberman.data"
# dataset = pd.read_csv(url, sep=",", header=None)

## Carregar base de dados: breast cancer
url = "https://raw.githubusercontent.com/fricaro/ml/development/bases/breast%20cancer%20coimbra.csv"
dataset = pd.read_csv(url, sep=",", header=None)

## Carregar base de dados: divorce
# url = "https://raw.githubusercontent.com/fricaro/ml/development/bases/divorce.csv"
# dataset = pd.read_csv(url, sep=";", header=None)

##

columns = len(dataset.columns)

y = dataset[columns-1] # extrai a primeira coluna, que é o label

X = dataset.loc[:,1:columns-2]

# Transforma para Array NumPy
X = np.array(X)
y = np.array(y)

### FOLDS

folds = 10

kf = StratifiedKFold(n_splits = folds, shuffle=True, random_state=100)

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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#########
## kMeans
#########
print(f"{bcolors.WARNING}###    kMeans{bcolors.ENDC}") 
results, show = runKMeans(X_train, y_train, X_test, y_test)
print(f"{bcolors.WARNING}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.WARNING}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.WARNING}###    média{bcolors.ENDC}") 
print(f"{bcolors.WARNING}###    {show}%{bcolors.ENDC}") 
print("")
print("")

#########
## kNN ##
#########
print(f"{bcolors.OKBLUE}###    kNN 5{bcolors.ENDC}") 
results, show = runKNN(X_train, y_train, X_test, y_test, 5)
print(f"{bcolors.OKBLUE}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.OKBLUE}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.OKBLUE}###    média{bcolors.ENDC}") 
print(f"{bcolors.OKBLUE}###    {show}%{bcolors.ENDC}") 
print("")

print(f"{bcolors.OKBLUE}###    kNN 10{bcolors.ENDC}") 
results, show = runKNN(X_train, y_train, X_test, y_test, 10)
print(f"{bcolors.OKBLUE}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.OKBLUE}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.OKBLUE}###    média{bcolors.ENDC}") 
print(f"{bcolors.OKBLUE}###    {show}%{bcolors.ENDC}") 
print("")
print("")

#########
## MLP###
#########
print(f"{bcolors.OKGREEN}###    MLP TAHN 3,2{bcolors.ENDC}") 
results, show = runMLP(X_train, y_train, X_test, y_test, 3, 2, 'tanh')
print(f"{bcolors.OKGREEN}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    média{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {show}%{bcolors.ENDC}") 
print("")
print("")

print(f"{bcolors.OKGREEN}###    MLP TAHN 4,3{bcolors.ENDC}") 
results, show = runMLP(X_train, y_train, X_test, y_test, 4, 3, 'tanh')
print(f"{bcolors.OKGREEN}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    média{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {show}%{bcolors.ENDC}") 
print("")
print("")

print(f"{bcolors.OKGREEN}###    MLP Relu 3,2{bcolors.ENDC}") 
results, show = runMLP(X_train, y_train, X_test, y_test, 3, 2, 'relu')
print(f"{bcolors.OKGREEN}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    média{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {show}%{bcolors.ENDC}") 
print("")
print("")

print(f"{bcolors.OKGREEN}###    MLP Relu 4,3{bcolors.ENDC}") 
results, show = runMLP(X_train, y_train, X_test, y_test, 4, 3, 'relu')
print(f"{bcolors.OKGREEN}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    média{bcolors.ENDC}") 
print(f"{bcolors.OKGREEN}###    {show}%{bcolors.ENDC}") 
print("")
print("")

##############
##searchTree##
##############
print(f"{bcolors.HEADER}###    Search Tree Gini{bcolors.ENDC}") 
results, show = runSearchTree(X_train, y_train, X_test, y_test, "gini")
print(f"{bcolors.HEADER}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.HEADER}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.HEADER}###    média{bcolors.ENDC}") 
print(f"{bcolors.HEADER}###    {show}%{bcolors.ENDC}") 
print("")
print("")


print(f"{bcolors.HEADER}###    Search Tree Entropy{bcolors.ENDC}") 
results, show = runSearchTree(X_train, y_train, X_test, y_test, "entropy")
print(f"{bcolors.HEADER}###    resultados{bcolors.ENDC}") 
print(f"{bcolors.HEADER}###    {results}{bcolors.ENDC}") 
print(f"{bcolors.HEADER}###    média{bcolors.ENDC}") 
print(f"{bcolors.HEADER}###    {show}%{bcolors.ENDC}") 
print("")
print("")

