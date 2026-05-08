# Importando librerias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz # https://graphviz.org/download/
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score, precision_score
from imblearn.metrics import sensitivity_score, specificity_score

def readData(rutaArchivo, formato='csv'):
    if formato == 'csv':
        data = pd.read_csv(rutaArchivo)
    return data

def getNombreClases(data, nombreClase, formato='csv'):
    nombreClases = data[nombreClase].unique()
    return nombreClases

#--------------------------------------------
#------------------ Main --------------------
#--------------------------------------------
archivo = './datasets/Iris.csv' 
nombreEtiqueta = 'Species'
columnasEliminar = [nombreEtiqueta,'Id']
criterio = 'gini'  # 'gini' o 'entropy'
profundidadMaxima = 3  # None para sin limite

data = readData(archivo)
nombreClases = getNombreClases(data, nombreEtiqueta)

# Separacion de datos en atributos y etiquetas
X = data.drop(columns=columnasEliminar)
y = data[nombreEtiqueta]

# Division de los datos en entrenamiento y prueba con hold-out estrateficado en 70%-30%
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.3,  
    random_state = 1, 
    stratify = y       
)

# Arbol de decision con sklearn
clf = DecisionTreeClassifier(criterion=criterio, max_depth=profundidadMaxima, random_state=1)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')

