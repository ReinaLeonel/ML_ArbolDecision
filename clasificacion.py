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

data = readData(archivo)
nombreClases = getNombreClases(data, nombreEtiqueta)

print("Nombre de clases:", nombreClases)
print("Dataset head: ", data.head())

