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

def getGraph_graphviz(modelo, X, nombreClases, archivoSalida):
    dot_data = export_graphviz(
        modelo,
        out_file=None,
        feature_names=X.columns,
        class_names=nombreClases,
        filled=True
    )

    dot_data = dot_data.replace(
        "digraph Tree {",
        'digraph Tree {\n dpi=300;\n size="20,20!";\n node [fontname="Helvetica", fontsize=12];'
    )

    graph = graphviz.Source(dot_data)
    graph.render(archivoSalida, format="png", cleanup=True)

#--------------------------------------------
#------------------ Main --------------------
#--------------------------------------------
archivo = './datasets/Iris.csv' 
nombreEtiqueta = 'Species'
columnasEliminar = [nombreEtiqueta,'Id']
criterio = 'gini'  # 'gini' o 'entropy'
profundidadMaxima = 3  # None para sin limite
datasetName = 'Iris'
nombreSalidaGraphviz = f'Imagenes/{datasetName}_Depth-{profundidadMaxima}_graph_{criterio}'

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

getGraph_graphviz(clf, X, nombreClases, nombreSalidaGraphviz)

metrics = []

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred, labels=nombreClases)
conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=nombreClases)
conf_matrix_display.plot(cmap='Blues')
conf_matrix_display.ax_.set_title(f'Matriz de confusión - {datasetName}')
plt.tight_layout()
plt.savefig(f'Imagenes/{datasetName}_MatrizConfusion.png')

# Cálculo de métricas
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
sensitivity = sensitivity_score(y_test, y_pred, average='weighted')
specificity = specificity_score(y_test, y_pred, average='weighted')

metrics.append({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': sensitivity,
        'F1-Score': f1,
        'Balanced Accuracy': balanced_acc,
        'Specificity': specificity,
    })

df_metrics = pd.DataFrame(metrics)

with open(f'./metricas/metricas_{datasetName}.df', 'w', encoding='utf-8') as file:
    file.write(f'Métricas para el dataset {datasetName} con criterio {criterio} y profundidad máxima {profundidadMaxima}\n\n')
    file.write(df_metrics.to_string())

