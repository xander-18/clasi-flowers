# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Cargar el dataset
from sklearn.datasets import load_iris

# Cargar el dataset de Iris
data = load_iris()
# Crear un DataFrame con las características y etiquetas
dataset_flower = pd.DataFrame(data=data.data, columns=data.feature_names)
dataset_flower['species'] = data.target

# Renombrar las columnas para hacerlas más fáciles de manejar
dataset_flower.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Mapeo de las especies
species_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
dataset_flower["species"] = dataset_flower["species"].map(species_mapping)

# Dividir en características (X) y etiquetas (y)
X = dataset_flower[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = dataset_flower["species"]

# Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Accuracy del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Visualizar el árbol de decisión
plt.figure(figsize=(15, 10))
tree.plot_tree(model, feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
               class_names=["Setosa", "Versicolor", "Virginica"], filled=True)
plt.title("Árbol de decisión para clasificar flores")
plt.show()

# Predicción con nuevos datos
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Ejemplo de características de una nueva flor
prediction = model.predict(new_flower)
print("Predicción para la nueva flor:", prediction[0])
