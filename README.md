# Modelos de Aprendizaje Supervisado y No Supervisado en Python

Este repositorio contiene ejemplos prácticos de **modelos de Machine Learning** implementados en Python, enfocados en las dos grandes categorías:  
- **Aprendizaje Supervisado**  
- **Aprendizaje No Supervisado**

El objetivo es proporcionar una guía clara y sencilla para estudiantes y desarrolladores que quieran entender la diferencia entre ambos enfoques y cómo aplicarlos con librerías comunes como `scikit-learn`, `numpy` y `matplotlib`.

---

## 📚 Aprendizaje Supervisado
El **aprendizaje supervisado** utiliza datos de entrenamiento que incluyen **entradas (features)** y **salidas conocidas (labels)**.  
El modelo aprende una función que mapea las entradas a las salidas.

Ejemplos de algoritmos:
- Regresión Lineal
- Regresión Logística
- K-Nearest Neighbors (KNN)
- Árboles de Decisión
- Máquinas de Soporte Vectorial (SVM)

### Ejemplo en Python (Clasificación con KNN)
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset de ejemplo
X, y = load_iris(return_X_y=True)

# Dividir en entrena
