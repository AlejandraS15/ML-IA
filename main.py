import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Generar dataset simulado
# -----------------------------
X, y = make_classification(
    n_samples=300,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# Convertir a DataFrame para mostrar
df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 7)])
df["Target"] = y

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Clasificación Supervisada en Python")
st.write("Ejemplo con **EDA, KNN, Árbol de Decisión y Naive Bayes** sobre un dataset simulado.")

# -----------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------
st.header("🔍 Análisis Exploratorio de Datos (EDA)")

st.subheader("📊 Vista previa del Dataset")
st.dataframe(df.head())

st.subheader("📈 Estadísticas Descriptivas")
st.write(df.describe())

# Histogramas
st.subheader("📊 Histogramas de las Variables")
columna_hist = st.selectbox("Selecciona una columna para ver su histograma:", df.columns[:-1])
fig, ax = plt.subplots()
ax.hist(df[columna_hist], bins=20, color="skyblue", edgecolor="black")
ax.set_title(f"Histograma de {columna_hist}")
st.pyplot(fig)

# Boxplots
st.subheader("📦 Boxplot de Variables")
columna_box = st.selectbox("Selecciona una columna para ver su boxplot:", df.columns[:-1])
fig, ax = plt.subplots()
sns.boxplot(x=df[columna_box], ax=ax, color="lightcoral")
ax.set_title(f"Boxplot de {columna_box}")
st.pyplot(fig)

# Correlación
st.subheader("📊 Mapa de Calor de Correlaciones")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------
# División de datos
# -----------------------------
test_size = st.slider("Proporción de test (%)", 10, 50, 20, step=5) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# -----------------------------
# Selección de modelo
# -----------------------------
st.header("⚙️ Entrenamiento de Modelos")
modelo = st.selectbox("Escoge un clasificador:", ["KNN", "Árbol de Decisión", "Naive Bayes"])

if modelo == "KNN":
    n_neighbors = st.slider("Número de vecinos (k)", 1, 15, 3)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
elif modelo == "Árbol de Decisión":
    max_depth = st.slider("Profundidad máxima del árbol", 1, 10, 3)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
else:
    clf = GaussianNB()

# -----------------------------
# Entrenar y evaluar
# -----------------------------
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Resultados del Modelo")
st.write(f"**Precisión del modelo:** {accuracy:.2f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
st.pyplot(fig)

# -----------------------------
# Visualización 2D de clases
# -----------------------------
st.subheader("🔎 Visualización 2D de Clases")
feat_x = st.selectbox("Eje X", df.columns[:-1], index=0)
feat_y = st.selectbox("Eje Y", df.columns[:-1], index=1)

fig, ax = plt.subplots()
scatter = ax.scatter(df[feat_x], df[feat_y], c=df["Target"], cmap="viridis", alpha=0.7)
legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
ax.add_artist(legend1)
ax.set_xlabel(feat_x)
ax.set_ylabel(feat_y)
st.pyplot(fig)
