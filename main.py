import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# -----------------------------
# Configuración inicial
# -----------------------------
st.title("🌱 Clasificación de Cultivos con ML")
st.write("Sube un archivo CSV con datos agrícolas para identificar el tipo de cultivo (Papa, Maíz, etc.)")

# -----------------------------
# Cargar CSV
# -----------------------------
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

required_columns = ["pH_suelo", "Humedad", "Temperatura", "Precipitacion", "RadiacionSolar", "Nutrientes", "Cultivo"]

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Validaciones
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ El archivo debe contener las columnas: {', '.join(required_columns)}")
            st.stop()

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

        # Correlación
        st.subheader("📊 Mapa de Calor de Correlaciones")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.drop("Cultivo", axis=1).corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # -----------------------------
        # Preparar datos para entrenamiento
        # -----------------------------
        X = df.drop("Cultivo", axis=1)
        y = df["Cultivo"]

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
            criterio = st.selectbox("Criterio", ["gini", "entropy"])
            clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterio, random_state=42)
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
        # Mostrar Árbol de Decisión
        # -----------------------------
        if modelo == "Árbol de Decisión":
            st.subheader("🌳 Visualización del Árbol de Decisión")
            fig, ax = plt.subplots(figsize=(12, 6))
            tree.plot_tree(
                clf,
                feature_names=X.columns,
                class_names=[str(c) for c in np.unique(y)],
                filled=True,
                rounded=True,
                fontsize=8,
                ax=ax
            )
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
else:
    st.info("👉 Por favor, sube un archivo CSV con tus datos agrícolas para continuar.")
