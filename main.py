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
# Título de la app
# -----------------------------
st.title("🌱 Clasificación de Cultivos con Machine Learning")
st.write("Sube un archivo CSV con datos agrícolas para identificar el tipo de cultivo (ej. papa, maíz, etc.).")

# -----------------------------
# Subir archivo CSV
# -----------------------------
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Validaciones básicas
        if df.shape[1] < 6:
            st.error("❌ El archivo debe tener al menos 6 columnas (5 características + Target).")
            st.stop()

        if "Target" not in df.columns:
            st.error("❌ El archivo debe contener una columna llamada **Target** con los cultivos (ej. papa, maíz...).")
            st.stop()

        # -----------------------------
        # Exploratory Data Analysis (EDA)
        # -----------------------------
        st.header("🔍 Análisis Exploratorio de Datos (EDA)")

        st.subheader("📊 Vista previa del Dataset")
        st.dataframe(df.head())

        st.subheader("📈 Estadísticas Descriptivas")
        st.write(df.describe())

        # Histograma
        st.subheader("📊 Histogramas")
        col_hist = st.selectbox("Selecciona una columna para histograma:", df.columns[:-1])
        fig, ax = plt.subplots()
        ax.hist(df[col_hist], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histograma de {col_hist}")
        st.pyplot(fig)

        # Boxplot
        st.subheader("📦 Boxplot")
        col_box = st.selectbox("Selecciona una columna para boxplot:", df.columns[:-1])
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col_box], ax=ax, color="lightcoral")
        ax.set_title(f"Boxplot de {col_box}")
        st.pyplot(fig)

        # Correlación
        st.subheader("📊 Mapa de Correlaciones")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # -----------------------------
        # Preparar datos
        # -----------------------------
        X = df.drop("Target", axis=1)
        y = df["Target"]

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
            max_depth = st.slider("Profundidad máxima del árbol", 1, 15, 5)
            criterion = st.selectbox("Criterio de división:", ["gini", "entropy"])
            clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
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
        # Visualización 2D
        # -----------------------------
        st.subheader("🔎 Visualización 2D de Clases")
        feat_x = st.selectbox("Eje X", df.columns[:-1], index=0)
        feat_y = st.selectbox("Eje Y", df.columns[:-1], index=1)

        fig, ax = plt.subplots()
        scatter = ax.scatter(df[feat_x], df[feat_y], c=pd.Categorical(df["Target"]).codes, cmap="viridis", alpha=0.7)
        legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
        ax.add_artist(legend1)
        ax.set_xlabel(feat_x)
        ax.set_ylabel(feat_y)
        st.pyplot(fig)

        # -----------------------------
        # Mostrar Árbol de Decisión
        # -----------------------------
        if modelo == "Árbol de Decisión":
            st.subheader("🌳 Visualización del Árbol de Decisión")

            fig, ax = plt.subplots(figsize=(14, 8))
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
        st.error(f"❌ Error al procesar el archivo: {e}")
