import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Cargar datos
# -----------------------------
st.title("🌱 Clasificación de Cultivos con ML")
st.write("Sube un archivo CSV con datos agrícolas para identificar el tipo de cultivo.")

uploaded_file = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Validaciones
        if df.shape[1] < 3:
            st.error("❌ El archivo debe tener al menos 3 columnas (2 características + Target).")
            st.stop()

        if "Target" not in df.columns:
            st.error("❌ El archivo debe contener una columna llamada **Target** con el tipo de cultivo.")
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
        st.subheader("📊 Histogramas de Variables")
        columna_hist = st.selectbox("Selecciona una columna numérica:", df.columns[:-1])
        fig, ax = plt.subplots()
        ax.hist(df[columna_hist], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histograma de {columna_hist}")
        st.pyplot(fig)

        # Mapa de calor
        st.subheader("📊 Correlaciones")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax)
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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(ax=ax, cmap="Greens", colorbar=False)
        st.pyplot(fig)

        # -----------------------------
        # Predicción con datos nuevos
        # -----------------------------
        st.header("🔮 Predicción de Cultivo Nuevo")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"Ingrese valor para {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        if st.button("Predecir"):
            new_data = pd.DataFrame([input_data])
            prediction = clf.predict(new_data)
            st.success(f"🌱 El modelo predice que el cultivo es: **{prediction[0]}**")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
