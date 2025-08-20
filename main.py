import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("🌱 Clasificación de Cultivos con Árbol de Decisión")

# Subida de archivo CSV
archivo = st.file_uploader("Sube un archivo CSV con los datos de agricultura", type=["csv"])

# Columnas requeridas
columnas_requeridas = [
    "pH_suelo", "Humedad", "Temperatura", 
    "Precipitacion", "RadiacionSolar", "Nutrientes", "Cultivo"
]

if archivo is not None:
    try:
        # Leer CSV
        df = pd.read_csv(archivo)

        # Verificación de columnas
        if not all(col in df.columns for col in columnas_requeridas):
            st.error(f"El archivo debe contener las siguientes columnas: {', '.join(columnas_requeridas)}")
        else:
            st.success("✅ Archivo cargado correctamente")
            
            # Mostrar primeros datos
            st.subheader("Vista previa de los datos")
            st.write(df.head())

            # Separar variables predictoras (X) y variable objetivo (y)
            X = df.drop("Cultivo", axis=1)
            y = df["Cultivo"]

            # Dividir en conjunto de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Entrenar árbol de decisión
            modelo = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
            modelo.fit(X_train, y_train)

            # Predicciones
            y_pred = modelo.predict(X_test)

            # Métricas de desempeño
            st.subheader("📊 Resultados del modelo")
            st.write("**Exactitud (Accuracy):**", accuracy_score(y_test, y_pred))
            st.text("Reporte de Clasificación:")
            st.text(classification_report(y_test, y_pred))

            # Mostrar árbol de decisión
            st.subheader("🌳 Visualización del Árbol de Decisión")
            fig, ax = plt.subplots(figsize=(16, 8))
            plot_tree(
                modelo, 
                feature_names=X.columns, 
                class_names=modelo.classes_, 
                filled=True, 
                rounded=True, 
                fontsize=10,
                ax=ax
            )
            st.pyplot(fig)

            # Predicción con nuevos datos
            st.subheader("🔎 Predicción de un nuevo cultivo")
            entradas = {}
            for col in X.columns:
                entradas[col] = st.number_input(f"Ingrese valor para {col}", float(df[col].min()), float(df[col].max()))
            
            if st.button("Predecir cultivo"):
                entrada_df = pd.DataFrame([entradas])
                prediccion = modelo.predict(entrada_df)[0]
                st.success(f"El modelo predice que el cultivo es: **{prediccion}**")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
else:
    st.info("Por favor, sube un archivo CSV para continuar.")
