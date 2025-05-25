
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import numpy as np

# Cargar modelo y binarizador
modelo = joblib.load("modelo_multietiqueta_cmo.pkl")
binarizador = joblib.load("binarizador_multietiqueta_cmo.pkl")

# Umbral ajustado para considerar una etiqueta como activa
UMBRAL = 0.3

st.title("Clasificador CMO multietiqueta (versión con depuración)")
st.write("Introduce un texto clínico. El modelo identificará una o varias intervenciones farmacéuticas CMO y permitirá añadir comentarios clínicos.")

usuario = st.text_input("Identificador del usuario (nombre o código):")
texto_input = st.text_area("Texto clínico libre:", "")
comentario_general = st.text_area("Comentario global sobre esta intervención (opcional):", "")

# Historial local
if "registro" not in st.session_state:
    st.session_state.registro = []

if st.button("Clasificar y registrar intervención"):
    if texto_input.strip() == "" or usuario.strip() == "":
        st.warning("Por favor, introduce un texto y un identificador de usuario.")
    else:
        probas = modelo.predict_proba([texto_input])[0]
        etiquetas_activas = [etiqueta for etiqueta, prob in zip(binarizador.classes_, probas) if prob >= UMBRAL]

        # Mostrar resultados
        if etiquetas_activas:
            st.success(f"Intervenciones detectadas (umbral {UMBRAL}): {', '.join(etiquetas_activas)}")
            fila = {
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Usuario": usuario,
                "Texto": texto_input,
                "Intervenciones CMO": ", ".join(etiquetas_activas),
                "Comentario": comentario_general
            }
            st.session_state.registro.append(fila)
        else:
            st.info("No se detectaron intervenciones con suficiente confianza.")

        # Mostrar tabla de probabilidades
        st.subheader("Probabilidades por etiqueta")
        df_probas = pd.DataFrame({
            "Código": binarizador.classes_,
            "Probabilidad": np.round(probas, 3)
        }).sort_values(by="Probabilidad", ascending=False)
        st.dataframe(df_probas)

# Mostrar historial
if st.session_state.registro:
    df_hist = pd.DataFrame(st.session_state.registro)
    st.subheader("Historial de intervenciones registradas")
    st.dataframe(df_hist)
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar historial en CSV", csv, "historial_intervenciones_cmo.csv", "text/csv")
