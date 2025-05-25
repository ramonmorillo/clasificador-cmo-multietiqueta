
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Cargar modelo y binarizador
modelo = joblib.load("modelo_multietiqueta_cmo.pkl")
binarizador = joblib.load("binarizador_multietiqueta_cmo.pkl")

# Título
st.title("Clasificador CMO multietiqueta")
st.write("Introduce un texto clínico. El modelo identificará una o varias intervenciones farmacéuticas CMO y permitirá añadir comentarios clínicos por intervención.")

# Entrada de usuario
usuario = st.text_input("Identificador del usuario (nombre o código):")
texto_input = st.text_area("Texto clínico libre:", "")
comentario_general = st.text_area("Comentario global sobre esta intervención (opcional):", "")

# Historial
if "registro" not in st.session_state:
    st.session_state.registro = []

if st.button("Clasificar y registrar intervención"):
    if texto_input.strip() == "" or usuario.strip() == "":
        st.warning("Por favor, introduce un texto y un identificador de usuario.")
    else:
        etiquetas_bin = modelo.predict([texto_input])
        etiquetas = binarizador.inverse_transform(etiquetas_bin)[0]

        if etiquetas:
            st.success(f"Intervenciones detectadas: {', '.join(etiquetas)}")
            fila = {
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Usuario": usuario,
                "Texto": texto_input,
                "Intervenciones CMO": ", ".join(etiquetas),
                "Comentario": comentario_general
            }
            st.session_state.registro.append(fila)
        else:
            st.info("No se detectaron intervenciones en el texto.")

# Mostrar historial
if st.session_state.registro:
    df_hist = pd.DataFrame(st.session_state.registro)
    st.subheader("Historial de intervenciones registradas")
    st.dataframe(df_hist)

    # Botón de exportación
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar historial en CSV", csv, "historial_intervenciones_cmo.csv", "text/csv")
