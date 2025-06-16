import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown

st.set_page_config(page_title="Diagnóstico Caña de Azúcar", page_icon="🌿", layout="centered")

@st.cache_resource
def load_model():
    model_path = os.path.abspath('best_sugarcane_model.keras')
    if not os.path.exists(model_path):
        # Reemplaza ENLACE_DIRECTO_O_ID por el enlace o ID real de Google Drive
        st.warning("Descargando el modelo desde Google Drive. Esto puede tardar unos minutos la primera vez...")
        try:
            gdown.download('1Uy22RdNdzZqc-96St6m7jreXZNJq761t', model_path, quiet=False)
        except Exception as e:
            st.error(f"No se pudo descargar el modelo: {str(e)}")
            return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

model = load_model()
if model is None:
    st.error("No se pudo cargar el modelo. Por favor, verifica que el archivo del modelo existe y está en el formato correcto.")
    st.stop()

DISEASE_INFO = {
    0: {'name': 'Healthy', 'color': '#4CAF50', 'treatment': 'No se requiere tratamiento.'},
    1: {'name': 'Red Rot', 'color': '#F44336', 'treatment': 'Aplicar carbendazim, eliminar plantas infectadas.'},
    2: {'name': 'Bacterial Blight', 'color': '#FF9800', 'treatment': 'Aplicar bactericidas y mejorar drenaje.'}
}

def preprocess_image(image: Image.Image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

st.title("🌿 Diagnóstico de Enfermedades en Hojas de Caña de Azúcar")
uploaded_file = st.file_uploader("📷 Sube una imagen de hoja", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("🔍 Diagnosticar"):
        with st.spinner("Procesando..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)
            disease = DISEASE_INFO[predicted_class]

        st.success("✅ Diagnóstico completado")
        st.markdown(f"### 🩺 Resultado: **{disease['name']}**")
        st.metric(label="Confianza", value=f"{confidence:.2f} %")
        st.markdown("### 💊 Tratamiento recomendado:")
        st.code(disease['treatment'])

        st.markdown("### 📊 Distribución de Probabilidades")
        fig, ax = plt.subplots()
        labels = [info['name'] for info in DISEASE_INFO.values()]
        colors = [info['color'] for info in DISEASE_INFO.values()]
        ax.bar(labels, prediction * 100, color=colors)
        ax.set_ylabel("Probabilidad (%)")
        ax.set_ylim([0, 100])
        plt.xticks(rotation=15)
        st.pyplot(fig)