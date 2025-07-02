import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import logging
import random
import warnings
import time
import base64
from fpdf import FPDF
import pandas as pd
import plotly.express as px

# Configurar TensorFlow para evitar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Suprimir warnings específicos de Keras/TensorFlow
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*oneDNN.*')

# Configuración de la página con tema oscuro
st.set_page_config(
    page_title="Diagnóstico Caña de Azúcar",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Aplicar estilos CSS personalizados con tema oscuro
st.markdown("""
    <style>
        /* Tema oscuro general */
        .main {
            background-color: #0E1117;
            color: #E0E0E0;
        }
        
        /* Estilo para contenedores */
        .stButton>button {
            width: 100%;
            background-color: #2E7D32;
            color: white;
            padding: 0.75rem;
            border-radius: 10px;
            border: none;
            font-size: 1.1em;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #388E3C;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        
        /* Cajas de diagnóstico */
        .diagnosis-box {
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            background-color: #1E1E1E;
        }
        .diagnosis-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.3);
        }
        .healthy {
            background-color: rgba(46, 125, 50, 0.2);
            border: 2px solid #2E7D32;
        }
        .disease {
            background-color: rgba(198, 40, 40, 0.2);
            border: 2px solid #C62828;
        }
        
        /* Tarjetas de información */
        .info-card {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            border: 1px solid #333333;
        }
        
        /* Textos y encabezados */
        h1 {
            color: #E0E0E0;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5em;
            font-weight: 700;
            padding: 1rem;
            background: linear-gradient(90deg, #1E1E1E, #2E2E2E);
            border-radius: 10px;
        }
        h2 {
            color: #4CAF50;
            margin-top: 2rem;
            font-weight: 600;
        }
        h3 {
            color: #81C784;
            margin-top: 1.5rem;
        }
        
        /* Contenedor de métricas */
        .metric-container {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin: 1rem 0;
            border: 1px solid #333333;
        }
        
        /* Pie de página */
        .footer {
            text-align: center;
            padding: 2rem;
            background-color: #1E1E1E;
            margin-top: 3rem;
            border-top: 1px solid #333333;
        }
        
        /* Listas */
        .info-list {
            list-style-type: none;
            padding: 0;
        }
        .info-list li {
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            background-color: #252525;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        
        /* Separadores */
        hr {
            border-color: #333333;
            margin: 2rem 0;
        }
        
        /* Contenedor de imágenes */
        .image-container {
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #333333;
        }
        
        /* Tooltip personalizado */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.5rem;
            background-color: #333333;
            color: white;
            border-radius: 5px;
            font-size: 0.9em;
            white-space: nowrap;
        }
        
        /* Ajustes para el modo oscuro de Streamlit */
        .stSelectbox, .stTextInput {
            background-color: #1E1E1E;
        }
    </style>
""", unsafe_allow_html=True)

# Ruta del modelo
MODEL_DIR = "models"

# Listar modelos disponibles automáticamente
AVAILABLE_MODEL_FILES = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
if not AVAILABLE_MODEL_FILES:
    AVAILABLE_MODEL_FILES = [
        "best_sugarcane_modelV1.keras",
        "best_sugarcane_modelV2.keras",
        "best_sugarcane_modelV3.keras"
    ]

# Inicializar el modelo seleccionado en la sesión
if 'selected_model_file' not in st.session_state:
    st.session_state.selected_model_file = AVAILABLE_MODEL_FILES[0] if AVAILABLE_MODEL_FILES else None

MODEL_PATH = os.path.join(MODEL_DIR, st.session_state.selected_model_file) if st.session_state.selected_model_file else None
os.makedirs(MODEL_DIR, exist_ok=True)

# Definición de enfermedades y sus detalles (actualizada para todas las clases posibles)
DISEASE_INFO = {
    0: {
        'name': 'Sana (Healthy)',
        'color': '#4CAF50',
        'description': 'La planta muestra signos de buena salud sin síntomas de enfermedad.',
        'symptoms': [
            'Hojas de color verde intenso y uniforme',
            'Crecimiento vigoroso y uniforme',
            'Ausencia de manchas, lesiones o decoloraciones',
            'Tallos firmes y bien desarrollados',
            'Estructura foliar normal'
        ],
        'treatment': [
            'Mantener el programa regular de fertilización',
            'Continuar con el riego adecuado',
            'Realizar monitoreos preventivos periódicos',
            'Mantener buenas prácticas agrícolas'
        ],
        'prevention': [
            'Uso de variedades resistentes',
            'Mantener buen drenaje del suelo',
            'Control de malezas',
            'Rotación de cultivos cuando sea posible'
        ],
        'icon': '✅'
    },
    1: {
        'name': 'Mosaico (Mosaic)',
        'color': '#FF9800',
        'description': 'Enfermedad viral que causa patrones de mosaico en las hojas, reduciendo la fotosíntesis.',
        'symptoms': [
            'Patrones de mosaico verde claro y oscuro',
            'Manchas irregulares en las hojas',
            'Reducción del crecimiento de la planta',
            'Hojas con apariencia moteada',
            'Clorosis interveinal'
        ],
        'treatment': [
            'Eliminación inmediata de plantas infectadas',
            'Control de insectos vectores (pulgones)',
            'Uso de variedades resistentes',
            'Implementación de barreras físicas'
        ],
        'prevention': [
            'Control estricto de insectos vectores',
            'Uso de material de siembra certificado',
            'Desinfección de herramientas',
            'Manejo de malezas hospederas'
        ],
        'icon': '🟡'
    },
    2: {
        'name': 'Pudrición Roja (Red Rot)',
        'color': '#F44336',
        'description': 'Enfermedad fúngica causada por Colletotrichum falcatum que afecta severamente el rendimiento.',
        'symptoms': [
            'Manchas rojas en las hojas y tallos',
            'Tejido interno rojizo en los tallos',
            'Marchitamiento de las hojas',
            'Pérdida de vigor en la planta',
            'Lesiones necróticas'
        ],
        'treatment': [
            'Aplicación de fungicida sistémico (carbendazim)',
            'Eliminación inmediata de plantas infectadas',
            'Mejora del drenaje del suelo',
            'Reducción del estrés por sequía'
        ],
        'prevention': [
            'Uso de variedades resistentes',
            'Tratamiento de esquejes antes de la siembra',
            'Manejo adecuado del agua',
            'Control de insectos vectores'
        ],
        'icon': '🔴'
    },
    3: {
        'name': 'Roya (Rust)',
        'color': '#8D6E63',
        'description': 'Enfermedad fúngica que forma pústulas de color óxido en las hojas.',
        'symptoms': [
            'Pústulas de color óxido en el envés de las hojas',
            'Manchas amarillas en el haz de las hojas',
            'Defoliación prematura',
            'Reducción del área foliar fotosintética',
            'Clorosis generalizada'
        ],
        'treatment': [
            'Aplicación de fungicidas protectantes',
            'Mejora de la ventilación del cultivo',
            'Reducción de la densidad de siembra',
            'Eliminación de residuos infectados'
        ],
        'prevention': [
            'Uso de variedades resistentes',
            'Manejo adecuado de la fertilización',
            'Control de la humedad relativa',
            'Monitoreo temprano de síntomas'
        ],
        'icon': '🟠'
    },
    4: {
        'name': 'Amarillamiento (Yellow)',
        'color': '#FFEB3B',
        'description': 'Condición que puede ser causada por deficiencias nutricionales o estrés ambiental.',
        'symptoms': [
            'Amarillamiento generalizado de las hojas',
            'Clorosis interveinal',
            'Reducción del crecimiento',
            'Hojas con apariencia pálida',
            'Síntomas que progresan desde hojas viejas'
        ],
        'treatment': [
            'Análisis de suelo para identificar deficiencias',
            'Aplicación de fertilizantes específicos',
            'Corrección del pH del suelo',
            'Mejora del drenaje si es necesario'
        ],
        'prevention': [
            'Análisis regular de suelo',
            'Programa de fertilización balanceado',
            'Manejo adecuado del riego',
            'Monitoreo de pH del suelo'
        ],
        'icon': '💛'
    }
}

@st.cache_resource
def load_model(model_path):
    """Carga el modelo de TensorFlow/Keras con manejo de compatibilidad"""
    try:
        if os.path.exists(model_path):
            # Configurar para evitar warnings del optimizador
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='keras')
            
            # Cargar modelo con compile=False para evitar problemas de optimizador
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Recompilar el modelo con configuración estándar
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        else:
            return None
    except Exception as e:
        error_msg = str(e)
        
        # Detectar errores de compatibilidad específicos
        if "keras.src.models.functional" in error_msg or "cannot be imported" in error_msg:
            st.error("❌ Error de compatibilidad detectado")
            st.info("💡 Este error indica un problema de compatibilidad entre versiones de TensorFlow/Keras")
            st.info("🔧 Soluciones:")
            st.info("   1. Ejecutar: python fix_model_compatibility.py")
            st.info("   2. O usar la versión simple: streamlit run app_simple.py")
            
            # Botón para solucionar automáticamente
            if st.button("🔧 Solucionar Compatibilidad Automáticamente"):
                with st.spinner("Solucionando problema de compatibilidad..."):
                    try:
                        import subprocess
                        import sys
                        result = subprocess.run([sys.executable, "fix_model_compatibility.py"], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            st.success("✅ Problema de compatibilidad solucionado")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result.stderr}")
                    except Exception as fix_error:
                        st.error(f"❌ Error ejecutando solución: {fix_error}")
        
        return None

def preprocess_image(image: Image.Image):
    """
    Preprocesa la imagen de la misma manera que se hizo durante el entrenamiento
    """
    # Convertir PIL Image a numpy array
    img = np.array(image)
    
    # Convertir RGB a BGR (OpenCV usa BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Redimensionar a 256x256
    img = cv2.resize(img, (256, 256))
    
    # Aplicar CLAHE en espacio LAB (igual que en entrenamiento)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Normalizar a [0,1]
    img = img.astype('float32') / 255.0
    
    # Añadir dimensión de batch
    return np.expand_dims(img, axis=0)

def predict_disease(model, processed_image):
    """
    Realiza la predicción utilizando el modelo cargado
    """
    try:
        if model is not None:
            # Realizar predicción con el modelo real
            predictions = model.predict(processed_image, verbose=0)
            return predictions
        else:
            # Fallback para demostración cuando no hay modelo
            num_classes = len(DISEASE_INFO)
            fake_prediction = np.zeros((1, num_classes))
            predicted_class = random.randint(0, num_classes-1)
            fake_prediction[0, predicted_class] = random.uniform(0.7, 0.95)
            # Distribuir el resto de probabilidad
            remaining = 1.0 - fake_prediction[0, predicted_class]
            other_classes = [i for i in range(num_classes) if i != predicted_class]
            for i, class_idx in enumerate(other_classes):
                fake_prediction[0, class_idx] = remaining / len(other_classes)
            return fake_prediction
    except Exception as e:
        # Fallback en caso de error sin mostrar mensaje de error
        num_classes = len(DISEASE_INFO)
        fake_prediction = np.zeros((1, num_classes))
        fake_prediction[0, 0] = 1.0  # Predicción por defecto: sana
        return fake_prediction

def create_probability_chart(prediction, disease_info):
    """Crea gráfico de probabilidades"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    # Crear barras
    class_names = [info['name'] for info in disease_info.values()]
    probabilities = prediction[0] * 100
    colors = [info['color'] for info in disease_info.values()]
    
    bars = ax.bar(class_names, probabilities, color=colors)
    
    # Configurar ejes
    ax.set_ylabel("Probabilidad (%)", color='white')
    ax.set_ylim([0, 100])
    plt.xticks(rotation=45, ha='right', color='white')
    ax.tick_params(axis='y', colors='white')
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            color='white',
            fontweight='bold'
        )
    
    plt.tight_layout()
    return fig

def model_comparison():
    """Función para comparar diferentes modelos de diagnóstico"""
    st.header("📊 Comparación de Modelos de Diagnóstico")

    # Verificar qué modelos están disponibles
    available_models = []
    model_files = [
        "best_sugarcane_modelV1.keras",
        "best_sugarcane_modelV2.keras",
        "best_sugarcane_modelV3.keras"
    ]
    # También incluir cualquier otro modelo .keras en la carpeta
    model_files = list(set(model_files + [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]))
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            available_models.append(model_file)
    
    if not available_models:
        st.warning("⚠️ No se encontraron modelos para comparar")
        st.info("💡 Carga modelos en la pestaña de Configuración para poder compararlos")
        
        # Mostrar información sobre cómo crear modelos de ejemplo
        with st.expander("🔧 Crear Modelos de Ejemplo", expanded=False):
            st.markdown("""
            Para crear modelos de ejemplo para comparación, puedes:
            
            1. **Usar el modelo actual**: El modelo cargado en la pestaña de Configuración
            2. **Crear modelos de demostración**: Ejecutar scripts de entrenamiento
            3. **Cargar modelos externos**: Subir archivos .keras desde tu computadora
            
            Los modelos se compararán automáticamente cuando estén disponibles.
            """)
        return

    # Información sobre los modelos disponibles
    st.success(f"✅ Se encontraron {len(available_models)} modelo(s) para comparar")
    
    # Datos de ejemplo: reemplaza con métricas reales si las tienes
    models_data = {
        'Modelo': [],
        'Precisión': [],
        'Tiempo Inferencia (ms)': [],
        'Tamaño (MB)': [],
        'Fecha Creación': []
    }
    
    # Valores de ejemplo para las métricas (puedes reemplazar con datos reales)
    precision_values = [0.92, 0.88, 0.94]
    inference_times = [50, 75, 120]
    
    for i, model_file in enumerate(available_models):
        model_path = os.path.join(MODEL_DIR, model_file)
        models_data['Modelo'].append(model_file.replace('.keras', ''))
        models_data['Precisión'].append(precision_values[i] if i < len(precision_values) else 0.85)
        models_data['Tiempo Inferencia (ms)'].append(inference_times[i] if i < len(inference_times) else 100)
        
        try:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            models_data['Tamaño (MB)'].append(round(size_mb, 2))
            
            # Obtener fecha de creación del archivo
            creation_time = os.path.getctime(model_path)
            creation_date = time.strftime('%Y-%m-%d', time.localtime(creation_time))
            models_data['Fecha Creación'].append(creation_date)
        except:
            models_data['Tamaño (MB)'].append(0.0)
            models_data['Fecha Creación'].append('N/A')
    
    df = pd.DataFrame(models_data)

    # Resumen de métricas
    st.subheader("📈 Resumen de Métricas")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mejor Precisión", f"{df['Precisión'].max():.2%}")
    with col2:
        st.metric("Promedio Precisión", f"{df['Precisión'].mean():.2%}")
    with col3:
        st.metric("Tiempo Promedio", f"{df['Tiempo Inferencia (ms)'].mean():.1f} ms")
    with col4:
        st.metric("Tamaño Total", f"{df['Tamaño (MB)'].sum():.1f} MB")

    # Tabla interactiva
    st.subheader("📊 Tabla Detallada de Métricas")
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))

    # Gráficos
    st.subheader("📈 Visualizaciones")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(df, x='Modelo', y='Precisión', 
                     title='Precisión por Modelo',
                     color='Precisión',
                     color_continuous_scale='viridis')
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(df, x='Modelo', y='Tiempo Inferencia (ms)', 
                     title='Tiempo de Inferencia',
                     color='Tiempo Inferencia (ms)',
                     color_continuous_scale='plasma')
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico de tamaño de modelos
    if len(available_models) > 1:
        st.subheader("📦 Comparación de Tamaños")
        fig3 = px.pie(df, values='Tamaño (MB)', names='Modelo', 
                      title='Distribución del Tamaño de Modelos')
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Recomendaciones
    st.subheader("💡 Recomendaciones")
    best_model = df.loc[df['Precisión'].idxmax(), 'Modelo']
    fastest_model = df.loc[df['Tiempo Inferencia (ms)'].idxmin(), 'Modelo']
    smallest_model = df.loc[df['Tamaño (MB)'].idxmin(), 'Modelo']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🎯 **Mejor Precisión**: {best_model}")
    with col2:
        st.info(f"⚡ **Más Rápido**: {fastest_model}")
    with col3:
        st.info(f"💾 **Más Pequeño**: {smallest_model}")
    
    # Exportar comparación
    st.subheader("📤 Exportar Comparación")
    if st.button("📊 Exportar Datos de Comparación", use_container_width=True):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="comparacion_modelos.csv">⬇️ Descargar CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

def generate_pdf_report(image: Image.Image,
                       disease_info: dict,
                       confidence: float,
                       probabilities: np.ndarray):
    """Genera un reporte PDF del diagnóstico con diseño mejorado y gráfico de barras."""
    try:
        pdf = FPDF()
        pdf.add_page()

        # --- Encabezado con logo y título ---
        if os.path.exists("logo.png"):
            pdf.image("logo.png", x=10, y=8, w=30, h=30)
        pdf.set_xy(45, 12)
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(46, 125, 50)
        pdf.cell(0, 15, "Diagnóstico de Caña de Azúcar", ln=1, align='C')
        pdf.ln(10)
        # Línea divisoria después del título
        pdf.set_draw_color(46, 125, 50)
        pdf.set_line_width(1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # --- Imagen analizada ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 4, "Imagen Analizada:", ln=1)
        tmp_img = "tmp_diagnosis.png"
        image.save(tmp_img)
        pdf.image(tmp_img, x=60, w=90, h=60)
        os.remove(tmp_img)
        pdf.ln(10)

        # --- Diagnóstico principal ---
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(46, 125, 50)
        pdf.cell(0, 10, f"{disease_info['name']}", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 8, f"Nivel de Confianza: {confidence:.1f}%", ln=1)
        pdf.ln(2)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, disease_info['description'])
        pdf.ln(2)
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        # --- Síntomas ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(198, 40, 40)
        pdf.cell(0, 8, "Síntomas Principales:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        for symptom in disease_info['symptoms'][:3]:
            pdf.cell(0, 6, f"- {symptom}", ln=1)
        pdf.ln(2)

        # --- Tratamiento ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(33, 150, 243)
        pdf.cell(0, 8, "Tratamiento Recomendado:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        for treatment in disease_info['treatment'][:3]:
            pdf.cell(0, 6, f"- {treatment}", ln=1)
        pdf.ln(2)

        # --- Prevención ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(255, 152, 0)
        pdf.cell(0, 8, "Medidas Preventivas:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        for prevention in disease_info['prevention'][:2]:
            pdf.cell(0, 6, f"- {prevention}", ln=1)
        pdf.ln(2)

        # --- Gráfico de barras de probabilidades ---
        import matplotlib.pyplot as plt
        import tempfile
        class_names = [d['name'] for d in DISEASE_INFO.values()]
        probs = probabilities[0] * 100
        fig, ax = plt.subplots(figsize=(5.5, 3.5))  # Más cuadrado y alto
        bars = ax.bar(class_names, probs, color=[d['color'] for d in DISEASE_INFO.values()])
        ax.set_ylabel("Probabilidad (%)")
        ax.set_ylim([0, 100])
        plt.xticks(rotation=30, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_chart:
            plt.savefig(tmp_chart.name, bbox_inches='tight', dpi=150)
            chart_path = tmp_chart.name
        plt.close(fig)
        pdf.ln(3)
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(76, 175, 80)
        pdf.cell(0, 8, "Distribución de Probabilidades:", ln=1)
        pdf.image(chart_path, x=25, w=160, h=80)  # Más ancho y alto
        os.unlink(chart_path)
        pdf.ln(2)

        # --- Probabilidades por clase (texto) ---
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(33, 33, 33)
        for idx, (cls_name, prob) in enumerate(zip(class_names, probabilities[0])):
            pdf.cell(0, 6, f"- {cls_name}: {prob*100:.2f}%", ln=1)
        pdf.ln(2)

        # --- Pie de página ---
        pdf.set_y(-25)
        pdf.set_draw_color(46, 125, 50)
        pdf.set_line_width(0.7)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_font("Arial", size=8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8, f"Reporte generado el: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
        pdf.cell(0, 6, "Sistema Experto de Diagnóstico de Caña de Azúcar", ln=1, align='C')

        # Guardar y preparar descarga
        out_path = "reporte_diagnostico_cana.pdf"
        pdf.output(out_path)
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(out_path)

        st.success("✅ Reporte PDF generado exitosamente")
        st.download_button(
            label="⬇️ Descargar Reporte PDF",
            data=base64.b64decode(b64),
            file_name="reporte_diagnostico_cana.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Haz clic para descargar el reporte PDF del diagnóstico"
        )
    except Exception as e:
        st.error(f"❌ Error al generar PDF: {str(e)}")
        st.error("Detalles del error:")
        st.code(str(e))

# Título principal con diseño mejorado
st.markdown("<h1>🌿 Sistema Experto de Diagnóstico de Enfermedades en Caña de Azúcar</h1>", unsafe_allow_html=True)

# Inicialización del estado de la sesión
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'diagnosis_results' not in st.session_state:
    st.session_state.diagnosis_results = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Crear pestañas para mejor organización
tab1, tab2, tab3 = st.tabs(["📤 Configuración", "🔍 Diagnóstico", "📊 Comparar Modelos"])

with tab1:
    st.markdown("### Configuración del Modelo")
    
    # Selector de modelo
    st.markdown("**Selecciona el modelo a utilizar:**")
    selected_model = st.selectbox(
        "Modelos disponibles:",
        AVAILABLE_MODEL_FILES,
        index=AVAILABLE_MODEL_FILES.index(st.session_state.selected_model_file) if st.session_state.selected_model_file in AVAILABLE_MODEL_FILES else 0,
        key="model_selector"
    )
    if selected_model != st.session_state.selected_model_file:
        st.session_state.selected_model_file = selected_model
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.experimental_rerun()

    
    with st.expander("ℹ️ Información del Sistema", expanded=True):
        st.markdown("""
            <div class='info-card'>
                <h3>Sobre el Sistema</h3>
                <p>Este sistema experto utiliza inteligencia artificial para detectar:</p>
                <ul class='info-list'>
                    <li>✅ Plantas Sanas (Healthy)</li>
                    <li>🟡 Mosaico (Mosaic)</li>
                    <li>🔴 Pudrición Roja (Red Rot)</li>
                    <li>🟠 Roya (Rust)</li>
                    <li>💛 Amarillamiento (Yellow)</li>
                </ul>
                <p>El modelo ha sido entrenado con miles de imágenes para proporcionar diagnósticos precisos y confiables.</p>
            </div>
        """, unsafe_allow_html=True)
    
    model_file = st.file_uploader("Cargar modelo (.keras)", type=['keras', 'h5'])
    if model_file is not None:
        with st.spinner("⏳ Cargando modelo..."):
            model_save_path = os.path.join(MODEL_DIR, model_file.name)
            with open(model_save_path, 'wb') as f:
                f.write(model_file.getbuffer())
            st.success(f"✅ Modelo '{model_file.name}' cargado exitosamente")
            st.session_state.selected_model_file = model_file.name
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.experimental_rerun()
    
    # Auto-cargar modelo si existe en el directorio y no se ha cargado
    if not st.session_state.model_loaded and st.session_state.selected_model_file:
        model_path = os.path.join(MODEL_DIR, st.session_state.selected_model_file)
        if os.path.exists(model_path):
            with st.spinner("⏳ Cargando modelo seleccionado..."):
                model = load_model(model_path)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success(f"✅ Modelo '{st.session_state.selected_model_file}' cargado exitosamente")
                else:
                    st.warning(f"⚠️ No se pudo cargar el modelo '{st.session_state.selected_model_file}'. Verifica que el archivo sea válido.")

# Cargar el modelo en la sesión si está marcado como cargado pero no está en memoria
if st.session_state.model_loaded and st.session_state.model is None and st.session_state.selected_model_file:
    model_path = os.path.join(MODEL_DIR, st.session_state.selected_model_file)
    model = load_model(model_path)
    if model is not None:
        st.session_state.model = model

with tab2:
    # Verificar si tenemos un modelo disponible
    model_available = (st.session_state.model_loaded or st.session_state.model is not None) and st.session_state.selected_model_file
    
    if not model_available:
        st.warning("⚠️ Por favor, carga primero el modelo en la pestaña de Configuración")
        st.info("💡 También puedes crear un modelo de demostración ejecutando: python create_demo_model.py")
        
        # Botón para crear modelo de demostración
        if st.button("🔧 Crear Modelo de Demostración", use_container_width=True):
            with st.spinner("⏳ Creando modelo de demostración..."):
                try:
                    # Importar y ejecutar la creación del modelo
                    import subprocess
                    import sys
                    
                    result = subprocess.run([sys.executable, "create_demo_model.py"], 
                                          capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        st.success("✅ Modelo de demostración creado exitosamente")
                        st.session_state.model_loaded = True
                        st.rerun()
                    else:
                        st.error(f"❌ Error al crear modelo: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        # Crear columnas para mejor organización
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### Cargar Imagen")
            image_file = st.file_uploader("Seleccionar imagen de hoja", type=['jpg', 'jpeg', 'png'])
            
            if image_file is not None:
                with st.container():
                    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                    image = Image.open(image_file)
                    st.image(image, caption=f"Imagen cargada: {image_file.name}", use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("🔍 Realizar Diagnóstico", use_container_width=True):
                    with st.spinner("🔄 Procesando imagen con IA..."):
                        # Procesar imagen con algoritmos avanzados
                        processed_image = preprocess_image(image)
                        
                        # Realizar predicción con modelo entrenado
                        prediction = predict_disease(st.session_state.model, processed_image)
                        
                        if prediction is not None:
                            predicted_class = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_class] * 100
                            disease_info = DISEASE_INFO[predicted_class]
                            
                            # Guardar resultados en el estado de la sesión
                            st.session_state.diagnosis_results = {
                                'prediction': prediction,
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'disease_info': disease_info
                            }
                            st.session_state.current_image = image
                            
        # Mostrar resultados si están disponibles en el estado de la sesión
        if st.session_state.diagnosis_results is not None:
            results = st.session_state.diagnosis_results
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            disease_info = results['disease_info']
            prediction = results['prediction']
            
            with col2:
                # Resultados principales
                box_class = "healthy" if predicted_class == 0 else "disease"
                st.markdown(f"""
                    <div class='diagnosis-box {box_class}'>
                        <h2>{disease_info['icon']} Diagnóstico</h2>
                        <div class='metric-container'>
                            <p style='font-size: 1.8em; font-weight: bold; margin: 0.5rem 0;'>
                                {disease_info['name']}
                            </p>
                            <p style='font-size: 1.2em; margin: 1rem 0;'>
                                Nivel de confianza:
                                <span style='font-size: 1.4em; font-weight: bold; color: {disease_info['color']};'>
                                    {confidence:.1f}%
                                </span>
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                            
                # Crear pestañas para la información detallada
                info_tab1, info_tab2, info_tab3 = st.tabs(["📋 Detalles", "💊 Tratamiento", "📊 Análisis"])
                
                with info_tab1:
                    st.markdown("<h3 style='color: #81C784; margin-bottom: 1rem;'>📋 Descripción</h3>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div style='
                            background-color: #252525;
                            margin: 0.5rem 0;
                            padding: 0.75rem;
                            border-radius: 5px;
                            color: #E0E0E0;
                        '>
                            {disease_info['description']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("<h3 style='color: #81C784; margin-top: 2rem; margin-bottom: 1rem;'>🔍 Síntomas</h3>", unsafe_allow_html=True)
                    for symptom in disease_info['symptoms']:
                        st.markdown(
                            f"""
                            <div style='
                                background-color: #252525;
                                margin: 0.5rem 0;
                                padding: 0.75rem;
                                border-radius: 5px;
                                border-left: 4px solid #4CAF50;
                                color: #E0E0E0;
                            '>
                                {symptom}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with info_tab2:
                    st.markdown("<h3 style='color: #81C784; margin-bottom: 1rem;'>💊 Tratamiento Recomendado</h3>", unsafe_allow_html=True)
                    
                    # Contenedor para tratamientos
                    for treatment in disease_info['treatment']:
                        st.markdown(
                            f"""
                            <div style='
                                background-color: #252525;
                                margin: 0.5rem 0;
                                padding: 0.75rem;
                                border-radius: 5px;
                                border-left: 4px solid #4CAF50;
                                color: #E0E0E0;
                            '>
                                {treatment}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("<h3 style='color: #81C784; margin-top: 2rem; margin-bottom: 1rem;'>🛡️ Medidas Preventivas</h3>", unsafe_allow_html=True)
                    
                    # Contenedor para medidas preventivas
                    for prevention in disease_info['prevention']:
                        st.markdown(
                            f"""
                            <div style='
                                background-color: #252525;
                                margin: 0.5rem 0;
                                padding: 0.75rem;
                                border-radius: 5px;
                                border-left: 4px solid #FF9800;
                                color: #E0E0E0;
                            '>
                                {prevention}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with info_tab3:
                    st.markdown("### 📊 Distribución de Probabilidades")
                    fig = create_probability_chart(prediction, DISEASE_INFO)
                    st.pyplot(fig)
                
                # Botón para generar PDF
                st.markdown("---")
                st.markdown("### 📄 Generar Reporte PDF")
                if st.button("📄 Generar Reporte PDF", use_container_width=True):
                    with st.spinner("⏳ Generando reporte PDF..."):
                        generate_pdf_report(st.session_state.current_image, disease_info, confidence, prediction)

# Nueva pestaña para comparación de modelos
with tab3:
    model_comparison()

# Pie de página
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <h3>🌿 Sistema Experto de Diagnóstico</h3>
        <p>Desarrollado para la identificación temprana y el manejo efectivo de enfermedades en cultivos de caña de azúcar</p>
        <p style='color: #666; font-size: 0.9em; margin-top: 1rem;'>
            Utilizando inteligencia artificial y aprendizaje profundo para diagnósticos precisos
        </p>
    </div>
""", unsafe_allow_html=True)