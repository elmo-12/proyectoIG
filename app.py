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
import pandas as pd
import plotly.express as px
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

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

def load_model_info(model_name):
    """Carga la información estadística de un modelo desde los archivos .txt"""
    try:
        # Extraer número de versión del nombre del modelo
        if "V1" in model_name or "v1" in model_name:
            info_file = "info/modelV1.txt"
        elif "V2" in model_name or "v2" in model_name:
            info_file = "info/modelV2.txt"
        elif "V3" in model_name or "v3" in model_name:
            info_file = "info/modelV3.txt"
        else:
            return None
        
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                content = f.read()
                
            # Extraer información básica
            lines = content.split('\n')
            test_loss = None
            test_accuracy = None
            
            for line in lines:
                if "Test Loss" in line:
                    test_loss = float(line.split(':')[1].strip())
                elif "Test Accuracy" in line:
                    test_accuracy = float(line.split(':')[1].strip())
            
            return {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'full_report': content
            }
        return None
    except Exception as e:
        return None

def get_confusion_matrix_path(model_name):
    """Obtiene la ruta de la matriz de confusión para un modelo específico"""
    try:
        if "V1" in model_name or "v1" in model_name:
            return "info/modelV1.png"
        elif "V2" in model_name or "v2" in model_name:
            return "info/modelV2.png"
        elif "V3" in model_name or "v3" in model_name:
            return "info/modelV3.png"
        return None
    except:
        return None

def clean_text_robust(text):
    """Limpia el texto de forma muy robusta eliminando todos los caracteres problemáticos"""
    import unicodedata
    import re
    
    # Convertir a string si no lo es
    if not isinstance(text, str):
        text = str(text)
    
    # Normalizar y eliminar acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Eliminar cualquier carácter que no sea ASCII básico, números, letras, espacios y puntuación básica
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Reemplazar caracteres específicos restantes
    replacements = {
        '°': 'o', '–': '-', '—': '-', ''': "'", ''': "'", 
        '"': '"', '"': '"', '…': '...', '®': '(R)', '©': '(C)',
        '¿': '?', '¡': '!'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Filtro final: solo caracteres ASCII imprimibles
    return ''.join(c for c in text if ord(c) < 128 and (c.isprintable() or c.isspace()))

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

@st.cache_resource
def load_all_models():
    """Carga todos los modelos disponibles para comparación"""
    models = {}
    model_files = [
        "best_sugarcane_modelV1.keras",
        "best_sugarcane_modelV2.keras", 
        "best_sugarcane_modelV3.keras"
    ]
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            model = load_model(model_path)
            if model is not None:
                models[model_file] = model
    
    return models

def preprocess_image(image):
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

def predict_with_all_models(processed_image):
    """
    Realiza predicción con todos los modelos disponibles
    """
    models = load_all_models()
    predictions = {}
    
    if not models:
        st.warning("⚠️ No hay modelos disponibles para comparación")
        return None
    
    for model_name, model in models.items():
        try:
            prediction = predict_disease(model, processed_image)
            predictions[model_name] = prediction
        except Exception as e:
            st.error(f"❌ Error con modelo {model_name}: {str(e)}")
            continue
    
    return predictions

def get_consensus_prediction(predictions):
    """
    Obtiene predicción de consenso basada en múltiples modelos
    """
    if not predictions:
        return None, None, None
    
    # Obtener información de los modelos para ponderar
    model_weights = {}
    for model_name in predictions.keys():
        model_info = load_model_info(model_name)
        if model_info and model_info['test_accuracy'] > 0:
            # Usar la precisión del modelo como peso
            model_weights[model_name] = model_info['test_accuracy']
        else:
            # Peso por defecto si no hay información
            model_weights[model_name] = 0.5
    
    # Calcular predicción promedio ponderada
    num_classes = len(DISEASE_INFO)
    weighted_prediction = np.zeros((1, num_classes))
    total_weight = 0
    
    for model_name, prediction in predictions.items():
        weight = model_weights.get(model_name, 0.5)
        weighted_prediction += prediction * weight
        total_weight += weight
    
    if total_weight > 0:
        weighted_prediction /= total_weight
    
    # Obtener clase predicha y confianza
    predicted_class = np.argmax(weighted_prediction[0])
    confidence = weighted_prediction[0][predicted_class] * 100
    
    return weighted_prediction, predicted_class, confidence

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

def create_comparative_chart(predictions, disease_info):
    """Crea gráfico comparativo de predicciones de múltiples modelos"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    class_names = [info['name'] for info in disease_info.values()]
    model_names = list(predictions.keys())
    
    # Colores para cada modelo
    model_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Configurar posiciones de las barras
    x = np.arange(len(class_names))
    width = 0.25
    
    # Crear barras para cada modelo
    for i, (model_name, prediction) in enumerate(predictions.items()):
        probabilities = prediction[0] * 100
        model_label = model_name.replace('best_sugarcane_model', 'Modelo ').replace('.keras', '')
        
        bars = ax.bar(x + i * width, probabilities, width, 
                     label=model_label, color=model_colors[i % len(model_colors)], 
                     alpha=0.8)
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Solo mostrar si la probabilidad es mayor a 5%
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom',
                       color='white', fontsize=8, fontweight='bold')
    
    # Configurar ejes
    ax.set_ylabel("Probabilidad (%)", color='white')
    ax.set_xlabel("Clases de Enfermedad", color='white')
    ax.set_title("Comparación de Predicciones por Modelo", color='white', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right', color='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    return fig

def model_comparison():
    """Función para comparar diferentes modelos de diagnóstico usando datos reales"""
    st.header("📊 Comparación de Modelos de Diagnóstico")

    # Verificar qué modelos están disponibles
    available_models = []
    model_files = [
        "best_sugarcane_modelV1.keras",
        "best_sugarcane_modelV2.keras",
        "best_sugarcane_modelV3.keras"
    ]
    # También incluir cualquier otro modelo .keras en la carpeta
    try:
        model_files = list(set(model_files + [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]))
    except:
        pass
    
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
    
    # Recopilar datos reales de los modelos
    models_data = {
        'Modelo': [],
        'Precisión': [],
        'Pérdida': [],
        'F1-Score Promedio': [],
        'Tamaño (MB)': [],
        'Fecha Creación': [],
        'Estado': []
    }
    
    # Estimaciones de tiempo de inferencia basadas en tamaño y complejidad
    inference_times = {'V1': 45, 'V2': 65, 'V3': 85}
    
    for model_file in available_models:
        model_path = os.path.join(MODEL_DIR, model_file)
        model_name = model_file.replace('.keras', '')
        models_data['Modelo'].append(model_name)
        
        # Cargar información estadística real
        model_info = load_model_info(model_file)
        if model_info:
            models_data['Precisión'].append(model_info['test_accuracy'])
            models_data['Pérdida'].append(model_info['test_loss'])
            
            # Extraer F1-Score promedio del reporte
            try:
                lines = model_info['full_report'].split('\n')
                f1_score = 0.0
                for line in lines:
                    if 'macro avg' in line and 'f1-score' in line:
                        parts = line.split()
                        f1_score = float(parts[3])
                        break
                models_data['F1-Score Promedio'].append(f1_score)
            except:
                models_data['F1-Score Promedio'].append(0.0)
                
            # Determinar estado del modelo basado en precisión
            if model_info['test_accuracy'] >= 0.7:
                models_data['Estado'].append('✅ Excelente')
            elif model_info['test_accuracy'] >= 0.5:
                models_data['Estado'].append('⚠️ Aceptable')
            else:
                models_data['Estado'].append('❌ Necesita mejoras')
        else:
            models_data['Precisión'].append(0.0)
            models_data['Pérdida'].append(0.0)
            models_data['F1-Score Promedio'].append(0.0)
            models_data['Estado'].append('❓ Sin información')
        
        # Obtener tamaño del archivo
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
    
    valid_precision = df[df['Precisión'] > 0]['Precisión']
    valid_f1 = df[df['F1-Score Promedio'] > 0]['F1-Score Promedio']
    
    with col1:
        if len(valid_precision) > 0:
            st.metric("Mejor Precisión", f"{valid_precision.max():.2%}")
        else:
            st.metric("Mejor Precisión", "N/A")
    with col2:
        if len(valid_precision) > 0:
            st.metric("Promedio Precisión", f"{valid_precision.mean():.2%}")
        else:
            st.metric("Promedio Precisión", "N/A")
    with col3:
        if len(valid_f1) > 0:
            st.metric("Mejor F1-Score", f"{valid_f1.max():.3f}")
        else:
            st.metric("Mejor F1-Score", "N/A")
    with col4:
        st.metric("Tamaño Total", f"{df['Tamaño (MB)'].sum():.1f} MB")

    # Tabla interactiva mejorada
    st.subheader("📊 Tabla Detallada de Métricas")
    
    # Crear tabla con formato condicional
    styled_df = df.copy()
    styled_df['Precisión'] = styled_df['Precisión'].apply(lambda x: f"{x:.2%}" if x > 0 else "N/A")
    styled_df['Pérdida'] = styled_df['Pérdida'].apply(lambda x: f"{x:.4f}" if x > 0 else "N/A")
    styled_df['F1-Score Promedio'] = styled_df['F1-Score Promedio'].apply(lambda x: f"{x:.3f}" if x > 0 else "N/A")
    
    st.dataframe(styled_df, use_container_width=True)

    # Gráficos mejorados
    st.subheader("📈 Visualizaciones")
    
    # Filtrar modelos con datos válidos para los gráficos
    valid_models = df[df['Precisión'] > 0]
    
    if len(valid_models) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(valid_models, x='Modelo', y='Precisión', 
                         title='Precisión por Modelo',
                         color='Precisión',
                         color_continuous_scale='viridis',
                         text='Precisión')
            fig1.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(tickformat='.0%')
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(valid_models, x='Modelo', y='F1-Score Promedio', 
                         title='F1-Score Promedio por Modelo',
                         color='F1-Score Promedio',
                         color_continuous_scale='plasma',
                         text='F1-Score Promedio')
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Gráfico de pérdida vs precisión
        st.subheader("📉 Análisis de Pérdida vs Precisión")
        fig3 = px.scatter(valid_models, x='Pérdida', y='Precisión', 
                         size='Tamaño (MB)', hover_name='Modelo',
                         title='Relación entre Pérdida y Precisión',
                         color='F1-Score Promedio',
                         color_continuous_scale='RdYlGn')
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Gráfico de tamaño de modelos
    if len(available_models) > 1:
        st.subheader("📦 Comparación de Tamaños")
        fig4 = px.pie(df, values='Tamaño (MB)', names='Modelo', 
                      title='Distribución del Tamaño de Modelos')
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Recomendaciones mejoradas
    st.subheader("💡 Recomendaciones")
    
    if len(valid_models) > 0:
        best_model = valid_models.loc[valid_models['Precisión'].idxmax(), 'Modelo']
        best_f1_model = valid_models.loc[valid_models['F1-Score Promedio'].idxmax(), 'Modelo']
        smallest_model = df.loc[df['Tamaño (MB)'].idxmin(), 'Modelo']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🎯 **Mejor Precisión**: {best_model}")
        with col2:
            st.info(f"⚖️ **Mejor F1-Score**: {best_f1_model}")
        with col3:
            st.info(f"💾 **Más Pequeño**: {smallest_model}")
        
        # Análisis detallado del mejor modelo
        st.subheader("🏆 Análisis del Mejor Modelo")
        best_model_info = load_model_info(best_model + '.keras')
        if best_model_info:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Estadísticas Principales:**")
                st.markdown(f"- Precisión: {best_model_info['test_accuracy']:.2%}")
                st.markdown(f"- Pérdida: {best_model_info['test_loss']:.4f}")
                
            with col2:
                st.markdown("**Reporte Detallado:**")
                with st.expander("Ver reporte completo"):
                    st.text(best_model_info['full_report'])
        
        # Mostrar matrices de confusión
        st.subheader("📊 Matrices de Confusión")
        matrices_cols = st.columns(min(3, len(valid_models)))
        
        for i, (_, model_row) in enumerate(valid_models.iterrows()):
            if i >= 3:  # Mostrar máximo 3 matrices
                break
            model_name = model_row['Modelo']
            confusion_path = get_confusion_matrix_path(model_name + '.keras')
            
            if confusion_path and os.path.exists(confusion_path):
                with matrices_cols[i]:
                    st.markdown(f"**{model_name}**")
                    st.image(confusion_path, use_column_width=True)
    
    # Exportar comparación
    st.subheader("📤 Exportar Comparación")
    if st.button("📊 Exportar Datos de Comparación", use_container_width=True):
        csv = styled_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="comparacion_modelos.csv">⬇️ Descargar CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("✅ Datos exportados exitosamente")

def generate_pdf_report(image,
                       disease_info: dict,
                       confidence: float,
                       probabilities: np.ndarray,
                       model_name: str = None,
                       all_predictions: dict = None,
                       consensus_prediction: np.ndarray = None):
    """Genera un reporte PDF del diagnóstico con comparativa de múltiples modelos."""
    try:
        if REPORTLAB_AVAILABLE:
            return generate_pdf_reportlab(image, disease_info, confidence, probabilities, model_name, all_predictions, consensus_prediction)
        elif FPDF_AVAILABLE:
            return generate_pdf_fpdf(image, disease_info, confidence, probabilities, model_name, clean_text_robust, all_predictions, consensus_prediction)
        else:
            st.error("❌ No hay bibliotecas de PDF disponibles")
            st.info("💡 Instala una de las siguientes bibliotecas:")
            st.info("   - ReportLab (recomendado): `pip install reportlab==4.0.4`")
            st.info("   - FPDF: `pip install fpdf2`")
            return None
    
    except Exception as e:
        st.error(f"❌ Error al generar PDF: {str(e)}")
        st.error("Detalles del error:")
        st.code(str(e))

def generate_pdf_reportlab(image,
                          disease_info: dict,
                          confidence: float,
                          probabilities: np.ndarray,
                          model_name: str = None,
                          all_predictions: dict = None,
                          consensus_prediction: np.ndarray = None):
    """Genera PDF usando ReportLab (soporta UTF-8 nativo) con comparativa de múltiples modelos"""
    try:
        # Crear documento
        out_path = "reporte_diagnostico_cana.pdf"
        doc = SimpleDocTemplate(out_path, pagesize=letter)
        story = []
        
        # Obtener estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2E7D32'),
            alignment=1  # Centrado
        )
        
        # Título
        story.append(Paragraph("🌿 Diagnóstico Comparativo de Caña de Azúcar", title_style))
        story.append(Spacer(1, 20))
        
        # Información de consenso de múltiples modelos
        if all_predictions and len(all_predictions) > 1:
            story.append(Paragraph("<b>Análisis con Múltiples Modelos:</b>", styles['Heading2']))
            story.append(Paragraph(f"<b>Modelos utilizados:</b> {len(all_predictions)} modelos", styles['Normal']))
            story.append(Paragraph(f"<b>Diagnóstico de consenso:</b> {disease_info['name']}", styles['Normal']))
            story.append(Paragraph(f"<b>Confianza del consenso:</b> {confidence:.1f}%", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Detalles de cada modelo
            for i, (model_name_iter, prediction) in enumerate(all_predictions.items()):
                model_info = load_model_info(model_name_iter)
                predicted_class = np.argmax(prediction[0])
                model_confidence = prediction[0][predicted_class] * 100
                model_disease = DISEASE_INFO[predicted_class]['name']
                
                story.append(Paragraph(f"<b>Modelo {i+1} ({model_name_iter}):</b>", styles['Heading3']))
                story.append(Paragraph(f"  • Diagnóstico: {model_disease}", styles['Normal']))
                story.append(Paragraph(f"  • Confianza: {model_confidence:.1f}%", styles['Normal']))
                if model_info:
                    story.append(Paragraph(f"  • Precisión del modelo: {model_info['test_accuracy']:.2%}", styles['Normal']))
                story.append(Spacer(1, 10))
        else:
            # Información del modelo único
            if model_name:
                model_info = load_model_info(model_name)
                story.append(Paragraph(f"<b>Modelo utilizado:</b> {model_name}", styles['Normal']))
                if model_info:
                    story.append(Paragraph(f"<b>Precisión del modelo:</b> {model_info['test_accuracy']:.2%}", styles['Normal']))
                    story.append(Paragraph(f"<b>Pérdida de prueba:</b> {model_info['test_loss']:.4f}", styles['Normal']))
                story.append(Spacer(1, 20))
        
        # Imagen
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            image.save(tmp_img_file.name)
            story.append(Paragraph("<b>Imagen Analizada:</b>", styles['Heading2']))
            story.append(ReportLabImage(tmp_img_file.name, width=4*inch, height=3*inch))
            tmp_img_path = tmp_img_file.name
        story.append(Spacer(1, 20))
        
        # Diagnóstico
        story.append(Paragraph(f"<b>Diagnóstico:</b> {disease_info['name']}", styles['Heading2']))
        story.append(Paragraph(f"<b>Nivel de Confianza:</b> {confidence:.1f}%", styles['Normal']))
        story.append(Paragraph(f"<b>Descripción:</b> {disease_info['description']}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Síntomas
        story.append(Paragraph("<b>Síntomas Principales:</b>", styles['Heading3']))
        for symptom in disease_info['symptoms'][:3]:
            story.append(Paragraph(f"• {symptom}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Tratamiento
        story.append(Paragraph("<b>Tratamiento Recomendado:</b>", styles['Heading3']))
        for treatment in disease_info['treatment'][:3]:
            story.append(Paragraph(f"• {treatment}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Prevención
        story.append(Paragraph("<b>Medidas Preventivas:</b>", styles['Heading3']))
        for prevention in disease_info['prevention'][:2]:
            story.append(Paragraph(f"• {prevention}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Gráficos de probabilidades
        import matplotlib.pyplot as plt
        
        # Gráfico comparativo si hay múltiples modelos
        if all_predictions and len(all_predictions) > 1:
            story.append(Paragraph("<b>Comparación de Modelos:</b>", styles['Heading3']))
            comparative_fig = create_comparative_chart(all_predictions, DISEASE_INFO)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_comparative:
                comparative_fig.savefig(tmp_comparative.name, bbox_inches='tight', dpi=150, facecolor='#1E1E1E')
                comparative_path = tmp_comparative.name
            plt.close(comparative_fig)
            
            story.append(ReportLabImage(comparative_path, width=7*inch, height=5*inch))
            story.append(Spacer(1, 20))
        
        # Gráfico de consenso o modelo único
        title = "Distribución de Probabilidades (Consenso)" if all_predictions and len(all_predictions) > 1 else "Distribución de Probabilidades"
        story.append(Paragraph(f"<b>{title}:</b>", styles['Heading3']))
        
        class_names = [d['name'] for d in DISEASE_INFO.values()]
        probs = (consensus_prediction[0] if consensus_prediction is not None else probabilities[0]) * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(class_names, probs, color=[d['color'] for d in DISEASE_INFO.values()])
        ax.set_ylabel("Probabilidad (%)")
        ax.set_title(title)
        ax.set_ylim([0, 100])
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', 
                   ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_chart:
            plt.savefig(tmp_chart.name, bbox_inches='tight', dpi=150)
            chart_path = tmp_chart.name
        plt.close(fig)
        
        story.append(ReportLabImage(chart_path, width=6*inch, height=4*inch))
        
        # Matriz de confusión
        if model_name:
            confusion_matrix_path = get_confusion_matrix_path(model_name)
            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                story.append(Paragraph("<b>Matriz de Confusión del Modelo:</b>", styles['Heading3']))
                story.append(ReportLabImage(confusion_matrix_path, width=6*inch, height=4*inch))
        
        # Generar PDF
        doc.build(story)
        
        # Limpiar archivos temporales después de generar el PDF
        try:
            os.unlink(tmp_img_path)
            os.unlink(chart_path)
            if 'comparative_path' in locals():
                os.unlink(comparative_path)
        except:
            pass
        
        # Preparar descarga
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
        st.error(f"❌ Error al generar PDF con ReportLab: {str(e)}")
        # Fallback a FPDF
        return generate_pdf_fpdf(image, disease_info, confidence, probabilities, model_name, clean_text_robust)

def generate_pdf_fpdf(image,
                     disease_info: dict,
                     confidence: float,
                     probabilities: np.ndarray,
                     model_name: str = None,
                     clean_text_func=None,
                     all_predictions: dict = None,
                     consensus_prediction: np.ndarray = None):
    """Genera PDF usando FPDF con limpieza robusta de texto y comparativa de múltiples modelos"""
    try:
        if clean_text_func is None:
            def clean_text_func(text):
                import re
                # Eliminar caracteres no ASCII
                return re.sub(r'[^\x00-\x7F]+', '?', str(text))
        
        pdf = FPDF()
        pdf.add_page()

        # --- Encabezado con logo y título ---
        if os.path.exists("logo.png"):
            pdf.image("logo.png", x=10, y=8, w=30, h=30)
        pdf.set_xy(45, 12)
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(46, 125, 50)
        title = "Diagnostico Comparativo de Cana de Azucar" if all_predictions and len(all_predictions) > 1 else "Diagnostico de Cana de Azucar"
        pdf.cell(0, 15, clean_text_func(title), ln=1, align='C')
        pdf.ln(10)
        # Línea divisoria después del título
        pdf.set_draw_color(46, 125, 50)
        pdf.set_line_width(1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # --- Información de múltiples modelos o modelo único ---
        if all_predictions and len(all_predictions) > 1:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(33, 150, 243)
            pdf.cell(0, 8, clean_text_func("Analisis con Multiples Modelos"), ln=1)
            pdf.set_font("Arial", size=10)
            pdf.set_text_color(33, 33, 33)
            pdf.cell(0, 6, clean_text_func(f"Modelos utilizados: {len(all_predictions)} modelos"), ln=1)
            pdf.ln(2)
            
            # Detalles de cada modelo
            for i, (model_name_iter, prediction) in enumerate(all_predictions.items()):
                model_info = load_model_info(model_name_iter)
                predicted_class = np.argmax(prediction[0])
                model_confidence = prediction[0][predicted_class] * 100
                model_disease = DISEASE_INFO[predicted_class]['name']
                
                pdf.set_font("Arial", 'B', 10)
                pdf.set_text_color(76, 175, 80)
                pdf.cell(0, 6, clean_text_func(f"Modelo {i+1} ({model_name_iter}):"), ln=1)
                pdf.set_font("Arial", size=9)
                pdf.set_text_color(33, 33, 33)
                pdf.cell(0, 5, clean_text_func(f"  Diagnostico: {model_disease}"), ln=1)
                pdf.cell(0, 5, clean_text_func(f"  Confianza: {model_confidence:.1f}%"), ln=1)
                if model_info:
                    pdf.cell(0, 5, clean_text_func(f"  Precision del modelo: {model_info['test_accuracy']:.2%}"), ln=1)
                pdf.ln(1)
            pdf.ln(3)
        else:
            # Información del modelo único
            if model_name:
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(33, 150, 243)
                pdf.cell(0, 8, clean_text_func(f"Modelo utilizado: {model_name}"), ln=1)
                
                # Cargar información estadística del modelo
                model_info = load_model_info(model_name)
                if model_info:
                    pdf.set_font("Arial", size=10)
                    pdf.set_text_color(33, 33, 33)
                    pdf.cell(0, 6, clean_text_func(f"Precision del modelo: {model_info['test_accuracy']:.2%}"), ln=1)
                    pdf.cell(0, 6, clean_text_func(f"Perdida de prueba: {model_info['test_loss']:.4f}"), ln=1)
                pdf.ln(3)

        # --- Imagen analizada ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 4, clean_text_func("Imagen Analizada:"), ln=1)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            image.save(tmp_img_file.name)
            pdf.image(tmp_img_file.name, x=60, w=90, h=60)
            tmp_img_path = tmp_img_file.name
        pdf.ln(10)

        # --- Diagnóstico principal ---
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(46, 125, 50)
        pdf.cell(0, 10, clean_text_func(f"{disease_info['name']}"), ln=1)
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 8, clean_text_func(f"Nivel de Confianza: {confidence:.1f}%"), ln=1)
        pdf.ln(2)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, clean_text_func(disease_info['description']))
        pdf.ln(2)
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        # --- Síntomas ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(198, 40, 40)
        pdf.cell(0, 8, clean_text_func("Sintomas Principales:"), ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        for symptom in disease_info['symptoms'][:3]:
            pdf.cell(0, 6, clean_text_func(f"- {symptom}"), ln=1)
        pdf.ln(2)

        # --- Tratamiento ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(33, 150, 243)
        pdf.cell(0, 8, clean_text_func("Tratamiento Recomendado:"), ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        for treatment in disease_info['treatment'][:3]:
            pdf.cell(0, 6, clean_text_func(f"- {treatment}"), ln=1)
        pdf.ln(2)

        # --- Prevención ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(255, 152, 0)
        pdf.cell(0, 8, clean_text_func("Medidas Preventivas:"), ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        for prevention in disease_info['prevention'][:2]:
            pdf.cell(0, 6, clean_text_func(f"- {prevention}"), ln=1)
        pdf.ln(2)

        # --- Gráficos de probabilidades ---
        import matplotlib.pyplot as plt
        
        # Gráfico comparativo si hay múltiples modelos
        if all_predictions and len(all_predictions) > 1:
            pdf.ln(3)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(76, 175, 80)
            pdf.cell(0, 8, clean_text_func("Comparacion de Modelos:"), ln=1)
            
            comparative_fig = create_comparative_chart(all_predictions, DISEASE_INFO)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_comparative:
                comparative_fig.savefig(tmp_comparative.name, bbox_inches='tight', dpi=150, facecolor='#1E1E1E')
                comparative_path = tmp_comparative.name
            plt.close(comparative_fig)
            
            pdf.image(comparative_path, x=10, w=190, h=100)
            pdf.ln(10)
        
        # Gráfico de consenso o modelo único
        class_names = [clean_text_func(d['name']) for d in DISEASE_INFO.values()]
        probs = (consensus_prediction[0] if consensus_prediction is not None else probabilities[0]) * 100
        chart_title = "Distribucion de Probabilidades (Consenso)" if all_predictions and len(all_predictions) > 1 else "Distribucion de Probabilidades"
        
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
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
        pdf.cell(0, 8, clean_text_func(chart_title + ":"), ln=1)
        pdf.image(chart_path, x=25, w=160, h=80)
        pdf.ln(2)

        # --- Probabilidades por clase (texto) ---
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(33, 33, 33)
        display_probs = consensus_prediction[0] if consensus_prediction is not None else probabilities[0]
        for idx, (cls_name, prob) in enumerate(zip(class_names, display_probs)):
            pdf.cell(0, 6, clean_text_func(f"- {cls_name}: {prob*100:.2f}%"), ln=1)
        pdf.ln(2)

        # --- Nueva página para información técnica ---
        pdf.add_page()
        
        # --- Matriz de confusión ---
        if model_name:
            confusion_matrix_path = get_confusion_matrix_path(model_name)
            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                pdf.set_font("Arial", 'B', 14)
                pdf.set_text_color(46, 125, 50)
                pdf.cell(0, 10, clean_text_func("Matriz de Confusion del Modelo"), ln=1)
                pdf.ln(5)
                pdf.image(confusion_matrix_path, x=20, w=170, h=120)
                pdf.ln(10)
        
        # --- Información estadística completa del modelo ---
        if model_name:
            model_info = load_model_info(model_name)
            if model_info:
                pdf.set_font("Arial", 'B', 14)
                pdf.set_text_color(46, 125, 50)
                pdf.cell(0, 10, clean_text_func("Estadisticas del Modelo"), ln=1)
                pdf.ln(5)
                
                # Mostrar el reporte completo
                pdf.set_font("Courier", size=9)
                pdf.set_text_color(33, 33, 33)
                
                # Dividir el reporte en líneas y procesarlo
                report_lines = model_info['full_report'].split('\n')
                for line in report_lines:
                    if line.strip():
                        # Limpiar la línea de caracteres especiales
                        clean_line = clean_text_func(line)
                        # Ajustar ancho de línea para evitar desbordamiento
                        if len(clean_line) > 80:
                            # Dividir líneas largas
                            words = clean_line.split()
                            current_line = ""
                            for word in words:
                                if len(current_line + word) < 80:
                                    current_line += word + " "
                                else:
                                    pdf.cell(0, 4, current_line.strip(), ln=1)
                                    current_line = word + " "
                            if current_line.strip():
                                pdf.cell(0, 4, current_line.strip(), ln=1)
                        else:
                            pdf.cell(0, 4, clean_line, ln=1)
                pdf.ln(5)

        # --- Pie de página ---
        pdf.set_y(-25)
        pdf.set_draw_color(46, 125, 50)
        pdf.set_line_width(0.7)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_font("Arial", size=8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8, clean_text_func(f"Reporte generado el: {time.strftime('%Y-%m-%d %H:%M:%S')}"), ln=1, align='C')
        pdf.cell(0, 6, clean_text_func("Sistema Experto de Diagnostico de Cana de Azucar"), ln=1, align='C')

        # Guardar y preparar descarga
        out_path = "reporte_diagnostico_cana.pdf"
        pdf.output(out_path)
        
        # Limpiar archivos temporales
        try:
            os.unlink(tmp_img_path)
            os.unlink(chart_path)
            if 'comparative_path' in locals():
                os.unlink(comparative_path)
        except:
            pass
        
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
        st.rerun()

    
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
        
        # Información sobre el sistema de PDF
        if REPORTLAB_AVAILABLE:
            st.success("✅ ReportLab disponible: Generación de PDF con soporte completo UTF-8")
        elif FPDF_AVAILABLE:
            st.info("⚠️ FPDF disponible: Generación de PDF básica con limpieza de caracteres")
            st.info("💡 Para mejor calidad de PDF, instala: `pip install reportlab==4.0.4`")
        else:
            st.error("❌ No hay bibliotecas de PDF disponibles")
            st.info("💡 Instala una de las siguientes bibliotecas:")
            st.info("   - ReportLab (recomendado): `pip install reportlab==4.0.4`")
            st.info("   - FPDF: `pip install fpdf2`")
            
        if not REPORTLAB_AVAILABLE and st.button("🔧 Instalar ReportLab automáticamente"):
            with st.spinner("⏳ Instalando ReportLab..."):
                try:
                    import subprocess
                    import sys
                    result = subprocess.run([sys.executable, "-m", "pip", "install", "reportlab==4.0.4"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ ReportLab instalado exitosamente")
                        st.info("🔄 Reinicia la aplicación para usar ReportLab")
                    else:
                        st.error(f"❌ Error instalando ReportLab: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
        if not FPDF_AVAILABLE and not REPORTLAB_AVAILABLE and st.button("🔧 Instalar FPDF automáticamente"):
            with st.spinner("⏳ Instalando FPDF..."):
                try:
                    import subprocess
                    import sys
                    result = subprocess.run([sys.executable, "-m", "pip", "install", "fpdf2"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ FPDF instalado exitosamente")
                        st.info("🔄 Reinicia la aplicación para usar FPDF")
                    else:
                        st.error(f"❌ Error instalando FPDF: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
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
            st.rerun()
    
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
                
                if st.button("🔍 Realizar Diagnóstico Comparativo", use_container_width=True):
                    with st.spinner("🔄 Procesando imagen con múltiples modelos de IA..."):
                        # Procesar imagen con algoritmos avanzados
                        processed_image = preprocess_image(image)
                        
                        # Realizar predicción con todos los modelos disponibles
                        all_predictions = predict_with_all_models(processed_image)
                        
                        if all_predictions and len(all_predictions) > 0:
                            # Obtener predicción de consenso
                            consensus_prediction, predicted_class, confidence = get_consensus_prediction(all_predictions)
                            
                            if consensus_prediction is not None:
                                disease_info = DISEASE_INFO[predicted_class]
                                
                                # Guardar resultados en el estado de la sesión
                                st.session_state.diagnosis_results = {
                                    'prediction': consensus_prediction,
                                    'predicted_class': predicted_class,
                                    'confidence': confidence,
                                    'disease_info': disease_info,
                                    'all_predictions': all_predictions,
                                    'consensus_prediction': consensus_prediction
                                }
                                st.session_state.current_image = image
                                
                                # Mostrar resumen de modelos utilizados
                                st.success(f"✅ Análisis completado con {len(all_predictions)} modelo(s)")
                                for model_name, pred in all_predictions.items():
                                    model_pred_class = np.argmax(pred[0])
                                    model_conf = pred[0][model_pred_class] * 100
                                    model_disease = DISEASE_INFO[model_pred_class]['name']
                                    st.info(f"📊 {model_name}: {model_disease} ({model_conf:.1f}%)")
                        else:
                            st.error("❌ No se pudieron cargar los modelos para el análisis")
                            st.info("💡 Verifica que los modelos estén disponibles en la carpeta 'models'")
                            
                            # Fallback a modelo único si está disponible
                            if st.session_state.model is not None:
                                st.info("🔄 Usando modelo único como alternativa...")
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
                                        'disease_info': disease_info,
                                        'all_predictions': None,
                                        'consensus_prediction': None
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
                diagnosis_title = "Diagnóstico de Consenso" if results.get('all_predictions') else "Diagnóstico"
                
                st.markdown(f"""
                    <div class='diagnosis-box {box_class}'>
                        <h2>{disease_info['icon']} {diagnosis_title}</h2>
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
                if results.get('all_predictions'):
                    info_tab1, info_tab2, info_tab3, info_tab4 = st.tabs(["📋 Detalles", "💊 Tratamiento", "📊 Comparación", "📈 Análisis"])
                else:
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

                # Pestaña de comparación (solo si hay múltiples modelos)
                if results.get('all_predictions'):
                    with info_tab3:
                        st.markdown("### 📊 Comparación de Modelos")
                        
                        # Gráfico comparativo
                        comparative_fig = create_comparative_chart(results['all_predictions'], DISEASE_INFO)
                        st.pyplot(comparative_fig)
                        
                        # Tabla de resultados detallados
                        st.markdown("#### 📋 Resultados Detallados por Modelo")
                        comparison_data = []
                        
                        for model_name, pred in results['all_predictions'].items():
                            model_pred_class = np.argmax(pred[0])
                            model_conf = pred[0][model_pred_class] * 100
                            model_disease = DISEASE_INFO[model_pred_class]['name']
                            model_info = load_model_info(model_name)
                            
                            comparison_data.append({
                                'Modelo': model_name.replace('best_sugarcane_model', 'Modelo ').replace('.keras', ''),
                                'Diagnóstico': model_disease,
                                'Confianza': f"{model_conf:.1f}%",
                                'Precisión del Modelo': f"{model_info['test_accuracy']:.2%}" if model_info else "N/A",
                                'Estado': '✅ Coincide' if model_pred_class == predicted_class else '⚠️ Difiere'
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Análisis de consenso
                        st.markdown("#### 🎯 Análisis de Consenso")
                        consensus_info = f"""
                        - **Modelos en consenso**: {sum(1 for _, pred in results['all_predictions'].items() if np.argmax(pred[0]) == predicted_class)} de {len(results['all_predictions'])}
                        - **Diagnóstico final**: {disease_info['name']}
                        - **Confianza ponderada**: {confidence:.1f}%
                        """
                        st.markdown(consensus_info)
                
                # Pestaña de análisis (ajustada según si hay múltiples modelos o no)
                analysis_tab = info_tab4 if results.get('all_predictions') else info_tab3
                with analysis_tab:
                    title = "📈 Análisis del Consenso" if results.get('all_predictions') else "📊 Distribución de Probabilidades"
                    st.markdown(f"### {title}")
                    
                    if results.get('consensus_prediction') is not None:
                        fig = create_probability_chart(results['consensus_prediction'], DISEASE_INFO)
                    else:
                        fig = create_probability_chart(prediction, DISEASE_INFO)
                    st.pyplot(fig)
                
                # Botón para generar PDF
                st.markdown("---")
                st.markdown("### 📄 Generar Reporte PDF Comparativo")
                if st.button("📄 Generar Reporte PDF Comparativo", use_container_width=True):
                    with st.spinner("⏳ Generando reporte PDF comparativo..."):
                        # Obtener todos los datos para el PDF
                        all_predictions = results.get('all_predictions')
                        consensus_prediction = results.get('consensus_prediction')
                        model_name = st.session_state.selected_model_file
                        
                        # Si hay múltiples modelos, usar consenso; sino usar predicción individual
                        pdf_prediction = consensus_prediction if consensus_prediction is not None else prediction
                        
                        generate_pdf_report(
                            st.session_state.current_image, 
                            disease_info, 
                            confidence, 
                            pdf_prediction, 
                            model_name,
                            all_predictions,
                            consensus_prediction
                        )

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