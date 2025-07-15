"""
Configuración y constantes de la aplicación
"""
import os
import tempfile
import streamlit as st

# Configuración de la aplicación
APP_CONFIG = {
    "title": "Sistema Experto de Diagnóstico de Enfermedades en Caña de Azúcar",
    "page_title": "Diagnóstico Caña de Azúcar",
    "page_icon": "🌿",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

# Configuración de TensorFlow
TENSORFLOW_CONFIG = {
    "tf_cpp_min_log_level": "3",
    "tf_warnings_filter": [
        {'category': 'UserWarning', 'module': 'keras'},
        {'category': 'FutureWarning', 'module': 'tensorflow'},
        {'message': '.*oneDNN.*'}
    ]
}

# Configuración de directorios
DIRECTORIES = {
    "models": "models",
    "info": "info",
    "temp": os.path.join(tempfile.gettempdir(), 'sugarcane_diagnosis')
}

# Configuración de modelos
MODEL_CONFIG = {
    "default_models": [
        "best_sugarcane_modelV1.keras",
        "best_sugarcane_modelV2.keras",
        "best_sugarcane_modelV3.keras"
    ],
    "supported_formats": ['.keras', '.h5'],
    "image_size": (256, 256),
    "batch_size": 1,
    "compile_config": {
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"]
    }
}

# Configuración de procesamiento de imágenes
IMAGE_CONFIG = {
    "input_size": (256, 256),
    "supported_formats": ['jpg', 'jpeg', 'png'],
    "clahe_config": {
        "clip_limit": 3.0,
        "tile_grid_size": (8, 8)
    },
    "normalization_range": (0, 1)
}

# Configuración de PDF
PDF_CONFIG = {
    "output_filename": "reporte_diagnostico_cana.pdf",
    "page_size": "letter",
    "font_sizes": {
        "title": 24,
        "header": 14,
        "normal": 11,
        "small": 9
    },
    "colors": {
        "primary": "#2E7D32",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336"
    }
}

# Configuración de visualización
VISUALIZATION_CONFIG = {
    "chart_style": "dark_background",
    "figure_size": (12, 6),
    "dpi": 150,
    "chart_colors": ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
    "background_color": "#1E1E1E"
}

# Configuración de sesión
SESSION_CONFIG = {
    "model_loaded": False,
    "model": None,
    "diagnosis_results": None,
    "current_image": None,
    "selected_model_file": None
}

# Configuración de límites
LIMITS_CONFIG = {
    "max_file_size_mb": 500,
    "max_models_comparison": 3,
    "max_probability_display": 5
}

# Configuración de UI
UI_CONFIG = {
    "tabs": {
        "config": "📤 Configuración",
        "diagnosis": "🔍 Diagnóstico",
        "comparison": "📊 Comparar Modelos"
    },
    "theme": {
        "background_color": "#0E1117",
        "text_color": "#E0E0E0",
        "primary_color": "#2E7D32",
        "secondary_color": "#4CAF50"
    }
}

def get_model_dir():
    """Obtener directorio de modelos"""
    return DIRECTORIES["models"]

def get_info_dir():
    """Obtener directorio de información"""
    return DIRECTORIES["info"]

def get_temp_dir():
    """Obtener directorio temporal de la aplicación"""
    temp_dir = DIRECTORIES["temp"]
    os.makedirs(temp_dir, exist_ok=True, mode=0o777)
    return temp_dir

def ensure_directories():
    """Crear directorios necesarios si no existen"""
    # Crear directorios de la aplicación
    for key, directory in DIRECTORIES.items():
        if key == "temp":
            # Manejar directorio temporal de forma especial
            os.makedirs(directory, exist_ok=True, mode=0o777)
        else:
            os.makedirs(directory, exist_ok=True)
    
    # Verificar permisos del directorio temporal
    temp_dir = DIRECTORIES["temp"]
    try:
        # Intentar crear un archivo de prueba
        test_file = os.path.join(temp_dir, 'test.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except (IOError, OSError) as e:
        st.error(f"❌ Error de permisos en directorio temporal: {str(e)}")
        st.error(f"Directorio temporal actual: {temp_dir}")
        # Intentar usar un directorio temporal alternativo
        alt_temp_dir = os.path.join(os.path.expanduser('~'), '.sugarcane_temp')
        st.warning(f"Intentando usar directorio temporal alternativo: {alt_temp_dir}")
        os.makedirs(alt_temp_dir, exist_ok=True, mode=0o777)
        DIRECTORIES["temp"] = alt_temp_dir

def initialize_session_state():
    """Inicializar estado de sesión de Streamlit"""
    # Asegurar que el directorio temporal existe y es accesible
    temp_dir = get_temp_dir()
    
    # Inicializar variables de sesión
    for key, value in SESSION_CONFIG.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Añadir directorio temporal al estado de la sesión
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = temp_dir

def get_available_models():
    """Obtener lista de modelos disponibles"""
    model_dir = get_model_dir()
    if os.path.exists(model_dir):
        available = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        return available if available else MODEL_CONFIG["default_models"]
    return MODEL_CONFIG["default_models"] 