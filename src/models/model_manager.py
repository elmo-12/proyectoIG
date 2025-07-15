"""
Módulo para manejo de modelos de machine learning
"""
import os
import warnings
import tensorflow as tf
import streamlit as st
from typing import Dict, Optional, List, Any
import numpy as np

from ..config.settings import (
    MODEL_CONFIG, 
    get_model_dir, 
    get_info_dir,
    TENSORFLOW_CONFIG
)

class ModelManager:
    """Clase para gestionar modelos de machine learning"""
    
    def __init__(self):
        self._configure_tensorflow()
        self._models_cache = {}
        
    def _configure_tensorflow(self):
        """Configurar TensorFlow para evitar warnings"""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = TENSORFLOW_CONFIG['tf_cpp_min_log_level']
        tf.get_logger().setLevel('ERROR')
        
        # Suprimir warnings específicos
        for filter_config in TENSORFLOW_CONFIG['tf_warnings_filter']:
            if 'message' in filter_config:
                warnings.filterwarnings('ignore', message=filter_config['message'])
            else:
                # Manejar categorías de warnings de forma segura
                category_name = filter_config['category']
                if hasattr(warnings, category_name):
                    category = getattr(warnings, category_name)
                    warnings.filterwarnings(
                        'ignore', 
                        category=category, 
                        module=filter_config['module']
                    )
                else:
                    # Fallback: filtrar por módulo solo
                    warnings.filterwarnings('ignore', module=filter_config['module'])
    
    @st.cache_resource
    def load_model(_self, model_path: str) -> Optional[tf.keras.Model]:
        """
        Cargar modelo de TensorFlow/Keras con manejo de compatibilidad
        
        Args:
            model_path (str): Ruta al archivo del modelo
            
        Returns:
            tf.keras.Model: Modelo cargado o None si hay error
        """
        try:
            if not os.path.exists(model_path):
                return None
                
            # Cargar modelo sin compilar para evitar problemas de optimizador
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Recompilar con configuración estándar
            model.compile(**MODEL_CONFIG['compile_config'])
            
            return model
            
        except Exception as e:
            _self._handle_model_error(e, model_path)
            return None
    
    def _handle_model_error(self, error: Exception, model_path: str):
        """Manejar errores de carga de modelo"""
        error_msg = str(error)
        
        if "keras.src.models.functional" in error_msg or "cannot be imported" in error_msg:
            st.error("❌ Error de compatibilidad detectado")
            st.info("💡 Este error indica un problema de compatibilidad entre versiones de TensorFlow/Keras")
            st.info("🔧 Soluciones:")
            st.info("   1. Ejecutar: python fix_model_compatibility.py")
            st.info("   2. O usar la versión simple: streamlit run app_simple.py")
            
            if st.button("🔧 Solucionar Compatibilidad Automáticamente"):
                self._fix_compatibility()
        else:
            st.error(f"❌ Error al cargar modelo: {error_msg}")
    
    def _fix_compatibility(self):
        """Intentar solucionar problemas de compatibilidad"""
        with st.spinner("Solucionando problema de compatibilidad..."):
            try:
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, "fix_model_compatibility.py"], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    st.success("✅ Problema de compatibilidad solucionado")
                    st.experimental_rerun()
                else:
                    st.error(f"❌ Error: {result.stderr}")
            except Exception as fix_error:
                st.error(f"❌ Error ejecutando solución: {fix_error}")
    
    @st.cache_resource
    def load_all_models(_self) -> Dict[str, tf.keras.Model]:
        """
        Cargar todos los modelos disponibles para comparación
        
        Returns:
            dict: Diccionario con modelos cargados
        """
        models = {}
        model_dir = get_model_dir()
        
        for model_file in MODEL_CONFIG['default_models']:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                model = _self.load_model(model_path)
                if model is not None:
                    models[model_file] = model
        
        return models
    
    def get_available_models(self) -> List[str]:
        """
        Obtener lista de modelos disponibles
        
        Returns:
            list: Lista de nombres de archivos de modelos
        """
        model_dir = get_model_dir()
        if os.path.exists(model_dir):
            available = [
                f for f in os.listdir(model_dir) 
                if any(f.endswith(fmt) for fmt in MODEL_CONFIG['supported_formats'])
            ]
            return available if available else MODEL_CONFIG['default_models']
        return MODEL_CONFIG['default_models']
    
    def save_uploaded_model(self, uploaded_file, filename: str) -> str:
        """
        Guardar modelo subido por el usuario
        
        Args:
            uploaded_file: Archivo subido
            filename (str): Nombre del archivo
            
        Returns:
            str: Ruta del modelo guardado
        """
        model_dir = get_model_dir()
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, filename)
        with open(model_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return model_path
    
    def load_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Cargar información estadística de un modelo
        
        Args:
            model_name (str): Nombre del modelo
            
        Returns:
            dict: Información del modelo o None si no existe
        """
        try:
            info_file = self._get_model_info_path(model_name)
            
            if info_file and os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    content = f.read()
                
                # Extraer métricas básicas
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
            
        except Exception:
            return None
    
    def _get_model_info_path(self, model_name: str) -> Optional[str]:
        """
        Obtener ruta del archivo de información del modelo
        
        Args:
            model_name (str): Nombre del modelo
            
        Returns:
            str: Ruta del archivo de información
        """
        info_dir = get_info_dir()
        
        if "V1" in model_name or "v1" in model_name:
            return os.path.join(info_dir, "modelV1.txt")
        elif "V2" in model_name or "v2" in model_name:
            return os.path.join(info_dir, "modelV2.txt")
        elif "V3" in model_name or "v3" in model_name:
            return os.path.join(info_dir, "modelV3.txt")
        
        return None
    
    def get_confusion_matrix_path(self, model_name: str) -> Optional[str]:
        """
        Obtener ruta de la matriz de confusión para un modelo
        
        Args:
            model_name (str): Nombre del modelo
            
        Returns:
            str: Ruta de la matriz de confusión
        """
        try:
            info_dir = get_info_dir()
            
            if "V1" in model_name or "v1" in model_name:
                return os.path.join(info_dir, "modelV1.png")
            elif "V2" in model_name or "v2" in model_name:
                return os.path.join(info_dir, "modelV2.png")
            elif "V3" in model_name or "v3" in model_name:
                return os.path.join(info_dir, "modelV3.png")
            
            return None
            
        except Exception:
            return None
    
    def get_model_size(self, model_name: str) -> float:
        """
        Obtener tamaño de un modelo en MB
        
        Args:
            model_name (str): Nombre del modelo
            
        Returns:
            float: Tamaño en MB
        """
        try:
            model_path = os.path.join(get_model_dir(), model_name)
            if os.path.exists(model_path):
                size_bytes = os.path.getsize(model_path)
                return size_bytes / (1024 * 1024)  # Convertir a MB
            return 0.0
        except Exception:
            return 0.0
    
    def validate_model_format(self, filename: str) -> bool:
        """
        Validar formato de archivo de modelo
        
        Args:
            filename (str): Nombre del archivo
            
        Returns:
            bool: True si el formato es válido
        """
        return any(filename.endswith(fmt) for fmt in MODEL_CONFIG['supported_formats']) 

# Crear instancia global del ModelManager
model_manager = ModelManager() 