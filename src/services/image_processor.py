"""
Módulo para procesamiento de imágenes
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import streamlit as st

from ..config.settings import IMAGE_CONFIG

class ImageProcessor:
    """Clase para procesar imágenes para diagnóstico"""
    
    def __init__(self):
        self.input_size = IMAGE_CONFIG['input_size']
        self.clahe_config = IMAGE_CONFIG['clahe_config']
        self.normalization_range = IMAGE_CONFIG['normalization_range']
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesar imagen para predicción del modelo
        
        Args:
            image (PIL.Image): Imagen original
            
        Returns:
            np.ndarray: Imagen procesada lista para predicción
        """
        # Convertir PIL Image a numpy array
        img = np.array(image)
        
        # Convertir RGB a BGR (OpenCV usa BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Redimensionar a tamaño requerido
        img = cv2.resize(img, self.input_size)
        
        # Aplicar CLAHE en espacio LAB para mejorar contraste
        img = self._apply_clahe(img)
        
        # Normalizar valores de pixel
        img = self._normalize_image(img)
        
        # Añadir dimensión de batch
        return np.expand_dims(img, axis=0)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image (np.ndarray): Imagen en formato BGR
            
        Returns:
            np.ndarray: Imagen con CLAHE aplicado
        """
        # Convertir BGR a LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Separar canales
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE al canal L (luminancia)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_config['clip_limit'],
            tileGridSize=self.clahe_config['tile_grid_size']
        )
        cl = clahe.apply(l)
        
        # Combinar canales
        limg = cv2.merge((cl, a, b))
        
        # Convertir de vuelta a RGB
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return img
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizar valores de pixel al rango especificado
        
        Args:
            image (np.ndarray): Imagen a normalizar
            
        Returns:
            np.ndarray: Imagen normalizada
        """
        # Convertir a float32 y normalizar a [0,1]
        img = image.astype('float32') / 255.0
        
        # Aplicar rango de normalización si es diferente de [0,1]
        if self.normalization_range != (0, 1):
            min_val, max_val = self.normalization_range
            img = img * (max_val - min_val) + min_val
        
        return img
    
    def validate_image(self, image: Optional[Image.Image]) -> bool:
        """
        Validar imagen subida
        
        Args:
            image (PIL.Image): Imagen a validar
            
        Returns:
            bool: True si la imagen es válida
        """
        if image is None:
            return False
        
        # Verificar formato
        if image.format.lower() not in ['jpeg', 'jpg', 'png']:
            st.error("❌ Formato de imagen no soportado. Use JPEG o PNG.")
            return False
        
        # Verificar tamaño mínimo
        if image.size[0] < 50 or image.size[1] < 50:
            st.error("❌ Imagen demasiado pequeña. Mínimo 50x50 pixels.")
            return False
        
        # Verificar que no sea demasiado grande
        if image.size[0] > 5000 or image.size[1] > 5000:
            st.error("❌ Imagen demasiado grande. Máximo 5000x5000 pixels.")
            return False
        
        return True
    
    def resize_for_display(self, image: Image.Image, max_size: Tuple[int, int] = (800, 600)) -> Image.Image:
        """
        Redimensionar imagen para visualización
        
        Args:
            image (PIL.Image): Imagen original
            max_size (tuple): Tamaño máximo (ancho, alto)
            
        Returns:
            PIL.Image: Imagen redimensionada
        """
        # Calcular nuevo tamaño manteniendo relación de aspecto
        width, height = image.size
        max_width, max_height = max_size
        
        # Calcular factor de escala
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)  # No hacer más grande
        
        # Nuevo tamaño
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def extract_image_features(self, image: Image.Image) -> dict:
        """
        Extraer características básicas de la imagen
        
        Args:
            image (PIL.Image): Imagen a analizar
            
        Returns:
            dict: Características de la imagen
        """
        # Convertir a numpy array
        img_array = np.array(image)
        
        # Características básicas
        features = {
            'width': image.size[0],
            'height': image.size[1],
            'channels': len(img_array.shape),
            'format': image.format,
            'mode': image.mode
        }
        
        # Estadísticas de color
        if len(img_array.shape) == 3:
            features.update({
                'mean_r': np.mean(img_array[:, :, 0]),
                'mean_g': np.mean(img_array[:, :, 1]),
                'mean_b': np.mean(img_array[:, :, 2]),
                'std_r': np.std(img_array[:, :, 0]),
                'std_g': np.std(img_array[:, :, 1]),
                'std_b': np.std(img_array[:, :, 2])
            })
        
        # Brillo promedio
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        features['brightness'] = np.mean(gray)
        features['contrast'] = np.std(gray)
        
        return features
    
    def create_image_preview(self, image: Image.Image) -> dict:
        """
        Crear vista previa de la imagen con información
        
        Args:
            image (PIL.Image): Imagen original
            
        Returns:
            dict: Información de vista previa
        """
        # Redimensionar para vista previa
        preview_image = self.resize_for_display(image)
        
        # Extraer características
        features = self.extract_image_features(image)
        
        # Información de procesamiento
        processing_info = {
            'original_size': image.size,
            'preview_size': preview_image.size,
            'model_input_size': self.input_size,
            'preprocessing_steps': [
                'Redimensionado a 256x256',
                'Aplicación de CLAHE',
                'Normalización [0,1]',
                'Conversión a batch'
            ]
        }
        
        return {
            'preview_image': preview_image,
            'features': features,
            'processing_info': processing_info
        }
    
    def apply_data_augmentation(self, image: np.ndarray, augmentation_type: str = 'basic') -> np.ndarray:
        """
        Aplicar aumento de datos a la imagen
        
        Args:
            image (np.ndarray): Imagen procesada
            augmentation_type (str): Tipo de aumento
            
        Returns:
            np.ndarray: Imagen con aumento aplicado
        """
        if augmentation_type == 'basic':
            # Rotación aleatoria pequeña
            angle = np.random.uniform(-10, 10)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            # Ajuste de brillo
            brightness = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        return image 