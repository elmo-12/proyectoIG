"""
Servicio de diagnóstico de enfermedades
"""
import numpy as np
import random
from typing import Dict, Optional, Tuple, List
from PIL import Image
import streamlit as st

from ..models.model_manager import ModelManager
from ..services.image_processor import ImageProcessor
from ..data.diseases import get_disease_info, get_all_diseases

class DiagnosisService:
    """Servicio para realizar diagnósticos de enfermedades"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.image_processor = ImageProcessor()
        self.disease_info = get_all_diseases()
        
    def predict_disease(self, model, processed_image: np.ndarray) -> np.ndarray:
        """
        Realizar predicción usando el modelo especificado
        
        Args:
            model: Modelo de TensorFlow/Keras
            processed_image (np.ndarray): Imagen preprocesada
            
        Returns:
            np.ndarray: Predicciones del modelo
        """
        try:
            if model is not None:
                # Realizar predicción con el modelo real
                predictions = model.predict(processed_image, verbose=0)
                return predictions
            else:
                # Fallback para demostración cuando no hay modelo
                return self._generate_fallback_prediction()
                
        except Exception as e:
            st.error(f"❌ Error en predicción: {str(e)}")
            return self._generate_fallback_prediction()
    
    def _generate_fallback_prediction(self) -> np.ndarray:
        """
        Generar predicción de demostración cuando no hay modelo disponible
        
        Returns:
            np.ndarray: Predicción simulada
        """
        num_classes = len(self.disease_info)
        fake_prediction = np.zeros((1, num_classes))
        predicted_class = random.randint(0, num_classes-1)
        fake_prediction[0, predicted_class] = random.uniform(0.7, 0.95)
        
        # Distribuir el resto de probabilidad
        remaining = 1.0 - fake_prediction[0, predicted_class]
        other_classes = [i for i in range(num_classes) if i != predicted_class]
        
        for i, class_idx in enumerate(other_classes):
            fake_prediction[0, class_idx] = remaining / len(other_classes)
            
        return fake_prediction
    
    def predict_with_multiple_models(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """
        Realizar predicción con múltiples modelos
        
        Args:
            image (PIL.Image): Imagen a analizar
            
        Returns:
            dict: Predicciones de múltiples modelos
        """
        # Procesar imagen
        processed_image = self.image_processor.preprocess_image(image)
        
        # Cargar todos los modelos
        models = self.model_manager.load_all_models()
        
        predictions = {}
        
        if not models:
            st.warning("⚠️ No hay modelos disponibles para comparación")
            return {}
        
        # Realizar predicción con cada modelo
        for model_name, model in models.items():
            try:
                prediction = self.predict_disease(model, processed_image)
                predictions[model_name] = prediction
            except Exception as e:
                st.error(f"❌ Error con modelo {model_name}: {str(e)}")
                continue
        
        return predictions
    
    def get_consensus_prediction(self, predictions: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
        """
        Obtener predicción de consenso basada en múltiples modelos
        
        Args:
            predictions (dict): Predicciones de múltiples modelos
            
        Returns:
            tuple: (predicción_consenso, clase_predicha, confianza)
        """
        if not predictions:
            return None, None, None
        
        # Obtener información de los modelos para ponderar
        model_weights = {}
        for model_name in predictions.keys():
            model_info = self.model_manager.load_model_info(model_name)
            if model_info and model_info.get('test_accuracy', 0) > 0:
                # Usar la precisión del modelo como peso
                model_weights[model_name] = model_info['test_accuracy']
            else:
                # Peso por defecto si no hay información
                model_weights[model_name] = 0.5
        
        # Calcular predicción promedio ponderada
        num_classes = len(self.disease_info)
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
    
    def analyze_image(self, image: Image.Image, model_name: str = None) -> Dict:
        """
        Analizar imagen completa con diagnóstico
        
        Args:
            image (PIL.Image): Imagen a analizar
            model_name (str): Nombre del modelo específico (opcional)
            
        Returns:
            dict: Resultados completos del análisis
        """
        # Validar imagen
        if not self.image_processor.validate_image(image):
            return None
        
        # Procesar imagen
        processed_image = self.image_processor.preprocess_image(image)
        
        # Realizar predicción
        if model_name:
            # Usar modelo específico
            model_path = f"models/{model_name}"
            model = self.model_manager.load_model(model_path)
            prediction = self.predict_disease(model, processed_image)
            
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class] * 100
            
            return {
                'prediction': prediction,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'disease_info': get_disease_info(predicted_class),
                'all_predictions': None,
                'consensus_prediction': None,
                'model_used': model_name
            }
        else:
            # Usar múltiples modelos
            all_predictions = self.predict_with_multiple_models(image)
            
            if all_predictions:
                # Obtener consenso
                consensus_prediction, predicted_class, confidence = self.get_consensus_prediction(all_predictions)
                
                return {
                    'prediction': consensus_prediction,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'disease_info': get_disease_info(predicted_class),
                    'all_predictions': all_predictions,
                    'consensus_prediction': consensus_prediction,
                    'model_used': 'multiple'
                }
            else:
                return None
    
    def get_prediction_summary(self, results: Dict) -> Dict:
        """
        Obtener resumen de predicción
        
        Args:
            results (dict): Resultados del análisis
            
        Returns:
            dict: Resumen de la predicción
        """
        if not results:
            return {}
        
        disease_info = results['disease_info']
        confidence = results['confidence']
        
        # Determinar nivel de confianza
        if confidence >= 80:
            confidence_level = "Alta"
            confidence_color = "#4CAF50"
        elif confidence >= 60:
            confidence_level = "Media"
            confidence_color = "#FF9800"
        else:
            confidence_level = "Baja"
            confidence_color = "#F44336"
        
        # Determinar prioridad de tratamiento
        severity = disease_info.get('severity', 'low')
        if severity == 'high':
            treatment_priority = "Urgente"
        elif severity == 'medium':
            treatment_priority = "Moderada"
        else:
            treatment_priority = "Baja"
        
        return {
            'disease_name': disease_info['name'],
            'confidence': confidence,
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'treatment_priority': treatment_priority,
            'category': disease_info.get('category', 'unknown'),
            'severity': severity,
            'is_healthy': results['predicted_class'] == 0
        }
    
    def compare_model_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict:
        """
        Comparar predicciones de múltiples modelos
        
        Args:
            predictions (dict): Predicciones de múltiples modelos
            
        Returns:
            dict: Análisis comparativo
        """
        if not predictions:
            return {}
        
        comparison_data = []
        consensus_class = None
        
        # Obtener clase de consenso
        consensus_pred, consensus_class, _ = self.get_consensus_prediction(predictions)
        
        # Analizar cada modelo
        for model_name, prediction in predictions.items():
            model_pred_class = np.argmax(prediction[0])
            model_confidence = prediction[0][model_pred_class] * 100
            model_disease = get_disease_info(model_pred_class)
            model_info = self.model_manager.load_model_info(model_name)
            
            comparison_data.append({
                'model_name': model_name,
                'predicted_class': model_pred_class,
                'confidence': model_confidence,
                'disease_name': model_disease['name'],
                'agrees_with_consensus': model_pred_class == consensus_class,
                'model_accuracy': model_info.get('test_accuracy', 0) if model_info else 0
            })
        
        # Estadísticas de consenso
        agreement_count = sum(1 for data in comparison_data if data['agrees_with_consensus'])
        agreement_percentage = (agreement_count / len(comparison_data)) * 100
        
        return {
            'individual_predictions': comparison_data,
            'consensus_class': consensus_class,
            'agreement_percentage': agreement_percentage,
            'total_models': len(predictions),
            'models_in_agreement': agreement_count
        }
    
    def get_confidence_interpretation(self, confidence: float) -> Dict:
        """
        Interpretar nivel de confianza
        
        Args:
            confidence (float): Nivel de confianza
            
        Returns:
            dict: Interpretación del nivel de confianza
        """
        if confidence >= 90:
            return {
                'level': 'Muy Alta',
                'color': '#4CAF50',
                'interpretation': 'Diagnóstico muy confiable',
                'recommendation': 'Proceder con el tratamiento recomendado'
            }
        elif confidence >= 80:
            return {
                'level': 'Alta',
                'color': '#8BC34A',
                'interpretation': 'Diagnóstico confiable',
                'recommendation': 'Proceder con precaución'
            }
        elif confidence >= 70:
            return {
                'level': 'Media',
                'color': '#FF9800',
                'interpretation': 'Diagnóstico moderadamente confiable',
                'recommendation': 'Considerar análisis adicional'
            }
        elif confidence >= 60:
            return {
                'level': 'Baja',
                'color': '#FF5722',
                'interpretation': 'Diagnóstico poco confiable',
                'recommendation': 'Requiere análisis adicional'
            }
        else:
            return {
                'level': 'Muy Baja',
                'color': '#F44336',
                'interpretation': 'Diagnóstico no confiable',
                'recommendation': 'Requiere evaluación manual'
            } 