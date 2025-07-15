"""
Módulo de visualización para gráficos y charts
"""
import matplotlib
matplotlib.use('Agg')  # Necesario para entornos sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import tempfile
import os

from ..config.settings import VISUALIZATION_CONFIG
from ..data.diseases import get_all_diseases, get_disease_names, get_disease_colors, get_disease_info_translated, get_disease_names_translated
from ..utils.i18n import t

class ChartGenerator:
    """Generador de gráficos para análisis de diagnóstico"""
    
    def __init__(self):
        self.config = VISUALIZATION_CONFIG
        self.disease_info = get_all_diseases()
        self.chart_colors = self.config['chart_colors']
        plt.style.use(self.config['chart_style'])
        
    def create_probability_chart(self, prediction: np.ndarray, 
                               title: str = None) -> plt.Figure:
        """
        Crear gráfico de barras con probabilidades por clase
        
        Args:
            prediction (np.ndarray): Predicciones del modelo
            title (str): Título del gráfico
            
        Returns:
            plt.Figure: Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        fig.patch.set_facecolor(self.config['background_color'])
        ax.set_facecolor(self.config['background_color'])
        
        # Obtener nombres y probabilidades traducidos
        class_names = get_disease_names_translated()
        probabilities = prediction[0] * 100
        colors = get_disease_colors()
        
        # Crear barras
        bars = ax.bar(class_names, probabilities, color=colors, alpha=0.8)
        
        # Configurar ejes con traducciones
        ax.set_ylabel(t('charts.probability_percent'), color='white', fontsize=12)
        chart_title = title or t('charts.probability_distribution')
        ax.set_title(chart_title, color='white', fontsize=14, fontweight='bold')
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
                fontweight='bold',
                fontsize=10
            )
        
        # Ajustar diseño
        plt.tight_layout()
        
        return fig
    
    def create_probability_chart_for_pdf(self, prediction: np.ndarray, 
                                       title: str = None) -> plt.Figure:
        """
        Crear gráfico de barras con probabilidades por clase optimizado para PDF
        
        Args:
            prediction (np.ndarray): Predicciones del modelo
            title (str): Título del gráfico
            
        Returns:
            plt.Figure: Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Obtener nombres y probabilidades traducidos
        class_names = get_disease_names_translated()
        probabilities = prediction[0] * 100
        colors = get_disease_colors()
        
        # Crear barras
        bars = ax.bar(class_names, probabilities, color=colors, alpha=0.8)
        
        # Configurar ejes con traducciones
        ax.set_ylabel(t('charts.probability_percent'), color='black', fontsize=12)
        chart_title = title or t('charts.probability_distribution')
        ax.set_title(chart_title, color='black', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        plt.xticks(rotation=45, ha='right', color='black')
        ax.tick_params(axis='y', colors='black')
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                color='black',
                fontweight='bold',
                fontsize=10
            )
        
        # Añadir grid suave para mejor legibilidad
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Ajustar diseño
        plt.tight_layout()
        
        return fig
    
    def create_comparative_chart(self, predictions: Dict[str, np.ndarray], 
                               title: str = None) -> plt.Figure:
        """
        Crear gráfico comparativo de múltiples modelos
        
        Args:
            predictions (dict): Predicciones de múltiples modelos
            title (str): Título del gráfico
            
        Returns:
            plt.Figure: Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.patch.set_facecolor(self.config['background_color'])
        ax.set_facecolor(self.config['background_color'])
        
        class_names = get_disease_names_translated()
        model_names = list(predictions.keys())
        
        # Configurar posiciones de las barras
        x = np.arange(len(class_names))
        width = 0.25
        
        # Crear barras para cada modelo
        for i, (model_name, prediction) in enumerate(predictions.items()):
            probabilities = prediction[0] * 100
            model_label = model_name.replace('best_sugarcane_model', t('charts.model_label')).replace('.keras', '')
            
            bars = ax.bar(x + i * width, probabilities, width, 
                         label=model_label, 
                         color=self.chart_colors[i % len(self.chart_colors)], 
                         alpha=0.8)
            
            # Añadir valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                if height > 5:  # Solo mostrar si la probabilidad es mayor a 5%
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom',
                           color='white', fontsize=8, fontweight='bold')
        
        # Configurar ejes con traducciones
        ax.set_ylabel(t('charts.probability_percent'), color='white', fontsize=12)
        ax.set_xlabel(t('charts.disease_classes'), color='white', fontsize=12)
        chart_title = title or t('charts.model_comparison')
        ax.set_title(chart_title, color='white', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45, ha='right', color='white')
        ax.tick_params(axis='y', colors='white')
        ax.legend(loc='upper right', facecolor=self.config['background_color'], 
                 edgecolor='white', labelcolor='white')
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        return fig
    
    def create_comparative_chart_for_pdf(self, predictions: Dict[str, np.ndarray], 
                                       title: str = None) -> plt.Figure:
        """
        Crear gráfico comparativo de múltiples modelos optimizado para PDF
        
        Args:
            predictions (dict): Predicciones de múltiples modelos
            title (str): Título del gráfico
            
        Returns:
            plt.Figure: Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        class_names = get_disease_names_translated()
        model_names = list(predictions.keys())
        
        # Configurar posiciones de las barras
        x = np.arange(len(class_names))
        width = 0.25
        
        # Crear barras para cada modelo
        for i, (model_name, prediction) in enumerate(predictions.items()):
            probabilities = prediction[0] * 100
            model_label = model_name.replace('best_sugarcane_model', t('charts.model_label')).replace('.keras', '')
            
            bars = ax.bar(x + i * width, probabilities, width, 
                         label=model_label, 
                         color=self.chart_colors[i % len(self.chart_colors)], 
                         alpha=0.8)
            
            # Añadir valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                if height > 5:  # Solo mostrar si la probabilidad es mayor a 5%
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom',
                           color='black', fontsize=8, fontweight='bold')
        
        # Configurar ejes con traducciones
        ax.set_ylabel(t('charts.probability_percent'), color='black', fontsize=12)
        ax.set_xlabel(t('charts.disease_classes'), color='black', fontsize=12)
        chart_title = title or t('charts.model_comparison')
        ax.set_title(chart_title, color='black', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45, ha='right', color='black')
        ax.tick_params(axis='y', colors='black')
        ax.legend(loc='upper right', facecolor='white', 
                 edgecolor='black', labelcolor='black')
        ax.set_ylim([0, 100])
        
        # Añadir grid suave para mejor legibilidad
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        plt.tight_layout()
        return fig
    
    def create_confidence_gauge(self, confidence: float) -> go.Figure:
        """
        Crear gauge de confianza con Plotly
        
        Args:
            confidence (float): Nivel de confianza (0-100)
            
        Returns:
            go.Figure: Figura de Plotly
        """
        # Determinar color según confianza
        if confidence >= 80:
            color = "green"
        elif confidence >= 60:
            color = "yellow"
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': t('charts.confidence_level')},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor=self.config['background_color'],
            plot_bgcolor=self.config['background_color'],
            font={'color': 'white'}
        )
        
        return fig
    
    def create_model_comparison_table(self, models_data: pd.DataFrame) -> go.Figure:
        """
        Crear tabla comparativa de modelos
        
        Args:
            models_data (pd.DataFrame): Datos de los modelos
            
        Returns:
            go.Figure: Tabla de Plotly
        """
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(models_data.columns),
                       fill_color='darkgray',
                       align='left',
                       font=dict(color='white', size=12)),
            cells=dict(values=[models_data[col] for col in models_data.columns],
                      fill_color='lightgray',
                      align='left',
                      font=dict(color='black', size=11))
        )])
        
        fig.update_layout(
            title=t('charts.detailed_model_comparison'),
            paper_bgcolor=self.config['background_color'],
            plot_bgcolor=self.config['background_color'],
            font={'color': 'white'}
        )
        
        return fig
    
    def create_performance_radar(self, model_metrics: Dict[str, float]) -> go.Figure:
        """
        Crear gráfico radar de rendimiento del modelo
        
        Args:
            model_metrics (dict): Métricas del modelo
            
        Returns:
            go.Figure: Gráfico radar de Plotly
        """
        # Métricas para el radar traducidas
        categories = [t('charts.accuracy'), t('charts.recall'), t('charts.f1_score'), t('charts.accuracy')]
        
        # Valores del modelo (convertir a escala 0-100)
        values = [
            model_metrics.get('precision', 0.5) * 100,
            model_metrics.get('recall', 0.5) * 100,
            model_metrics.get('f1_score', 0.5) * 100,
            model_metrics.get('accuracy', 0.5) * 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=t('charts.current_model'),
            line_color='rgb(1,90,100)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='white'
                ),
                angularaxis=dict(
                    gridcolor='white'
                )
            ),
            showlegend=True,
            title=t('charts.model_performance'),
            paper_bgcolor=self.config['background_color'],
            plot_bgcolor=self.config['background_color'],
            font={'color': 'white'}
        )
        
        return fig
    
    def create_disease_distribution_pie(self, predictions: Dict[str, np.ndarray]) -> go.Figure:
        """
        Crear gráfico de torta con distribución de predicciones
        
        Args:
            predictions (dict): Predicciones de múltiples modelos
            
        Returns:
            go.Figure: Gráfico de torta de Plotly
        """
        # Calcular promedio de predicciones
        avg_predictions = np.mean([pred[0] for pred in predictions.values()], axis=0)
        
        # Preparar datos con traducciones
        labels = get_disease_names_translated()
        values = avg_predictions * 100
        colors = get_disease_colors()
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hovertemplate=f'<b>%{{label}}</b><br>{t("charts.probability")}: %{{value:.1f}}%<extra></extra>'
        )])
        
        fig.update_layout(
            title=t('charts.average_predictions_distribution'),
            paper_bgcolor=self.config['background_color'],
            plot_bgcolor=self.config['background_color'],
            font={'color': 'white'}
        )
        
        return fig
    
    def create_accuracy_trend(self, model_history: Dict[str, List[float]]) -> go.Figure:
        """
        Crear gráfico de tendencia de precisión
        
        Args:
            model_history (dict): Historial de entrenamiento
            
        Returns:
            go.Figure: Gráfico de líneas de Plotly
        """
        fig = go.Figure()
        
        # Línea de precisión de entrenamiento
        if 'train_accuracy' in model_history:
            fig.add_trace(go.Scatter(
                y=model_history['train_accuracy'],
                mode='lines+markers',
                name=t('charts.training_accuracy'),
                line=dict(color='blue')
            ))
        
        # Línea de precisión de validación
        if 'val_accuracy' in model_history:
            fig.add_trace(go.Scatter(
                y=model_history['val_accuracy'],
                mode='lines+markers',
                name=t('charts.validation_accuracy'),
                line=dict(color='orange')
            ))
        
        fig.update_layout(
            title=t('charts.training_evolution'),
            xaxis_title=t('charts.epoch'),
            yaxis_title=t('charts.accuracy'),
            paper_bgcolor=self.config['background_color'],
            plot_bgcolor=self.config['background_color'],
            font={'color': 'white'}
        )
        
        return fig
    
    def save_chart_temp(self, fig: plt.Figure) -> str:
        """
        Guardar gráfico en archivo temporal
        
        Args:
            fig (plt.Figure): Figura de matplotlib
            
        Returns:
            str: Ruta del archivo temporal
        """
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.png', 
            delete=False,
            dir=tempfile.gettempdir()
        )
        
        fig.savefig(
            temp_file.name, 
            bbox_inches='tight', 
            dpi=self.config['dpi'],
            facecolor=self.config['background_color']
        )
        
        plt.close(fig)
        return temp_file.name
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """
        Limpiar archivos temporales
        
        Args:
            file_paths (list): Lista de rutas de archivos a eliminar
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass  # Ignorar errores al limpiar 

# Crear instancia global del ChartGenerator
chart_generator = ChartGenerator() 