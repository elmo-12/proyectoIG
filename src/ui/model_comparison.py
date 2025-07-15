"""
Componente UI para comparación de modelos
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
import base64
from typing import Dict, List

from ..models.model_manager import model_manager
from ..config.settings import get_model_dir
from ..visualization.charts import chart_generator

class ModelComparisonUI:
    """Interfaz de usuario para comparación de modelos"""
    
    def __init__(self):
        self.model_manager = model_manager
        self.chart_generator = chart_generator
        
    def render(self):
        """Renderizar interfaz de comparación de modelos"""
        st.header("📊 Comparación de Modelos de Diagnóstico")
        
        # Verificar modelos disponibles
        available_models = self._get_available_models()
        
        if not available_models:
            self._render_no_models_message()
            return
        
        # Mostrar información de modelos disponibles
        st.success(f"✅ Se encontraron {len(available_models)} modelo(s) para comparar")
        
        # Recopilar datos de los modelos
        models_data = self._collect_models_data(available_models)
        
        if not models_data['Modelo']:
            st.error("❌ No se pudieron cargar los datos de los modelos")
            return
        
        # Crear DataFrame
        df = pd.DataFrame(models_data)
        
        # Renderizar métricas de resumen
        self._render_summary_metrics(df)
        
        # Renderizar tabla detallada
        self._render_detailed_table(df)
        
        # Renderizar visualizaciones
        self._render_visualizations(df)
        
        # Renderizar recomendaciones
        self._render_recommendations(df)
        
        # Renderizar opción de exportar
        self._render_export_option(df)
    
    def _get_available_models(self) -> List[str]:
        """Obtener lista de modelos disponibles"""
        available_models = []
        model_files = [
            "best_sugarcane_modelV1.keras",
            "best_sugarcane_modelV2.keras",
            "best_sugarcane_modelV3.keras"
        ]
        
        # Incluir cualquier otro modelo .keras en la carpeta
        model_dir = get_model_dir()
        try:
            if os.path.exists(model_dir):
                additional_models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
                model_files = list(set(model_files + additional_models))
        except:
            pass
        
        # Verificar cuáles existen
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                available_models.append(model_file)
        
        return available_models
    
    def _render_no_models_message(self):
        """Renderizar mensaje cuando no hay modelos"""
        st.warning("⚠️ No se encontraron modelos para comparar")
        st.info("💡 Carga modelos en la pestaña de Configuración para poder compararlos")
        
        with st.expander("🔧 Crear Modelos de Ejemplo", expanded=False):
            st.markdown("""
            Para crear modelos de ejemplo para comparación, puedes:
            
            1. **Usar el modelo actual**: El modelo cargado en la pestaña de Configuración
            2. **Crear modelos de demostración**: Ejecutar scripts de entrenamiento
            3. **Cargar modelos externos**: Subir archivos .keras desde tu computadora
            
            Los modelos se compararán automáticamente cuando estén disponibles.
            """)
    
    def _collect_models_data(self, available_models: List[str]) -> Dict[str, List]:
        """Recopilar datos de los modelos"""
        models_data = {
            'Modelo': [],
            'Precisión': [],
            'Pérdida': [],
            'F1-Score Promedio': [],
            'Tamaño (MB)': [],
            'Fecha Creación': [],
            'Estado': []
        }
        
        for model_file in available_models:
            model_path = os.path.join(get_model_dir(), model_file)
            model_name = model_file.replace('.keras', '')
            models_data['Modelo'].append(model_name)
            
            # Cargar información estadística
            model_info = self.model_manager.load_model_info(model_file)
            if model_info:
                models_data['Precisión'].append(model_info['test_accuracy'])
                models_data['Pérdida'].append(model_info['test_loss'])
                
                # Extraer F1-Score
                f1_score = self._extract_f1_score(model_info['full_report'])
                models_data['F1-Score Promedio'].append(f1_score)
                
                # Determinar estado
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
            
            # Obtener tamaño y fecha
            try:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                models_data['Tamaño (MB)'].append(round(size_mb, 2))
                
                creation_time = os.path.getctime(model_path)
                creation_date = time.strftime('%Y-%m-%d', time.localtime(creation_time))
                models_data['Fecha Creación'].append(creation_date)
            except:
                models_data['Tamaño (MB)'].append(0.0)
                models_data['Fecha Creación'].append('N/A')
        
        return models_data
    
    def _extract_f1_score(self, report: str) -> float:
        """Extraer F1-Score del reporte"""
        try:
            lines = report.split('\n')
            for line in lines:
                if 'macro avg' in line and 'f1-score' in line:
                    parts = line.split()
                    return float(parts[3])
            return 0.0
        except:
            return 0.0
    
    def _render_summary_metrics(self, df: pd.DataFrame):
        """Renderizar métricas de resumen"""
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
    
    def _render_detailed_table(self, df: pd.DataFrame):
        """Renderizar tabla detallada"""
        st.subheader("📊 Tabla Detallada de Métricas")
        
        # Formatear datos para mostrar
        display_df = df.copy()
        display_df['Precisión'] = display_df['Precisión'].apply(lambda x: f"{x:.2%}" if x > 0 else "N/A")
        display_df['Pérdida'] = display_df['Pérdida'].apply(lambda x: f"{x:.4f}" if x > 0 else "N/A")
        display_df['F1-Score Promedio'] = display_df['F1-Score Promedio'].apply(lambda x: f"{x:.3f}" if x > 0 else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
    
    def _render_visualizations(self, df: pd.DataFrame):
        """Renderizar visualizaciones"""
        st.subheader("📈 Visualizaciones")
        
        # Filtrar modelos con datos válidos
        valid_models = df[df['Precisión'] > 0]
        
        if len(valid_models) == 0:
            st.warning("⚠️ No hay datos suficientes para mostrar gráficos")
            return
        
        # Gráficos de barras
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                valid_models, 
                x='Modelo', 
                y='Precisión',
                title='Precisión por Modelo',
                color='Precisión',
                color_continuous_scale='viridis'
            )
            fig1.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(tickformat='.0%')
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                valid_models, 
                x='Modelo', 
                y='F1-Score Promedio',
                title='F1-Score Promedio por Modelo',
                color='F1-Score Promedio',
                color_continuous_scale='plasma'
            )
            fig2.update_traces(texttemplate='%{y:.3f}', textposition='outside')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Gráfico de dispersión
        if len(valid_models) > 1:
            st.subheader("📉 Análisis de Pérdida vs Precisión")
            fig3 = px.scatter(
                valid_models, 
                x='Pérdida', 
                y='Precisión',
                size='Tamaño (MB)', 
                hover_name='Modelo',
                title='Relación entre Pérdida y Precisión',
                color='F1-Score Promedio',
                color_continuous_scale='RdYlGn'
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Gráfico de tamaños
        if len(df) > 1:
            st.subheader("📦 Comparación de Tamaños")
            fig4 = px.pie(
                df, 
                values='Tamaño (MB)', 
                names='Modelo',
                title='Distribución del Tamaño de Modelos'
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    def _render_recommendations(self, df: pd.DataFrame):
        """Renderizar recomendaciones"""
        st.subheader("💡 Recomendaciones")
        
        valid_models = df[df['Precisión'] > 0]
        
        if len(valid_models) == 0:
            st.info("No hay suficientes datos para generar recomendaciones")
            return
        
        # Mejores modelos
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
        
        # Análisis del mejor modelo
        st.subheader("🏆 Análisis del Mejor Modelo")
        best_model_info = self.model_manager.load_model_info(best_model + '.keras')
        
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
        
        # Matrices de confusión
        self._render_confusion_matrices(valid_models)
    
    def _render_confusion_matrices(self, valid_models: pd.DataFrame):
        """Renderizar matrices de confusión"""
        st.subheader("📊 Matrices de Confusión")
        
        matrices_cols = st.columns(min(3, len(valid_models)))
        
        for i, (_, model_row) in enumerate(valid_models.iterrows()):
            if i >= 3:  # Mostrar máximo 3 matrices
                break
                
            model_name = model_row['Modelo']
            confusion_path = self.model_manager.get_confusion_matrix_path(model_name + '.keras')
            
            if confusion_path and os.path.exists(confusion_path):
                with matrices_cols[i]:
                    st.markdown(f"**{model_name}**")
                    st.image(confusion_path, use_column_width=True)
    
    def _render_export_option(self, df: pd.DataFrame):
        """Renderizar opción de exportar"""
        st.subheader("📤 Exportar Comparación")
        
        if st.button("📊 Exportar Datos de Comparación", use_container_width=True):
            # Formatear datos para exportar
            export_df = df.copy()
            export_df['Precisión'] = export_df['Precisión'].apply(lambda x: f"{x:.2%}" if x > 0 else "N/A")
            export_df['Pérdida'] = export_df['Pérdida'].apply(lambda x: f"{x:.4f}" if x > 0 else "N/A")
            export_df['F1-Score Promedio'] = export_df['F1-Score Promedio'].apply(lambda x: f"{x:.3f}" if x > 0 else "N/A")
            
            # Crear CSV
            csv = export_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="comparacion_modelos.csv">⬇️ Descargar CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Datos exportados exitosamente") 