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
from ..utils.i18n import t

class ModelComparisonUI:
    """Interfaz de usuario para comparación de modelos"""
    
    def __init__(self):
        """Inicializar UI de comparación"""
        self.model_manager = model_manager
    
    def render(self):
        """Renderizar interfaz de comparación"""
        st.title(f"📊 {t('comparison.title')}")
        
        # Obtener modelos disponibles
        available_models = self._get_available_models()
        
        if not available_models:
            self._render_no_models_message()
            return
            
        st.info(t('comparison.models_found').format(count=len(available_models)))
        
        try:
            # Recopilar datos de los modelos
            models_data = self._collect_models_data(available_models)
            df = pd.DataFrame(models_data)
            
            # Renderizar componentes
            self._render_summary_metrics(df)
            self._render_detailed_table(df)
            self._render_visualizations(df)
            self._render_recommendations(df)
            self._render_export_option(df)
            
        except Exception as e:
            st.error(t('comparison.models_load_error'))
            st.error(f"Error: {str(e)}")
    
    def _get_available_models(self) -> List[str]:
        """Obtener lista de modelos disponibles"""
        try:
            model_files = [f for f in os.listdir(get_model_dir()) 
                         if f.endswith('.keras')]
            return sorted(model_files)
        except:
            return []
    
    def _render_no_models_message(self):
        """Renderizar mensaje cuando no hay modelos"""
        st.warning(t('comparison.no_models'))
        st.markdown(f"""
        ### {t('comparison.load_models_info')}
        
        #### {t('comparison.create_examples')}
        
        {t('comparison.examples_info')}
        
        1. **{t('comparison.use_current_model')}**
        2. **{t('comparison.create_demo_models')}**
        3. **{t('comparison.load_external_models')}**
        
        {t('comparison.auto_comparison')}
        """)
    
    def _collect_models_data(self, available_models: List[str]) -> Dict[str, List]:
        """Recopilar datos de los modelos"""
        models_data = {
            t('comparison.metrics.model'): [],
            t('comparison.metrics.accuracy'): [],
            t('comparison.metrics.loss'): [],
            t('comparison.metrics.f1_score'): [],
            t('comparison.metrics.size'): [],
            t('comparison.metrics.creation_date'): [],
            t('comparison.metrics.state'): []
        }
        
        for model_file in available_models:
            model_path = os.path.join(get_model_dir(), model_file)
            model_name = model_file.replace('.keras', '')
            models_data[t('comparison.metrics.model')].append(model_name)
            
            # Cargar información estadística
            model_info = self.model_manager.load_model_info(model_file)
            if model_info:
                models_data[t('comparison.metrics.accuracy')].append(model_info['test_accuracy'])
                models_data[t('comparison.metrics.loss')].append(model_info['test_loss'])
                
                # Extraer F1-Score
                f1_score = self._extract_f1_score(model_info['full_report'])
                models_data[t('comparison.metrics.f1_score')].append(f1_score)
                
                # Determinar estado
                if model_info['test_accuracy'] >= 0.7:
                    models_data[t('comparison.metrics.state')].append(f'✅ {t("comparison.model_states.excellent")}')
                elif model_info['test_accuracy'] >= 0.5:
                    models_data[t('comparison.metrics.state')].append(f'⚠️ {t("comparison.model_states.acceptable")}')
                else:
                    models_data[t('comparison.metrics.state')].append(f'❌ {t("comparison.model_states.needs_improvement")}')
            else:
                models_data[t('comparison.metrics.accuracy')].append(0.0)
                models_data[t('comparison.metrics.loss')].append(0.0)
                models_data[t('comparison.metrics.f1_score')].append(0.0)
                models_data[t('comparison.metrics.state')].append(f'❓ {t("comparison.model_states.no_info")}')
            
            # Obtener tamaño y fecha
            try:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                models_data[t('comparison.metrics.size')].append(round(size_mb, 2))
                
                creation_time = os.path.getctime(model_path)
                creation_date = time.strftime('%Y-%m-%d', time.localtime(creation_time))
                models_data[t('comparison.metrics.creation_date')].append(creation_date)
            except:
                models_data[t('comparison.metrics.size')].append(0.0)
                models_data[t('comparison.metrics.creation_date')].append('N/A')
        
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
        st.subheader(f"📈 {t('comparison.summary_metrics')}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy_col = t('comparison.metrics.accuracy')
        f1_col = t('comparison.metrics.f1_score')
        size_col = t('comparison.metrics.size')
        
        valid_precision = df[df[accuracy_col] > 0][accuracy_col]
        valid_f1 = df[df[f1_col] > 0][f1_col]
        
        with col1:
            if len(valid_precision) > 0:
                st.metric(t('comparison.best_accuracy'), f"{valid_precision.max():.2%}")
            else:
                st.metric(t('comparison.best_accuracy'), "N/A")
                
        with col2:
            if len(valid_precision) > 0:
                st.metric(t('comparison.average_accuracy'), f"{valid_precision.mean():.2%}")
            else:
                st.metric(t('comparison.average_accuracy'), "N/A")
                
        with col3:
            if len(valid_f1) > 0:
                st.metric(t('comparison.best_f1'), f"{valid_f1.max():.3f}")
            else:
                st.metric(t('comparison.best_f1'), "N/A")
                
        with col4:
            st.metric(t('comparison.total_size'), f"{df[size_col].sum():.1f} MB")
    
    def _render_detailed_table(self, df: pd.DataFrame):
        """Renderizar tabla detallada"""
        st.subheader(f"📊 {t('comparison.detailed_table')}")
        
        # Formatear datos para mostrar
        display_df = df.copy()
        accuracy_col = t('comparison.metrics.accuracy')
        loss_col = t('comparison.metrics.loss')
        f1_col = t('comparison.metrics.f1_score')
        
        display_df[accuracy_col] = display_df[accuracy_col].apply(lambda x: f"{x:.2%}" if x > 0 else "N/A")
        display_df[loss_col] = display_df[loss_col].apply(lambda x: f"{x:.4f}" if x > 0 else "N/A")
        display_df[f1_col] = display_df[f1_col].apply(lambda x: f"{x:.3f}" if x > 0 else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
    
    def _render_visualizations(self, df: pd.DataFrame):
        """Renderizar visualizaciones"""
        st.subheader(f"📈 {t('comparison.charts.title')}")
        
        # Obtener nombres de columnas traducidos
        accuracy_col = t('comparison.metrics.accuracy')
        loss_col = t('comparison.metrics.loss')
        f1_col = t('comparison.metrics.f1_score')
        model_col = t('comparison.metrics.model')
        size_col = t('comparison.metrics.size')
        
        # Filtrar modelos con datos válidos
        valid_models = df[df[accuracy_col] > 0]
        
        if len(valid_models) == 0:
            st.warning(t('comparison.charts.no_data'))
            return
        
        # Gráficos de barras
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                valid_models, 
                x=model_col, 
                y=accuracy_col,
                title=t('comparison.charts.accuracy_by_model'),
                color=accuracy_col,
                color_continuous_scale='viridis'
            )
            fig1.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black'),
                yaxis=dict(tickformat='.0%')
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                valid_models, 
                x=model_col, 
                y=f1_col,
                title=t('comparison.charts.f1_by_model'),
                color=f1_col,
                color_continuous_scale='plasma'
            )
            fig2.update_traces(texttemplate='%{y:.3f}', textposition='outside')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black')
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Gráfico de dispersión
        if len(valid_models) > 1:
            st.subheader(f"📉 {t('comparison.charts.loss_vs_accuracy_title')}")
            fig3 = px.scatter(
                valid_models, 
                x=loss_col, 
                y=accuracy_col,
                size=size_col, 
                hover_name=model_col,
                title=t('comparison.charts.loss_vs_accuracy'),
                color=f1_col,
                color_continuous_scale='RdYlGn'
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black')
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Gráfico de tamaños
        if len(df) > 1:
            st.subheader(f"📦 {t('comparison.charts.size_comparison_title')}")
            fig4 = px.pie(
                df, 
                values=size_col, 
                names=model_col,
                title=t('comparison.charts.model_size_distribution')
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black')
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    def _render_recommendations(self, df: pd.DataFrame):
        """Renderizar recomendaciones"""
        st.subheader(f"💡 {t('comparison.recommendations')}")
        
        accuracy_col = t('comparison.metrics.accuracy')
        f1_col = t('comparison.metrics.f1_score')
        size_col = t('comparison.metrics.size')
        model_col = t('comparison.metrics.model')
        
        valid_models = df[df[accuracy_col] > 0]
        
        if len(valid_models) < 2:
            st.info(t('comparison.no_recommendations'))
            return
        
        # Encontrar mejor modelo por precisión
        best_accuracy_idx = valid_models[accuracy_col].idxmax()
        best_accuracy_model = valid_models.loc[best_accuracy_idx]
        
        # Encontrar mejor modelo por F1-Score
        best_f1_idx = valid_models[f1_col].idxmax()
        best_f1_model = valid_models.loc[best_f1_idx]
        
        # Encontrar modelo más pequeño
        smallest_idx = valid_models[size_col].idxmin()
        smallest_model = valid_models.loc[smallest_idx]
        
        # Mostrar recomendaciones
        st.markdown(f"#### 🎯 {t('comparison.best_model_analysis')}")
        
        # Mejor por precisión
        st.markdown(f"**{t('comparison.best_accuracy')}**: {best_accuracy_model[model_col]}")
        st.markdown(t('comparison.accuracy_metric').format(accuracy=best_accuracy_model[accuracy_col]))
        st.markdown(t('comparison.loss_metric').format(loss=best_accuracy_model[loss_col]))
        
        # Mejor por F1-Score
        if best_f1_idx != best_accuracy_idx:
            st.markdown(f"\n**{t('comparison.best_f1')}**: {best_f1_model[model_col]}")
            st.markdown(t('comparison.accuracy_metric').format(accuracy=best_f1_model[accuracy_col]))
            st.markdown(t('comparison.loss_metric').format(loss=best_f1_model[loss_col]))
        
        # Modelo más pequeño
        if smallest_idx not in [best_accuracy_idx, best_f1_idx]:
            st.markdown(f"\n**{t('comparison.smallest')}**: {smallest_model[model_col]}")
            st.markdown(t('comparison.accuracy_metric').format(accuracy=smallest_model[accuracy_col]))
            st.markdown(t('comparison.loss_metric').format(loss=smallest_model[loss_col]))
    
    def _render_export_option(self, df: pd.DataFrame):
        """Renderizar opción de exportar datos"""
        st.markdown("---")
        
        # Convertir DataFrame a CSV
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        
        # Botón de descarga
        href = f'<a href="data:file/csv;base64,{b64}" download="model_comparison.csv" class="download-button">📥 {t("comparison.detailed_report")}</a>'
        st.markdown(href, unsafe_allow_html=True) 