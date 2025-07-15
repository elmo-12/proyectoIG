"""
Componentes de interfaz de usuario para la aplicación
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np
import os

from ..config.settings import UI_CONFIG, get_available_models, ensure_directories, initialize_session_state
from ..models.model_manager import ModelManager
from ..services.diagnosis_service import DiagnosisService
from ..services.image_processor import ImageProcessor
from ..visualization.charts import ChartGenerator
from ..reports.pdf_generator import PDFReportGenerator
from ..ui.styles import apply_custom_styles, create_diagnosis_box, create_info_card, create_metric_container
from ..data.diseases import get_disease_info, get_disease_info_translated
from ..utils.i18n import t, i18n

class UIComponents:
    """Componentes de interfaz de usuario"""
    
    def __init__(self):
        self.tabs = UI_CONFIG['tabs']
        self.model_manager = ModelManager()
        self.diagnosis_service = DiagnosisService()
        self.image_processor = ImageProcessor()
        self.chart_generator = ChartGenerator()
        self.pdf_generator = PDFReportGenerator()
        ensure_directories()
        initialize_session_state()
        apply_custom_styles()
    
    def render_language_selector(self):
        """Renderizar selector de idioma"""
        i18n.render_language_selector()
    
    def render_header(self):
        """Renderizar encabezado principal"""
        st.markdown(
            f"<h1 style='color: {UI_CONFIG['theme']['primary_color']}'>🌿 {t('app.title')}</h1>",
            unsafe_allow_html=True
        )
    
    def render_configuration_tab(self):
        """Renderizar pestaña de configuración"""
        st.markdown(f"### {t('config.title')}")
        
        # Selector de modelo
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                t('config.model_selector'),
                available_models,
                index=available_models.index(st.session_state.selected_model_file) if st.session_state.selected_model_file in available_models else 0,
                key="model_selector"
            )
            
            if selected_model != st.session_state.selected_model_file:
                st.session_state.selected_model_file = selected_model
                st.session_state.model_loaded = False
                st.session_state.model = None
                st.experimental_rerun()
        
        # Información del sistema
        self._render_system_info()
        
        # Carga de modelo
        self._render_model_upload()
        
        # Auto-carga de modelo
        self._auto_load_model()
    
    def _render_system_info(self):
        """Renderizar información del sistema"""
        with st.expander(f"ℹ️ {t('config.system_info')}", expanded=True):
            info_content = f"""
            <div class='info-card'>
                <h3>{t('config.about_system')}</h3>
                <p>{t('config.system_description')}</p>
                <ul class='info-list'>
                    <li>✅ {t('config.diseases.healthy')}</li>
                    <li>🟡 {t('config.diseases.mosaic')}</li>
                    <li>🔴 {t('config.diseases.red_rot')}</li>
                    <li>🟠 {t('config.diseases.rust')}</li>
                    <li>💛 {t('config.diseases.yellow')}</li>
                </ul>
                <p>{t('config.model_description')}</p>
            </div>
            """
            st.markdown(info_content, unsafe_allow_html=True)
            
            # Información de PDF
            self._render_pdf_info()
    
    def _render_pdf_info(self):
        """Renderizar información sobre generación de PDF"""
        from ..reports.pdf_generator import REPORTLAB_AVAILABLE, FPDF_AVAILABLE
        
        if REPORTLAB_AVAILABLE:
            st.success(f"✅ {t('config.reportlab_available')}")
        elif FPDF_AVAILABLE:
            st.info(f"⚠️ {t('config.fpdf_available')}")
            st.info(f"💡 {t('config.fpdf_recommendation')}")
        else:
            st.error(f"❌ {t('config.no_pdf_libraries')}")
            st.info(f"💡 {t('config.pdf_install_info')}")
    
    def _render_model_upload(self):
        """Renderizar sección de carga de modelo"""
        model_file = st.file_uploader(t('config.upload_model'), type=['keras', 'h5'])
        
        if model_file is not None:
            if self.model_manager.validate_model_format(model_file.name):
                with st.spinner(f"⏳ {t('config.model_loading')}"):
                    model_path = self.model_manager.save_uploaded_model(model_file, model_file.name)
                    st.success(f"✅ {t('config.model_loaded', model_name=model_file.name)}")
                    st.session_state.selected_model_file = model_file.name
                    st.session_state.model_loaded = False
                    st.session_state.model = None
                    st.experimental_rerun()
            else:
                st.error(f"❌ {t('config.model_format_error')}")
    
    def _auto_load_model(self):
        """Auto-cargar modelo si existe"""
        if not st.session_state.model_loaded and st.session_state.selected_model_file:
            model_path = f"models/{st.session_state.selected_model_file}"
            with st.spinner(f"⏳ {t('config.model_selected_loading')}"):
                model = self.model_manager.load_model(model_path)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success(f"✅ {t('config.model_loaded', model_name=st.session_state.selected_model_file)}")
                else:
                    st.warning(f"⚠️ {t('config.model_load_warning', model_name=st.session_state.selected_model_file)}")
    
    def render_diagnosis_tab(self):
        """Renderizar pestaña de diagnóstico"""
        # Verificar disponibilidad del modelo
        model_available = (st.session_state.model_loaded or st.session_state.model is not None) and st.session_state.selected_model_file
        
        if not model_available:
            self._render_no_model_available()
            return
        
        # Interfaz de diagnóstico
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            self._render_image_upload()
        
        # Mostrar resultados si están disponibles
        if st.session_state.diagnosis_results is not None:
            with col2:
                self._render_diagnosis_results()
    
    def _render_no_model_available(self):
        """Renderizar cuando no hay modelo disponible"""
        st.warning(f"⚠️ {t('diagnosis.no_model')}")
        st.info(f"💡 {t('diagnosis.demo_model_info')}")
        
        if st.button(f"🔧 {t('diagnosis.create_demo_model')}", use_container_width=True):
            self._create_demo_model()
    
    def _create_demo_model(self):
        """Crear modelo de demostración"""
        with st.spinner(f"⏳ {t('diagnosis.demo_model_creating')}"):
            try:
                import subprocess
                import sys
                
                result = subprocess.run(
                    [sys.executable, "create_demo_model.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    st.success(f"✅ {t('diagnosis.demo_model_success')}")
                    st.session_state.model_loaded = True
                    st.experimental_rerun()
                else:
                    st.error(f"❌ {t('diagnosis.demo_model_error', error=result.stderr)}")
            except Exception as e:
                st.error(f"❌ {t('app.error')}: {str(e)}")
    
    def _render_image_upload(self):
        """Renderizar sección de carga de imagen"""
        st.markdown(f"### {t('diagnosis.upload_image')}")
        image_file = st.file_uploader(t('diagnosis.select_image'), type=['jpg', 'jpeg', 'png'])
        
        if image_file is not None:
            image = Image.open(image_file)
            
            # Validar imagen
            if self.image_processor.validate_image(image):
                # Mostrar imagen con estilo
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(image, caption=t('diagnosis.image_loaded', filename=image_file.name), use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Botón de diagnóstico
                if st.button(f"🔍 {t('diagnosis.perform_diagnosis')}", use_container_width=True):
                    self._perform_diagnosis(image)
    
    def _perform_diagnosis(self, image: Image.Image):
        """Realizar diagnóstico de la imagen"""
        with st.spinner(f"🔄 {t('diagnosis.processing_image')}"):
            try:
                # Realizar análisis completo
                results = self.diagnosis_service.analyze_image(image)
                
                if results:
                    # Guardar resultados
                    st.session_state.diagnosis_results = results
                    st.session_state.current_image = image
                    
                    # Mostrar resumen
                    model_used = results.get('model_used', 'unknown')
                    if model_used == 'multiple':
                        all_predictions = results.get('all_predictions', {})
                        st.success(f"✅ {t('diagnosis.analysis_complete_multiple', count=len(all_predictions))}")
                        
                        # Mostrar predicciones individuales
                        for model_name, pred in all_predictions.items():
                            model_pred_class = np.argmax(pred[0])
                            model_conf = pred[0][model_pred_class] * 100
                            model_disease = get_disease_info_translated(model_pred_class)['name']
                            st.info(f"📊 {model_name}: {model_disease} ({model_conf:.1f}%)")
                    else:
                        st.success(f"✅ {t('diagnosis.analysis_complete_single')}")
                        
                    st.experimental_rerun()
                else:
                    st.error(f"❌ {t('diagnosis.analysis_error')}")
                    
            except Exception as e:
                st.error(f"❌ {t('diagnosis.diagnosis_error', error=str(e))}")
    
    def _render_diagnosis_results(self):
        """Renderizar resultados del diagnóstico"""
        results = st.session_state.diagnosis_results
        
        if not results:
            return
        
        predicted_class = results['predicted_class']
        confidence = results['confidence']
        disease_info = get_disease_info_translated(predicted_class)
        
        # Caja de diagnóstico principal
        box_type = "healthy" if predicted_class == 0 else "disease"
        diagnosis_title = t('diagnosis.consensus_diagnosis') if results.get('all_predictions') else t('diagnosis.diagnosis')
        
        # Crear contenido del diagnóstico
        diagnosis_content = f"""
            <div class='diagnosis-box {box_type}'>
                <h2 style='color: #4CAF50; margin-top: 1rem; font-weight: 600;'>{disease_info['icon']} {diagnosis_title}</h2>
                <div class='metric-container'>
                    <p style='font-size: 1.8em; font-weight: bold; margin: 0.5rem 0; color: #E0E0E0;'>
                        {disease_info['name']}
                    </p>
                    <p style='font-size: 1.2em; margin: 1rem 0; color: #E0E0E0;'>
                        {t('diagnosis.confidence_level')}
                        <span style='font-size: 1.4em; font-weight: bold; color: {disease_info['color']};'>
                            {confidence:.1f}%
                        </span>
                    </p>
                </div>
            </div>
        """
        
        st.markdown(diagnosis_content, unsafe_allow_html=True)
        
        # Pestañas de información detallada
        self._render_detailed_info_tabs(results)
        
        # Botón de reporte PDF
        self._render_pdf_button(results)

    def _render_detailed_info_tabs(self, results: Dict):
        """Renderizar pestañas de información detallada"""
        disease_info = get_disease_info_translated(results['predicted_class'])
        
        if results.get('all_predictions'):
            info_tabs = st.tabs([
                f"📋 {t('diagnosis.details')}", 
                f"💊 {t('diagnosis.treatment')}", 
                f"📊 {t('diagnosis.comparison')}", 
                f"📈 {t('diagnosis.analysis')}"
            ])
            tab_comparison = info_tabs[2]
            tab_analysis = info_tabs[3]
        else:
            info_tabs = st.tabs([
                f"📋 {t('diagnosis.details')}", 
                f"💊 {t('diagnosis.treatment')}", 
                f"📊 {t('diagnosis.analysis')}"
            ])
            tab_analysis = info_tabs[2]
        
        # Pestaña de detalles
        with info_tabs[0]:
            self._render_details_tab(disease_info)
            
        # Pestaña de tratamiento
        with info_tabs[1]:
            self._render_treatment_tab(disease_info)
            
        # Pestaña de comparación (si hay múltiples modelos)
        if results.get('all_predictions'):
            with tab_comparison:
                self._render_comparison_tab(results)
            
        # Pestaña de análisis
        with tab_analysis:
            self._render_analysis_tab(results)

    def _render_details_tab(self, disease_info: Dict):
        """Renderizar pestaña de detalles"""
        st.markdown(f"### 📋 {t('diagnosis.description')}")
        st.markdown(create_info_card(t('diagnosis.description'), disease_info['description']), unsafe_allow_html=True)
        
        st.markdown(f"### 🔍 {t('diagnosis.symptoms')}")
        for symptom in disease_info['symptoms']:
            st.markdown(f"• {symptom}")

    def _render_treatment_tab(self, disease_info: Dict):
        """Renderizar pestaña de tratamiento"""
        st.markdown(f"### 💊 {t('diagnosis.recommended_treatment')}")
        for treatment in disease_info['treatment']:
            st.markdown(f"• {treatment}")
            
        st.markdown(f"### 🛡️ {t('diagnosis.preventive_measures')}")
        for prevention in disease_info['prevention']:
            st.markdown(f"• {prevention}")

    def _render_comparison_tab(self, results: Dict):
        """Renderizar pestaña de comparación"""
        st.markdown(f"### 📊 {t('diagnosis.model_comparison')}")
        
        all_predictions = results['all_predictions']
        
        # Gráfico comparativo
        comparative_fig = self.chart_generator.create_comparative_chart(all_predictions)
        st.pyplot(comparative_fig)
        
        # Tabla de resultados
        self._render_comparison_table(results)

    def _render_comparison_table(self, results: Dict):
        """Renderizar tabla de comparación"""
        st.markdown(f"#### 📋 {t('diagnosis.detailed_results')}")
        
        comparison_data = []
        predicted_class = results['predicted_class']
        
        for model_name, pred in results['all_predictions'].items():
            model_pred_class = np.argmax(pred[0])
            model_conf = pred[0][model_pred_class] * 100
            model_disease = get_disease_info_translated(model_pred_class)['name']
            model_info = self.model_manager.load_model_info(model_name)
            
            comparison_data.append({
                t('comparison.metrics.model'): model_name.replace('best_sugarcane_model', t('comparison.model_prefix')).replace('.keras', ''),
                t('diagnosis.diagnosis'): model_disease,
                t('diagnosis.confidence_level'): f"{model_conf:.1f}%",
                t('comparison.metrics.accuracy'): f"{model_info['test_accuracy']:.2%}" if model_info else "N/A",
                t('diagnosis.model_agreement'): t('diagnosis.model_match') if model_pred_class == predicted_class else t('diagnosis.model_differs')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

    def _render_analysis_tab(self, results: Dict):
        """Renderizar pestaña de análisis"""
        if 'all_predictions' in results:
            st.markdown(f"#### {t('diagnosis.consensus_analysis')}")
            
            # Gráfico de distribución de probabilidades
            st.markdown(f"##### {t('diagnosis.probability_distribution')}")
            chart = self.chart_generator.create_probability_chart(results['prediction'])
            st.plotly_chart(chart, use_container_width=True)

    def _render_pdf_button(self, results: Dict):
        """Renderizar botón de generación de PDF"""
        if st.button(f"📄 {t('diagnosis.generate_pdf')}", use_container_width=True):
            self._generate_pdf_report(results)

    def _generate_pdf_report(self, results: Dict):
        """Generar reporte PDF"""
        try:
            with st.spinner(t('diagnosis.generating_pdf')):
                # Obtener información traducida de la enfermedad
                disease_info = get_disease_info_translated(results['predicted_class'])
                
                # Generar PDF
                pdf_path = self.pdf_generator.generate_report(
                    image=st.session_state.current_image,
                    disease_info=disease_info,
                    confidence=results['confidence'],
                    probabilities=results['prediction'],
                    model_name=st.session_state.selected_model_file,
                    all_predictions=results.get('all_predictions'),
                    consensus_prediction=results.get('consensus_prediction')
                )
                
                if pdf_path and os.path.exists(pdf_path):
                    try:
                        # Leer el archivo PDF en memoria
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                        
                        # Mostrar mensaje de éxito y botón de descarga
                        st.success(t('pdf.pdf_generated_success'))
                        st.download_button(
                            label=f"📥 {t('pdf.download_report')}",
                            data=pdf_bytes,
                            file_name="diagnosis_report.pdf",
                            mime="application/pdf",
                            help=t('pdf.download_help'),
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ {t('pdf.download_error')}: {str(e)}")
                    finally:
                        # Limpiar archivo temporal
                        try:
                            if os.path.exists(pdf_path):
                                os.unlink(pdf_path)
                        except Exception as e:
                            st.warning(f"⚠️ {t('pdf.cleanup_warning')}: {str(e)}")
                else:
                    st.error(t('pdf.generation_failed'))
        except Exception as e:
            st.error(t('diagnosis.pdf_generation_error', error=str(e)))
    
    def render_comparison_tab(self):
        """Renderizar pestaña de comparación de modelos"""
        from .model_comparison import ModelComparisonUI
        comparison_ui = ModelComparisonUI()
        comparison_ui.render()
    
    def render_footer(self):
        """Renderizar pie de página"""
        st.markdown("---")
        st.markdown(f"### {t('footer.title')}")
        st.markdown(f"*{t('footer.description')}*")
        st.markdown(f"*{t('footer.subtitle')}*") 