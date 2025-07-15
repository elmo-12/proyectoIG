"""
Componentes de interfaz de usuario para la aplicación
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np

from ..config.settings import UI_CONFIG, get_available_models, ensure_directories, initialize_session_state
from ..models.model_manager import ModelManager
from ..services.diagnosis_service import DiagnosisService
from ..services.image_processor import ImageProcessor
from ..visualization.charts import ChartGenerator
from ..reports.pdf_generator import PDFReportGenerator
from ..ui.styles import apply_custom_styles, create_diagnosis_box, create_info_card, create_metric_container
from ..data.diseases import get_disease_info

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
    
    def render_header(self):
        """Renderizar encabezado principal"""
        st.markdown(
            f"<h1 style='color: {UI_CONFIG['theme']['primary_color']}'>🌿 Sistema Experto de Diagnóstico de Enfermedades en Caña de Azúcar</h1>",
            unsafe_allow_html=True
        )
    
    def render_configuration_tab(self):
        """Renderizar pestaña de configuración"""
        st.markdown("### Configuración del Modelo")
        
        # Selector de modelo
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "Selecciona el modelo a utilizar:",
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
        with st.expander("ℹ️ Información del Sistema", expanded=True):
            info_content = """
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
            """
            st.markdown(info_content, unsafe_allow_html=True)
            
            # Información de PDF
            self._render_pdf_info()
    
    def _render_pdf_info(self):
        """Renderizar información sobre generación de PDF"""
        from ..reports.pdf_generator import REPORTLAB_AVAILABLE, FPDF_AVAILABLE
        
        if REPORTLAB_AVAILABLE:
            st.success("✅ ReportLab disponible: Generación de PDF con soporte completo UTF-8")
        elif FPDF_AVAILABLE:
            st.info("⚠️ FPDF disponible: Generación de PDF básica")
            st.info("💡 Para mejor calidad de PDF, instala: `pip install reportlab==4.0.4`")
        else:
            st.error("❌ No hay bibliotecas de PDF disponibles")
            st.info("💡 Instala ReportLab o FPDF para generar reportes PDF")
    
    def _render_model_upload(self):
        """Renderizar sección de carga de modelo"""
        model_file = st.file_uploader("Cargar modelo (.keras)", type=['keras', 'h5'])
        
        if model_file is not None:
            if self.model_manager.validate_model_format(model_file.name):
                with st.spinner("⏳ Cargando modelo..."):
                    model_path = self.model_manager.save_uploaded_model(model_file, model_file.name)
                    st.success(f"✅ Modelo '{model_file.name}' cargado exitosamente")
                    st.session_state.selected_model_file = model_file.name
                    st.session_state.model_loaded = False
                    st.session_state.model = None
                    st.experimental_rerun()
            else:
                st.error("❌ Formato de modelo no soportado")
    
    def _auto_load_model(self):
        """Auto-cargar modelo si existe"""
        if not st.session_state.model_loaded and st.session_state.selected_model_file:
            model_path = f"models/{st.session_state.selected_model_file}"
            with st.spinner("⏳ Cargando modelo seleccionado..."):
                model = self.model_manager.load_model(model_path)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success(f"✅ Modelo '{st.session_state.selected_model_file}' cargado exitosamente")
                else:
                    st.warning(f"⚠️ No se pudo cargar el modelo '{st.session_state.selected_model_file}'")
    
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
        st.warning("⚠️ Por favor, carga primero el modelo en la pestaña de Configuración")
        st.info("💡 También puedes crear un modelo de demostración")
        
        if st.button("🔧 Crear Modelo de Demostración", use_container_width=True):
            self._create_demo_model()
    
    def _create_demo_model(self):
        """Crear modelo de demostración"""
        with st.spinner("⏳ Creando modelo de demostración..."):
            try:
                import subprocess
                import sys
                
                result = subprocess.run(
                    [sys.executable, "create_demo_model.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    st.success("✅ Modelo de demostración creado exitosamente")
                    st.session_state.model_loaded = True
                    st.experimental_rerun()
                else:
                    st.error(f"❌ Error al crear modelo: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def _render_image_upload(self):
        """Renderizar sección de carga de imagen"""
        st.markdown("### Cargar Imagen")
        image_file = st.file_uploader("Seleccionar imagen de hoja", type=['jpg', 'jpeg', 'png'])
        
        if image_file is not None:
            image = Image.open(image_file)
            
            # Validar imagen
            if self.image_processor.validate_image(image):
                # Mostrar imagen con estilo
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(image, caption=f"Imagen cargada: {image_file.name}", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Botón de diagnóstico
                if st.button("🔍 Realizar Diagnóstico Comparativo", use_container_width=True):
                    self._perform_diagnosis(image)
    
    def _perform_diagnosis(self, image: Image.Image):
        """Realizar diagnóstico de la imagen"""
        with st.spinner("🔄 Procesando imagen con múltiples modelos de IA..."):
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
                        st.success(f"✅ Análisis completado con {len(all_predictions)} modelo(s)")
                        
                        # Mostrar predicciones individuales
                        for model_name, pred in all_predictions.items():
                            model_pred_class = np.argmax(pred[0])
                            model_conf = pred[0][model_pred_class] * 100
                            model_disease = get_disease_info(model_pred_class)['name']
                            st.info(f"📊 {model_name}: {model_disease} ({model_conf:.1f}%)")
                    else:
                        st.success("✅ Análisis completado con modelo único")
                        
                    st.experimental_rerun()
                else:
                    st.error("❌ No se pudo realizar el análisis")
                    
            except Exception as e:
                st.error(f"❌ Error en diagnóstico: {str(e)}")
    
    def _render_diagnosis_results(self):
        """Renderizar resultados del diagnóstico"""
        results = st.session_state.diagnosis_results
        
        if not results:
            return
        
        predicted_class = results['predicted_class']
        confidence = results['confidence']
        disease_info = results['disease_info']
        
        # Caja de diagnóstico principal
        box_type = "healthy" if predicted_class == 0 else "disease"
        diagnosis_title = "Diagnóstico de Consenso" if results.get('all_predictions') else "Diagnóstico"
        
        # Crear contenido del diagnóstico
        diagnosis_content = f"""
            <div class='diagnosis-box {box_type}'>
                <h2 style='color: #4CAF50; margin-top: 1rem; font-weight: 600;'>{disease_info['icon']} {diagnosis_title}</h2>
                <div class='metric-container'>
                    <p style='font-size: 1.8em; font-weight: bold; margin: 0.5rem 0; color: #E0E0E0;'>
                        {disease_info['name']}
                    </p>
                    <p style='font-size: 1.2em; margin: 1rem 0; color: #E0E0E0;'>
                        Nivel de confianza:
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
        disease_info = results['disease_info']
        
        if results.get('all_predictions'):
            info_tabs = st.tabs(["📋 Detalles", "💊 Tratamiento", "📊 Comparación", "📈 Análisis"])
            tab_comparison = info_tabs[2]
            tab_analysis = info_tabs[3]
        else:
            info_tabs = st.tabs(["📋 Detalles", "💊 Tratamiento", "📊 Análisis"])
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
        st.markdown("### 📋 Descripción")
        st.markdown(create_info_card("Descripción", disease_info['description']), unsafe_allow_html=True)
        
        st.markdown("### 🔍 Síntomas")
        for symptom in disease_info['symptoms']:
            st.markdown(f"• {symptom}")
    
    def _render_treatment_tab(self, disease_info: Dict):
        """Renderizar pestaña de tratamiento"""
        st.markdown("### 💊 Tratamiento Recomendado")
        for treatment in disease_info['treatment']:
            st.markdown(f"• {treatment}")
        
        st.markdown("### 🛡️ Medidas Preventivas")
        for prevention in disease_info['prevention']:
            st.markdown(f"• {prevention}")
    
    def _render_comparison_tab(self, results: Dict):
        """Renderizar pestaña de comparación"""
        st.markdown("### 📊 Comparación de Modelos")
        
        all_predictions = results['all_predictions']
        
        # Gráfico comparativo
        comparative_fig = self.chart_generator.create_comparative_chart(all_predictions)
        st.pyplot(comparative_fig)
        
        # Tabla de resultados
        self._render_comparison_table(results)
    
    def _render_comparison_table(self, results: Dict):
        """Renderizar tabla de comparación"""
        st.markdown("#### 📋 Resultados Detallados por Modelo")
        
        comparison_data = []
        predicted_class = results['predicted_class']
        
        for model_name, pred in results['all_predictions'].items():
            model_pred_class = np.argmax(pred[0])
            model_conf = pred[0][model_pred_class] * 100
            model_disease = get_disease_info(model_pred_class)['name']
            model_info = self.model_manager.load_model_info(model_name)
            
            comparison_data.append({
                'Modelo': model_name.replace('best_sugarcane_model', 'Modelo ').replace('.keras', ''),
                'Diagnóstico': model_disease,
                'Confianza': f"{model_conf:.1f}%",
                'Precisión del Modelo': f"{model_info['test_accuracy']:.2%}" if model_info else "N/A",
                'Estado': '✅ Coincide' if model_pred_class == predicted_class else '⚠️ Difiere'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    def _render_analysis_tab(self, results: Dict):
        """Renderizar pestaña de análisis"""
        title = "📈 Análisis del Consenso" if results.get('all_predictions') else "📊 Distribución de Probabilidades"
        st.markdown(f"### {title}")
        
        # Gráfico de probabilidades
        if results.get('consensus_prediction') is not None:
            fig = self.chart_generator.create_probability_chart(results['consensus_prediction'])
        else:
            fig = self.chart_generator.create_probability_chart(results['prediction'])
        
        st.pyplot(fig)
    
    def _render_pdf_button(self, results: Dict):
        """Renderizar botón de generación de PDF"""
        st.markdown("---")
        st.markdown("### 📄 Generar Reporte PDF")
        
        if st.button("📄 Generar Reporte PDF Comparativo", use_container_width=True):
            self._generate_pdf_report(results)
    
    def _generate_pdf_report(self, results: Dict):
        """Generar reporte PDF"""
        with st.spinner("⏳ Generando reporte PDF..."):
            try:
                pdf_path = self.pdf_generator.generate_report(
                    image=st.session_state.current_image,
                    disease_info=results['disease_info'],
                    confidence=results['confidence'],
                    probabilities=results['prediction'],
                    model_name=st.session_state.selected_model_file,
                    all_predictions=results.get('all_predictions'),
                    consensus_prediction=results.get('consensus_prediction')
                )
                
                if pdf_path:
                    self.pdf_generator.create_download_button(pdf_path)
                    
            except Exception as e:
                st.error(f"❌ Error al generar PDF: {str(e)}")
    
    def render_comparison_tab(self):
        """Renderizar pestaña de comparación de modelos"""
        from ..ui.model_comparison import ModelComparisonUI
        comparison_ui = ModelComparisonUI()
        comparison_ui.render()
    
    def render_footer(self):
        """Renderizar pie de página"""
        st.markdown("---")
        footer_content = """
        <div class='footer'>
            <h3>🌿 Sistema Experto de Diagnóstico</h3>
            <p>Desarrollado para la identificación temprana y el manejo efectivo de enfermedades en cultivos de caña de azúcar</p>
            <p style='color: #666; font-size: 0.9em; margin-top: 1rem;'>
                Utilizando inteligencia artificial y aprendizaje profundo para diagnósticos precisos
            </p>
        </div>
        """
        st.markdown(footer_content, unsafe_allow_html=True) 