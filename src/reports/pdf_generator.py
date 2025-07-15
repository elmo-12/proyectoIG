"""
Módulo para generación de reportes PDF
"""
import os
import time
import base64
import tempfile
from typing import Dict, Optional, Any
import numpy as np
from PIL import Image
import streamlit as st

from ..config.settings import PDF_CONFIG
from ..utils.text_utils import clean_text_robust
from ..visualization.charts import ChartGenerator
from ..models.model_manager import ModelManager

# Detectar bibliotecas PDF disponibles
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

class PDFReportGenerator:
    """Generador de reportes PDF para diagnósticos"""
    
    def __init__(self):
        self.config = PDF_CONFIG
        self.model_manager = ModelManager()
        self.chart_generator = ChartGenerator()
        
    def generate_report(self, 
                       image: Image.Image,
                       disease_info: Dict[str, Any],
                       confidence: float,
                       probabilities: np.ndarray,
                       model_name: str = None,
                       all_predictions: Dict[str, np.ndarray] = None,
                       consensus_prediction: np.ndarray = None) -> Optional[str]:
        """
        Generar reporte PDF del diagnóstico
        
        Args:
            image (PIL.Image): Imagen analizada
            disease_info (dict): Información de la enfermedad
            confidence (float): Nivel de confianza
            probabilities (np.ndarray): Probabilidades de predicción
            model_name (str): Nombre del modelo usado
            all_predictions (dict): Predicciones de múltiples modelos
            consensus_prediction (np.ndarray): Predicción de consenso
            
        Returns:
            str: Ruta del archivo PDF o None si hay error
        """
        try:
            pdf_path = None
            
            if REPORTLAB_AVAILABLE:
                pdf_path = self._generate_reportlab_pdf(
                    image, disease_info, confidence, probabilities,
                    model_name, all_predictions, consensus_prediction
                )
            elif FPDF_AVAILABLE:
                pdf_path = self._generate_fpdf_pdf(
                    image, disease_info, confidence, probabilities,
                    model_name, all_predictions, consensus_prediction
                )
            else:
                self._show_pdf_error()
                return None
            
            # Crear botón de descarga si se generó el PDF exitosamente
            if pdf_path:
                self.create_download_button(pdf_path)
                return pdf_path
            else:
                return None
                
        except Exception as e:
            st.error(f"❌ Error al generar PDF: {str(e)}")
            return None
    
    def _show_pdf_error(self):
        """Mostrar error y opciones de instalación de librerías PDF"""
        st.error("❌ No hay bibliotecas de PDF disponibles")
        st.info("💡 Instala una de las siguientes bibliotecas:")
        st.info("   - ReportLab (recomendado): `pip install reportlab==4.0.4`")
        st.info("   - FPDF: `pip install fpdf2`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Instalar ReportLab"):
                self._install_package("reportlab==4.0.4")
        with col2:
            if st.button("🔧 Instalar FPDF"):
                self._install_package("fpdf2")
    
    def _install_package(self, package_name: str):
        """Instalar paquete automáticamente"""
        with st.spinner(f"⏳ Instalando {package_name}..."):
            try:
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success(f"✅ {package_name} instalado exitosamente")
                    st.info("🔄 Reinicia la aplicación para usar la nueva biblioteca")
                else:
                    st.error(f"❌ Error instalando {package_name}: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def _generate_reportlab_pdf(self, 
                              image: Image.Image,
                              disease_info: Dict[str, Any],
                              confidence: float,
                              probabilities: np.ndarray,
                              model_name: str = None,
                              all_predictions: Dict[str, np.ndarray] = None,
                              consensus_prediction: np.ndarray = None) -> str:
        """Generar PDF usando ReportLab"""
        
        # Lista para mantener archivos temporales hasta el final
        temp_files = []
        
        # Crear documento
        out_path = self.config['output_filename']
        doc = SimpleDocTemplate(out_path, pagesize=letter)
        story = []
        
        # Obtener estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=self.config['font_sizes']['title'],
            spaceAfter=30,
            textColor=HexColor(self.config['colors']['primary']),
            alignment=1  # Centrado
        )
        
        # Título
        title = "🌿 Diagnóstico Comparativo de Caña de Azúcar" if all_predictions and len(all_predictions) > 1 else "🌿 Diagnóstico de Caña de Azúcar"
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        
        # Información del análisis
        self._add_analysis_info(story, styles, model_name, all_predictions)
        
        # Imagen analizada
        self._add_image_section(story, styles, image, temp_files)
        
        # Diagnóstico principal
        self._add_diagnosis_section(story, styles, disease_info, confidence)
        
        # Gráficos
        self._add_charts_section(story, styles, probabilities, all_predictions, consensus_prediction, temp_files)
        
        # Información técnica
        self._add_technical_info(story, styles, model_name)
        
        # Generar PDF
        doc.build(story)
        
        # Limpiar archivos temporales después de generar el PDF
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return out_path
    
    def _generate_fpdf_pdf(self,
                          image: Image.Image,
                          disease_info: Dict[str, Any],
                          confidence: float,
                          probabilities: np.ndarray,
                          model_name: str = None,
                          all_predictions: Dict[str, np.ndarray] = None,
                          consensus_prediction: np.ndarray = None) -> str:
        """Generar PDF usando FPDF"""
        
        # Lista para mantener archivos temporales hasta el final
        temp_files = []
        
        pdf = FPDF()
        pdf.add_page()
        
        # Encabezado
        self._add_fpdf_header(pdf, all_predictions)
        
        # Información de modelos
        self._add_fpdf_model_info(pdf, model_name, all_predictions)
        
        # Imagen
        self._add_fpdf_image(pdf, image, temp_files)
        
        # Diagnóstico
        self._add_fpdf_diagnosis(pdf, disease_info, confidence)
        
        # Tratamiento y prevención
        self._add_fpdf_treatment(pdf, disease_info)
        
        # Gráficos
        self._add_fpdf_charts(pdf, probabilities, all_predictions, consensus_prediction, temp_files)
        
        # Información técnica
        self._add_fpdf_technical_info(pdf, model_name)
        
        # Pie de página
        self._add_fpdf_footer(pdf)
        
        # Guardar PDF
        out_path = self.config['output_filename']
        pdf.output(out_path)
        
        # Limpiar archivos temporales después de generar el PDF
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return out_path
    
    def _add_analysis_info(self, story, styles, model_name, all_predictions):
        """Añadir información del análisis (ReportLab)"""
        if all_predictions and len(all_predictions) > 1:
            story.append(Paragraph("<b>Análisis con Múltiples Modelos:</b>", styles['Heading2']))
            story.append(Paragraph(f"<b>Modelos utilizados:</b> {len(all_predictions)} modelos", styles['Normal']))
            
            for i, (model_name_iter, prediction) in enumerate(all_predictions.items()):
                model_info = self.model_manager.load_model_info(model_name_iter)
                story.append(Paragraph(f"<b>Modelo {i+1} ({model_name_iter}):</b>", styles['Heading3']))
                if model_info:
                    story.append(Paragraph(f"  • Precisión: {model_info['test_accuracy']:.2%}", styles['Normal']))
        else:
            if model_name:
                story.append(Paragraph(f"<b>Modelo utilizado:</b> {model_name}", styles['Normal']))
                model_info = self.model_manager.load_model_info(model_name)
                if model_info:
                    story.append(Paragraph(f"<b>Precisión:</b> {model_info['test_accuracy']:.2%}", styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    def _add_image_section(self, story, styles, image, temp_files):
        """Añadir sección de imagen (ReportLab)"""
        story.append(Paragraph("<b>Imagen Analizada:</b>", styles['Heading2']))
        
        # Crear archivo temporal sin eliminarlo inmediatamente
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            image.save(tmp_img_file.name)
            story.append(ReportLabImage(tmp_img_file.name, width=4*inch, height=3*inch))
            temp_files.append(tmp_img_file.name)  # Agregar a la lista para limpiar después
        
        story.append(Spacer(1, 20))

    def _add_diagnosis_section(self, story, styles, disease_info, confidence):
        """Añadir sección de diagnóstico (ReportLab)"""
        story.append(Paragraph(f"<b>Diagnóstico:</b> {disease_info['name']}", styles['Heading2']))
        story.append(Paragraph(f"<b>Confianza:</b> {confidence:.1f}%", styles['Normal']))
        story.append(Paragraph(f"<b>Descripción:</b> {disease_info['description']}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Síntomas
        story.append(Paragraph("<b>Síntomas:</b>", styles['Heading3']))
        for symptom in disease_info['symptoms'][:3]:
            story.append(Paragraph(f"• {symptom}", styles['Normal']))
        
        story.append(Spacer(1, 15))
    
    def _add_charts_section(self, story, styles, probabilities, all_predictions, consensus_prediction, temp_files):
        """Añadir sección de gráficos (ReportLab)"""
        story.append(Paragraph("<b>Análisis Visual:</b>", styles['Heading2']))
        
        # Crear y guardar gráfico
        if consensus_prediction is not None:
            fig = self.chart_generator.create_probability_chart(
                consensus_prediction, 
                "Distribución de Probabilidades (Consenso)"
            )
        else:
            fig = self.chart_generator.create_probability_chart(
                probabilities, 
                "Distribución de Probabilidades"
            )
        
        # Crear archivo temporal sin eliminarlo inmediatamente
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_chart:
            fig.savefig(tmp_chart.name, bbox_inches='tight', dpi=150)
            temp_files.append(tmp_chart.name)  # Agregar a la lista para limpiar después
        
        # Cerrar la figura
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # Añadir al PDF
        story.append(ReportLabImage(tmp_chart.name, width=6*inch, height=4*inch))
        story.append(Spacer(1, 20))
    
    def _add_technical_info(self, story, styles, model_name):
        """Añadir información técnica (ReportLab)"""
        if model_name:
            model_info = self.model_manager.load_model_info(model_name)
            if model_info:
                story.append(Paragraph("<b>Información Técnica:</b>", styles['Heading2']))
                story.append(Paragraph(f"Precisión: {model_info['test_accuracy']:.2%}", styles['Normal']))
                story.append(Paragraph(f"Pérdida: {model_info['test_loss']:.4f}", styles['Normal']))
            
            # Matriz de confusión
            confusion_path = self.model_manager.get_confusion_matrix_path(model_name)
            if confusion_path and os.path.exists(confusion_path):
                story.append(Paragraph("<b>Matriz de Confusión:</b>", styles['Heading3']))
                story.append(ReportLabImage(confusion_path, width=6*inch, height=4*inch))
    
    def _add_fpdf_header(self, pdf, all_predictions):
        """Añadir encabezado FPDF"""
        if os.path.exists("logo.png"):
            pdf.image("logo.png", x=10, y=8, w=30, h=30)
        
        pdf.set_xy(45, 12)
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(46, 125, 50)
        
        title = "Diagnostico Comparativo de Cana de Azucar" if all_predictions and len(all_predictions) > 1 else "Diagnostico de Cana de Azucar"
        pdf.cell(0, 15, clean_text_robust(title), ln=1, align='C')
        pdf.ln(10)
        
        # Línea divisoria
        pdf.set_draw_color(46, 125, 50)
        pdf.set_line_width(1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
    
    def _add_fpdf_model_info(self, pdf, model_name, all_predictions):
        """Añadir información de modelos FPDF"""
        if all_predictions and len(all_predictions) > 1:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(33, 150, 243)
            pdf.cell(0, 8, clean_text_robust(f"Modelos utilizados: {len(all_predictions)}"), ln=1)
            pdf.ln(2)
        elif model_name:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(33, 150, 243)
            pdf.cell(0, 8, clean_text_robust(f"Modelo: {model_name}"), ln=1)
            pdf.ln(2)
    
    def _add_fpdf_image(self, pdf, image, temp_files):
        """Añadir imagen FPDF"""
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 4, clean_text_robust("Imagen Analizada:"), ln=1)
        
        # Crear archivo temporal sin eliminarlo inmediatamente
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            image.save(tmp_img_file.name)
            pdf.image(tmp_img_file.name, x=60, w=90, h=60)
            temp_files.append(tmp_img_file.name)  # Agregar a la lista para limpiar después
        
        pdf.ln(10)
    
    def _add_fpdf_diagnosis(self, pdf, disease_info, confidence):
        """Añadir diagnóstico FPDF"""
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(46, 125, 50)
        pdf.cell(0, 10, clean_text_robust(disease_info['name']), ln=1)
        
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 8, clean_text_robust(f"Confianza: {confidence:.1f}%"), ln=1)
        pdf.ln(2)
        
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, clean_text_robust(disease_info['description']))
        pdf.ln(5)
    
    def _add_fpdf_treatment(self, pdf, disease_info):
        """Añadir tratamiento FPDF"""
        # Síntomas
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(198, 40, 40)
        pdf.cell(0, 8, clean_text_robust("Sintomas:"), ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        
        for symptom in disease_info['symptoms'][:3]:
            pdf.cell(0, 6, clean_text_robust(f"- {symptom}"), ln=1)
        pdf.ln(3)
        
        # Tratamiento
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(33, 150, 243)
        pdf.cell(0, 8, clean_text_robust("Tratamiento:"), ln=1)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(33, 33, 33)
        
        for treatment in disease_info['treatment'][:3]:
            pdf.cell(0, 6, clean_text_robust(f"- {treatment}"), ln=1)
        pdf.ln(3)
    
    def _add_fpdf_charts(self, pdf, probabilities, all_predictions, consensus_prediction, temp_files):
        """Añadir gráficos FPDF"""
        # Gráfico de probabilidades
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(76, 175, 80)
        pdf.cell(0, 8, clean_text_robust("Analisis de Probabilidades:"), ln=1)
        
        # Crear y guardar gráfico
        if consensus_prediction is not None:
            fig = self.chart_generator.create_probability_chart(consensus_prediction)
        else:
            fig = self.chart_generator.create_probability_chart(probabilities)
        
        # Crear archivo temporal sin eliminarlo inmediatamente
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_chart:
            fig.savefig(tmp_chart.name, bbox_inches='tight', dpi=150)
            temp_files.append(tmp_chart.name)  # Agregar a la lista para limpiar después
        
        # Cerrar la figura
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # Añadir al PDF
        pdf.image(tmp_chart.name, x=25, w=160, h=80)
        pdf.ln(10)
    
    def _add_fpdf_technical_info(self, pdf, model_name):
        """Añadir información técnica FPDF"""
        if model_name:
            model_info = self.model_manager.load_model_info(model_name)
            if model_info:
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(76, 175, 80)
                pdf.cell(0, 8, clean_text_robust("Informacion Tecnica:"), ln=1)
                
                pdf.set_font("Arial", size=10)
                pdf.set_text_color(33, 33, 33)
                pdf.cell(0, 6, clean_text_robust(f"Precision: {model_info['test_accuracy']:.2%}"), ln=1)
                pdf.cell(0, 6, clean_text_robust(f"Perdida: {model_info['test_loss']:.4f}"), ln=1)
                pdf.ln(5)
    
    def _add_fpdf_footer(self, pdf):
        """Añadir pie de página FPDF"""
        pdf.set_y(-25)
        pdf.set_draw_color(46, 125, 50)
        pdf.set_line_width(0.7)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        
        pdf.set_font("Arial", size=8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8, clean_text_robust(f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}"), ln=1, align='C')
        pdf.cell(0, 6, clean_text_robust("Sistema de Diagnostico de Cana de Azucar"), ln=1, align='C')
    
    def create_download_button(self, pdf_path: str):
        """Crear botón de descarga para el PDF"""
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            
            # Limpiar archivo temporal
            try:
                os.remove(pdf_path)
            except:
                pass
            
            st.success("✅ Reporte PDF generado exitosamente")
            st.download_button(
                label="⬇️ Descargar Reporte PDF",
                data=base64.b64decode(b64),
                file_name=self.config['output_filename'],
                mime="application/pdf",
                use_container_width=True,
                help="Haz clic para descargar el reporte PDF del diagnóstico"
            ) 