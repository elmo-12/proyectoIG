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
import matplotlib.pyplot as plt

from ..config.settings import PDF_CONFIG, get_temp_dir
from ..utils.text_utils import clean_text_robust
from ..visualization.charts import ChartGenerator
from ..models.model_manager import ModelManager
from ..utils.i18n import t

# Detectar bibliotecas PDF disponibles
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, colors
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
            # Asegurar que el directorio temporal existe y es accesible
            temp_dir = get_temp_dir()
            
            # Generar un nombre único para el archivo PDF usando el idioma actual
            timestamp = int(time.time())
            pdf_filename = f"{t('pdf.report_filename')}_{timestamp}.pdf"
            pdf_path = os.path.join(temp_dir, pdf_filename)
            
            if REPORTLAB_AVAILABLE:
                self.config['output_filename'] = pdf_path  # Actualizar la ruta del archivo
                pdf_path = self._generate_reportlab_pdf(
                    image, disease_info, confidence, probabilities,
                    model_name, all_predictions, consensus_prediction
                )
            elif FPDF_AVAILABLE:
                self.config['output_filename'] = pdf_path  # Actualizar la ruta del archivo
                pdf_path = self._generate_fpdf_pdf(
                    image, disease_info, confidence, probabilities,
                    model_name, all_predictions, consensus_prediction
                )
            else:
                self._show_pdf_error()
                return None
            
            # Verificar que el archivo se creó correctamente
            if pdf_path and os.path.exists(pdf_path):
                return pdf_path
            else:
                st.error(f"❌ {t('pdf.generation_failed')}")
                return None
                
        except Exception as e:
            st.error(f"❌ {t('pdf.generation_error')}: {str(e)}")
            return None
    
    def _show_pdf_error(self):
        """Mostrar error y opciones de instalación de librerías PDF"""
        st.error(f"❌ {t('config.no_pdf_libraries')}")
        st.info(f"💡 {t('pdf.install_libraries')}")
        st.info(f"   - ReportLab ({t('pdf.recommended')}): `pip install reportlab==4.0.4`")
        st.info(f"   - FPDF: `pip install fpdf2`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🔧 {t('pdf.install_reportlab')}"):
                self._install_package("reportlab==4.0.4")
        with col2:
            if st.button(f"🔧 {t('pdf.install_fpdf')}"):
                self._install_package("fpdf2")
    
    def _install_package(self, package_name: str):
        """Instalar paquete automáticamente"""
        with st.spinner(f"⏳ {t('pdf.installing', package=package_name)}"):
            try:
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success(f"✅ {t('pdf.install_success', package=package_name)}")
                    st.info(f"🔄 {t('pdf.restart_app')}")
                else:
                    st.error(f"❌ {t('pdf.install_error', package=package_name, error=result.stderr)}")
            except Exception as e:
                st.error(f"❌ {t('app.error')}: {str(e)}")
    
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
        title = f"🌿 {t('pdf.comparative_diagnosis_title')}" if all_predictions and len(all_predictions) > 1 else f"�� {t('pdf.diagnosis_title')}"
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
            story.append(Paragraph(f"<b>{t('pdf.multiple_models_analysis')}:</b>", styles['Heading2']))
            story.append(Paragraph(f"<b>{t('pdf.models_used')}:</b> {len(all_predictions)} {t('pdf.models')}", styles['Normal']))
            
            for i, (model_name_iter, prediction) in enumerate(all_predictions.items()):
                model_info = self.model_manager.load_model_info(model_name_iter)
                story.append(Paragraph(f"<b>{t('pdf.model_number', number=i+1)} ({model_name_iter}):</b>", styles['Heading3']))
                if model_info:
                    story.append(Paragraph(f"  • {t('pdf.accuracy')}: {model_info['test_accuracy']:.2%}", styles['Normal']))
        else:
            if model_name:
                story.append(Paragraph(f"<b>{t('pdf.model_used')}:</b> {model_name}", styles['Normal']))
                model_info = self.model_manager.load_model_info(model_name)
                if model_info:
                    story.append(Paragraph(f"<b>{t('pdf.accuracy')}:</b> {model_info['test_accuracy']:.2%}", styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    def _add_image_section(self, story, styles, image, temp_files):
        """Añadir sección de imagen (ReportLab)"""
        story.append(Paragraph(f"<b>{t('pdf.analyzed_image')}:</b>", styles['Heading2']))
        
        # Crear archivo temporal sin eliminarlo inmediatamente
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            image.save(tmp_img_file.name)
            story.append(ReportLabImage(tmp_img_file.name, width=4*inch, height=3*inch))
            temp_files.append(tmp_img_file.name)  # Agregar a la lista para limpiar después
        
        story.append(Spacer(1, 20))

    def _add_diagnosis_section(self, story, styles, disease_info, confidence):
        """Añadir sección de diagnóstico (ReportLab)"""
        story.append(Paragraph(f"<b>{t('pdf.diagnosis')}:</b> {disease_info['name']}", styles['Heading2']))
        story.append(Paragraph(f"<b>{t('pdf.confidence')}:</b> {confidence:.1f}%", styles['Normal']))
        story.append(Paragraph(f"<b>{t('pdf.description')}:</b> {disease_info['description']}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Síntomas
        story.append(Paragraph(f"<b>{t('pdf.symptoms')}:</b>", styles['Heading3']))
        for symptom in disease_info['symptoms'][:3]:
            story.append(Paragraph(f"• {symptom}", styles['Normal']))
        
        story.append(Spacer(1, 15))
    
    def _add_charts_section(self, story, styles, probabilities, all_predictions, consensus_prediction, temp_files):
        """Añadir sección de gráficos (ReportLab)"""
        story.append(Paragraph(f"<b>{t('pdf.visual_analysis')}:</b>", styles['Heading2']))
        
        try:
            # Gráfico de probabilidades
            if consensus_prediction is not None:
                fig = self.chart_generator.create_probability_chart_for_pdf(
                    consensus_prediction, 
                    t('pdf.probability_distribution_consensus')
                )
            else:
                fig = self.chart_generator.create_probability_chart_for_pdf(
                    probabilities, 
                    t('pdf.probability_distribution')
                )
            
            # Guardar gráfico de probabilidades
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_chart:
                fig.savefig(tmp_chart.name, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
                temp_files.append(tmp_chart.name)
                plt.close(fig)
                story.append(ReportLabImage(tmp_chart.name, width=6*inch, height=4*inch))
                story.append(Spacer(1, 20))
            
            # Gráfico comparativo si hay múltiples modelos
            if all_predictions and len(all_predictions) > 1:
                story.append(Paragraph(f"<b>{t('pdf.model_comparison')}:</b>", styles['Heading3']))
                fig_comp = self.chart_generator.create_comparative_chart_for_pdf(all_predictions)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_comp:
                    fig_comp.savefig(tmp_comp.name, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
                    temp_files.append(tmp_comp.name)
                    plt.close(fig_comp)
                    story.append(ReportLabImage(tmp_comp.name, width=7*inch, height=4*inch))
                    story.append(Spacer(1, 20))
            
            # Matriz de confusión
            model_name = st.session_state.get('selected_model_file', None)
            if model_name is None and all_predictions:
                # Si no hay modelo seleccionado pero hay predicciones, usar el primer modelo
                model_name = list(all_predictions.keys())[0]
            
            if model_name:
                confusion_path = self.model_manager.get_confusion_matrix_path(model_name)
                if confusion_path and os.path.exists(confusion_path):
                    story.append(PageBreak())  # Nueva página para la matriz
                    story.append(Paragraph(f"<b>{t('pdf.confusion_matrix')}</b>", styles['Heading3']))
                    story.append(ReportLabImage(confusion_path, width=6*inch, height=4*inch))
                    story.append(Spacer(1, 20))
        
        except Exception as e:
            st.error(f"Error generando gráficos: {str(e)}")
            plt.close('all')  # Cerrar todas las figuras en caso de error
            
        # Información técnica y métricas
        if model_name := st.session_state.get('selected_model_file'):
            model_info = self.model_manager.load_model_info(model_name)
            if model_info:
                story.append(Paragraph(f"<b>{t('pdf.model_metrics')}:</b>", styles['Heading3']))
                
                # Métricas principales
                metrics_style = ParagraphStyle(
                    'Metrics',
                    parent=styles['Normal'],
                    fontSize=10,
                    leading=14
                )
                story.append(Paragraph(f"Test Loss: {model_info['test_loss']:.4f}", metrics_style))
                story.append(Paragraph(f"Test Accuracy: {model_info['test_accuracy']:.4f}", metrics_style))
                story.append(Spacer(1, 10))
                
                # Tabla de clasificación
                if 'classification_report' in model_info:
                    report_style = ParagraphStyle(
                        'Report',
                        parent=styles['Code'],
                        fontSize=8,
                        leading=10,
                        fontName='Courier'
                    )
                    story.append(Paragraph(f"<b>{t('pdf.classification_report')}:</b>", styles['Heading4']))
                    story.append(Spacer(1, 5))
                    
                    # Crear tabla de clasificación
                    table_data = [
                        ['', 'precision', 'recall', 'f1-score', 'support'],
                    ]
                    
                    # Añadir datos de cada clase
                    for class_name in ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']:
                        if class_name in model_info['classification_report']:
                            metrics = model_info['classification_report'][class_name]
                            table_data.append([
                                class_name,
                                f"{metrics['precision']:.2f}",
                                f"{metrics['recall']:.2f}",
                                f"{metrics['f1-score']:.2f}",
                                str(metrics['support'])
                            ])
                    
                    # Añadir promedios
                    for avg_type in ['accuracy', 'macro avg', 'weighted avg']:
                        if avg_type in model_info['classification_report']:
                            metrics = model_info['classification_report'][avg_type]
                            row = [avg_type]
                            if avg_type == 'accuracy':
                                row.extend(['', '', f"{metrics:.2f}", str(model_info['classification_report']['support'])])
                            else:
                                row.extend([
                                    f"{metrics['precision']:.2f}",
                                    f"{metrics['recall']:.2f}",
                                    f"{metrics['f1-score']:.2f}",
                                    str(metrics['support'])
                                ])
                            table_data.append(row)
                    
                    # Crear y estilizar la tabla
                    table_style = TableStyle([
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('BOX', (0, 0), (-1, -1), 2, colors.black),
                        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
                    ])
                    
                    table = Table(table_data)
                    table.setStyle(table_style)
                    story.append(table)
                    story.append(Spacer(1, 20))

    def _add_technical_info(self, story, styles, model_name):
        """Añadir información técnica (ReportLab)"""
        if model_name:
            model_info = self.model_manager.load_model_info(model_name)
            if model_info:
                story.append(Paragraph(f"<b>{t('pdf.technical_info')}:</b>", styles['Heading2']))
                story.append(Paragraph(f"{t('pdf.accuracy')}: {model_info['test_accuracy']:.2%}", styles['Normal']))
                story.append(Paragraph(f"{t('pdf.loss')}: {model_info['test_loss']:.4f}", styles['Normal']))
    
    def _add_fpdf_header(self, pdf, all_predictions):
        """Añadir encabezado FPDF"""
        if os.path.exists("logo.png"):
            pdf.image("logo.png", x=10, y=8, w=30, h=30)
        
        pdf.set_xy(45, 12)
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(46, 125, 50)
        
        title = t('pdf.comparative_diagnosis_title') if all_predictions and len(all_predictions) > 1 else t('pdf.diagnosis_title')
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
        try:
            # Gráfico de probabilidades
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(76, 175, 80)
            pdf.cell(0, 8, clean_text_robust(t('pdf.probability_analysis')), ln=1)
            
            # Crear y guardar gráfico de probabilidades
            if consensus_prediction is not None:
                fig = self.chart_generator.create_probability_chart_for_pdf(consensus_prediction)
            else:
                fig = self.chart_generator.create_probability_chart_for_pdf(probabilities)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_chart:
                fig.savefig(tmp_chart.name, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
                temp_files.append(tmp_chart.name)
                plt.close(fig)
                pdf.image(tmp_chart.name, x=25, w=160, h=80)
                pdf.ln(10)
            
            # Gráfico comparativo si hay múltiples modelos
            if all_predictions and len(all_predictions) > 1:
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(76, 175, 80)
                pdf.cell(0, 8, clean_text_robust(t('pdf.model_comparison')), ln=1)
                
                fig_comp = self.chart_generator.create_comparative_chart_for_pdf(all_predictions)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_comp:
                    fig_comp.savefig(tmp_comp.name, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
                    temp_files.append(tmp_comp.name)
                    plt.close(fig_comp)
                    pdf.image(tmp_comp.name, x=15, w=180, h=90)
                    pdf.ln(10)
            
            # Matriz de confusión
            model_name = st.session_state.get('selected_model_file', None)
            if model_name is None and all_predictions:
                # Si no hay modelo seleccionado pero hay predicciones, usar el primer modelo
                model_name = list(all_predictions.keys())[0]
            
            if model_name:
                confusion_path = self.model_manager.get_confusion_matrix_path(model_name)
                if confusion_path and os.path.exists(confusion_path):
                    pdf.add_page()  # Nueva página para la matriz
                    pdf.set_font("Arial", 'B', 14)
                    pdf.set_text_color(46, 125, 50)
                    pdf.cell(0, 10, clean_text_robust(t('pdf.confusion_matrix')), ln=1)
                    pdf.ln(5)
                    pdf.image(confusion_path, x=20, w=170, h=120)
                    pdf.ln(10)
        
        except Exception as e:
            st.error(f"Error generando gráficos: {str(e)}")
            plt.close('all')  # Cerrar todas las figuras en caso de error
            
        # Información técnica y métricas
        model_name = st.session_state.get('selected_model_file', None)
        if model_name is None and all_predictions:
            # Si no hay modelo seleccionado pero hay predicciones, usar el primer modelo
            model_name = list(all_predictions.keys())[0]
            
        if model_name:
            model_info = self.model_manager.load_model_info(model_name)
            if model_info:
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(76, 175, 80)
                pdf.cell(0, 8, clean_text_robust(t('pdf.model_metrics')), ln=1)
                
                # Métricas principales
                pdf.set_font("Arial", '', 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 6, clean_text_robust(f"Test Loss: {model_info['test_loss']:.4f}"), ln=1)
                pdf.cell(0, 6, clean_text_robust(f"Test Accuracy: {model_info['test_accuracy']:.4f}"), ln=1)
                pdf.ln(5)
                
                # Tabla de clasificación
                if 'classification_report' in model_info:
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(0, 8, clean_text_robust(t('pdf.classification_report')), ln=1)
                    pdf.ln(2)
                    
                    # Encabezados de la tabla
                    pdf.set_font("Arial", 'B', 8)
                    col_width = 35
                    pdf.cell(col_width, 6, "Clase", 1, 0, 'C')
                    for header in ['Precision', 'Recall', 'F1-Score', 'Support']:
                        pdf.cell(col_width, 6, header, 1, 0, 'C')
                    pdf.ln()
                    
                    # Datos de la tabla
                    pdf.set_font("Arial", '', 8)
                    for class_name in ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']:
                        if class_name in model_info['classification_report']:
                            metrics = model_info['classification_report'][class_name]
                            pdf.cell(col_width, 6, class_name, 1, 0, 'L')
                            pdf.cell(col_width, 6, f"{metrics['precision']:.2f}", 1, 0, 'C')
                            pdf.cell(col_width, 6, f"{metrics['recall']:.2f}", 1, 0, 'C')
                            pdf.cell(col_width, 6, f"{metrics['f1-score']:.2f}", 1, 0, 'C')
                            pdf.cell(col_width, 6, str(metrics['support']), 1, 1, 'C')
                    
                    # Promedios
                    for avg_type in ['accuracy', 'macro avg', 'weighted avg']:
                        if avg_type in model_info['classification_report']:
                            metrics = model_info['classification_report'][avg_type]
                            pdf.cell(col_width, 6, avg_type, 1, 0, 'L')
                            if avg_type == 'accuracy':
                                pdf.cell(col_width, 6, '', 1, 0, 'C')
                                pdf.cell(col_width, 6, '', 1, 0, 'C')
                                pdf.cell(col_width, 6, f"{metrics:.2f}", 1, 0, 'C')
                                pdf.cell(col_width, 6, str(model_info['classification_report']['support']), 1, 1, 'C')
                            else:
                                pdf.cell(col_width, 6, f"{metrics['precision']:.2f}", 1, 0, 'C')
                                pdf.cell(col_width, 6, f"{metrics['recall']:.2f}", 1, 0, 'C')
                                pdf.cell(col_width, 6, f"{metrics['f1-score']:.2f}", 1, 0, 'C')
                                pdf.cell(col_width, 6, str(metrics['support']), 1, 1, 'C')
                    pdf.ln(10)
                    
                    # Matriz de confusión
                    confusion_matrix_path = self.model_manager.get_confusion_matrix_path(model_name)
                    if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                        try:
                            # Intentar cargar la imagen con PIL primero para verificar que es válida
                            from PIL import Image
                            confusion_img = Image.open(confusion_matrix_path)
                            
                            # Si la imagen se carga correctamente, agregarla al PDF
                            pdf.add_page()  # Añadir nueva página para la matriz
                            pdf.set_font("Arial", 'B', 14)
                            pdf.set_text_color(76, 175, 80)
                            pdf.cell(0, 10, clean_text_robust(t('pdf.confusion_matrix')), ln=1, align='C')
                            pdf.ln(5)
                            
                            # Crear una copia temporal de la imagen
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_confusion:
                                confusion_img.save(tmp_confusion.name, format='PNG')
                                temp_files.append(tmp_confusion.name)
                                
                                # Calcular dimensiones para centrar la imagen
                                page_width = pdf.w - 2 * pdf.l_margin
                                image_width = page_width * 0.9  # 90% del ancho de la página
                                image_height = image_width * 0.75  # Mantener proporción
                                x = (pdf.w - image_width) / 2
                                
                                pdf.image(tmp_confusion.name, x=x, w=image_width, h=image_height)
                                pdf.ln(10)
                        except Exception as img_error:
                            st.error(f"Error al cargar la matriz de confusión: {str(img_error)}")
    
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
            
            st.success(f"✅ {t('pdf.pdf_generated_success')}")
            st.download_button(
                label=f"⬇️ {t('pdf.download_report')}",
                data=base64.b64decode(b64),
                file_name=self.config['output_filename'],
                mime="application/pdf",
                use_container_width=True,
                help=t('pdf.download_help')
            ) 