"""
Aplicación principal del Sistema Experto de Diagnóstico de Enfermedades en Caña de Azúcar
"""
import streamlit as st
from src.config.settings import APP_CONFIG, initialize_session_state, ensure_directories
from src.ui.components import UIComponents
from src.utils.i18n import t

def main():
    """Función principal de la aplicación"""
    # Configurar página
    st.set_page_config(
        page_title=APP_CONFIG["page_title"],
        page_icon=APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
    )
    
    # Inicializar directorios y estado de sesión
    ensure_directories()
    initialize_session_state()
    
    # Crear instancia de componentes UI
    ui = UIComponents()
    
    # Renderizar selector de idioma primero
    ui.render_language_selector()
    
    # Renderizar encabezado
    ui.render_header()
    
    # Crear pestañas principales
    tab1, tab2, tab3 = st.tabs([
        t("tabs.config"),
        t("tabs.diagnosis"), 
        t("tabs.comparison")
    ])
    
    # Renderizar contenido de cada pestaña
    with tab1:
        ui.render_configuration_tab()
    
    with tab2:
        ui.render_diagnosis_tab()
    
    with tab3:
        ui.render_comparison_tab()
    
    # Renderizar pie de página
    ui.render_footer()

if __name__ == "__main__":
    main() 