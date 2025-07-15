"""
Estilos CSS para la interfaz de usuario
"""
import streamlit as st
from ..config.settings import UI_CONFIG

def apply_custom_styles():
    """Aplicar estilos CSS personalizados"""
    theme = UI_CONFIG['theme']
    
    css_styles = f"""
    <style>
        /* Tema oscuro general */
        .main {{
            background-color: {theme['background_color']};
            color: {theme['text_color']};
        }}
        
        /* Estilo para contenedores */
        .stButton>button {{
            width: 100%;
            background-color: {theme['primary_color']};
            color: white;
            padding: 0.75rem;
            border-radius: 10px;
            border: none;
            font-size: 1.1em;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {theme['secondary_color']};
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }}
        
        /* Cajas de diagnóstico */
        .diagnosis-box {{
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            background-color: #1E1E1E;
        }}
        .diagnosis-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.3);
        }}
        .healthy {{
            background-color: rgba(76, 175, 80, 0.2);
            border: 2px solid #4CAF50;
        }}
        .disease {{
            background-color: rgba(244, 67, 54, 0.2);
            border: 2px solid #F44336;
        }}
        
        /* Tarjetas de información */
        .info-card {{
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            border: 1px solid #333333;
        }}
        
        /* Textos y encabezados */
        h1 {{
            color: {theme['text_color']};
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5em;
            font-weight: 700;
            padding: 1rem;
            background: linear-gradient(90deg, #1E1E1E, #2E2E2E);
            border-radius: 10px;
        }}
        h2 {{
            color: {theme['secondary_color']};
            margin-top: 2rem;
            font-weight: 600;
        }}
        h3 {{
            color: #81C784;
            margin-top: 1.5rem;
        }}
        
        /* Contenedor de métricas */
        .metric-container {{
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin: 1rem 0;
            border: 1px solid #333333;
        }}
        
        /* Pie de página */
        .footer {{
            text-align: center;
            padding: 2rem;
            background-color: #1E1E1E;
            margin-top: 3rem;
            border-top: 1px solid #333333;
        }}
        
        /* Listas */
        .info-list {{
            list-style-type: none;
            padding: 0;
        }}
        .info-list li {{
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            background-color: #252525;
            border-radius: 5px;
            border-left: 4px solid {theme['secondary_color']};
        }}
        
        /* Separadores */
        hr {{
            border-color: #333333;
            margin: 2rem 0;
        }}
        
        /* Contenedor de imágenes */
        .image-container {{
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #333333;
        }}
        
        /* Tooltip personalizado */
        .tooltip {{
            position: relative;
            display: inline-block;
        }}
        .tooltip:hover::after {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.5rem;
            background-color: #333333;
            color: white;
            border-radius: 5px;
            font-size: 0.9em;
            white-space: nowrap;
        }}
        
        /* Ajustes para el modo oscuro de Streamlit */
        .stSelectbox, .stTextInput {{
            background-color: #1E1E1E;
        }}
        
        /* Alertas personalizadas */
        .alert {{
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid;
        }}
        .alert-success {{
            background-color: rgba(76, 175, 80, 0.1);
            border-color: #4CAF50;
            color: #4CAF50;
        }}
        .alert-warning {{
            background-color: rgba(255, 152, 0, 0.1);
            border-color: #FF9800;
            color: #FF9800;
        }}
        .alert-error {{
            background-color: rgba(244, 67, 54, 0.1);
            border-color: #F44336;
            color: #F44336;
        }}
        
        /* Barra de progreso */
        .progress-bar {{
            background-color: #333333;
            border-radius: 10px;
            overflow: hidden;
            height: 10px;
            margin: 0.5rem 0;
        }}
        .progress-fill {{
            height: 100%;
            background-color: {theme['secondary_color']};
            transition: width 0.3s ease;
        }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            font-size: 0.8em;
            font-weight: 600;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .badge-success {{
            background-color: #4CAF50;
            color: white;
        }}
        .badge-warning {{
            background-color: #FF9800;
            color: white;
        }}
        .badge-error {{
            background-color: #F44336;
            color: white;
        }}
        
        /* Spinner personalizado */
        .spinner {{
            border: 4px solid #333333;
            border-top: 4px solid {theme['secondary_color']};
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Animaciones */
        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .slide-up {{
            animation: slideUp 0.3s ease-out;
        }}
        
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
    </style>
    """
    
    st.markdown(css_styles, unsafe_allow_html=True)

def create_diagnosis_box(content: str, box_type: str = "healthy") -> str:
    """
    Crear caja de diagnóstico con estilos
    
    Args:
        content (str): Contenido HTML
        box_type (str): Tipo de caja ('healthy' o 'disease')
        
    Returns:
        str: HTML con estilos aplicados
    """
    return f"""
    <div class='diagnosis-box {box_type} fade-in'>
        {content}
    </div>
    """

def create_info_card(content: str, card_type: str = "info") -> str:
    """
    Crear tarjeta de información con estilos
    
    Args:
        content (str): Contenido de la tarjeta
        card_type (str): Tipo de tarjeta ('info', 'warning', 'error')
        
    Returns:
        str: HTML con estilos aplicados
    """
    return f"""
    <div class='info-card {card_type}'>
        <p style='margin: 0; font-size: 1.1em;'>{content}</p>
    </div>
    """

def create_metric_container(metrics: list) -> str:
    """
    Crear contenedor de métricas
    
    Args:
        metrics (list): Lista de métricas [(label, value), ...]
        
    Returns:
        str: HTML del contenedor
    """
    metrics_html = ""
    for label, value in metrics:
        metrics_html += f"""
        <div style='margin: 0.5rem 0;'>
            <strong>{label}:</strong> {value}
        </div>
        """
    
    return f"""
    <div class='metric-container'>
        {metrics_html}
    </div>
    """

def create_alert(message: str, alert_type: str = "success") -> str:
    """
    Crear alerta personalizada
    
    Args:
        message (str): Mensaje de la alerta
        alert_type (str): Tipo de alerta ('success', 'warning', 'error')
        
    Returns:
        str: HTML de la alerta
    """
    return f"""
    <div class='alert alert-{alert_type}'>
        {message}
    </div>
    """

def create_progress_bar(percentage: float, color: str = None) -> str:
    """
    Crear barra de progreso
    
    Args:
        percentage (float): Porcentaje de progreso (0-100)
        color (str): Color personalizado
        
    Returns:
        str: HTML de la barra de progreso
    """
    if color:
        style = f"background-color: {color};"
    else:
        style = ""
    
    return f"""
    <div class='progress-bar'>
        <div class='progress-fill' style='width: {percentage}%; {style}'></div>
    </div>
    """

def create_badge(text: str, badge_type: str = "success") -> str:
    """
    Crear badge/etiqueta
    
    Args:
        text (str): Texto del badge
        badge_type (str): Tipo de badge ('success', 'warning', 'error')
        
    Returns:
        str: HTML del badge
    """
    return f"""
    <span class='badge badge-{badge_type}'>{text}</span>
    """

def create_spinner(text: str = "Cargando...") -> str:
    """
    Crear spinner de carga
    
    Args:
        text (str): Texto debajo del spinner
        
    Returns:
        str: HTML del spinner
    """
    return f"""
    <div style='text-align: center; margin: 2rem 0;'>
        <div class='spinner'></div>
        <p>{text}</p>
    </div>
    """ 