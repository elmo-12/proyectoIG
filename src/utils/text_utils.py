"""
Utilidades para procesamiento de texto
"""
import unicodedata
import re
from typing import Any, Optional

def clean_text_robust(text: Any) -> str:
    """
    Limpia el texto de forma robusta eliminando caracteres problemáticos
    
    Args:
        text: Texto a limpiar (puede ser cualquier tipo)
        
    Returns:
        str: Texto limpio
    """
    # Convertir a string si no lo es
    if not isinstance(text, str):
        text = str(text)
    
    # Normalizar y eliminar acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Eliminar cualquier carácter que no sea ASCII básico
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Reemplazar caracteres específicos
    replacements = {
        '°': 'o', '–': '-', '—': '-', ''': "'", ''': "'", 
        '"': '"', '"': '"', '…': '...', '®': '(R)', '©': '(C)',
        '¿': '?', '¡': '!'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Filtro final: solo caracteres ASCII imprimibles
    return ''.join(c for c in text if ord(c) < 128 and (c.isprintable() or c.isspace()))

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formatear valor como porcentaje
    
    Args:
        value (float): Valor a formatear
        decimals (int): Número de decimales
        
    Returns:
        str: Porcentaje formateado
    """
    return f"{value:.{decimals}f}%"

def format_confidence_level(confidence: float) -> str:
    """
    Formatear nivel de confianza con descripción
    
    Args:
        confidence (float): Nivel de confianza (0-100)
        
    Returns:
        str: Descripción del nivel de confianza
    """
    if confidence >= 90:
        return f"{confidence:.1f}% (Muy Alta)"
    elif confidence >= 80:
        return f"{confidence:.1f}% (Alta)"
    elif confidence >= 70:
        return f"{confidence:.1f}% (Media)"
    elif confidence >= 60:
        return f"{confidence:.1f}% (Baja)"
    else:
        return f"{confidence:.1f}% (Muy Baja)"

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncar texto a una longitud máxima
    
    Args:
        text (str): Texto a truncar
        max_length (int): Longitud máxima
        suffix (str): Sufijo a añadir si se trunca
        
    Returns:
        str: Texto truncado
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def capitalize_first_letter(text: str) -> str:
    """
    Capitalizar primera letra de cada palabra
    
    Args:
        text (str): Texto a capitalizar
        
    Returns:
        str: Texto capitalizado
    """
    return ' '.join(word.capitalize() for word in text.split())

def clean_filename(filename: str) -> str:
    """
    Limpiar nombre de archivo eliminando caracteres no válidos
    
    Args:
        filename (str): Nombre de archivo
        
    Returns:
        str: Nombre de archivo limpio
    """
    # Eliminar caracteres no válidos para nombres de archivo
    invalid_chars = r'[<>:"/\\|?*]'
    filename = re.sub(invalid_chars, '_', filename)
    
    # Eliminar espacios múltiples
    filename = re.sub(r'\s+', '_', filename)
    
    # Eliminar puntos al inicio y final
    filename = filename.strip('.')
    
    return filename 