"""
Información detallada sobre enfermedades de caña de azúcar
"""
from ..utils.i18n import t

# Información detallada de enfermedades y sus tratamientos
DISEASE_INFO = {
    0: {
        'name': 'Sana (Healthy)',
        'color': '#4CAF50',
        'description': 'La planta muestra signos de buena salud sin síntomas de enfermedad.',
        'symptoms': [
            'Hojas de color verde intenso y uniforme',
            'Crecimiento vigoroso y uniforme',
            'Ausencia de manchas, lesiones o decoloraciones',
            'Tallos firmes y bien desarrollados',
            'Estructura foliar normal'
        ],
        'treatment': [
            'Mantener el programa regular de fertilización',
            'Continuar con el riego adecuado',
            'Realizar monitoreos preventivos periódicos',
            'Mantener buenas prácticas agrícolas'
        ],
        'prevention': [
            'Uso de variedades resistentes',
            'Mantener buen drenaje del suelo',
            'Control de malezas',
            'Rotación de cultivos cuando sea posible'
        ],
        'icon': '✅',
        'severity': 'low',
        'category': 'healthy'
    },
    1: {
        'name': 'Mosaico (Mosaic)',
        'color': '#FF9800',
        'description': 'Enfermedad viral que causa patrones de mosaico en las hojas, reduciendo la fotosíntesis.',
        'symptoms': [
            'Patrones de mosaico verde claro y oscuro',
            'Manchas irregulares en las hojas',
            'Reducción del crecimiento de la planta',
            'Hojas con apariencia moteada',
            'Clorosis interveinal'
        ],
        'treatment': [
            'Eliminación inmediata de plantas infectadas',
            'Control de insectos vectores (pulgones)',
            'Uso de variedades resistentes',
            'Implementación de barreras físicas'
        ],
        'prevention': [
            'Control estricto de insectos vectores',
            'Uso de material de siembra certificado',
            'Desinfección de herramientas',
            'Manejo de malezas hospederas'
        ],
        'icon': '🟡',
        'severity': 'medium',
        'category': 'viral'
    },
    2: {
        'name': 'Pudrición Roja (Red Rot)',
        'color': '#F44336',
        'description': 'Enfermedad fúngica causada por Colletotrichum falcatum que afecta severamente el rendimiento.',
        'symptoms': [
            'Manchas rojas en las hojas y tallos',
            'Tejido interno rojizo en los tallos',
            'Marchitamiento de las hojas',
            'Pérdida de vigor en la planta',
            'Lesiones necróticas'
        ],
        'treatment': [
            'Aplicación de fungicida sistémico (carbendazim)',
            'Eliminación inmediata de plantas infectadas',
            'Mejora del drenaje del suelo',
            'Reducción del estrés por sequía'
        ],
        'prevention': [
            'Uso de variedades resistentes',
            'Tratamiento de esquejes antes de la siembra',
            'Manejo adecuado del agua',
            'Control de insectos vectores'
        ],
        'icon': '🔴',
        'severity': 'high',
        'category': 'fungal'
    },
    3: {
        'name': 'Roya (Rust)',
        'color': '#8D6E63',
        'description': 'Enfermedad fúngica que forma pústulas de color óxido en las hojas.',
        'symptoms': [
            'Pústulas de color óxido en el envés de las hojas',
            'Manchas amarillas en el haz de las hojas',
            'Defoliación prematura',
            'Reducción del área foliar fotosintética',
            'Clorosis generalizada'
        ],
        'treatment': [
            'Aplicación de fungicidas protectantes',
            'Mejora de la ventilación del cultivo',
            'Reducción de la densidad de siembra',
            'Eliminación de residuos infectados'
        ],
        'prevention': [
            'Uso de variedades resistentes',
            'Manejo adecuado de la fertilización',
            'Control de la humedad relativa',
            'Monitoreo temprano de síntomas'
        ],
        'icon': '🟠',
        'severity': 'medium',
        'category': 'fungal'
    },
    4: {
        'name': 'Amarillamiento (Yellow)',
        'color': '#FFEB3B',
        'description': 'Condición que puede ser causada por deficiencias nutricionales o estrés ambiental.',
        'symptoms': [
            'Amarillamiento generalizado de las hojas',
            'Clorosis interveinal',
            'Reducción del crecimiento',
            'Hojas con apariencia pálida',
            'Síntomas que progresan desde hojas viejas'
        ],
        'treatment': [
            'Análisis de suelo para identificar deficiencias',
            'Aplicación de fertilizantes específicos',
            'Corrección del pH del suelo',
            'Mejora del drenaje si es necesario'
        ],
        'prevention': [
            'Análisis regular de suelo',
            'Programa de fertilización balanceado',
            'Manejo adecuado del riego',
            'Monitoreo de pH del suelo'
        ],
        'icon': '💛',
        'severity': 'low',
        'category': 'nutritional'
    }
}

# Mapeo de categorías para análisis
DISEASE_CATEGORIES = {
    'healthy': 'Plantas Sanas',
    'viral': 'Enfermedades Virales',
    'fungal': 'Enfermedades Fúngicas',
    'nutritional': 'Deficiencias Nutricionales'
}

# Niveles de severidad
SEVERITY_LEVELS = {
    'low': {'name': 'Baja', 'color': '#4CAF50', 'priority': 1},
    'medium': {'name': 'Media', 'color': '#FF9800', 'priority': 2},
    'high': {'name': 'Alta', 'color': '#F44336', 'priority': 3}
}

def get_disease_info(disease_id):
    """
    Obtener información de una enfermedad específica
    
    Args:
        disease_id (int): ID de la enfermedad
        
    Returns:
        dict: Información de la enfermedad
    """
    return DISEASE_INFO.get(disease_id, None)

def get_all_diseases():
    """
    Obtener información de todas las enfermedades
    
    Returns:
        dict: Diccionario con toda la información de enfermedades
    """
    return DISEASE_INFO

def get_disease_names():
    """
    Obtener lista de nombres de enfermedades
    
    Returns:
        list: Lista de nombres de enfermedades
    """
    return [disease['name'] for disease in DISEASE_INFO.values()]

def get_disease_colors():
    """
    Obtener lista de colores para cada enfermedad
    
    Returns:
        list: Lista de colores hexadecimales
    """
    return [disease['color'] for disease in DISEASE_INFO.values()]

def get_diseases_by_category(category):
    """
    Obtener enfermedades por categoría
    
    Args:
        category (str): Categoría de enfermedad
        
    Returns:
        dict: Enfermedades de la categoría especificada
    """
    return {
        disease_id: disease_info 
        for disease_id, disease_info in DISEASE_INFO.items()
        if disease_info['category'] == category
    }

def get_diseases_by_severity(severity):
    """
    Obtener enfermedades por nivel de severidad
    
    Args:
        severity (str): Nivel de severidad
        
    Returns:
        dict: Enfermedades del nivel de severidad especificado
    """
    return {
        disease_id: disease_info 
        for disease_id, disease_info in DISEASE_INFO.items()
        if disease_info['severity'] == severity
    }

def get_treatment_recommendations(disease_id):
    """
    Obtener recomendaciones de tratamiento para una enfermedad
    
    Args:
        disease_id (int): ID de la enfermedad
        
    Returns:
        list: Lista de recomendaciones de tratamiento
    """
    disease_info = get_disease_info(disease_id)
    return disease_info['treatment'] if disease_info else []

def get_prevention_measures(disease_id):
    """
    Obtener medidas preventivas para una enfermedad
    
    Args:
        disease_id (int): ID de la enfermedad
        
    Returns:
        list: Lista de medidas preventivas
    """
    disease_info = get_disease_info(disease_id)
    return disease_info['prevention'] if disease_info else []

def is_healthy(disease_id):
    """
    Verificar si una enfermedad corresponde a planta sana
    
    Args:
        disease_id (int): ID de la enfermedad
        
    Returns:
        bool: True si es planta sana, False si es enfermedad
    """
    return disease_id == 0

def get_disease_info_translated(disease_id):
    """
    Obtener información traducida de una enfermedad específica
    
    Args:
        disease_id (int): ID de la enfermedad
        
    Returns:
        dict: Información traducida de la enfermedad
    """
    disease_info = get_disease_info(disease_id)
    if not disease_info:
        return None
        
    # Traducir información básica
    translated_info = {
        'name': t(f'diseases.{disease_info["category"]}.name'),
        'color': disease_info['color'],
        'description': t(f'diseases.{disease_info["category"]}.description'),
        'symptoms': [
            t(f'diseases.{disease_info["category"]}.symptoms.{i}')
            for i in range(len(disease_info['symptoms']))
        ],
        'treatment': [
            t(f'diseases.{disease_info["category"]}.treatment.{i}')
            for i in range(len(disease_info['treatment']))
        ],
        'prevention': [
            t(f'diseases.{disease_info["category"]}.prevention.{i}')
            for i in range(len(disease_info['prevention']))
        ],
        'icon': disease_info['icon'],
        'severity': disease_info['severity'],
        'category': disease_info['category']
    }
    
    return translated_info

def get_disease_names_translated():
    """
    Obtener nombres de enfermedades traducidos
    
    Returns:
        list: Lista de nombres de enfermedades traducidos
    """
    disease_names = []
    for disease_id in range(5):  # 0-4 para las 5 enfermedades
        disease_info = get_disease_info_translated(disease_id)
        if disease_info:
            disease_names.append(disease_info['name'])
        else:
            # Fallback a nombres originales
            original_info = get_disease_info(disease_id)
            disease_names.append(original_info['name'] if original_info else f"Clase {disease_id}")
    
    return disease_names 