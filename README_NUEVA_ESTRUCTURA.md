# 🌿 Sistema Experto de Diagnóstico de Caña de Azúcar - Estructura Modular

## 📁 Estructura del Proyecto

```
proyectoIG/
├── src/                           # Código fuente modular
│   ├── __init__.py               # Inicialización del paquete
│   ├── config/                   # Configuración y constantes
│   │   ├── __init__.py
│   │   └── settings.py           # Configuraciones centralizadas
│   ├── data/                     # Datos y modelos de dominio
│   │   ├── __init__.py
│   │   └── diseases.py           # Información de enfermedades
│   ├── models/                   # Gestión de modelos ML
│   │   ├── __init__.py
│   │   └── model_manager.py      # Carga y gestión de modelos
│   ├── services/                 # Servicios de negocio
│   │   ├── __init__.py
│   │   ├── diagnosis_service.py  # Lógica de diagnóstico
│   │   └── image_processor.py    # Procesamiento de imágenes
│   ├── visualization/            # Generación de gráficos
│   │   ├── __init__.py
│   │   └── charts.py             # Generador de gráficos
│   ├── reports/                  # Generación de reportes
│   │   ├── __init__.py
│   │   └── pdf_generator.py      # Generador de PDF
│   ├── ui/                       # Interfaz de usuario
│   │   ├── __init__.py
│   │   ├── components.py         # Componentes UI principales
│   │   ├── styles.py             # Estilos CSS
│   │   └── model_comparison.py   # UI de comparación de modelos
│   └── utils/                    # Utilidades auxiliares
│       ├── __init__.py
│       └── text_utils.py         # Utilidades de texto
├── models/                       # Modelos ML (.keras)
├── info/                         # Información de modelos
├── main_app.py                   # Aplicación principal modular
├── app.py                        # Aplicación original (backup)
├── requirements.txt              # Dependencias
├── Dockerfile                    # Configuración Docker
├── docker-compose.yml           # Configuración Docker Compose
└── README_NUEVA_ESTRUCTURA.md   # Esta documentación
```

## 🏗️ Arquitectura Modular

### 1. **Configuración (`src/config/`)**
- **`settings.py`**: Configuraciones centralizadas, constantes y configuración de la aplicación
- Gestión de directorios, configuración de modelos, UI, PDF, etc.

### 2. **Datos (`src/data/`)**
- **`diseases.py`**: Información completa sobre enfermedades, síntomas, tratamientos y prevención
- Funciones para consultar información de enfermedades por categoría, severidad, etc.

### 3. **Modelos (`src/models/`)**
- **`model_manager.py`**: Gestión completa de modelos ML
- Carga, validación, información estadística y manejo de errores de compatibilidad

### 4. **Servicios (`src/services/`)**
- **`diagnosis_service.py`**: Lógica de diagnóstico, consenso de múltiples modelos
- **`image_processor.py`**: Procesamiento de imágenes con CLAHE, normalización y validación

### 5. **Visualización (`src/visualization/`)**
- **`charts.py`**: Generación de gráficos con matplotlib y plotly
- Charts de probabilidades, comparativos, gauges de confianza, etc.

### 6. **Reportes (`src/reports/`)**
- **`pdf_generator.py`**: Generación de reportes PDF con ReportLab y FPDF
- Soporte para reportes comparativos de múltiples modelos

### 7. **Interfaz de Usuario (`src/ui/`)**
- **`components.py`**: Componentes UI principales y lógica de renderizado
- **`styles.py`**: Estilos CSS personalizados y funciones de UI
- **`model_comparison.py`**: Interfaz específica para comparación de modelos

### 8. **Utilidades (`src/utils/`)**
- **`text_utils.py`**: Utilidades de procesamiento de texto y limpieza

## 🚀 Cómo Usar la Nueva Estructura

### Ejecutar la Aplicación Modular

```bash
# Aplicación modular (recomendada)
streamlit run main_app.py

# Aplicación original (backup)
streamlit run app.py
```

### Importar Módulos

```python
# Configuración
from src.config.settings import APP_CONFIG, MODEL_CONFIG

# Datos
from src.data.diseases import get_disease_info, get_all_diseases

# Servicios
from src.services.diagnosis_service import diagnosis_service
from src.services.image_processor import image_processor

# Modelos
from src.models.model_manager import model_manager

# Visualización
from src.visualization.charts import chart_generator

# Reportes
from src.reports.pdf_generator import pdf_generator
```

## 🎯 Beneficios de la Estructura Modular

### 1. **Mantenibilidad**
- Código organizado en módulos específicos
- Separación clara de responsabilidades
- Fácil localización de funcionalidades

### 2. **Escalabilidad**
- Módulos independientes y reutilizables
- Fácil adición de nuevas funcionalidades
- Arquitectura preparada para crecimiento

### 3. **Testabilidad**
- Cada módulo puede ser probado independientemente
- Dependencias claras y manejables
- Código más fácil de debuggear

### 4. **Reutilización**
- Componentes reutilizables entre diferentes partes
- Servicios centralizados
- Configuración unificada

## 📦 Principales Componentes

### ConfigManager (`src/config/settings.py`)
```python
# Configuración centralizada
APP_CONFIG = {
    "title": "Sistema Experto de Diagnóstico...",
    "page_title": "Diagnóstico Caña de Azúcar",
    # ...
}

# Funciones útiles
ensure_directories()
initialize_session_state()
get_available_models()
```

### DiagnosisService (`src/services/diagnosis_service.py`)
```python
# Servicio principal de diagnóstico
diagnosis_service.analyze_image(image)
diagnosis_service.predict_with_multiple_models(image)
diagnosis_service.get_consensus_prediction(predictions)
```

### ModelManager (`src/models/model_manager.py`)
```python
# Gestión de modelos
model_manager.load_model(model_path)
model_manager.load_all_models()
model_manager.load_model_info(model_name)
```

### ChartGenerator (`src/visualization/charts.py`)
```python
# Generación de gráficos
chart_generator.create_probability_chart(prediction)
chart_generator.create_comparative_chart(predictions)
chart_generator.create_confidence_gauge(confidence)
```

### PDFGenerator (`src/reports/pdf_generator.py`)
```python
# Generación de reportes PDF
pdf_generator.generate_report(image, disease_info, confidence, ...)
pdf_generator.create_download_button(pdf_path)
```

## 🔧 Configuración

### Variables de Entorno
```python
# TensorFlow
TF_CPP_MIN_LOG_LEVEL = "3"

# Directorios
MODEL_DIR = "models"
INFO_DIR = "info"
TEMP_DIR = "temp"
```

### Dependencias
Las mismas que en `requirements.txt`:
- streamlit
- tensorflow
- opencv-python
- matplotlib
- plotly
- reportlab/fpdf2
- pandas
- numpy
- pillow

## 🐳 Docker

La configuración de Docker funciona igual:

```bash
# Construir imagen
docker build -t sugarcane-app .

# Ejecutar contenedor
docker run -p 8123:8123 sugarcane-app

# Docker Compose
docker-compose up
```

## 🔄 Migración desde Aplicación Original

### Equivalencias de Funciones

| Función Original | Nuevo Módulo |
|------------------|--------------|
| `load_model()` | `model_manager.load_model()` |
| `preprocess_image()` | `image_processor.preprocess_image()` |
| `predict_disease()` | `diagnosis_service.predict_disease()` |
| `generate_pdf_report()` | `pdf_generator.generate_report()` |
| `create_probability_chart()` | `chart_generator.create_probability_chart()` |
| `DISEASE_INFO` | `diseases.get_all_diseases()` |

### Pasos de Migración

1. **Backup**: La aplicación original `app.py` se mantiene como backup
2. **Gradual**: Puedes usar ambas aplicaciones en paralelo
3. **Testeo**: Prueba la nueva estructura con `main_app.py`
4. **Migración**: Cuando esté listo, usa solo la nueva estructura

## 🧪 Testing

### Estructura de Tests (Recomendada)
```
tests/
├── test_config/
├── test_data/
├── test_models/
├── test_services/
├── test_visualization/
├── test_reports/
└── test_ui/
```

### Ejemplo de Test
```python
import pytest
from src.services.diagnosis_service import diagnosis_service
from src.models.model_manager import model_manager

def test_diagnosis_service():
    # Test del servicio de diagnóstico
    assert diagnosis_service is not None
    
def test_model_manager():
    # Test del gestor de modelos
    available_models = model_manager.get_available_models()
    assert isinstance(available_models, list)
```

## 📈 Próximos Pasos

### Mejoras Sugeridas

1. **API REST**: Convertir en API con FastAPI
2. **Base de Datos**: Integrar almacenamiento persistente
3. **Autenticación**: Sistema de usuarios y roles
4. **Métricas**: Logging y monitoreo avanzado
5. **CI/CD**: Pipeline de integración continua
6. **Tests**: Suite completa de pruebas unitarias

### Nuevas Funcionalidades

1. **Análisis Histórico**: Tracking de diagnósticos
2. **Alertas**: Notificaciones automáticas
3. **Exportación**: Múltiples formatos (Excel, JSON)
4. **Integración**: APIs externas de agricultura
5. **Mobile**: Aplicación móvil complementaria

## 🤝 Contribución

### Estructura para Desarrolladores

1. **Fork** el proyecto
2. **Crear** rama para nueva funcionalidad
3. **Seguir** la estructura modular establecida
4. **Añadir** tests para nuevo código
5. **Documentar** cambios en README
6. **Crear** Pull Request

### Convenciones de Código

- **Python**: PEP 8
- **Docstrings**: Google Style
- **Imports**: Absolutos dentro del paquete
- **Nombres**: Descriptivos en español/inglés
- **Comentarios**: Español para lógica de negocio

## 📞 Soporte

Para dudas sobre la nueva estructura:

1. **Documentación**: Este README
2. **Código**: Comentarios en cada módulo
3. **Ejemplos**: Ver `main_app.py` y módulos
4. **Issues**: Crear issue en el repositorio

---

## 🎉 Conclusión

La nueva estructura modular proporciona:

- ✅ **Mejor organización** del código
- ✅ **Mayor mantenibilidad** a largo plazo
- ✅ **Facilidad para testing** y debugging
- ✅ **Preparación para escalar** el proyecto
- ✅ **Reutilización** de componentes
- ✅ **Separación clara** de responsabilidades

¡La aplicación mantiene toda su funcionalidad original pero ahora es más robusta, mantenible y escalable! 🚀 