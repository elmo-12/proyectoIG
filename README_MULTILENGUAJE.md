# Sistema Multilenguaje - Diagnóstico de Enfermedades en Caña de Azúcar

## 🌐 Descripción

Se ha implementado un sistema multilenguaje completo para la aplicación de diagnóstico de enfermedades en caña de azúcar. El sistema soporta **español** e **inglés** y permite cambiar el idioma de la interfaz dinámicamente.

## 📁 Estructura del Sistema

```
src/
├── translations/           # Archivos de traducciones
│   ├── es.json            # Traducciones en español
│   └── en.json            # Traducciones en inglés
├── utils/
│   └── i18n.py           # Sistema de internacionalización
├── config/
│   └── settings.py       # Configuración actualizada
├── data/
│   └── diseases.py       # Información de enfermedades traducida
└── ui/
    └── components.py     # Componentes UI actualizados
```

## 🚀 Características Implementadas

### ✅ Idiomas Soportados
- **Español (es)** - Idioma por defecto
- **English (en)** - Idioma secundario

### ✅ Elementos Traducidos
- **Interfaz principal**: Títulos, botones, etiquetas
- **Pestañas**: Configuración, Diagnóstico, Comparación
- **Información de enfermedades**: Nombres, descripciones, síntomas, tratamientos
- **Mensajes del sistema**: Errores, éxitos, advertencias
- **Reportes PDF**: Contenido generado en el idioma seleccionado

### ✅ Funcionalidades
- **Selector de idioma**: En la barra lateral con banderas
- **Cambio dinámico**: Sin necesidad de reiniciar la aplicación
- **Persistencia**: El idioma se mantiene durante la sesión
- **Fallback**: Si no hay traducción, usa español como respaldo

## 🛠️ Uso del Sistema

### Cambiar Idioma
1. Busca el selector "🌐 Idioma / Language" en la barra lateral
2. Selecciona el idioma deseado
3. La interfaz se actualizará automáticamente

### Para Desarrolladores

#### Agregar Nuevas Traducciones
1. Edita los archivos `src/translations/es.json` y `src/translations/en.json`
2. Usa la función `t()` en tu código:
```python
from src.utils.i18n import t

# Usar traducción
texto = t('clave.traduccion')

# Con variables
texto = t('mensaje.con.variable', nombre='Juan')
```

#### Agregar Nuevo Idioma
1. Crea un archivo `src/translations/[codigo].json`
2. Actualiza `src/utils/i18n.py` para incluir el nuevo idioma:
```python
self.supported_languages = {
    'es': {'name': 'Español', 'flag': '🇪🇸'},
    'en': {'name': 'English', 'flag': '🇺🇸'},
    'fr': {'name': 'Français', 'flag': '🇫🇷'}  # Nuevo idioma
}
```

## 📝 Estructura de Traducciones

Los archivos JSON están organizados por secciones:

```json
{
  "app": {
    "title": "Título de la aplicación",
    "loading": "Cargando"
  },
  "tabs": {
    "config": "Configuración",
    "diagnosis": "Diagnóstico"
  },
  "diseases": {
    "healthy": {
      "name": "Sana",
      "description": "Descripción...",
      "symptoms": ["Síntoma 1", "Síntoma 2"],
      "treatment": ["Tratamiento 1", "Tratamiento 2"]
    }
  }
}
```

## 🔧 Configuración

### Idioma por Defecto
Se puede cambiar en `src/config/settings.py`:
```python
I18N_CONFIG = {
    "default_language": "es",  # Cambiar a "en" para inglés por defecto
    # ...
}
```

### Agregar Más Idiomas
1. Crear archivo de traducción
2. Actualizar `supported_languages` en `i18n.py`
3. Añadir traducciones para todas las claves existentes

## 🎯 Beneficios

- **Accesibilidad**: Usuarios de diferentes idiomas pueden usar la aplicación
- **Mantenibilidad**: Sistema centralizado y fácil de mantener
- **Escalabilidad**: Fácil agregar nuevos idiomas
- **Profesionalismo**: Interfaz completa y coherente en múltiples idiomas

## 🐛 Resolución de Problemas

### Texto no Traducido
- Verificar que la clave existe en ambos archivos JSON
- Comprobar que la sintaxis JSON es correcta
- Revisar que se usa la función `t()` correctamente

### Error al Cambiar Idioma
- Reiniciar la aplicación
- Verificar que los archivos JSON no tienen errores de sintaxis
- Comprobar que el directorio `src/translations/` existe

## 🚀 Ejecutar la Aplicación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run main_app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 📊 Estado del Proyecto

✅ **Completado**: Sistema multilenguaje completamente funcional
✅ **Probado**: Interfaz en español e inglés
✅ **Documentado**: README y comentarios en código
✅ **Gráficos**: Títulos y etiquetas traducidos en todos los gráficos
✅ **Comparación**: Tabla de comparación de modelos completamente traducida
✅ **Reportes PDF**: Generación de reportes PDF en el idioma seleccionado
✅ **Enfermedades**: Información de enfermedades (síntomas, tratamientos) traducida

## 🌟 Características Completas

- **100% Traducido**: Todos los elementos de la interfaz están traducidos
- **Gráficos Multilenguaje**: Títulos, ejes y etiquetas en el idioma seleccionado
- **PDFs Traducidos**: Reportes generados completamente en el idioma seleccionado
- **Información Médica**: Nombres de enfermedades, síntomas y tratamientos traducidos
- **Comparación de Modelos**: Tabla de métricas y análisis completamente traducida
- **Selector Mejorado**: Botones de idioma con banderas en la barra lateral

## 🎯 Elementos Traducidos

### Interfaz Principal
- [x] Título de la aplicación
- [x] Pestañas principales
- [x] Mensajes de estado (éxito, error, advertencia)
- [x] Botones y controles

### Gráficos y Visualizaciones
- [x] Títulos de gráficos
- [x] Etiquetas de ejes
- [x] Leyendas
- [x] Nombres de enfermedades en gráficos
- [x] Tooltips y hover information

### Información Médica
- [x] Nombres de enfermedades
- [x] Descripciones de enfermedades
- [x] Listas de síntomas
- [x] Tratamientos recomendados
- [x] Medidas preventivas

### Comparación de Modelos
- [x] Encabezados de tabla
- [x] Métricas de rendimiento
- [x] Estados de modelos
- [x] Recomendaciones
- [x] Análisis estadístico

### Reportes PDF
- [x] Títulos de documento
- [x] Secciones de contenido
- [x] Información técnica
- [x] Gráficos embebidos
- [x] Botones de descarga

---

*Sistema desarrollado con Streamlit, Python y archivos JSON para traducciones* 