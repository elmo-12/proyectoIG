# 🌿 Sistema Experto de Diagnóstico de Enfermedades en Caña de Azúcar

## 📝 Descripción

Sistema experto basado en inteligencia artificial para el diagnóstico de enfermedades en cultivos de caña de azúcar. Utiliza modelos de deep learning para analizar imágenes y detectar diferentes patologías, proporcionando diagnósticos precisos y recomendaciones de tratamiento.

## ✨ Características Principales

- 🔍 Diagnóstico automático de enfermedades mediante análisis de imágenes
- 🌐 Soporte multilenguaje (Español, English, Français, Português)
- 📊 Comparación de múltiples modelos de IA
- 📱 Interfaz web responsive y amigable
- 📄 Generación de reportes PDF detallados
- 🔄 Procesamiento avanzado de imágenes con CLAHE
- 📈 Visualización de resultados con gráficos interactivos

## 🔬 Enfermedades Detectadas

- ✅ Plantas Sanas (Healthy)
- 🟡 Mosaico (Mosaic)
- 🔴 Pudrición Roja (Red Rot)
- 🟠 Roya (Rust)
- 💛 Amarillamiento (Yellow)

## 🛠️ Tecnologías Utilizadas

- **Frontend**: Streamlit
- **Backend**: Python
- **IA/ML**: TensorFlow, Keras
- **Procesamiento de Imágenes**: OpenCV, Pillow
- **Visualización**: Matplotlib, Plotly
- **Reportes**: ReportLab, FPDF
- **Containerización**: Docker

## 📁 Estructura del Proyecto

```
proyectoIG/
├── src/                           # Código fuente modular
│   ├── config/                   # Configuración y constantes
│   ├── data/                     # Datos y modelos de dominio
│   ├── models/                   # Gestión de modelos ML
│   ├── services/                 # Servicios de negocio
│   ├── visualization/            # Generación de gráficos
│   ├── reports/                  # Generación de reportes
│   ├── ui/                       # Interfaz de usuario
│   ├── utils/                    # Utilidades auxiliares
│   └── translations/             # Archivos de idiomas
├── models/                       # Modelos ML entrenados
├── info/                        # Documentación de modelos
├── temp/                        # Archivos temporales
├── main_app.py                  # Aplicación principal
├── requirements.txt             # Dependencias
└── Dockerfile                   # Configuración Docker
```

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Git

### Instalación Local

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd proyectoIG
```

2. Crear y activar entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Unix o MacOS:
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación:
```bash
streamlit run main_app.py
```

### Usando Docker

1. Construir la imagen:
```bash
docker build -t sugarcane-app .
```

2. Ejecutar el contenedor:
```bash
docker run -p 8123:8123 sugarcane-app
```

O usando Docker Compose:
```bash
docker-compose up
```

## 💻 Uso de la Aplicación

1. **Configuración**:
   - Seleccionar el modelo de IA a utilizar
   - Elegir el idioma de la interfaz
   - Verificar la disponibilidad de las bibliotecas PDF

2. **Diagnóstico**:
   - Cargar imagen de la planta
   - Obtener diagnóstico automático
   - Ver probabilidades de cada enfermedad
   - Generar reporte PDF

3. **Comparación de Modelos**:
   - Analizar imagen con múltiples modelos
   - Comparar resultados y precisión
   - Visualizar métricas comparativas

## 🌐 Soporte Multilenguaje

El sistema soporta cuatro idiomas:
- 🇪🇸 Español (por defecto)
- 🇬🇧 English
- 🇫🇷 Français
- 🇧🇷 Português

## 📊 Modelos de IA Disponibles

- **ModelV1**: Modelo base con arquitectura CNN
- **ModelV2**: Modelo mejorado con transfer learning
- **ModelV3**: Modelo avanzado con fine-tuning

## 📄 Generación de Reportes

El sistema puede generar reportes PDF detallados que incluyen:
- Imagen analizada
- Diagnóstico principal
- Probabilidades de cada enfermedad
- Recomendaciones de tratamiento
- Gráficos y visualizaciones

## 🔧 Configuración Avanzada

### Variables de Entorno
```python
TF_CPP_MIN_LOG_LEVEL = "3"  # Reducir logs de TensorFlow
CUDA_VISIBLE_DEVICES = "-1"  # Deshabilitar GPU si es necesario
```

### Configuración de Streamlit
```toml
[server]
maxUploadSize = 500
port = 8123
address = "0.0.0.0"
```

## 🤝 Contribución

1. Fork del proyecto
2. Crear rama para nueva funcionalidad
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Commit de cambios
   ```bash
   git commit -m "Añadir nueva funcionalidad"
   ```
4. Push a la rama
   ```bash
   git push origin feature/nueva-funcionalidad
   ```
5. Crear Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

### Equipo de Desarrollo
- **Elmo Tirado Ruiz** - Líder del Proyecto - [@elmo-12](https://github.com/elmo-12)
- **Kevin Rivas Verastegui** - Desarrollador ML/IA
- **Fatima Avila Juarez** - Desarrolladora Backend
- **Pamela Alayo Gamboa** - Desarrolladora Frontend

### Contacto
Para consultas y soporte, puedes contactar al equipo a través de:
- GitHub: [@elmo-12](https://github.com/elmo-12)

## 🙏 Agradecimientos

Agradecemos especialmente a:
- La comunidad de agricultores de caña de azúcar por su retroalimentación y datos de prueba
- La Universidad por proporcionar los recursos y el apoyo necesario
- La comunidad open source de TensorFlow y Streamlit

## 📞 Soporte

Para soporte y consultas:
- Crear un issue en el repositorio
- Contactar al equipo de desarrollo

---

Desarrollado con ❤️ para la comunidad agrícola 