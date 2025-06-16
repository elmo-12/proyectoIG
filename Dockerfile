# Utiliza una imagen oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Variables de entorno para reducir mensajes de TensorFlow
ENV PYTHONWARNINGS="ignore"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS="0"

# Instalar dependencias del sistema para OpenCV y otras herramientas
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de dependencias
COPY requirements.txt ./

# Instala las dependencias Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless --no-cache-dir

# Crear directorio para modelos y configuración
RUN mkdir -p /app/models
RUN mkdir -p /root/.streamlit

# Copia el código de la aplicación y la configuración
COPY . /app/
COPY .streamlit/config.toml /root/.streamlit/

# Asegura que el directorio app y models tengan los permisos correctos
RUN chmod -R 777 /app

# Expone el puerto por defecto de Streamlit
EXPOSE 8123

# Comando para ejecutar la app con límite de carga aumentado
CMD ["streamlit", "run", "app.py", "--server.maxUploadSize=500", "--server.port=8123", "--server.address=0.0.0.0"]