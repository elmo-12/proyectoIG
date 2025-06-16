# Utiliza una imagen oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de dependencias
COPY requirements.txt ./

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos del proyecto
COPY app.py ./
# Descarga el modelo desde Google Drive (usando el enlace completo)
RUN gdown "https://drive.google.com/file/d/1Uy22RdNdzZqc-96St6m7jreXZNJq761t/view" -O best_sugarcane_model.keras

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para ejecutar la app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 