# Dockerfile
# Web service de predicción de ACV

FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias primero (aprovecha caché de Docker)
COPY pyproject.toml uv.lock ./

# Instalar uv y dependencias
RUN pip install uv --quiet && \
    uv sync --frozen --no-dev

# Copiar el código de la app y el modelo
COPY app/ ./app/
COPY model/ ./model/

# Puerto que expone la app
EXPOSE 8080

# Comando para arrancar la app
CMD ["uv", "run", "python", "app/app.py"]
