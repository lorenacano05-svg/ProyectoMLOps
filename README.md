# Predicción de Riesgo de ACV
Proyecto de MLOps para predecir el riesgo de Accidente Cerebrovascular (ACV) a partir de datos clínicos y sociales de pacientes. Usa Prefect para orquestación, MLflow para tracking de experimentos y FastAPI para el web service.

---

## Estructura del proyecto

```
Trabajofinal_MLOps/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── notebooks/
│   ├── EDA_trabajofinal.ipynb
│   └── model_stroke.ipynb
├── src/
│   ├── pipeline.py
│   ├── preprocess.py
│   └── train.py
├── app/
│   ├── app.py
│   └── index.html
├── model/
│   ├── metadata.json
│   └── model/
│       └── model.pkl
├── pyproject.toml
└── README.md
```

---

## Requisitos

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) instalado

Instalar dependencias:

```bash
uv sync
```

---

## Cómo correr el proyecto

Necesitas **3 terminales abiertas al mismo tiempo**, siempre desde la raíz del proyecto.

### Terminal 1 — MLflow

```bash
uv run python -m mlflow server --host 127.0.0.1 --port 5001
```

Verifica en: http://127.0.0.1:5001

### Terminal 2 — Prefect

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run prefect server start
```

Verifica en: http://127.0.0.1:4200

### Terminal 3 — Pipeline de entrenamiento

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run python pipeline.py
```

> **Importante:** respetar el orden. MLflow primero, Prefect segundo, pipeline tercero.

Cuando el pipeline termine verás:

```
Pipeline completado exitosamente.
Metadata disponible en: model/metadata.json
```

---

## Web service

Una vez que el pipeline haya corrido y el modelo esté guardado en `model/`, levanta el web service:

```bash
uv run python app/app.py
```

Abre http://localhost:8080 para usar el formulario de predicción.

---

## Pipeline de entrenamiento

El pipeline está compuesto por 5 tasks orquestadas con Prefect:

| Task | Descripción |
|---|---|
| `cargar_datos` | Lee el CSV y elimina la columna id |
| `limpiar_datos` | Hace el split y construye el preprocesador |
| `entrenar_modelos` | Entrena Logistic Regression y Random Forest con tracking en MLflow |
| `seleccionar_mejor_modelo` | Elige el modelo con mayor recall en la clase 1 (stroke=1) |
| `guardar_modelo` | Descarga los artefactos del mejor modelo y genera metadata.json |

---

## Modelos entrenados

Se comparan dos modelos. El criterio de selección es el **recall de la clase 1** (stroke=1), ya que en contexto clínico es más importante detectar todos los casos positivos que minimizar falsas alarmas.

| Modelo | Descripción |
|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=2000` |
| Random Forest | `n_estimators=500`, `random_state=42` |

---

## Variables de entorno

Se pueden configurar mediante variables de entorno:

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `DATA_PATH` | `data/healthcare-dataset-stroke-data.csv` | Ruta al dataset |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5001` | URI del servidor MLflow |
| `EXPERIMENT_NAME` | `stroke-prediction2` | Nombre del experimento en MLflow |
| `MODEL_OUTPUT_DIR` | `model` | Carpeta donde se guarda el modelo |