# app.py
# Web service para predicción de riesgo de ACV
# Herramienta: FastAPI + interfaz web

import json
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_PATH    = Path("model/model/model.pkl")
METADATA_PATH = Path("model/metadata.json")


# ── Cargar modelo al iniciar ──────────────────────────────────────────────────

def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return modelo, metadata


modelo, metadata = cargar_modelo()


# ── Aplicación FastAPI ────────────────────────────────────────────────────────

app = FastAPI(title="Predicción de Riesgo de ACV", version="1.0.0")


# ── Esquemas ──────────────────────────────────────────────────────────────────

class PacienteInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


class PrediccionOutput(BaseModel):
    prediccion: int
    riesgo: str
    probabilidad_stroke: float
    mensaje: str
    modelo_usado: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Interfaz"])
def root():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html no encontrado</h1>", status_code=404)


@app.get("/health", tags=["Estado"])
def health():
    return {"status": "ok"}


@app.get("/modelo-info", tags=["Estado"])
def modelo_info():
    return {
        "nombre":         metadata.get("nombre"),
        "recall_clase_1": metadata.get("recall_class_1"),
        "roc_auc":        metadata.get("roc_auc"),
    }


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(paciente: PacienteInput):
    try:
        import pandas as pd

        datos        = pd.DataFrame([paciente.model_dump()])
        prediccion   = int(modelo.predict(datos)[0])
        probabilidad = float(modelo.predict_proba(datos)[0][1])

        if prediccion == 1:
            riesgo  = "ALTO"
            mensaje = (
                f"El paciente tiene riesgo ALTO de ACV. "
                f"Probabilidad estimada: {probabilidad:.1%}. "
                f"Se recomienda evaluación médica urgente."
            )
        else:
            riesgo  = "BAJO"
            mensaje = (
                f"El paciente tiene riesgo BAJO de ACV. "
                f"Probabilidad estimada: {probabilidad:.1%}. "
                f"Se recomienda mantener controles médicos regulares."
            )

        return PrediccionOutput(
            prediccion=prediccion,
            riesgo=riesgo,
            probabilidad_stroke=round(probabilidad, 4),
            mensaje=mensaje,
            modelo_usado=metadata.get("nombre", "No disponible")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
