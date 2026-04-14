# pipeline.py
# Pipeline de entrenamiento para predicción de riesgo de ACV
# Herramientas: Prefect (orquestación) + MLflow (tracking)

import os
import json
import pandas as pd
from pathlib import Path

from prefect import task, flow

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# ── Configuración ─────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME     = os.getenv("EXPERIMENT_NAME", "stroke-predictionX")
DATA_PATH           = os.getenv("DATA_PATH", "data/healthcare-dataset-stroke-data.csv")
MODEL_OUTPUT_DIR    = os.getenv("MODEL_OUTPUT_DIR", "model")


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="cargar_datos", log_prints=True)
def cargar_datos(path: str) -> pd.DataFrame:
    """Lee el CSV y elimina la columna id."""
    df = pd.read_csv(path)
    df = df.drop(columns=["id"])

    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    print(f"Valores nulos:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df


@task(name="limpiar_datos", log_prints=True)
def limpiar_datos(df: pd.DataFrame) -> tuple:
    """Separa X e y, hace el split y construye el preprocesador."""

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    print(f"Distribución de clase:\n{y.value_counts()}")
    print(f"Proporción positivos (stroke=1): {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    # Columnas por tipo
    num_cols = ["age", "avg_glucose_level", "bmi"]
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    bin_cols = ["hypertension", "heart_disease"]

    # Pipelines de preprocesamiento
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore"))
    ])
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
        ("bin", bin_pipeline, bin_cols)
    ])

    return X_train, X_test, y_train, y_test, preprocessor


@task(name="entrenar_modelos", log_prints=True)
def entrenar_modelos(
    X_train, X_test, y_train, y_test, preprocessor
) -> list:
    """Entrena LR y RF, loguea métricas y modelos en MLflow."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    modelos = [
        {
            "nombre":       "Logistic Regression",
            "clasificador": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params":       {"max_iter": 1000, "class_weight": "balanced"},
            "familia":      "LR"
        },
        {
            "nombre":       "Random Forest",
            "clasificador": RandomForestClassifier(n_estimators=500, random_state=42),
            "params":       {"n_estimators": 500, "random_state": 42},
            "familia":      "RF"
        }
    ]

    resultados = []

    for m in modelos:
        print(f"\nEntrenando: {m['nombre']}...")

        with mlflow.start_run(run_name=m["nombre"]) as run:

            # Tags
            mlflow.set_tag("problem_type", "binary classification")
            mlflow.set_tag("model_family", m["familia"])
            mlflow.set_tag("dataset",      "healthcare-dataset-stroke-data")

            # Parámetros
            for k, v in m["params"].items():
                mlflow.log_param(k, v)

            # Pipeline completo: preprocesador + clasificador
            modelo_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier",   m["clasificador"])
            ])

            # Entrenar y predecir
            modelo_pipeline.fit(X_train, y_train)
            y_pred  = modelo_pipeline.predict(X_test)
            y_proba = modelo_pipeline.predict_proba(X_test)[:, 1]

            # Métricas
            report = classification_report(y_test, y_pred, output_dict=True)
            auc    = roc_auc_score(y_test, y_proba)

            mlflow.log_metric("roc_auc",          auc)
            mlflow.log_metric("accuracy",          report["accuracy"])
            mlflow.log_metric("f1_macro",          report["macro avg"]["f1-score"])
            mlflow.log_metric("recall_class_1",    report["1"]["recall"])
            mlflow.log_metric("precision_class_1", report["1"]["precision"])
            mlflow.log_metric("f1_class_1",        report["1"]["f1-score"])

            # Guardar modelo en MLflow
            mlflow.sklearn.log_model(modelo_pipeline, name="model")

            print(f"  recall clase 1:    {report['1']['recall']:.3f}")
            print(f"  precision clase 1: {report['1']['precision']:.3f}")
            print(f"  ROC-AUC:           {auc:.3f}")

            resultados.append({
                "run_id":           run.info.run_id,
                "nombre":           m["nombre"],
                "recall_class_1":   report["1"]["recall"],
                "precision_class_1":report["1"]["precision"],
                "roc_auc":          auc
            })

    return resultados


@task(name="seleccionar_mejor_modelo", log_prints=True)
def seleccionar_mejor_modelo(resultados: list) -> dict:
    """Elige el modelo con mayor recall en la clase 1 (stroke=1)."""

    if not resultados:
        raise ValueError("No hay modelos entrenados para comparar")

    print("\nComparando modelos por recall clase 1:")
    for r in resultados:
        print(f"  {r['nombre']}: "
              f"recall={r['recall_class_1']:.3f} | "
              f"precision={r['precision_class_1']:.3f} | "
              f"roc_auc={r['roc_auc']:.3f}")

    mejor = max(resultados, key=lambda x: x["recall_class_1"])

    print(f"\nMejor modelo: {mejor['nombre']}")
    print(f"  recall clase 1: {mejor['recall_class_1']:.3f}")
    print(f"  run_id:         {mejor['run_id']}")

    return mejor


@task(name="guardar_modelo", log_prints=True)
def guardar_modelo(mejor: dict, output_dir: str = MODEL_OUTPUT_DIR) -> str:
    """Descarga los artefactos del mejor modelo y genera metadata.json."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Descargando modelo del run: {mejor['run_id']}...")

    modelo_path = mlflow.artifacts.download_artifacts(
        run_id=mejor["run_id"],
        artifact_path="model",
        dst_path=str(output_path)
    )

    print(f"Modelo guardado en: {modelo_path}")

    # Crear metadata.json
    metadata = {
        "run_id":           mejor["run_id"],
        "nombre":           mejor["nombre"],
        "recall_class_1":   mejor["recall_class_1"],
        "precision_class_1":mejor["precision_class_1"],
        "roc_auc":          mejor["roc_auc"],
        "model_path":       modelo_path,
        "experiment_name":  EXPERIMENT_NAME
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"metadata.json guardado en: {metadata_path}")
    print(f"\nResumen del modelo guardado:")
    print(f"  Nombre:         {mejor['nombre']}")
    print(f"  Recall clase 1: {mejor['recall_class_1']:.3f}")
    print(f"  Precision cl 1: {mejor['precision_class_1']:.3f}")
    print(f"  ROC-AUC:        {mejor['roc_auc']:.3f}")

    return str(metadata_path)


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="stroke_training_pipeline", log_prints=True)
def stroke_training_pipeline():
    """
    Pipeline completo de entrenamiento para predicción de ACV.

    Orden de ejecución:
        1. cargar_datos      → lee el CSV
        2. limpiar_datos     → split + preprocesador
        3. entrenar_modelos  → LR y RF con tracking en MLflow
        4. seleccionar_mejor_modelo → mayor recall clase 1
        5. guardar_modelo    → artefactos + metadata.json
    """
    df = cargar_datos(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = limpiar_datos(df)

    resultados = entrenar_modelos(X_train, X_test, y_train, y_test, preprocessor)

    mejor = seleccionar_mejor_modelo(resultados)

    metadata_path = guardar_modelo(mejor)

    print(f"\nPipeline completado exitosamente.")
    print(f"Metadata disponible en: {metadata_path}")


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    stroke_training_pipeline()
