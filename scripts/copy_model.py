import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def copy_model_to_app():
    """Copia el modelo entrenado a la carpeta de la app"""

    # 📁 raíz del proyecto
    project_root = Path(__file__).resolve().parent.parent

    # 📥 origen (tu modelo entrenado)
    source_model = project_root / "model"

    # 📤 destino (para la app)
    destination_model = project_root / "app" / "model"

    logging.info("Copiando modelo...")
    logging.info(f"Origen: {source_model}")
    logging.info(f"Destino: {destination_model}")

    # verificar que existe
    if not source_model.exists():
        raise FileNotFoundError(f"No existe el modelo en: {source_model}")

    # eliminar destino si existe
    if destination_model.exists():
        shutil.rmtree(destination_model)

    # copiar carpeta completa
    shutil.copytree(source_model, destination_model)

    logging.info("Modelo copiado correctamente")

    return destination_model


if __name__ == "__main__":
    copy_model_to_app()