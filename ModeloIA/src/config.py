"""Configuración centralizada de rutas, clases y constantes del pipeline."""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# config.py vive en ModeloIA/src/, dos niveles bajo la raíz del proyecto
# (ModeloIA/src/config.py -> ModeloIA/src -> ModeloIA -> raíz).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELO_IA_DIR = PROJECT_ROOT / "ModeloIA"

RECURSOS_DIR = PROJECT_ROOT / "RECURSOS"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = MODELO_IA_DIR / "modeloEntrenado"
RESULTS_DIR = PROJECT_ROOT / "resultados"

KAGGLE_DATASET_ID = "marcosgabriel/infrared-solar-modules"
KAGGLE_ROOT = DATA_DIR / "InfraredSolarModules"
KAGGLE_METADATA_FILE = KAGGLE_ROOT / "module_metadata.json"

MPPT_DIRS = [
    RECURSOS_DIR / "IR" / "IR" / "12 Julio" / "MPPT",
    RECURSOS_DIR / "IR" / "IR" / "27 junio" / "MPPT",
]

SPLIT_INDEX_CSV = DATA_DIR / "split_index.csv"

CLASSES = [
    "No-Anomaly",
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Diode",
    "Diode-Multi",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CLASSES_ES = {
    "No-Anomaly": "Sin anomalía",
    "Cell": "Celda dañada",
    "Cell-Multi": "Múltiples celdas dañadas",
    "Cracking": "Fisura / agrietamiento",
    "Hot-Spot": "Punto caliente",
    "Hot-Spot-Multi": "Múltiples puntos calientes",
    "Diode": "Falla de diodo",
    "Diode-Multi": "Múltiples fallas de diodo",
    "Offline-Module": "Módulo desconectado",
    "Shadowing": "Sombreado",
    "Soiling": "Suciedad / ensuciamiento",
    "Vegetation": "Vegetación (obstrucción)",
}

IMG_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 2
SEED = 42

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Overlay de cámara quemado en las imágenes de campo de RECURSOS (fecha/hora,
# barra de escala de temperatura): se recorta antes de usarlas.
OVERLAY_CROP = {"top": 0.0, "bottom": 0.08, "left": 0.0, "right": 0.14}

# Los pesos "balanced" completos sobrecorrigen el desbalance (evidencia: en la
# primera corrida, No-Anomaly tuvo precision=0.95 pero recall=0.66 — muchos
# paneles sanos terminaban clasificados como falla). Suavizar con una potencia
# < 1 conserva la compensación por desbalance sin empujar tan fuerte hacia las
# clases minoritarias.
CLASS_WEIGHT_POWER = 0.5

MODEL_PATH = MODEL_DIR / "infrared_solar_modules_model.pth"
