"""
Genera un JSON con todos los metadatos del entrenamiento (arquitectura,
hiperparámetros, datos usados, métricas finales) para poder compartir/revisar
los resultados sin tener que volver a entrenar el modelo.
"""
import json

import pandas as pd

from config import (
    BATCH_SIZE,
    CLASS_WEIGHT_POWER,
    CLASSES,
    IMG_SIZE,
    KAGGLE_DATASET_ID,
    MODEL_PATH,
    RESULTS_DIR,
    SEED,
    SPLIT_INDEX_CSV,
    TEST_FRACTION,
    VAL_FRACTION,
)

METADATA_PATH = MODEL_PATH.parent / "metadata_entrenamiento.json"


def build_metadata() -> dict:
    df = pd.read_csv(SPLIT_INDEX_CSV)
    split_counts = df.groupby("split").size().to_dict()
    class_counts = df["label"].value_counts().to_dict()
    source_counts = df["source"].value_counts().to_dict()

    with open(RESULTS_DIR / "metricas.json", encoding="utf-8") as f:
        metricas = json.load(f)
    confusion_matrix_data = metricas.pop("confusion_matrix", None)

    per_class_df = pd.read_csv(RESULTS_DIR / "metricas_por_clase.csv", index_col=0)
    metricas_por_clase = per_class_df.round(4).to_dict(orient="index")

    historial_path = RESULTS_DIR / "historial_entrenamiento.json"
    historial_entrenamiento = None
    if historial_path.exists():
        with open(historial_path, encoding="utf-8") as f:
            historial_entrenamiento = json.load(f)

    metadata = {
        "modelo": {
            "arquitectura": "ResNet18 (torchvision, preentrenado en ImageNet)",
            "capas_entrenables": ["layer3", "layer4", "fc (con Dropout 0.3 antes de la capa lineal)"],
            "num_clases": len(CLASSES),
            "clases": CLASSES,
            "class_to_idx": {c: i for i, c in enumerate(CLASSES)},
            "tamano_entrada_px": IMG_SIZE,
            "normalizacion": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        },
        "entrenamiento": {
            "optimizador": "Adam",
            "learning_rate": 1e-4,
            "loss": "CrossEntropyLoss con pesos por clase (class_weight='balanced' ** CLASS_WEIGHT_POWER)",
            "class_weight_power": CLASS_WEIGHT_POWER,
            "batch_size": BATCH_SIZE,
            "data_augmentation": "flip horizontal/vertical, rotación ±10°, color jitter (brillo/contraste)",
            "early_stopping": "paciencia 7 epochs sobre val_loss, restaura el mejor checkpoint",
            "seed": SEED,
        },
        "datos": {
            "fuente_principal": f"Kaggle: {KAGGLE_DATASET_ID} (20.000 imágenes originales, 12 clases)",
            "fuente_extra": "RECURSOS/IR MPPT (27 imágenes) agregadas como No-Anomaly extra",
            "total_imagenes": int(len(df)),
            "imagenes_por_fuente": {str(k): int(v) for k, v in source_counts.items()},
            "imagenes_por_clase": {str(k): int(v) for k, v in class_counts.items()},
            "split": {
                "train": int(split_counts.get("train", 0)),
                "val": int(split_counts.get("val", 0)),
                "test": int(split_counts.get("test", 0)),
                "val_fraction": VAL_FRACTION,
                "test_fraction": TEST_FRACTION,
            },
        },
        "resultados_test": {
            **{k: round(v, 4) for k, v in metricas.items()},
            "por_clase": metricas_por_clase,
            "matriz_confusion": confusion_matrix_data,
        },
        "curvas_entrenamiento": historial_entrenamiento,
        "archivos": {
            "modelo": str(MODEL_PATH),
            "matriz_confusion_png": str(RESULTS_DIR / "matriz_confusion.png"),
            "metricas_por_clase_csv": str(RESULTS_DIR / "metricas_por_clase.csv"),
        },
    }
    return metadata


def export() -> None:
    metadata = build_metadata()
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadatos guardados en {METADATA_PATH}")


if __name__ == "__main__":
    export()
