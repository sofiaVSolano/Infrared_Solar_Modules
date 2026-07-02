"""
Construye el índice único de entrenamiento (CSV) combinando:
  1. El dataset Kaggle "Infrared Solar Modules" (fuente principal, 12 clases reales).
  2. Las imágenes MPPT de RECURSOS/IR (condición sana / sin falla inducida), agregadas
     como ejemplos adicionales de "No-Anomaly". Esto es una asunción razonable pero
     no verificada por un experto: MPPT = módulo operando en su punto óptimo, sin
     falla eléctrica inducida, visualmente homogéneo en las muestras inspeccionadas.

No se copian las ~20.000 imágenes de Kaggle a otra carpeta: el índice solo guarda
sus rutas absolutas dentro de data/InfraredSolarModules.

El overlay de cámara (fecha/hora, barra de escala) solo existe en las imágenes de
RECURSOS, por eso el índice trae la columna `needs_crop`.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    KAGGLE_METADATA_FILE,
    KAGGLE_ROOT,
    MPPT_DIRS,
    SEED,
    SPLIT_INDEX_CSV,
    TEST_FRACTION,
    VAL_FRACTION,
)


def load_kaggle_rows() -> list[dict]:
    if not KAGGLE_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {KAGGLE_METADATA_FILE}. "
            "Ejecuta primero: python src/download_dataset.py"
        )
    with open(KAGGLE_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    rows = []
    for entry in metadata.values():
        filepath = KAGGLE_ROOT / entry["image_filepath"]
        rows.append(
            {
                "filepath": str(filepath),
                "label": entry["anomaly_class"],
                "source": "kaggle",
                "needs_crop": False,
            }
        )
    return rows


def load_mppt_rows() -> list[dict]:
    rows = []
    for d in MPPT_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.BMP")):
            rows.append(
                {
                    "filepath": str(f),
                    "label": "No-Anomaly",
                    "source": "recursos_mppt",
                    "needs_crop": True,
                }
            )
    return rows


def build_index() -> pd.DataFrame:
    rows = load_kaggle_rows() + load_mppt_rows()
    df = pd.DataFrame(rows)
    print(f"Total de imágenes indexadas: {len(df)}")
    print(df.groupby("label").size().sort_values(ascending=False))

    train_df, temp_df = train_test_split(
        df, test_size=VAL_FRACTION + TEST_FRACTION, stratify=df["label"], random_state=SEED
    )
    relative_test_size = TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, stratify=temp_df["label"], random_state=SEED
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    SPLIT_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(SPLIT_INDEX_CSV, index=False)
    print(f"\nÍndice guardado en {SPLIT_INDEX_CSV}")
    print(full_df.groupby("split").size())
    return full_df


if __name__ == "__main__":
    build_index()
