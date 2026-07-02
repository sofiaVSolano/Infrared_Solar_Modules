"""
Descarga el dataset público "Infrared Solar Modules" (Kaggle) usando directamente
la API REST v1 de Kaggle (Bearer auth), sin depender de la librería `kagglehub`.

Nota: se evaluó usar `kagglehub`, pero la versión publicada en PyPI (1.0.2) tiene
un bug de empaquetado real (le falta el submódulo `kagglesdk.competitions.legacy`,
necesario incluso solo para importar la librería) y la versión anterior (0.3.13)
no soporta el token nuevo tipo "KGAT_..." (solo el kaggle.json clásico de
usuario+key). El endpoint REST clásico sí soporta Bearer auth con el token nuevo
(verificado empíricamente), así que se implementa acá directamente.

Requiere credenciales de Kaggle configuradas de antemano, mediante UNA de estas vías:
  1. Variable de entorno KAGGLE_API_TOKEN
  2. Archivo ~/.kaggle/access_token (o ~/.kaggle/access_token.txt en Windows)
"""
import os
import shutil
import zipfile
from pathlib import Path

import requests

from config import DATA_DIR, KAGGLE_DATASET_ID

KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET_ID}"


def _get_api_token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token.strip()

    for filename in ("access_token", "access_token.txt"):
        token_path = Path.home() / ".kaggle" / filename
        if token_path.exists():
            return token_path.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "No se encontraron credenciales de Kaggle. Configura la variable de entorno "
        "KAGGLE_API_TOKEN o crea el archivo ~/.kaggle/access_token con tu token."
    )


def download() -> Path:
    target = DATA_DIR / "InfraredSolarModules"
    if target.exists() and (target / "module_metadata.json").exists():
        print(f"Ya existe {target}, se omite la descarga.")
        return target

    token = _get_api_token()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "infrared-solar-modules.zip"

    print(f"Descargando dataset desde {KAGGLE_DOWNLOAD_URL} ...")
    with requests.get(
        KAGGLE_DOWNLOAD_URL, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=60
    ) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    print(f"Descarga completa: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    print(f"Extrayendo a {DATA_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()

    metadata_matches = list(DATA_DIR.rglob("module_metadata.json"))
    if not metadata_matches:
        raise FileNotFoundError(
            f"No se encontró 'module_metadata.json' dentro de {DATA_DIR} tras extraer el dataset."
        )

    dataset_root = metadata_matches[0].parent
    if dataset_root != target:
        shutil.move(str(dataset_root), str(target))
        shutil.rmtree(dataset_root.parent, ignore_errors=True)

    print(f"Dataset listo en {target}")
    return target


if __name__ == "__main__":
    download()
