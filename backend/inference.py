"""Carga el modelo entrenado y clasifica una imagen térmica subida por el usuario."""
import base64
import io
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT / "ModeloIA" / "src"))

from config import CLASSES, CLASSES_ES, MODEL_PATH  # noqa: E402
from dataset import build_transforms  # noqa: E402
from model import build_model  # noqa: E402

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = build_transforms(train=False)
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    # pretrained=False: los pesos de ImageNet se sobrescriben en la línea de abajo
    # con el checkpoint entrenado, así que descargarlos solo agrega una dependencia
    # de red innecesaria (y el contenedor del backend no tiene salida a internet).
    model = build_model(freeze_backbone=True, pretrained=False).to(_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model
    return _model


def _encode_jpeg_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict(image_bytes: bytes) -> dict:
    model = load_model()
    # El modelo se entrenó casi en su totalidad (20.000/20.027 imágenes) con
    # fotos en escala de grises reales (canales R=G=B idénticos). Forzamos gris
    # acá para que una imagen a color (ej. paleta arcoíris de una cámara FLIR)
    # no quede fuera de esa distribución y degrade la predicción.
    image_gris = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")
    tensor = _transform(image_gris).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    ranked = sorted(range(len(CLASSES)), key=lambda i: probs[i], reverse=True)
    top_idx = ranked[0]

    return {
        "clase_predicha": {
            "en": CLASSES[top_idx],
            "es": CLASSES_ES[CLASSES[top_idx]],
        },
        "confianza": round(float(probs[top_idx]), 4),
        "probabilidades": [
            {
                "en": CLASSES[i],
                "es": CLASSES_ES[CLASSES[i]],
                "prob": round(float(probs[i]), 4),
            }
            for i in ranked
        ],
        # La imagen en gris tal como la ve el modelo (post-conversión, antes del
        # resize), para que el frontend pueda mostrarla junto a la original.
        "imagen_gris_base64": _encode_jpeg_base64(image_gris),
    }
