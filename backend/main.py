"""API para subir una imagen térmica, clasificarla y obtener retroalimentación de Claude."""
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from claude_feedback import get_explanation  # noqa: E402
from inference import load_model, predict  # noqa: E402

app = FastAPI(title="Clasificador térmico de módulos solares")

# El frontend corre en su propio contenedor/origen (ver docker-compose.yml),
# por eso hace falta CORS acá. CORS_ALLOW_ORIGINS permite restringirlo en
# producción (ej. "http://localhost:8080"); por defecto permite todo, para
# no trabarse en desarrollo local.
allow_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    load_model()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    image_bytes = await file.read()
    try:
        resultado = predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar la imagen: {exc}")

    try:
        resultado["explicacion_claude"] = get_explanation(resultado)
    except Exception as exc:
        resultado["explicacion_claude"] = None
        resultado["error_claude"] = str(exc)

    return resultado
