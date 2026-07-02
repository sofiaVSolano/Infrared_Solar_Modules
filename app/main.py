"""API para subir una imagen térmica, clasificarla y obtener retroalimentación de Claude."""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

from claude_feedback import get_explanation  # noqa: E402
from inference import load_model, predict  # noqa: E402

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Clasificador térmico de módulos solares")


@app.on_event("startup")
def _startup():
    load_model()


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


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")
