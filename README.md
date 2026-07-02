# Termoscan — Clasificador de fallas térmicas en módulos solares

Clasificador de Deep Learning (ResNet18, transfer learning) que detecta 12 tipos de anomalía en imágenes térmicas infrarrojas de módulos solares, con una herramienta web para subir una imagen, ver la clasificación (español/inglés) y recibir un diagnóstico en lenguaje natural generado por Claude.

Resultado del modelo actual (test set): **accuracy 0.791**, **F1-macro 0.662**. Ver `resultados/metricas.json` y `ModeloIA/modeloEntrenado/metadata_entrenamiento.json` para el detalle completo (incluye matriz de confusión y curvas de entrenamiento).

## Estructura del proyecto

```
ModeloIA/             Todo lo del modelo de IA:
  src/                   Pipeline de entrenamiento (descarga, preparación de datos, modelo, entrenamiento, evaluación)
  notebooks/             Notebook de entrenamiento interactivo (Jupyter)
  modeloEntrenado/       Modelo entrenado (.pth, no versionado) + metadata (json, sí versionado)
backend/               API FastAPI: clasifica la imagen y llama a Claude
frontend/              Interfaz web (HTML/CSS/JS, sin frameworks)
resultados/            Métricas, matriz de confusión, historial de entrenamiento
RECURSOS/              Datos propios de campo (no versionado, ver .gitignore)
data/                  Dataset de Kaggle descargado (no versionado, se regenera)
docker-compose.yml     Levanta backend + frontend con Docker
DOCUMENTACION.md       Análisis completo: qué datos se usaron, por qué, y el pipeline detallado
```

## 1. Entrenar el modelo

```
pip install -r requirements.txt
```

Configurar credenciales de Kaggle (`KAGGLE_API_TOKEN` o `~/.kaggle/access_token`), luego abrir y correr `ModeloIA/notebooks/entrenamiento.ipynb` de punta a punta. Genera:

- `ModeloIA/modeloEntrenado/infrared_solar_modules_model.pth` (no se versiona en git — se regenera corriendo el notebook)
- `ModeloIA/modeloEntrenado/metadata_entrenamiento.json` (arquitectura, hiperparámetros, métricas, matriz de confusión y curvas — sí versionado)
- `resultados/` (métricas, matriz de confusión, historial)

Detalle completo de qué datos se usaron y por qué en [DOCUMENTACION.md](DOCUMENTACION.md).

## 2. Correr la herramienta web

### Opción A: con Docker (recomendado)

```
cp backend/.env.example backend/.env   # pegar tu ANTHROPIC_API_KEY ahí
docker compose up --build
```

Abrir `http://localhost:8080`. El backend queda expuesto en `http://localhost:8000`. Requiere que `ModeloIA/modeloEntrenado/infrared_solar_modules_model.pth` exista (paso 1) — se monta como volumen, no se copia a la imagen.

### Opción B: sin Docker

```
cd backend
pip install -r requirements.txt
cp .env.example .env   # pegar tu ANTHROPIC_API_KEY ahí
uvicorn main:app --reload
```

Y en otra terminal, servir `frontend/` con cualquier servidor estático (ej. `python -m http.server 8080` desde adentro de `frontend/`) y abrir `http://localhost:8080`.

## Clases detectadas

`No-Anomaly`, `Cell`, `Cell-Multi`, `Cracking`, `Hot-Spot`, `Hot-Spot-Multi`, `Diode`, `Diode-Multi`, `Offline-Module`, `Shadowing`, `Soiling`, `Vegetation` — con nombre y descripción en español directamente en la herramienta web.
