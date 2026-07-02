# Termoscan — Clasificador de fallas térmicas en módulos solares

Clasificador de Deep Learning (ResNet18, transfer learning) que detecta 12 tipos de anomalía en imágenes térmicas infrarrojas de módulos solares, con una herramienta web para subir una imagen, ver la clasificación (español/inglés) y recibir un diagnóstico en lenguaje natural generado por Claude.

Resultado del modelo actual (test set): **accuracy 0.791**, **F1-macro 0.662**. Ver `resultados/metricas.json` y `modeloEntrenado/metadata_entrenamiento.json` para el detalle completo (incluye matriz de confusión y curvas de entrenamiento).

## Estructura del proyecto

```
src/                  Pipeline de entrenamiento (descarga, preparación de datos, modelo, entrenamiento, evaluación)
notebooks/            Notebook de entrenamiento interactivo (Jupyter)
app/                  Herramienta web: subir imagen → clasificar → diagnóstico con Claude
resultados/           Métricas, matriz de confusión, historial de entrenamiento
modeloEntrenado/      Metadata del modelo (el .pth no se versiona, ver más abajo)
RECURSOS/             Datos propios de campo (no versionado, ver .gitignore)
data/                 Dataset de Kaggle descargado (no versionado, se regenera)
DOCUMENTACION.md      Análisis completo: qué datos se usaron, por qué, y el pipeline detallado
```

## 1. Entrenar el modelo

```
pip install -r requirements.txt
```

Configurar credenciales de Kaggle (`KAGGLE_API_TOKEN` o `~/.kaggle/access_token`), luego abrir y correr `notebooks/entrenamiento.ipynb` de punta a punta. Genera:

- `modeloEntrenado/infrared_solar_modules_model.pth` (no se versiona en git — se regenera corriendo el notebook)
- `modeloEntrenado/metadata_entrenamiento.json` (arquitectura, hiperparámetros, métricas, matriz de confusión y curvas — sí versionado)
- `resultados/` (métricas, matriz de confusión, historial)

Detalle completo de qué datos se usaron y por qué en [DOCUMENTACION.md](DOCUMENTACION.md).

## 2. Correr la herramienta web

```
cd app
pip install -r requirements.txt
cp .env.example .env   # pegar tu ANTHROPIC_API_KEY ahí
uvicorn main:app --reload
```

Abrir `http://127.0.0.1:8000`. Requiere que `modeloEntrenado/infrared_solar_modules_model.pth` exista (paso 1).

## Clases detectadas

`No-Anomaly`, `Cell`, `Cell-Multi`, `Cracking`, `Hot-Spot`, `Hot-Spot-Multi`, `Diode`, `Diode-Multi`, `Offline-Module`, `Shadowing`, `Soiling`, `Vegetation` — con nombre y descripción en español directamente en la herramienta web.
