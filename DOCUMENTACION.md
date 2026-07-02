# Documentación — Clasificador de Fallas en Módulos Solares (Infrared Solar Modules)

## Qué hace la herramienta

Entrena un modelo de Deep Learning (ResNet18 por transfer learning) que clasifica una imagen térmica infrarroja de un módulo/celda solar en una de **12 clases** de estado/falla:

`No-Anomaly, Cell, Cell-Multi, Cracking, Hot-Spot, Hot-Spot-Multi, Diode, Diode-Multi, Offline-Module, Shadowing, Soiling, Vegetation`

El resultado de este pipeline es un **modelo entrenado y evaluado** (`modeloEntrenado/infrared_solar_modules_model.pth`), pensado para integrarse después en una herramienta donde:

1. Una persona sube una imagen térmica.
2. El modelo la clasifica (clase + nivel de confianza).
3. Esa clasificación se pasa como contexto a un LLM de Claude, que genera la retroalimentación/explicación para el usuario (ej. qué significa la falla detectada, qué acción tomar).

Este documento cubre la parte **1 (entrenamiento)**. La parte de inferencia + integración con Claude todavía no está construida — se detalla al final qué necesita esa siguiente etapa.

---

## Datos de entrada

### Fuentes usadas para entrenar

| Fuente | Rol | Cantidad | Por qué |
|---|---|---|---|
| Dataset público de Kaggle `marcosgabriel/infrared-solar-modules` | Fuente principal de las 12 clases | 20.000 imágenes (24×40 px, escala de grises) + `module_metadata.json` con la etiqueta de cada una | Es el único origen de datos con las 12 clases reales verificadas (ground truth), documentadas en el paper/dataset original. |
| `RECURSOS/IR/IR/12 Julio/MPPT` y `RECURSOS/IR/IR/27 junio/MPPT` | Extra de la clase `No-Anomaly` | 27 imágenes | MPPT = módulo operando en su punto óptimo, sin falla eléctrica inducida → visualmente sano, mapeo razonable a `No-Anomaly`. Es una asunción, no una etiqueta verificada por un experto (queda documentado como tal en `prepare_dataset.py`). |

### Fuentes de `RECURSOS` que se descartaron para este modelo (y por qué)

- `RECURSOS/IR` → carpetas `Circuito Abierto`, `Corto Circuito`, `Mixto`: son condiciones eléctricas inducidas experimentalmente, no corresponden de forma verificable a ninguna de las 11 clases de falla de Kaggle (`Cracking`, `Hot-Spot`, `Diode`, etc.). Usarlas como esas etiquetas sería inventar ground truth.
- `RECURSOS/archive/Eletroluminescence`: otra modalidad (electroluminiscencia, no térmica), solo 11 imágenes.
- `RECURSOS/archive/open_short`, `RECURSOS/TODAS_BMP 2022`: otras cámaras/formatos, sin mapeo confiable a las 12 clases de Kaggle.
- `RECURSOS/archive/Thermography/*`: carpetas vacías (huérfanas), sin contenido real.


### Formato de la imagen de entrada esperada

- Las imágenes de Kaggle son JPG en escala de grises reales (un solo canal, replicado a 3 canales al cargar para que calce con el backbone ImageNet — por eso se ven en blanco y negro, no es un error).
- Las imágenes de `RECURSOS` (MPPT) son BMP a color con overlay de cámara quemado (fecha/hora, barra de escala de temperatura); se les recorta automáticamente el borde derecho (14%) e inferior (8%) antes de usarlas (`config.OVERLAY_CROP`), para que el modelo no aprenda atajos a partir del overlay en vez del patrón térmico real.

---

## Proceso (pipeline de entrenamiento)

```
RECURSOS/IR (MPPT)  ─┐
                      ├─► prepare_dataset.py ─► data/split_index.csv ─► dataset.py ─► model.py ─► engine.py (train + early stopping) ─► evaluate.py ─► modeloEntrenado/*.pth
Kaggle (download_dataset.py) ─┘
```

| Paso | Archivo | Qué hace |
|---|---|---|
| 1. Descarga | `src/download_dataset.py` | Baja el dataset de Kaggle vía API REST (Bearer auth con tu token) a `data/InfraredSolarModules/`. Si ya existe, no vuelve a descargar. |
| 2. Preparación del índice | `src/prepare_dataset.py` | Lee `module_metadata.json` de Kaggle + las imágenes MPPT de `RECURSOS`, arma un único índice, y hace un split estratificado por clase: **70% train / 15% val / 15% test** (semilla fija = reproducible). Lo guarda en `data/split_index.csv`. |
| 3. Carga y augmentation | `src/dataset.py` | `Dataset` de PyTorch: lee cada imagen (`cv2`), recorta el overlay si corresponde, y aplica transformaciones. En train: flip horizontal/vertical, rotación ±10°, jitter de brillo/contraste. En val/test: solo resize + normalización. Todo a **128×128 px** (justificado porque las imágenes de Kaggle son nativamente 24×40, no hay información real que aproveche una resolución mayor). |
| 4. Modelo | `src/model.py` | ResNet18 preentrenado en ImageNet. Se congela todo el backbone excepto `layer4`, y se reemplaza la capa final por una `Linear(512, 12)`. |
| 5. Entrenamiento | `src/engine.py` + notebook | `CrossEntropyLoss` ponderada por clase (`class_weight="balanced"`, ver más abajo), optimizador Adam (`lr=1e-4`), hasta 50 epochs con **Early Stopping** (paciencia 7 sobre `val_loss`, se restaura el mejor checkpoint). |
| 6. Evaluación | `src/evaluate.py` | Sobre el split de test: Accuracy, Precision/Recall/F1 macro, métricas por clase, y matriz de confusión (`resultados/matriz_confusion.png`). |
| 7. Guardado | notebook, sección 8 | Guarda el modelo final. |

### Cómo se maneja el desbalance de clases

El dataset está fuertemente desbalanceado (`No-Anomaly` ≈10.000 vs `Diode-Multi` ≈175, ~57x de diferencia). Se usa `class_weight="balanced"` de scikit-learn para que la pérdida penalice más los errores en clases minoritarias. El notebook grafica estos pesos y las métricas por clase para decidir si hace falta reforzar esto con `WeightedRandomSampler`, más capas descongeladas, o Focal Loss (documentado como nota dentro del propio notebook, sección 5).

---

## Datos de salida

| Archivo | Contenido |
|---|---|
| `modeloEntrenado/infrared_solar_modules_model.pth` | Checkpoint de PyTorch con: `model_state_dict` (pesos del ResNet18 ya fine-tuneado), `classes` (lista ordenada de las 12 clases), `class_to_idx` (mapeo clase→índice, necesario para decodificar la predicción). |
| `resultados/metricas.json` | Accuracy, Precision/Recall/F1 macro sobre el test set. |
| `resultados/metricas_por_clase.csv` | Precision, Recall, F1 y soporte (n° de imágenes) por cada una de las 12 clases. |
| `resultados/matriz_confusion.png` | Matriz de confusión del test set. |

---

## Resumen Entrada → Proceso → Salida

| | Entrenamiento (lo que existe hoy) | Futuro: inferencia + Claude (no construido aún) |
|---|---|---|
| **Entrada** | 20.027 imágenes térmicas etiquetadas (Kaggle + MPPT) | 1 imagen térmica subida por el usuario |
| **Proceso** | Descarga → índice train/val/test → augmentation → transfer learning ResNet18 → early stopping → evaluación | Cargar `infrared_solar_modules_model.pth` → mismo preprocesamiento (resize 128×128 + normalización ImageNet, mismo recorte de overlay si aplica) → forward pass → softmax → clase + confianza → armar prompt con ese resultado → llamada a la API de Claude |
| **Salida** | Modelo entrenado (`.pth`) + métricas + matriz de confusión | Clase predicha + confianza + explicación/recomendación generada por Claude en lenguaje natural |

### Lo que falta para la siguiente etapa (no incluido en este documento/pipeline)

- Un script/endpoint de **inferencia** que cargue el `.pth`, reciba una imagen nueva, aplique el mismo preprocesamiento que `dataset.py` (mismo `IMG_SIZE`, misma normalización, mismo criterio de recorte de overlay si la imagen viene de una cámara de campo) y devuelva clase + probabilidad.
- El **prompt/lógica de integración con Claude**: qué contexto exacto se le pasa al LLM (clase predicha, confianza, quizás las probabilidades de las otras clases, metadatos del panel) para que la retroalimentación sea útil y no solo repita el nombre de la clase.
- Definir qué pasa si la imagen subida no se parece a nada del dataset de entrenamiento (out-of-distribution) — por ejemplo, si alguien sube una foto que no es una imagen térmica.

Si quieres, en la próxima iteración armamos ese script de inferencia y el diseño del prompt para Claude.
