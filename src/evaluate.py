"""Cálculo de métricas y matriz de confusión sobre un conjunto de evaluación."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import CLASSES, CLASSES_ES


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    df = pd.DataFrame(report).T.loc[CLASSES]
    df["support"] = df["support"].astype(int)
    return df


def confusion_matrix_json(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    return {
        "clases_en": CLASSES,
        "clases_es": [CLASSES_ES[c] for c in CLASSES],
        "matriz": cm.tolist(),
    }


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_model(model, dataloader, device, confusion_matrix_path=None) -> dict:
    y_true, y_pred = collect_predictions(model, dataloader, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics["per_class"] = per_class_report(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix_json(y_true, y_pred)
    if confusion_matrix_path is not None:
        plot_confusion_matrix(y_true, y_pred, confusion_matrix_path)
    return metrics
