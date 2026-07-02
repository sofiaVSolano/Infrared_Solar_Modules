"""Dataset de PyTorch para el índice generado por prepare_dataset.py."""
import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, IMG_SIZE, OVERLAY_CROP

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _crop_overlay(image_bgr):
    h, w = image_bgr.shape[:2]
    top = int(h * OVERLAY_CROP["top"])
    bottom = h - int(h * OVERLAY_CROP["bottom"])
    left = int(w * OVERLAY_CROP["left"])
    right = w - int(w * OVERLAY_CROP["right"])
    return image_bgr[top:bottom, left:right]


class InfraredSolarModulesDataset(Dataset):
    def __init__(self, csv_path, split: str, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_bgr = cv2.imread(row["filepath"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {row['filepath']}")

        if bool(row["needs_crop"]):
            image_bgr = _crop_overlay(image_bgr)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)

        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_IDX[row["label"]]
        return image, torch.tensor(label, dtype=torch.long)
