"""Data pro Milestone 2: split, Dataset, transformace a DataLoadery.

Navazuje na Milestone 1 — stejný index (sea_animals_index.csv), stejný stratifikovaný split
(70/15/15, seed 42), stejná třída SeaAnimalsDataset. Mění se jen transformace: předtrénovaná
ResNet-50 očekává vstup 224 x 224 a normalizaci podle ImageNetu.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Statistiky ImageNetu — na nich byla ResNet-50 předtrénovaná, proto je musíme použít i my
# (ne vlastní mean/std z Milestone 1).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_splits(index_csv, out_dir, seed=42):
    """Stratifikovaný split 70/15/15 — stejný postup a seed jako v Milestone 1.

    Výsledek uloží do out_dir/{train,val,test}.csv. Pokud soubory už existují, jen je načte,
    takže split je pevný i při dalších spuštěních.
    """
    out_dir = Path(out_dir)
    paths = {name: out_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()):
        return {name: pd.read_csv(p) for name, p in paths.items()}

    data = pd.read_csv(index_csv)[["path", "label", "label_idx"]]
    train_df, temp_df = train_test_split(
        data, test_size=0.30, stratify=data["label_idx"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_idx"], random_state=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, df in splits.items():
        df = df.reset_index(drop=True)
        df.to_csv(paths[name], index=False)
        splits[name] = df
    return splits


class SeaAnimalsDataset(Dataset):
    """Stejný Dataset jako v Milestone 1: (obrázek po transformaci, label jako torch.long)."""

    def __init__(self, df, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def build_transforms(img_size=224, random_erasing=0.0):
    """Vrátí (train_tf, eval_tf).

    train: náhodný výřez + flip + mírná změna barev (jako vylepšený model v M1), volitelně
           Random Erasing (vymaže náhodný obdélník — až PO normalizaci, pracuje s tenzorem).
    eval:  Resize na 232 + středový výřez 224 — přesně preprocessing, se kterým byly váhy
           ResNet50_Weights.IMAGENET1K_V2 vyhodnocené (weights.transforms()).
    """
    train_steps = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if random_erasing > 0:
        train_steps.append(transforms.RandomErasing(p=random_erasing))

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 232 / 224)),     # 224 -> 232 (kratší strana)
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(train_steps), eval_tf


def make_loaders(splits, train_tf, eval_tf, batch_size=32, num_workers=4):
    """DataLoadery pro train/val/test. Shuffle jen u train; workery zůstávají běžet mezi epochami."""
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    return {
        "train": DataLoader(SeaAnimalsDataset(splits["train"], train_tf), shuffle=True, **kwargs),
        "val": DataLoader(SeaAnimalsDataset(splits["val"], eval_tf), shuffle=False, **kwargs),
        "test": DataLoader(SeaAnimalsDataset(splits["test"], eval_tf), shuffle=False, **kwargs),
    }
