import torch
from PIL import Image
from torch.utils.data import Dataset


class SeaAnimalsDataset(Dataset):
    def __init__(self, df, transform=None):
        # df: tabulka se sloupci "path" a "label_idx"
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")   # vždy 3 kanály
        if self.transform is not None:
            img = self.transform(img)                       # PIL obrázek -> tenzor (3, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
