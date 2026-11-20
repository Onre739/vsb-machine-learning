"""Analýza chyb pro Milestone 2: predikce pro celou sadu a Grad-CAM."""
import torch
import torch.nn.functional as F


def predict(model, loader, device):
    """Logity a skutečné labely pro celý loader (v pořadí loaderu — u val/test bez shuffle = pořadí v CSV)."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


class GradCAM:
    """Grad-CAM: kam se síť „dívala“, když rozhodovala o dané třídě.

    1. Při průchodu sítí si uložíme výstup posledního konvolučního bloku (layer4: 2048 map 7 x 7).
       Každá mapa říká, KDE v obrázku je nějaký vzor.
    2. Spočítáme gradient skóre zvolené třídy podle těchto map. Průměr gradientu přes celou mapu = váha
       mapy: jak moc daný vzor zvyšuje skóre třídy.
    3. Vážený součet map (jen kladná část, ReLU) = heatmapa 7 x 7, kterou zvětšíme na velikost obrázku.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:       # pod torch.no_grad() gradient není (např. běžná predikce) — nic neukládat
            output.register_hook(lambda grad: setattr(self, "gradients", grad))   # gradient při backward()

    def remove(self):
        """Odpojí hook od modelu — po analýze se model chová zase úplně normálně."""
        self.handle.remove()

    def __call__(self, x, class_idx):
        """x: (1, 3, H, W) normalizovaný obrázek, class_idx: třída. Vrací heatmapu (H, W) v rozsahu 0–1."""
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)            # (1, 2048, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu()
