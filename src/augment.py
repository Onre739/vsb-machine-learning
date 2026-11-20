"""Dynamická augmentace na úrovni batche: CutMix a Mixup (Milestone 2, experiment E3)."""
import math
import random

import numpy as np
import torch


def mix_batch(x, y, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0):
    """Smíchá obrázky v batchi s jinými obrázky téhož batche.

    Mixup:  nový obrázek = lam * obrázek_A + (1 - lam) * obrázek_B   (průhledné prolnutí)
    CutMix: do obrázku A se vloží obdélník z obrázku B; lam = podíl plochy, která zůstala z A
    lam se losuje z Beta(alpha, alpha). Label je pak „lam díl třídy A + (1 - lam) dílu třídy B“.

    Vrací (x, y_a, y_b, lam). Bez míchání je y_b = y_a a lam = 1.
    """
    use_mixup, use_cutmix = mixup_alpha > 0, cutmix_alpha > 0
    if not (use_mixup or use_cutmix) or random.random() > prob:
        return x, y, y, 1.0

    # když jsou zapnuté obě metody, pro každý batch se náhodně vybere jedna z nich
    do_cutmix = use_cutmix and (not use_mixup or random.random() < 0.5)
    alpha = cutmix_alpha if do_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)     # s kým se který obrázek smíchá

    if do_cutmix:
        H, W = x.shape[2], x.shape[3]
        cut = math.sqrt(1.0 - lam)                         # strana obdélníku tak, aby měl plochu (1 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = random.randrange(H), random.randrange(W)  # střed obdélníku
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)        # oprava: obdélník se mohl oříznout o okraj
    else:
        x = lam * x + (1.0 - lam) * x[perm]

    return x, y, y[perm], lam
