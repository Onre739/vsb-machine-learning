"""Trénink pro Milestone 2: CutMix/Mixup, warmup + cosine, postupné odmrazování a logy experimentů.

Každý experiment je popsaný jedním slovníkem `config` a jeho výsledky se ukládají do runs/<name>/:
    config.json  — kompletní nastavení (pro reprodukci)
    log.csv      — metriky po každé epoše
    best.pt      — váhy z epochy s nejlepší validační accuracy (mimo git)
Když už experiment se stejnou konfigurací proběhl, fit() ho netrénuje znovu, jen načte výsledky.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import build_transforms, make_loaders
from src.models import build_resnet50, count_params, freeze_bn_stats, param_groups, set_trainable


def set_seed(seed):
    """Pevný seed + deterministické cuDNN => stejná čísla při každém spuštění (jako v M1)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------------------------
# CutMix / Mixup
# ---------------------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------------------------
# Plánovač learning rate: lineární warmup + cosine
# ---------------------------------------------------------------------------------------------
def make_scheduler(optimizer, total_steps, warmup_steps):
    """Násobitel lr podle kroku: 0 -> 1 lineárně během warmupu, pak po kosinu z 1 k 0.

    Plánovač se volá po každém batchi. Násobitel se aplikuje na základní lr KAŽDÉ skupiny
    parametrů zvlášť, takže poměry mezi bloky (discriminative LR) zůstávají zachované.
    """
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------------------------
# Jedna epocha tréninku / validace
# ---------------------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, mix_cfg=None):
    model.train()
    freeze_bn_stats(model)                  # BatchNorm ve zmrazených blocích zůstane v eval režimu
    mix_cfg = mix_cfg or {}
    total_loss, correct, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x, y_a, y_b, lam = mix_batch(x, y, **mix_cfg)

        optimizer.zero_grad()
        logits = model(x)
        # loss smíchaného obrázku = stejně smíchané loss vůči oběma labelům
        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        scheduler.step()                    # plánovač po každém batchi (warmup je v krocích)

        pred = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        # u smíchaného batche počítáme zásah vážený stejně jako label
        correct += (lam * (pred == y_a).sum() + (1.0 - lam) * (pred == y_b).sum()).item()
        n += y.size(0)

    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------------------------
# Celý experiment
# ---------------------------------------------------------------------------------------------
def blocks_for_epoch(unfreeze_schedule, epoch):
    """Které bloky jsou v dané epoše odmrazené. Plán {"1": [], "4": ["layer4"]} znamená:
    epochy 1–3 jen hlava, od epochy 4 navíc layer4."""
    start = max(int(e) for e in unfreeze_schedule if int(e) <= epoch)
    return unfreeze_schedule[str(start)]


def fit(config, splits, n_classes, device, runs_dir="runs", retrain=False):
    """Natrénuje (nebo načte) experiment. Vrací (model s nejlepšími vahami, log jako DataFrame)."""
    cfg = json.loads(json.dumps(config))                 # kopie ve stejné podobě, jakou má config.json
    run_dir = Path(runs_dir) / cfg["name"]
    cfg_path, log_path, ckpt_path = run_dir / "config.json", run_dir / "log.csv", run_dir / "best.pt"

    # --- experiment už proběhl: jen načíst -------------------------------------------------------
    if not retrain and ckpt_path.exists() and log_path.exists():
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        if saved != cfg:
            raise ValueError(f"runs/{cfg['name']} už existuje s jinou konfigurací — "
                             "změň name, nebo spusť s retrain=True.")
        model = build_resnet50(n_classes, dropout=cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log = pd.read_csv(log_path)
        print(f"[{cfg['name']}] načteno z {run_dir} (nejlepší val acc {log['val_acc'].max():.1%})")
        return model, log

    # --- nový trénink ---------------------------------------------------------------------------
    set_seed(cfg["seed"])
    model = build_resnet50(n_classes, dropout=cfg["dropout"]).to(device)   # až po seedu => stejná hlava
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    train_tf, eval_tf = build_transforms(cfg["img_size"], cfg["random_erasing"])
    loaders = make_loaders(splits, train_tf, eval_tf, cfg["batch_size"], cfg["num_workers"])

    optimizer = torch.optim.AdamW(param_groups(model, cfg["lrs"]), weight_decay=cfg["weight_decay"])
    head_group = next(g for g in optimizer.param_groups if g["name"] == "fc")   # pro log lr hlavy
    steps_per_epoch = len(loaders["train"])
    scheduler = make_scheduler(optimizer, cfg["epochs"] * steps_per_epoch,
                               int(cfg["warmup_epochs"] * steps_per_epoch))
    train_crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    eval_crit = nn.CrossEntropyLoss()                    # validace bez smoothingu => srovnatelná loss
    mix_cfg = {"mixup_alpha": cfg["mixup_alpha"], "cutmix_alpha": cfg["cutmix_alpha"],
               "prob": cfg["mix_prob"]}

    rows, best_acc, prev_blocks = [], -1.0, None
    t_start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        blocks = blocks_for_epoch(cfg["unfreeze_schedule"], epoch)
        if blocks != prev_blocks:
            set_trainable(model, blocks)
            print(f"[{cfg['name']}] epocha {epoch}: odmrazeno {blocks or 'nic'} + hlava "
                  f"-> trénovatelných {count_params(model)[1]:,} parametrů")
            prev_blocks = blocks

        t0 = time.perf_counter()
        lr_head = head_group["lr"]                        # lr hlavy na začátku epochy (pro graf)
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], train_crit, optimizer,
                                          scheduler, device, mix_cfg)
        va_loss, va_acc = evaluate(model, loaders["val"], eval_crit, device)

        is_best = va_acc > best_acc
        if is_best:
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
        rows.append({"epoch": epoch, "blocks": "+".join(blocks) or "-", "lr_head": lr_head,
                     "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                     "val_acc": va_acc, "seconds": round(time.perf_counter() - t0, 1)})
        pd.DataFrame(rows).to_csv(log_path, index=False)  # průběžně — log přežije i přerušený trénink
        print(f"[{cfg['name']}] epocha {epoch:2d}/{cfg['epochs']} | train loss {tr_loss:.3f} acc {tr_acc:6.1%} "
              f"| val loss {va_loss:.3f} acc {va_acc:6.1%} | {rows[-1]['seconds']:5.1f} s{' ⭐' if is_best else ''}",
              flush=True)

    print(f"[{cfg['name']}] hotovo za {(time.perf_counter() - t_start) / 60:.1f} min, "
          f"nejlepší val acc {best_acc:.1%}")
    del loaders                                           # ukončí workery DataLoaderu
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, pd.DataFrame(rows)
