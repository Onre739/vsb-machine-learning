"""Model pro Milestone 2: předtrénovaná ResNet-50 (transfer learning, varianta A).

Navazuje na cvičení 4 (ML_04_TransferLearning_PyTorch): načtení vah z ImageNetu, výměna hlavy
`model.fc`, zmrazení přes `requires_grad = False`, zmrazené BatchNorm statistiky a různé learning
rate pro různé části sítě (parameter groups).
"""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

# Části ResNet-50 od vstupu k výstupu. Čím dál od vstupu, tím specifičtější vzory blok rozpoznává:
#   stem (conv1 + bn1)  -> hrany, barvy           (obecné, hodí se pro jakákoliv data)
#   layer1 .. layer4    -> textury -> části -> objekty
#   fc                  -> naše nová hlava pro 23 tříd (náhodné váhy)
BLOCKS = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]


def build_resnet50(n_classes, dropout=0.2):
    """ResNet-50 s vahami z ImageNetu a novou hlavou Dropout -> Linear(2048 -> n_classes)."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Původní hlava: Linear(2048 -> 1000 tříd ImageNetu). Nahradíme ji vlastní (nová vrstva je trénovatelná).
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, n_classes),
    )
    return model


def get_blocks(model):
    """Slovník název bloku -> seznam jeho modulů (stem je složený ze dvou modulů)."""
    return {
        "stem": [model.conv1, model.bn1],
        "layer1": [model.layer1],
        "layer2": [model.layer2],
        "layer3": [model.layer3],
        "layer4": [model.layer4],
        "fc": [model.fc],
    }


def block_params(model, name):
    """Všechny parametry jednoho bloku."""
    return [p for module in get_blocks(model)[name] for p in module.parameters()]


def set_trainable(model, trainable_blocks):
    """Zmrazí celou síť a odmrazí jen vyjmenované bloky (hlava `fc` je trénovatelná vždy).

    Zmrazený parametr (requires_grad=False) nedostane gradient, takže ho optimizer nezmění.
    """
    for p in model.parameters():
        p.requires_grad = False
    for name in set(trainable_blocks) | {"fc"}:
        for p in block_params(model, name):
            p.requires_grad = True


def freeze_bn_stats(model):
    """BatchNorm ve zmrazených blocích přepne do eval režimu (bn_frozen ze cvičení 4).

    BatchNorm si kromě vah pamatuje i průměr a rozptyl dat (running statistiky), které se v režimu
    train() přepisují podle aktuálních batchů — i když jsou jeho váhy zmrazené. Ve zmrazených
    blocích chceme zachovat statistiky z ImageNetu, proto je přepneme do eval(). Volá se po
    každém model.train(), protože ten přepne do train režimu celou síť.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
            m.eval()


def count_params(model):
    """(všechny parametry, trénovatelné parametry) — nejužitečnější kontrola v transfer learningu."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_groups(model, lrs):
    """Skupiny parametrů s vlastním learning rate pro každý blok (discriminative learning rates).

    lrs: slovník blok -> lr, např. {"layer3": 1e-5, "layer4": 1e-4, "fc": 1e-3}.
    Do optimizeru dáme rovnou všechny bloky, které budeme během tréninku odmrazovat — dokud je
    blok zmrazený, jeho parametry nemají gradient a optimizer je přeskočí. Po odmrazení se začnou
    učit se svým lr, aniž bychom museli vytvářet nový optimizer.
    """
    return [{"params": block_params(model, name), "lr": lr, "name": name}
            for name, lr in lrs.items()]
