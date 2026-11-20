# 🌊 Klasifikace mořských živočichů

Semestrální projekt do předmětu **Machine Learning** (VŠB-TUO), student GEM0023.

Klasifikace fotografií mořských živočichů do **23 tříd** (želvy, medúzy, delfíni, chobotnice, korály…)
v čistém PyTorch — od vlastní datové pipeline a sítí trénovaných od nuly (Milestone 1) po transfer learning
s předtrénovanou ResNet-50 (Milestone 2).

## 📊 Výsledky (testovací sada, 2 040 obrázků)

| Model | Test accuracy | Test macro accuracy |
|---|---|---|
| Baseline CNN od nuly (M1) | 39,5 % | 35,9 % |
| Vylepšená CNN od nuly (M1) | 56,6 % | 54,1 % |
| ResNet-50, trénuje se jen hlava (M2 — E1) | 87,2 % | 85,3 % |
| + postupné odmrazování a různé lr (M2 — E2) | 90,3 % | 89,2 % |
| **+ CutMix / Mixup, Random Erasing, label smoothing (M2 — E3)** | **90,7 %** | **89,5 %** |

Hlavní zjištění: zdaleka nejvíc přinesl samotný **transfer learning** (+30,6 p. b.), fine-tuning přidal dalších
~3 p. b. a regularizace odstranila přeučení. Analýza chyb ukázala, že většina nejhorších „chyb“ modelu jsou
ve skutečnosti **chybné labely v datasetu**. Podrobnosti jsou v notebooku Milestone 2.

## 📁 Struktura repozitáře

| Cesta | Obsah |
|---|---|
| `ML_GEM0023_milestone_01.ipynb` | **Milestone 1:** EDA, vlastní `Dataset` a `DataLoader`, baseline CNN, ruční train/val smyčka, overfit sanity check, vylepšená CNN |
| `ML_GEM0023_milestone_02.ipynb` | **Milestone 2:** transfer learning (ResNet-50), experimenty E1–E3, srovnání, analýza chyb (confusion matrix, nejhorší vzorky, Grad-CAM), vyhodnocení na testu, shrnutí |
| `src/data.py` | split dat, `SeaAnimalsDataset`, transformace (224 px, ImageNet normalizace), DataLoadery |
| `src/models.py` | ResNet-50 s novou hlavou, zmrazování / odmrazování bloků, skupiny parametrů s různým lr |
| `src/train.py` | CutMix / Mixup, warmup + cosine plánovač, tréninková smyčka, `fit()` s logováním experimentů |
| `src/analysis.py` | predikce pro celou sadu, Grad-CAM |
| `runs/<experiment>/` | `config.json` (kompletní nastavení experimentu) a `log.csv` (metriky po epochách) |
| `splits/` | pevný stratifikovaný split train / val / test (70 / 15 / 15, seed 42) |
| `sea_animals_index.csv` | index obrázků po vyčištění (bez poškozených souborů a duplikátů) |
| `dataset.py` | `Dataset` pro Milestone 1 (notebook ho vytváří sám přes `%%writefile`) |

## ⚙️ Instalace

Vyzkoušeno na Windows 10, Python 3.13, NVIDIA GTX 1080 (8 GB).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

PyTorch se instaluje zvlášť z CUDA 12.6 wheelů — 2.14 je poslední verze s podporou GPU architektury Pascal.
Pro novější GPU stačí aktuální PyTorch podle [pytorch.org](https://pytorch.org/get-started/locally/).

## ▶️ Spuštění

1. Otevřít notebook a vybrat kernel z `.venv`.
2. **Dataset** ([Sea Animals Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste),
   ~340 MB) stáhne notebook Milestone 1 přes `kagglehub` do `data/sea_animals/`. Notebook Milestone 2
   předpokládá, že data už jsou stažená.
3. Notebook spustit přes **Run All**.

**Trénink v Milestone 2:** funkce `fit()` nejdřív zkontroluje `runs/<experiment>/`. Pokud tam jsou uložené
váhy a stejná konfigurace, experiment jen načte; jinak ho natrénuje. Natrénované váhy (`best.pt`, ~95 MB
na experiment) nejsou v gitu, takže na novém počítači se všechny tři experimenty natrénují znovu
(celkem ~35 minut na GTX 1080). Přetrénování lze vynutit proměnnou `RETRAIN = True`.

## 🔁 Reprodukovatelnost

- pevný seed a deterministické cuDNN (`torch.backends.cudnn.deterministic = True`) ⇒ opakovaný běh na stejném
  hardwaru dává stejná čísla (ověřeno),
- split uložený v `splits/`, konfigurace a logy každého experimentu v `runs/`,
- verze knihoven v `requirements.txt`.

Na jiné GPU nebo s jinou verzí knihoven se čísla mohou mírně lišit.

## 🛠️ Poznámky k prostředí

- **Windows + `num_workers > 0`:** `Dataset` definovaný přímo v buňce notebooku nejde použít ve workerech
  DataLoaderu (proces se spouští metodou *spawn* a třídu neumí naimportovat) ⇒ třídy jsou v `.py` souborech.
- **Antivirus kontrolující HTTPS** (např. Avast) způsobí při stahování chybu `CERTIFICATE_VERIFY_FAILED`.
  Řeší ji balíček `pip-system-certs` (je v `requirements.txt`).
- **Mixed precision (AMP)** se nepoužívá — na GPU bez tensor cores (Pascal) by trénink zpomalila.
