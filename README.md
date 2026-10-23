<div align="center">

# RSNA Knee Abnormality Detection

**Gold-batch fine-tuning on top of a public ensemble**

Kaggle · RSNA Knee Abnormality Detection 2026 · knee MRI, 12 findings, macro ROC AUC

**220th of 6,425 teams** · private 0.944 · public 0.960

Author: Semyon Davshits

![Kaggle](https://img.shields.io/badge/Kaggle-code%20competition-20BEFF?logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-CoAtNet%20%7C%20ConvNeXt-FFD21E)

</div>

## 1. Overview

* Five readers score a canonical 2.5D view of each study; a fixed per-finding rank blend combines them.
* The 384 px and 224 px readers are public checkpoints fine-tuned on our labels and averaged with the originals (weight soups).
* The 58 radiologist-labelled studies are drawn explicitly into the training batches (probability *p* per sample) instead of being up-weighted in the loss.
* Our own 3D models and Lateral-OA specialists add a small rank share.

## 2. Architecture

| Reader | Model | Input | Origin |
|---|---|---|---|
| A | CoAtNet, 2.5D multiple-instance, per-finding attention | 384 px, 94 windows | public checkpoint, fine-tuned by us |
| A′ | same, local crop | 224 px | public checkpoint, fine-tuned by us |
| B | compact 2.5D reader, flip TTA | as published | public (heliosli) |
| C | ConvNeXt-S, all data | 288 px | public (nartaa) |
| D | r(2+1)D-18 ×2 + Lateral-OA specialist ×3 | 192 px volumes / 320 px crops | ours |

![pipeline](img/01_pipeline.png)

## 3. Input

Series are sorted by physical position and resampled into five fixed slots: sagittal fat-sat 26, sagittal 22, coronal fat-sat 18, coronal 12, axial 18 slices (96 in total).

![volume](img/02_volume.png)

Every slice is cropped to a 140 mm field centred on the knee.

![field](img/03_field.png)

A window is three neighbouring slices stacked as RGB; one window is centred on every slice (94 per study).

![window](img/04_window.png)

Left knees are mirrored to look like right knees.

![mirror](img/07_mirror.png)

## 4. Readers

Reader A embeds every window with one shared backbone; each of the 12 findings has its own attention over the windows. The mirrored study is scored again and averaged.

![mil](img/05_mil.png)

The Lateral-OA specialist sees 82 mm boxes of the lateral compartment on coronal and lateral sagittal slices, resized to 320 px.

![specialist](img/06_specialist.png)

## 5. Labels

* Reports read by language models into per-finding probabilities, averaged with public report-derived label sets.
* Synovitis, Lateral OA, Effusion and Contusion re-read by larger models.

![label reading](img/08_label_reading.png)

* Loss weight 0.3 where the report is silent on a finding that radiologists often leave unmentioned, 1.0 otherwise.
* 3D models: 0.7 × reader-A predictions + 0.3 × report label.

![label target](img/09_label_target.png)

| StudyInstanceUID | ACL … Fracture | ACL__w … Fracture__w | is_gold |
|---|---|---|---|
| `1.2.3.example` | 0.08 … 0.06 | 1.0 … 0.3 | False |

## 6. Training

### 6.1 Fine-tuning and weight soups

One epoch from the public checkpoint: low LR, EMA, random windows, random mirror, random crop.
Final reader A = `0.25 × public checkpoint + 4 × 0.1875 × gold-batch fine-tunes`.

```bash
python src/training/make_soup.py soup.pt public.pt:0.25 ft1.pt:0.1875 ft2.pt:0.1875 ft3.pt:0.1875 ft4.pt:0.1875
```

### 6.2 Gold batches

With batch size 2 most batches contain no gold study, so a larger loss weight hardly changes their share; sampling with probability *p* does.

![gold share](img/10_gold_share.png)

```bash
python src/training/train_raptor_ft.py --holdout -1 --gold_w 1 --gold_frac 0.50 --epochs 1        --init public.pt --labels labels.csv --out runs/gf50
```

Each point is one submission with the 384 px soup fine-tuned at that gold share; final soups use *p* = 0.50.

![gold scores](img/11_gold_scores.png)

### 6.3 Specialists and 3D models

* Lateral-OA specialist: reader A trained on Lateral OA and Lateral Meniscus only, 3 seeds.
* 3D model: r(2+1)D-18 over each series slot, attention over slots, 12 epochs on the distilled labels, 2 seeds.

## 7. Ensemble

Per finding, every reader is converted to ranks, then:

| Step | Blend |
|---|---|
| A-side | 0.65 × A + 0.35 × A′ |
| A-side vs B | A-side share: Synovitis 0.85, PF OA 0.65, Baker's 0.30, Lateral OA 0.15, other findings 0.50 |
| + D (3D) | rank share 0.15, all findings |
| + D (specialists) | rank share 0.30, Lateral OA only |
| + C | rank share 0.20, all findings |

![runtime](img/15_runtime.png)

## 8. Results

| Final submission | Public | Private |
|---|---|---|
| A = soup of 4 gold-batch fine-tunes (p = 0.50) | 0.960 | 0.944 |
| A = soup of 2 gold-batch fine-tunes, A′ + gold-batch fine-tune | 0.960 | 0.944 |

![history](img/12_history.png)

![public vs private](img/13_public_private.png)

![timeline](img/14_timeline.png)

## Credits

* **nartaa**: public CoAtNet 384/224 checkpoints (readers A, A′), reader C.
* **heliosli**: reader B, per-finding blend shares, flip TTA.
* Public report-derived label sets of other participants.
* MRI in the figures: knee MRI by Ptrump16, CC BY-SA 4.0, via Wikimedia Commons.
