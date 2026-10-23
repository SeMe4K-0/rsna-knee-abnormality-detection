"""Anatomy-targeted windows on cache/corpus96 for the specialist models (user idea 2026-10-14, lumbar lesson: show the
region of interest large).

corpus96 keeps the raw DICOM pixel orientation. cache/geom_train.csv shows it is the same for every train study:
coronal columns run to patient Left, sagittal columns to Posterior (anterior = image left), sagittal slices are
ordered from patient Left to Right (normal x = -1), axial columns to Left and rows to Posterior (anterior = image top).
So for a RIGHT knee the lateral side is image-left on coronal and the last slices of a sagittal stack; for a LEFT knee
image-right and the first slices. Left knees are canonicalised to look like right knees (coronal/axial flipped in
width, sagittal triplets reversed), so the specialist never needs mirror augmentation.

spec "lat" (Lateral OA, Lateral Meniscus): coronal slots, lateral 224x224 box at the joint line
                                         + sagittal slots, lateral half of the stack, central 224x224 box.
spec "pf"  (PF OA):                       axial slot, anterior-central 224x224 box (patella + trochlea)
                                         + sagittal slots, central 50 % of the stack, anterior 224x224 box.
Each window = 3 adjacent slices of one slot, cropped (224 px = 82 mm), later resized to the model input (320).
"""
import numpy as np
import torch
import torch.nn.functional as F

SLOTS = [26, 22, 18, 12, 18]
B = np.cumsum([0] + SLOTS)
W = 224
# per spec: list of (slots, (rel_lo, rel_hi) for a RIGHT knee, (y0, x0) box for a RIGHT knee)
SPECS = {
    "lat": [((2, 3), (0.0, 1.0), (80, 16)),            # coronal: joint line, lateral (image-left) half
            ((0, 1), (0.45, 1.0), (80, 80))],          # sagittal: lateral slices, central box
    "pf": [((4,), (0.0, 1.0), (16, 80)),               # axial: anterior (top) centre
           ((0, 1), (0.25, 0.75), (48, 0))],           # sagittal: central slices, anterior (left) box
    "syn": [((0, 1), (0.3, 0.7), (0, 0)),              # sagittal central: suprapatellar pouch (anterior-superior)
            ((0, 1), (0.3, 0.7), (140, 0)),            # sagittal central: Hoffa fat pad (anterior-inferior)
            ((4,), (0.0, 1.0), (16, 80))],             # axial: anterior recesses around the patella
    "men": [((2, 3), (0.0, 1.0), (80, 16)),            # coronal: lateral joint line (as "lat")
            ((0, 1), (0.5, 1.0), (100, 120))],         # sagittal lateral slices: joint line, posterior horn side
}


def _rel(D):
    s = np.searchsorted(B, np.arange(D), side="right") - 1
    return s, (np.arange(D) - B[s]) / np.maximum(1, np.array(SLOTS)[s] - 1)


def candidates(mask, spec, side):
    """centre slices allowed for this spec (both neighbours inside the same slot and present)."""
    D = len(mask); s, rel = _rel(D)
    out = []
    for slots, (lo, hi), box in SPECS[spec]:
        if side == "L" and 0 in slots:                  # sagittal stacks run Left -> Right: mirror the slice range
            lo, hi = 1 - hi, 1 - lo
        for c in range(1, D - 1):
            if s[c] in slots and s[c - 1] == s[c] == s[c + 1] and mask[c - 1] and mask[c] and mask[c + 1] and lo <= rel[c] <= hi:
                out.append((c, box, s[c]))
    return out


def windows(vol, mask, spec, side, k, rng=None, jitter=0):
    """uint8 (k,3,W,W) canonicalised windows; rng=None -> k evenly spaced over the candidates (eval)."""
    cand = candidates(mask, spec, side)
    if not cand:                                        # missing series: fall back to the centre of the volume
        D = len(mask); c = D // 2
        cand = [(c, (80, 80), 0)]
    if rng is None:
        idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
    else:
        idx = rng.choice(len(cand), k, replace=len(cand) < k)
    out = np.empty((k, 3, W, W), np.uint8)
    for i, j in enumerate(idx):
        c, (y0, x0), sl = cand[j]
        if side == "L" and sl in (2, 3, 4):              # coronal / axial: lateral box sits on the other side
            x0 = 384 - W - x0
        if jitter and rng is not None:
            y0 = int(np.clip(y0 + rng.integers(-jitter, jitter + 1), 0, 384 - W))
            x0 = int(np.clip(x0 + rng.integers(-jitter, jitter + 1), 0, 384 - W))
        t = vol[c - 1:c + 2, y0:y0 + W, x0:x0 + W]
        if side == "L":
            t = t[::-1] if sl in (0, 1) else t[:, :, ::-1]   # canonicalise to a right knee
        out[i] = t
    return out


@torch.no_grad()
def predict_study(model, vol, mask, side, spec, dev, res=320, k=48, amp=torch.float16):
    from raptor_lib import MEAN, STD
    x = torch.from_numpy(windows(vol, mask, spec, side, k)).to(dev).float() / 255.0
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    x = ((x - MEAN.to(dev)) / STD.to(dev))[None]
    with torch.autocast("cuda", dtype=amp):
        return torch.sigmoid(model(x).float())[0].cpu().numpy()
