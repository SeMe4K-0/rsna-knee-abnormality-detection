"""3D model (approach B): every series slot of cache/corpus96 is a 3D volume read by a video ResNet (torchvision r3d_18,
Kinetics-400 weights, shared over slots) -> slot embedding -> per-finding attention over the 5 slots.
Different inductive bias from the 2.5D window models (Raptor, goodpjw, our v4-v6): the network sees the whole stack at once.

Slots of corpus96: Sag FS 26, Sag 22, Cor FS 18, Cor 12, Ax 18 slices; 384 px over 140 mm. Input: centre/random 320 crop -> 192 px.
python scripts/train_3d.py --fold 0 --out runs/r3d_f0          # fold 0 held out (CV vs report labels + gold58)
python scripts/train_3d.py --fold -1 --out runs/r3d_all        # all non-gold studies
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from train import LAB, macro_auc, make_folds

SLOTS = [26, 22, 18, 12, 18]
SAG = (0, 1)
BOUND = np.cumsum([0] + SLOTS)
IMG, CROP = 384, 320
MEAN, STD = torch.tensor([0.43216, 0.394666, 0.37645]), torch.tensor([0.22803, 0.22145, 0.216989])   # Kinetics


class Studies(Dataset):
    def __init__(self, uids, lab, root, train, res):
        self.uids, self.lab, self.root, self.train, self.res = list(uids), lab, root, train, res

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        vol = np.load(f"{self.root}/{u}.npy")                         # (96,384,384) uint8
        mask = np.array(json.load(open(f"{self.root}/{u}.json"))["mask"], np.uint8)
        rng = np.random.default_rng() if self.train else None
        lo = (IMG - CROP) // 2
        y = x = lo
        if self.train:
            y, x = (int(rng.integers(0, IMG - CROP + 1)) for _ in range(2))
        v = torch.from_numpy(vol[:, y:y + CROP, x:x + CROP].copy()).float()
        v = F.interpolate(v[None], size=(self.res, self.res), mode="area")[0]          # (96,res,res)
        if self.train and rng.random() < 0.5:                           # anatomical L/R mirror (as the public TTA)
            parts = [v[BOUND[s]:BOUND[s + 1]] for s in range(5)]
            parts = [p.flip(0) if s in SAG else p.flip(-1) for s, p in enumerate(parts)]
            v = torch.cat(parts, 0)
        present = torch.tensor([mask[BOUND[s]:BOUND[s + 1]].any() for s in range(5)])
        t = self.lab.loc[u, LAB].values.astype(np.float32); w = self.lab.loc[u, [l + "__w" for l in LAB]].values.astype(np.float32)
        return v.to(torch.uint8), present, torch.from_numpy(t), torch.from_numpy(w)


VIDEO = {  # torchvision video backbones (Kinetics-400): constructor, weights enum, feature size
    "r3d_18": ("r3d_18", "R3D_18_Weights", 512), "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights", 512),
    "mc3_18": ("mc3_18", "MC3_18_Weights", 512), "swin3d_t": ("swin3d_t", "Swin3D_T_Weights", 768),
}


class Net3D(nn.Module):
    def __init__(self, pretrained=True, drop=0.2, arch="r3d_18"):
        super().__init__()
        from torchvision.models import video as V
        ctor, wname, f = VIDEO[arch]
        bb = getattr(V, ctor)(weights=getattr(V, wname).DEFAULT if pretrained else None)
        if hasattr(bb, "fc"):
            bb.fc = nn.Identity()
        else:
            bb.head = nn.Identity()
        self.bb, self.arch = bb, arch
        self.slot_emb = nn.Parameter(torch.zeros(5, f))
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, f)); nn.init.trunc_normal_(self.clsW, std=0.02)
        self.clsb = nn.Parameter(torch.zeros(12))
        self.ckpt = False

    def encode(self, x):                                                # x (N,3,D,H,W)
        b = self.bb
        if not hasattr(b, "layer1"):                                    # swin3d: torchvision forward, stages checkpointed
            x = b.pos_drop(b.patch_embed(x))
            for st in b.features:
                x = checkpoint(st, x, use_reentrant=False) if (self.ckpt and self.training) else st(x)
            x = b.norm(x).permute(0, 4, 1, 2, 3)
            return b.avgpool(x).flatten(1)
        x = b.stem(x)
        for layer in (b.layer1, b.layer2, b.layer3, b.layer4):
            x = checkpoint(layer, x, use_reentrant=False) if (self.ckpt and self.training) else layer(x)
        return b.avgpool(x).flatten(1)

    def forward(self, v, present):                                     # v uint8 (B,96,H,W), present (B,5)
        B = v.shape[0]
        x = v.float() / 255.0
        feats = []
        for s in range(5):
            sl = x[:, BOUND[s]:BOUND[s + 1]]                            # (B,D,H,W)
            sl = sl[:, None].expand(-1, 3, -1, -1, -1)                   # grey -> 3 channels
            sl = (sl - MEAN.to(x.device).view(1, 3, 1, 1, 1)) / STD.to(x.device).view(1, 3, 1, 1, 1)
            feats.append(self.encode(sl).float())
        h = self.norm(torch.stack(feats, 1) + self.slot_emb)            # (B,5,512)
        a = self.att(h).masked_fill(~present[..., None], -1e4)
        a = torch.softmax(a, dim=1)
        return (torch.einsum("bsn,bsf->bnf", a, h) * self.clsW).sum(-1) + self.clsb


def jitter(v, dev):
    x = v.to(dev, non_blocking=True)
    B = x.shape[0]
    g = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.2
    gam = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.3
    return ((x.float() / 255.0).clamp(0, 1) ** gam * g).clamp(0, 1).mul(255).round().to(torch.uint8)


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); P = []
    for v, p, *_ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(torch.sigmoid(model(v.to(dev), p.to(dev)).float()).cpu().numpy())
    return np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="-1 = all non-gold studies")
    ap.add_argument("--labels", default="labels/labels_teach.csv")
    ap.add_argument("--eval_labels", default="labels/labels_v1.csv")
    ap.add_argument("--root", default="cache/corpus96")
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="r3d_18", help="r3d_18 | r2plus1d_18 | mc3_18 | swin3d_t")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--eval_from", type=int, default=4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr_bb", type=float, default=2e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=400)
    ap.add_argument("--out", default="runs/r3d_f0")
    ap.add_argument("--gold_frac", type=float, default=0.0, help="--fold -1: also train on the 58 gold studies, drawn with this probability per sample (e.g. .10)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    if os.name == "nt":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    lab = pd.read_csv(a.labels).set_index("StudyInstanceUID"); ev = pd.read_csv(a.eval_labels).set_index("StudyInstanceUID")
    folds = make_folds(lab); gold = list(lab.index[lab.is_gold])
    if a.fold >= 0:
        tr = list(lab.index[(folds != a.fold) & ~lab.is_gold]); va = list(lab.index[(folds == a.fold) & ~lab.is_gold])
    else:
        tr, va = list(lab.index[~lab.is_gold]), []
        if a.gold_frac > 0:
            tr = tr + gold                                                # gold studies join training (drawn with p = gold_frac)
    if a.limit:
        tr, va, gold = tr[:a.limit], va[:a.limit // 4], gold[:16]
    yv = ev.loc[va, LAB].values; wv = ev.loc[va, [l + "__w" for l in LAB]].values >= 1; yg = lab.loc[gold, LAB].values
    kw = dict(num_workers=a.workers, pin_memory=True, timeout=600 if a.workers else 0)
    if a.gold_frac > 0 and a.fold < 0:
        from torch.utils.data import WeightedRandomSampler
        isg = np.array([u in set(gold) for u in tr]); pw = np.where(isg, a.gold_frac / isg.sum(), (1 - a.gold_frac) / (~isg).sum())
        smp = WeightedRandomSampler(torch.as_tensor(pw, dtype=torch.double), num_samples=len(tr), replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
        print(f"gold oversampling: {int(isg.sum())} gold studies drawn with p={a.gold_frac} (gold58 below is then in-sample)", flush=True)
        dl = DataLoader(Studies(tr, lab, a.root, True, a.res), batch_size=a.bs, sampler=smp, drop_last=True, persistent_workers=False, **kw)
    else:
        dl = DataLoader(Studies(tr, lab, a.root, True, a.res), batch_size=a.bs, shuffle=True, drop_last=True, persistent_workers=False, **kw)
    dl_va = DataLoader(Studies(va, lab, a.root, False, a.res), batch_size=4, num_workers=2)
    dl_go = DataLoader(Studies(gold, lab, a.root, False, a.res), batch_size=4, num_workers=0)

    model = Net3D(arch=a.arch).to(dev); model.ckpt = a.ckpt
    bbp = list(model.bb.parameters()); ids = {id(p) for p in bbp}
    hd = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": bbp, "lr": a.lr_bb}, {"params": hd, "lr": a.lr_head}], weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_bb, a.lr_head], total_steps=a.epochs * len(dl), pct_start=0.1)
    best, hist, start, mid = -1.0, [], 0, 0
    ck = f"{a.out}/last.pt"
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        best, hist, start, mid = c["best"], c["hist"], c["epoch"] + 1, c.get("mid_it", 0)
        print(f"resumed after epoch {c['epoch']} + {mid} its", flush=True)

    def save(ep_done, mid_it=0):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                    "hist": hist, "epoch": ep_done, "mid_it": mid_it}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); tl = 0; n = 0; skip, mid = mid, 0
        for j, (v, p, t, w) in enumerate(dl):
            it = skip + j
            if it >= len(dl):
                break
            v = jitter(v, dev); p, t, w = p.to(dev), t.to(dev), w.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(v, p)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), t, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
            tl += loss.item(); n += 1
            if it % 100 == 0:
                print(f"  ep{ep} it{it}/{len(dl)} loss {tl / n:.4f} {time.time() - t0:.0f}s", flush=True)
            if a.save_every and it % a.save_every == a.save_every - 1:
                save(ep - 1, it + 1)
        if ep < a.eval_from and ep != a.epochs - 1:
            hist.append({"ep": ep, "loss": tl / max(n, 1)}); save(ep)
            print(f"ep{ep} loss {tl / max(n, 1):.4f} | (no eval) | {(time.time() - t0) / 60:.1f} min", flush=True)
            continue
        pg = predict(model, dl_go, dev); gv, _ = macro_auc(yg, pg)
        cv = float("nan")
        if va:
            pv = predict(model, dl_va, dev); cv, _ = macro_auc(yv, pv, wv)
        hist.append({"ep": ep, "loss": tl / max(n, 1), "cv": cv, "gold": gv})
        print(f"ep{ep} loss {tl / max(n, 1):.4f} | CV {cv:.4f} | gold58 {gv:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        score = gv if not va else cv
        torch.save(model.state_dict(), f"{a.out}/ep{ep}.pt")
        if score > best:
            best = score; torch.save(model.state_dict(), f"{a.out}/best.pt")
            if va:
                np.save(f"{a.out}/oof.npy", pv); pd.Series(va).to_csv(f"{a.out}/oof_uids.csv", index=False)
            np.save(f"{a.out}/gold.npy", pg)
        save(ep)
    json.dump(hist, open(f"{a.out}/hist.json", "w"), indent=1)
    print("R3D DONE", flush=True)


if __name__ == "__main__":
    main()
