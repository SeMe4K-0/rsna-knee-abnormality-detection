"""Fine-tune the public 0.949 Raptor/CoAtNet checkpoint (nartaa) on our labels, in its own input format (cache/corpus96).

Low LR, few epochs, EMA of the weights; random 3-slice windows, random L/R mirror, random 320 crop of the 384 volume.
--holdout 0 keeps fold 0 out (comparison base vs fine-tuned on it; the base saw fold 0 in its own training, so a win
for the fine-tuned model is conservative); --holdout -1 trains on all non-gold studies.

python scripts/train_raptor_ft.py --holdout 0 --labels labels/labels_v1.csv --out runs/ft_raptor_f0
Checkpoints are saved in the public format ({"model": state_dict, "arch", "res"}) so the public notebook can load them.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import raptor_lib as R
import spec_views as SV
from train import LAB, macro_auc, make_folds

BASE = "weights/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
SPEC, SIDE = None, {}                                                       # --spec: anatomy specialist (scripts/spec_views.py)


class Studies(Dataset):
    def __init__(self, uids, lab, k, root, seed):
        self.uids, self.lab, self.k, self.root, self.seed = list(uids), lab, k, root, seed
        self.spec, self.side = SPEC, SIDE                                   # kept on the instance: spawned workers re-import the module

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2**31))
        u = self.uids[i]
        vol, mask = R.load_study(self.root, u)
        D = vol.shape[0]
        if self.spec:                                                       # anatomy-targeted specialist windows
            tri = SV.windows(vol, mask, self.spec, self.side.get(u, "R"), self.k, rng, jitter=16)
            t = self.lab.loc[u, LAB].values.astype(np.float32); w = self.lab.loc[u, [l + "__w" for l in LAB]].values.astype(np.float32)
            return torch.from_numpy(tri), torch.from_numpy(t), torch.from_numpy(w)
        tri, cs = R.triplets(vol, R.random_centres(mask, D, self.k, rng))
        if rng.random() < 0.5:
            tri = R.mirror(tri, cs, D)
        span = R.IMG - R.CROP
        jit = span if R.CROP == R.INPUT_RES else max(8, span // 4)         # resized views: small jitter around the centre
        lo = (span - jit) // 2
        y, x = (lo + int(rng.integers(0, jit + 1)) for _ in range(2))
        tri = np.ascontiguousarray(tri[:, :, y:y + R.CROP, x:x + R.CROP])
        t = self.lab.loc[u, LAB].values.astype(np.float32); w = self.lab.loc[u, [l + "__w" for l in LAB]].values.astype(np.float32)
        return torch.from_numpy(tri), torch.from_numpy(t), torch.from_numpy(w)


def norm_batch(x, dev):
    x = x.to(dev, non_blocking=True).float() / 255.0                       # [B,K,3,crop,crop]
    B = x.shape[0]
    if R.CROP != R.INPUT_RES:
        K = x.shape[1]
        x = F.interpolate(x.flatten(0, 1), size=(R.INPUT_RES, R.INPUT_RES), mode="bilinear", align_corners=False).view(B, K, 3, R.INPUT_RES, R.INPUT_RES)
    g = 1 + (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5) * 0.2              # mild per-study intensity jitter
    gam = 1 + (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5) * 0.3
    x = (x.clamp(0, 1) ** gam * g).clamp(0, 1)
    return (x - R.MEAN.to(dev)[None]) / R.STD.to(dev)[None]


@torch.no_grad()
def evaluate(model, uids, root, dev, k=24, mirror=False):
    model.eval()
    if SPEC:
        return np.stack([SV.predict_study(model, *R.load_study(root, u), SIDE.get(u, "R"), SPEC, dev, res=R.INPUT_RES, k=max(k, 48), amp=torch.bfloat16) for u in uids])
    P = [R.predict_study(model, *R.load_study(root, u), dev, k=k, use_mirror=mirror, amp=torch.bfloat16) for u in uids]
    return np.stack(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0, help="fold kept out; -1 = train on all non-gold studies")
    ap.add_argument("--labels", default="labels/labels_v1.csv")
    ap.add_argument("--eval_labels", default="labels/labels_v1.csv")
    ap.add_argument("--root", default="cache/corpus96")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr_bb", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ft_raptor_f0")
    ap.add_argument("--save_every", type=int, default=400, help="mid-epoch resume checkpoint every N iterations")
    ap.add_argument("--freeze_bb", action="store_true", help="train only the head (attention + classifier); backbone frozen")
    ap.add_argument("--gold_full", action="store_true", help="gold58 evaluated like the Kaggle notebook: 94 windows + mirror")
    ap.add_argument("--gold_w", type=float, default=0.0, help="--holdout -1: also train on the 58 gold studies, loss weight x gold_w (0 = left out)")
    ap.add_argument("--gold_frac", type=float, default=0.0, help="with --gold_w > 0: draw the gold studies with this probability per sample (e.g. 0.10 = ~10%% gold batches); 0 = plain shuffle")
    ap.add_argument("--init", default=BASE, help="start checkpoint (public format), e.g. our soup4")
    ap.add_argument("--crop", type=int, default=320, help="px cropped from the 384 volume (nartaa 224 model: 307)")
    ap.add_argument("--res", type=int, default=320, help="model input size after resizing the crop (nartaa 224 model: 224)")
    ap.add_argument("--fw", default="", help='per-finding loss multipliers, e.g. "Lateral OA=3,Synovitis=3"; with --only the others get 0')
    ap.add_argument("--spec", default="", help="anatomy specialist windows: lat | pf (scripts/spec_views.py; sets the 224 -> --res view)")
    ap.add_argument("--stop_after", type=int, default=-1, help="end the run after this epoch index (the LR schedule still spans --epochs)")
    ap.add_argument("--only", action="store_true", help="specialist: loss only on the findings named in --fw")
    a = ap.parse_args()
    global SPEC, SIDE
    if a.spec:
        SPEC = a.spec; a.crop = SV.W
        SIDE = pd.read_csv("cache/geom_train.csv").set_index("StudyInstanceUID").side.to_dict()
    R.set_view(a.crop, a.res)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if os.name == "nt":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    dev = torch.device("cuda")

    lab = pd.read_csv(a.labels).set_index("StudyInstanceUID"); ev = pd.read_csv(a.eval_labels).set_index("StudyInstanceUID")
    folds = make_folds(lab)
    gold = list(lab.index[lab.is_gold])
    if a.holdout >= 0:
        tr = list(lab.index[(folds != a.holdout) & ~lab.is_gold]); va = list(lab.index[(folds == a.holdout) & ~lab.is_gold])
    else:
        tr, va = list(lab.index[~lab.is_gold]), []
        if a.gold_w > 0:                                                    # gold labels = the test semantics (radiologist image reads)
            g_ = list(lab.index[lab.is_gold]); tr = tr + g_
            for c_ in LAB:
                lab.loc[g_, c_ + "__w"] = lab.loc[g_, c_ + "__w"] * a.gold_w
    if a.limit:                                                            # smoke test: only studies already cached
        have = lambda us: [u for u in us if os.path.exists(f"{a.root}/{u}.npy")]
        tr, va, gold = have(tr)[:a.limit], have(va)[:a.limit // 4], have(gold)[:20]
    yv = ev.loc[va, LAB].values; wv = ev.loc[va, [l + "__w" for l in LAB]].values >= 1
    yg = lab.loc[gold, LAB].values

    model, ck0 = R.load_raptor(a.init, dev)
    fw = torch.full((len(LAB),), 0.0 if a.only else 1.0)
    for kv in filter(None, a.fw.split(",")):
        k_, v_ = kv.split("="); fw[LAB.index(k_.strip())] = float(v_)
    fw = fw.to(dev); print("finding loss weights", dict(zip(LAB, fw.tolist())), flush=True)
    if a.freeze_bb:                                                       # head-only: the public backbone cannot drift
        for p_ in model.backbone.parameters():
            p_.requires_grad_(False)
    else:
        model.backbone.set_grad_checkpointing(True)
    ema = {n: p.detach().clone() for n, p in model.state_dict().items()}
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = ([{"params": bb, "lr": a.lr_bb}] if bb else []) + [{"params": hd, "lr": a.lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    if a.gold_frac > 0 and a.gold_w > 0 and a.holdout < 0:                # explicit gold oversampling (loss-weight x30 barely changes the gold share per batch)
        from torch.utils.data import WeightedRandomSampler
        gset = set(lab.index[lab.is_gold]); isg = np.array([u in gset for u in tr]); ng = max(1, int(isg.sum()))
        pw = np.where(isg, a.gold_frac / ng, (1 - a.gold_frac) / max(1, int((~isg).sum())))
        sampler = WeightedRandomSampler(torch.as_tensor(pw, dtype=torch.double), num_samples=len(tr), replacement=True,
                                        generator=torch.Generator().manual_seed(a.seed))
        print(f"gold oversampling: {ng} gold studies drawn with p={a.gold_frac}", flush=True)
        dl = DataLoader(Studies(tr, lab, a.k, a.root, a.seed), batch_size=a.bs, sampler=sampler, drop_last=True,
                        num_workers=a.workers, pin_memory=True, timeout=600 if a.workers else 0)
    else:
        dl = DataLoader(Studies(tr, lab, a.k, a.root, a.seed), batch_size=a.bs, shuffle=True, drop_last=True,
                        num_workers=a.workers, pin_memory=True, timeout=600 if a.workers else 0)
    total = a.epochs * len(dl); warm = max(1, min(200, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

    gk, gm = (94, True) if a.gold_full else (24, False)
    start, hist, mid_it, mid_tl = 0, [], 0, 0.0
    ck_path = f"{a.out}/last.pt"
    if os.path.exists(ck_path):
        c = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        ema = c["ema"]; start, hist = c["epoch"] + 1, c["hist"]
        mid_it, mid_tl = c.get("mid_it", 0), c.get("mid_tl", 0.0)
        print(f"resumed after epoch {c['epoch']} + {mid_it} iterations of the next", flush=True)
    else:                                                                  # reference: the untouched base model
        pg = evaluate(model, gold, a.root, dev, gk, gm); gv, _ = macro_auc(yg, pg)
        cv = macro_auc(yv, evaluate(model, va, a.root, dev), wv)[0] if va else float("nan")
        hist.append({"ep": -1, "cv": cv, "gold": gv})
        print(f"base | holdout CV {cv:.4f} | gold58 {gv:.4f}", flush=True)

    def ema_model():
        m = R.RaptorClassifier(R.build_backbone(ck0.get("arch", R.ARCH)), F_dim=model.backbone.num_features).to(dev)
        m.load_state_dict(ema); return m

    def save_last(ep_done, mid=0, tl_mid=0.0):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "ema": ema,
                    "epoch": ep_done, "hist": hist, "mid_it": mid, "mid_tl": tl_mid}, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    for ep in range(start, a.epochs):
        if 0 <= a.stop_after < ep:
            break
        model.train(); t0 = time.time(); skip, tl = mid_it, mid_tl; mid_it, mid_tl = 0, 0.0
        if a.freeze_bb:
            model.backbone.eval()                                        # frozen: keep BN statistics too
        for j, (x, t, w) in enumerate(dl):
            it = skip + j
            if it >= len(dl):
                break
            x = norm_batch(x, dev); t, w = t.to(dev), w.to(dev) * fw
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), t, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for n, p in model.state_dict().items():
                    if p.dtype.is_floating_point:
                        ema[n].mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
                    else:
                        ema[n].copy_(p)
            tl += loss.item()
            if it % 100 == 0:
                print(f"  ep{ep} it{it}/{len(dl)} loss {tl / (it + 1):.4f} {time.time() - t0:.0f}s", flush=True)
            if a.save_every and it % a.save_every == a.save_every - 1:
                save_last(ep - 1, it + 1, tl)
        m = ema_model()
        pg = evaluate(m, gold, a.root, dev, gk, gm); gv, gper = macro_auc(yg, pg); np.save(f"{a.out}/gold_ep{ep}.npy", pg)
        cv = float("nan")
        if va:
            pv = evaluate(m, va, a.root, dev); cv, _ = macro_auc(yv, pv, wv); np.save(f"{a.out}/holdout_ep{ep}.npy", pv)
        hist.append({"ep": ep, "loss": tl / len(dl), "cv": cv, "gold": gv, "min": (time.time() - t0) / 60})
        print(f"ep{ep} loss {tl / len(dl):.4f} | holdout CV {cv:.4f} | gold58 {gv:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        out = {"model": {n: v.detach().cpu() for n, v in ema.items()}, "arch": ck0.get("arch", R.ARCH), "res": ck0.get("res", 384), "view": [a.crop, a.res], "spec": a.spec,
               "lab": LAB, "epoch": ep, "src": f"ft of {os.path.basename(a.init)} on {a.labels} fw[{a.fw}] only={a.only}", "gold_auc": gv}
        torch.save(out, f"{a.out}/ft_ep{ep}.pt")
        save_last(ep)
        del m; torch.cuda.empty_cache()
        if ep == a.stop_after:
            print(f"stopped after epoch {ep} (--stop_after)", flush=True); break
    json.dump(hist, open(f"{a.out}/hist.json", "w"), indent=1)
    print("FT DONE", flush=True)


if __name__ == "__main__":
    main()
