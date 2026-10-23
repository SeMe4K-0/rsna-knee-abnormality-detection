"""Kaggle inference for our own arms, one decode per study: test DICOM -> corpus96 volume (scripts/build_corpus96.py, files read
from disk) -> (1) 3D video models (weights/r3d_*.pt / r2plus1d_*.pt, plain + mirror) -> $KNEE_OUT/_r3d.csv (12 columns)
               (2) anatomy specialists (weights/spec_*.pt, scripts/spec_views.py windows) -> $KNEE_OUT/_spec.csv (one column
                   per finding, from spec_<view>_<finding>[_<seed>].pt files, e.g. spec_lat_LatOA.pt + spec_lat_LatOA_s2.pt are averaged).
Knee side and series orientation come from the DICOM headers of the chosen series (scripts/build_geom.py logic). Volumes are
brought to the orientation every train study had (coronal columns -> patient Left, sagittal columns -> Posterior and slices
Left -> Right, axial rows -> Posterior), so the specialists see what they were trained on. Runs as its own process.
"""
import glob
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG)
import build_corpus96 as C                                                   # noqa: E402
import raptor_lib as R                                                       # noqa: E402
import spec_views as SV                                                      # noqa: E402
from train_3d import BOUND, CROP, IMG, SAG, Net3D                            # noqa: E402

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TAG = {"LatOA": "Lateral OA", "LatMen": "Lateral Meniscus", "PFOA": "PF OA"}
INPUT = os.environ.get("KNEE_INPUT", "/kaggle/input")
OUT = os.environ.get("KNEE_OUT", "/kaggle/working")
RES3D = 192


class _Disk:
    @staticmethod
    def read(path):
        with open(path, "rb") as f:
            return f.read()


def _init(comp):
    C._z = _Disk()
    t = pd.read_csv(f"{comp}/test_series.csv", dtype=str)
    C._ser = {k: v.to_dict("records") for k, v in t.groupby("StudyInstanceUID")}
    C._index = defaultdict(list)
    for p in glob.glob(f"{comp}/test_series/*/*/*.dcm"):
        parts = p.replace("\\", "/").split("/")
        C._index[(parts[-3], parts[-2])].append(p)


def _geom(sid):
    """side (L/R) + per-slot orientation flags, same series choice as build_corpus96.build."""
    rows = C._ser.get(sid, []); used = set(); cxs = []; o = {}
    for k, (plane, fluid, _) in enumerate(C.SLOTS):
        r = C._pick(rows, plane, fluid, used)
        if r is None:
            continue
        used.add(r["SeriesInstanceUID"])
        files = C._index.get((sid, r["SeriesInstanceUID"]), [])
        if not files:
            continue
        try:
            h = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
            iop = np.array(h.ImageOrientationPatient, float); ipp = np.array(h.ImagePositionPatient, float)
            ps = np.array(h.PixelSpacing, float); rv, cv = iop[:3], iop[3:]
            cxs.append((ipp + rv * ps[1] * int(h.Columns) / 2 + cv * ps[0] * int(h.Rows) / 2)[0])
            o[k] = (rv, cv, np.cross(rv, cv))
        except Exception:
            continue
    side = ("L" if np.median(cxs) > 0 else "R") if cxs else "R"
    return side, o


def _canon(vol, o):
    """bring every slot to the train orientation (in place on a copy)."""
    vol = vol.copy()
    for k, (rv, cv, nv) in o.items():
        a, b = SV.B[k], SV.B[k + 1]
        if k in (0, 1):                                                       # sagittal
            if rv[1] < 0:
                vol[a:b] = vol[a:b, :, ::-1]
            if nv[0] > 0:                                                     # slices must run patient Left -> Right
                vol[a:b] = vol[a:b][::-1]
        else:                                                                 # coronal / axial: columns -> patient Left
            if rv[0] < 0:
                vol[a:b] = vol[a:b, :, ::-1]
            if k == 4 and cv[1] < 0:                                          # axial rows -> Posterior
                vol[a:b] = vol[a:b, ::-1, :]
    return vol


def _one(sid):
    try:
        vol, mask, _ = C.build(sid)
        side, o = _geom(sid)
        return sid, vol, mask, side, o, None
    except Exception as e:
        return sid, None, None, None, None, repr(e)[:200]


def to_input3d(vol, mask, mirror):
    lo = (IMG - CROP) // 2
    v = torch.from_numpy(vol[:, lo:lo + CROP, lo:lo + CROP].copy()).float()
    v = F.interpolate(v[None], size=(RES3D, RES3D), mode="area")[0]
    if mirror:
        parts = [v[BOUND[s]:BOUND[s + 1]] for s in range(5)]
        v = torch.cat([p.flip(0) if s in SAG else p.flip(-1) for s, p in enumerate(parts)], 0)
    present = torch.tensor([bool(mask[BOUND[s]:BOUND[s + 1]].any()) for s in range(5)])
    return v.round().clamp(0, 255).to(torch.uint8), present


def main():
    comp = os.path.dirname(sorted(glob.glob(f"{INPUT}/**/test_series.csv", recursive=True))[0])
    uids = pd.read_csv(f"{comp}/test.csv", dtype=str).StudyInstanceUID.tolist()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m3d = []
    for f in sorted(glob.glob(f"{PKG}/weights/*.pt")):
        b = os.path.basename(f)
        if b.startswith(("r3d", "r2plus1d")):
            m = Net3D(pretrained=False, arch="r2plus1d_18" if "r2plus1d" in b else "r3d_18")
            m.load_state_dict({k: v.float() for k, v in torch.load(f, map_location="cpu").items()})
            m3d.append(m.to(dev).eval())
    specs = []
    R.set_view(SV.W, 320)
    for f in sorted(glob.glob(f"{PKG}/weights/spec_*.pt")):
        view, tag = os.path.basename(f)[:-3].split("_")[1:3]
        ck = torch.load(f, map_location="cpu", weights_only=False)
        bb = R.build_backbone(ck.get("arch", R.ARCH)); m = R.RaptorClassifier(bb, F_dim=bb.num_features)
        m.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
        specs.append((view, TAG[tag], m.to(dev).eval()))
    print(f"ours: {len(m3d)} 3D model(s), specialists {[(v, t) for v, t, _ in specs]}, {len(uids)} studies, dev {dev}", flush=True)
    t0 = time.time(); p3d, psp, errs, sides = {}, {}, [], defaultdict(int)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(comp,)) as ex, torch.no_grad():
        for i, (sid, vol, mask, side, o, e) in enumerate(ex.map(_one, uids, chunksize=2)):
            if e:
                errs.append((sid, e)); continue
            if m3d:
                ps = []
                for mir in (False, True):
                    v, p = to_input3d(vol, mask, mir)
                    for m in m3d:                                            # fp32 on purpose (fp16 bit us once)
                        ps.append(torch.sigmoid(m(v[None].to(dev), p[None].to(dev)).float())[0].cpu().numpy())
                p3d[sid] = np.mean(ps, 0)
            if specs:
                cv_ = _canon(vol, o); sides[side] += 1
                acc = defaultdict(list)                                      # several seeds of one specialist -> mean
                for view, lab, m in specs:
                    acc[lab].append(float(SV.predict_study(m, cv_, mask, side, view, dev, res=320, k=48, amp=torch.float16)[LAB.index(lab)]))
                psp[sid] = {lab: float(np.mean(v)) for lab, v in acc.items()}
            if i % 100 == 0:
                print(f"  {i}/{len(uids)} {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"ours done {(time.time() - t0) / 60:.1f} min | 3D {len(p3d)} | spec {len(psp)} | errors {len(errs)} | sides {dict(sides)}", flush=True)
    for e in errs[:5]:
        print("ERR", e)
    if m3d:
        sub = pd.DataFrame([p3d.get(u, np.full(12, 0.5)) for u in uids], columns=LAB)
        sub.insert(0, "StudyInstanceUID", uids); sub.to_csv(f"{OUT}/_r3d.csv", index=False)
    if specs:
        cols = sorted({lab for _, lab, _ in specs})
        sp = pd.DataFrame([[psp.get(u, {}).get(c, 0.5) for c in cols] for u in uids], columns=cols)
        sp.insert(0, "StudyInstanceUID", uids); sp.to_csv(f"{OUT}/_spec.csv", index=False)


if __name__ == "__main__":
    main()
