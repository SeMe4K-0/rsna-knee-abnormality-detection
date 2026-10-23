"""Weight soup of Raptor-format checkpoints (no evaluation): python scripts/make_soup.py OUT.pt a.pt:0.5 b.pt:0.25 c.pt:0.25"""
import sys, torch
out, parts = sys.argv[1], [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in sys.argv[2:]]
acc, meta = None, None
for path, w in parts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if acc is None:
        acc = {k: v.float() * w if v.is_floating_point() else v.clone() for k, v in sd.items()}
        meta = {k: v for k, v in ck.items() if k != "model"}
    else:
        for k, v in sd.items():
            if v.is_floating_point():
                acc[k] += v.float() * w
meta["src"] = "soup " + " + ".join(f"{w}*{p}" for p, w in parts)
torch.save(dict(meta, model=acc), out); print("saved", out, meta.get("arch"), meta.get("res"))
