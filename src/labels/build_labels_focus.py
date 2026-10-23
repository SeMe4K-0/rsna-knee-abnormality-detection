"""labels_v7: labels_v1 with the 4 hard findings re-labelled from the focused big-LLM runs (labels/focus/INSTRUCTIONS_FOCUS.md).

Synovitis, Lateral OA <- mean(Opus, GPT-6.1 Sol); Effusion, Contusion <- mean(Opus, Sol, Sonnet)   (gold58 choice, 2026-10-13)
weight: 1 where any focused labeller saw the finding mentioned (status N/D/Q/P), else the v1 weight (silent cells 0.3).
Gold studies keep their gold labels. The other 8 findings are unchanged.
python scripts/build_labels_focus.py --out labels/labels_v7.csv
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

F4 = ["Synovitis", "Lateral OA", "Effusion", "Contusion"]
SRC = {"Synovitis": ["opus", "sol"], "Lateral OA": ["opus", "sol"],
       "Effusion": ["opus", "sol", "sonnet"], "Contusion": ["opus", "sol", "sonnet"]}
RUNS = {"opus": "transfer/labeling_pkg/out_focus", "sol": "transfer/labeling_pkg_sol61/out_focus"}

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="labels/labels_v7.csv")
a = ap.parse_args()
idmap = pd.read_csv("labels/sonnet/id_map.csv").set_index("rid").StudyInstanceUID
P, S = {}, {}
for name, d in RUNS.items():
    rows = [json.loads(l) for f in sorted(glob.glob(f"{d}/batch_*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
    assert len(rows) == len(idmap), (name, len(rows))
    uid = [idmap[o["id"]] for o in rows]
    P[name] = pd.DataFrame([[v / 100 for v in o["p"]] for o in rows], index=uid, columns=F4)
    S[name] = pd.DataFrame([list(o["s"]) for o in rows], index=uid, columns=F4)
P["sonnet"] = pd.read_csv("labels/sonnet_labels.csv").set_index("StudyInstanceUID")[F4]

v1 = pd.read_csv("labels/labels_v1.csv").set_index("StudyInstanceUID")
gold, ng = v1[v1.is_gold], ~v1.is_gold
out = v1.copy()
print("gold58 AUC of the new target (prob mean / rank mean) vs best single source:")
for l in F4:
    src = [P[s][l].reindex(v1.index) for s in SRC[l]]
    new = pd.concat(src, axis=1).mean(1)
    rk = pd.concat([x.rank(pct=True) for x in src], axis=1).mean(1)
    g = gold.index
    print(f"  {l:11s} prob {roc_auc_score(gold[l], new[g]):.3f} | rank {roc_auc_score(gold[l], rk[g]):.3f} | "
          + " ".join(f"{s} {roc_auc_score(gold[l], P[s][l].reindex(g)):.3f}" for s in SRC[l]))
    mentioned = pd.concat([S[s][l].reindex(v1.index).isin(list("NDQP")) for s in ("opus", "sol")], axis=1).any(axis=1)
    out.loc[ng, l] = new[ng].round(4)
    out.loc[ng & mentioned, l + "__w"] = 1.0
    print(f"     non-gold mean {v1.loc[ng, l].mean():.3f} -> {out.loc[ng, l].mean():.3f} | weight-1 cells "
          f"{(v1.loc[ng, l + '__w'] >= 1).mean():.2f} -> {(out.loc[ng, l + '__w'] >= 1).mean():.2f} | "
          f"spearman vs v1 {v1.loc[ng, l].corr(out.loc[ng, l], method='spearman'):.3f}")
assert out[F4].notna().all().all()
out.to_csv(a.out)
print("written", a.out)
