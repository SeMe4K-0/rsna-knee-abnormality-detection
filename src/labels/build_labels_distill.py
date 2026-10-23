"""labels_distill: knowledge distillation targets from our best model (Raptor soup) for training other students.
confident report cells: 0.3 * labels_v1 + 0.7 * teacher (weight 1); silent Syn/Fx/Contusion cells: teacher (weight 1).
python scripts/build_labels_distill.py --teacher cache/teacher_soup4.csv --out labels/labels_distill.csv"""
import argparse
import numpy as np
import pandas as pd
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser(); ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--base", default="labels/labels_v1.csv", help="report labels mixed with the teacher (v7 = focused Opus/Sol relabel)")
ap.add_argument("--alpha", type=float, default=0.7, help="teacher share on confident cells")
a = ap.parse_args()
v1 = pd.read_csv(a.base).set_index("StudyInstanceUID")
te = pd.read_csv(a.teacher).set_index("StudyInstanceUID").reindex(v1.index)
assert te[LAB].notna().all().all(), "teacher predictions missing"
out = v1.copy(); ng = ~v1.is_gold
for l in LAB:
    conf, silent = ng & (v1[l + "__w"] >= 1), ng & (v1[l + "__w"] < 1)
    out.loc[conf, l] = (1 - a.alpha) * v1.loc[conf, l] + a.alpha * te.loc[conf, l]
    out.loc[silent, l] = te.loc[silent, l]; out.loc[silent, l + "__w"] = 1.0
out.round(4).to_csv(a.out)
print("written", a.out, "| mean |change| vs v1:", np.abs(out.loc[ng, LAB].values - v1.loc[ng, LAB].values).mean().round(4))
