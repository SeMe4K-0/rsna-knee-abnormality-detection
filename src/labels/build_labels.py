"""Build training labels v1 -> labels/labels_v1.csv

target[l] = 0.5 * LLM p  +  0.25 * steven_v2  +  0.25 * pilkwang_v2
  LLM = Sonnet (batches 000-021) or Luna 6 (batches 022-036), probabilities 0..1 under the host criteria.
  steven_v2 uses exactly 0.5 for "not addressed": there it is replaced by the LLM value.
weight[l] = 1, except 0.3 where the LLM says the report is silent (status U) on a finding whose
  silence is uninformative (Synovitis, Fracture, Contusion: gold positives 18-33% when silent).
The 58 gold studies keep their expert labels (weight 1) and are flagged is_gold so they can be held out.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from labels_load import LAB, load_sources, load_train

SILENT_UNINFORMATIVE = {"Synovitis", "Fracture", "Contusion"}

tr, gold = load_train()
S = load_sources(tr.index)
idm = pd.read_csv("labels/sonnet/id_map.csv").set_index("rid")

rows = []
for model, batches in [("sonnet", range(0, 22)), ("luna6", range(22, 37))]:
    for b in batches:
        for line in open(f"labels/{model}/out/batch_{b:03d}.jsonl", encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                rows.append({"StudyInstanceUID": idm.loc[o["id"], "StudyInstanceUID"], "llm": model,
                             **{l: v / 100 for l, v in zip(LAB, o["p"])},
                             **{l + "__s": c for l, c in zip(LAB, o["s"])}})
llm = pd.DataFrame(rows).set_index("StudyInstanceUID").reindex(tr.index)
assert llm[LAB].notna().all().all(), "every study must have LLM labels"

st = S["steven_v2"].copy()
st = st.where(~np.isclose(st, 0.5), llm[LAB])
target = 0.5 * llm[LAB] + 0.25 * st + 0.25 * S["pilkwang_v2"]

weight = pd.DataFrame(1.0, index=tr.index, columns=LAB)
for l in SILENT_UNINFORMATIVE:
    weight.loc[llm[l + "__s"] == "U", l] = 0.3

out = target.add_suffix("").copy()
for l in LAB:
    out[l + "__w"] = weight[l]
out["llm"] = llm["llm"]
out["is_gold"] = out.index.isin(gold.index)
out.loc[gold.index, LAB] = gold.values
out.loc[gold.index, [l + "__w" for l in LAB]] = 1.0
out.round(4).to_csv("labels/labels_v1.csv")
print(out.shape, out.llm.value_counts().to_dict(), "gold", int(out.is_gold.sum()))
print("mean target", out.loc[~out.is_gold, LAB].mean().round(3).to_dict())
print("share of down-weighted cells", {l: round((weight[l] < 1).mean(), 2) for l in SILENT_UNINFORMATIVE})

# how good is the blended target itself on the gold studies (before gold overwrite)?
from sklearn.metrics import roc_auc_score
a = {l: roc_auc_score(gold[l], target.loc[gold.index, l]) for l in LAB}
print("blended target vs gold58: macro", round(np.mean(list(a.values())), 4), {k: round(v, 3) for k, v in a.items()})
