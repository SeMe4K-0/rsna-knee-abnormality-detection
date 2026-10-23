"""S6 blend cell fix : probe_dual fills studies the S6 worker did not score
(deadline!) with the column MEAN, and the old check (value != 0.5) counted those rows as real.  New cell: the scored set is
read from the worker predictions.csv files; S6 is ranked among its scored studies only, and an unscored study takes the
base rank (no S6 vote).  With full coverage the output is identical to the old cell.  Coverage is printed (S6_COVERAGE).
python kaggle/blend/s6_fix.py kaggle/<slug>/<slug>.ipynb"""
import json, sys
NEW = '''import csv as _csv
import numpy as np, pandas as pd
SUB = '/kaggle/working/submission.csv'
try:
    assert S6_OK, 'S6 stage did not run'
    base = pd.read_csv(SUB)
    s6 = pd.read_csv('/kaggle/working/_s6fall.csv').set_index('StudyInstanceUID').reindex(base.StudyInstanceUID)
    cols = [c for c in base.columns if c != 'StudyInstanceUID']
    scored6 = set()
    for _f in (TMP / 'prediction_s6').glob('worker*/predictions.csv'):
        with open(_f, encoding='utf-8') as _fh:
            for _r in _csv.DictReader(_fh):
                try:
                    if all(0. <= float(_r[c]) <= 1. for c in cols):
                        scored6.add(_r['StudyInstanceUID'])
                except (KeyError, ValueError, TypeError):
                    pass
    m = base.StudyInstanceUID.isin(scored6).values & s6[cols].notna().all(axis=1).values
    print(f"S6_COVERAGE {m.mean():.1%} of {len(m)} studies scored by the S6 worker", flush=True)
    assert m.mean() > 0.5, 'S6 arm mostly missing'
    out = base.copy()
    for c in cols:
        rb = base[c].rank(pct=True).values
        r6 = rb.copy()
        r6[m] = pd.Series(s6[c].values[m]).rank(pct=True).values          # S6 ranked among its own scored studies
        out[c] = (1 - W_S6) * rb + W_S6 * r6                              # unscored study: base rank, no S6 vote
    base.to_csv('/kaggle/working/_before_s6.csv', index=False)
    out.to_csv(SUB + '.tmp', index=False); os.replace(SUB + '.tmp', SUB)
    print('S6 blended submission written', out.shape, flush=True)
except Exception as e:
    print('S6 ARM SKIPPED, submission kept:', repr(e), flush=True)
'''
p = sys.argv[1]; nb = json.load(open(p, encoding="utf-8")); n = 0
for c in nb["cells"]:
    s = "".join(c["source"])
    if "assert S6_OK, 'S6 stage did not run'" in s:
        c["source"] = NEW.splitlines(True); n += 1
assert n == 1, n
json.dump(nb, open(p, "w", encoding="utf-8"), indent=1); print(p, "S6 blend cell replaced")
