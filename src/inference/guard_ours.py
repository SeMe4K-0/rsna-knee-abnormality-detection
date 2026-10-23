"""Time guard for the 'ours' arm cell (3D + specialists): it had none, so a slow compact stage (flip TTA) could push the
notebook past 9 h.  The runner now gets timeout = time left - 30 min (and is skipped below 20 min); a timeout or failure
leaves the submission as it is (the blend cells already skip on a missing _r3d.csv / _spec.csv).
python kaggle/blend/guard_ours.py kaggle/<slug>/<slug>.ipynb"""
import json, sys
p = sys.argv[1]; nb = json.load(open(p, encoding="utf-8")); n = 0
old = '_r = subprocess.run([sys.executable, _runner[0]], env={**os.environ, "KNEE_OUT": "/kaggle/working"}, capture_output=True, text=True)\nprint(_r.stdout[-2500:]); print(_r.stderr[-1500:])\n'
new = ('_left = 9 * 3600 - (time.time() - NB_T0) - 30 * 60          # [guard] keep 30 min for S6 skip logic, blends and margin\n'
       'if _left < 20 * 60:\n    print(f"OURS_SKIPPED only {_left / 60:.0f} min left", flush=True)\nelse:\n'
       '    try:\n'
       '        _r = subprocess.run([sys.executable, _runner[0]], env={**os.environ, "KNEE_OUT": "/kaggle/working"}, capture_output=True, text=True, timeout=_left)\n'
       '        print(_r.stdout[-2500:]); print(_r.stderr[-1500:])\n'
       '    except Exception as _e:\n        print("OURS_RUNNER_STOPPED", repr(_e)[:300], flush=True)\n')
for c in nb["cells"]:
    s = "".join(c["source"])
    if old in s:
        c["source"] = s.replace(old, new).splitlines(True); n += 1
assert n >= 1, "ours runner cell not found"
json.dump(nb, open(p, "w", encoding="utf-8"), indent=1); print(p, "guarded cells:", n)
