# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
import os, soundfile as sf

base_root = _paths.MUSDB18_ROOT
tot = 0.0
n = 0
durs = []
for sub in ["train", "test"]:
    base = os.path.join(base_root, sub)
    for s in sorted(os.listdir(base)):
        p = os.path.join(base, s, "mixture.wav")
        if os.path.exists(p):
            i = sf.info(p)
            durs.append(i.duration)
            tot += i.duration
            n += 1

print("songs:", n)
print("total_dur_min:", round(tot / 60, 1))
print("mean_dur_s:", round(tot / n, 1))
print("min/max dur s:", round(min(durs), 1), round(max(durs), 1))
