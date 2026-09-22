# -*- coding: utf-8 -*-

"""扫描 MUSDB18-HQ train 歌曲，找前 N 秒内 vocals/drums 非静音的"干净"样本。"""
from __future__ import annotations

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

ROOT = Path(_paths.MUSDB18_ROOT / "train")
SR = 44100
DUR = 30
N_SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def rms_db(path, n):
    y, _ = sf.read(str(path), dtype="float32", always_2d=True)
    y = y[:n]
    r = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    return 20 * np.log10(r)


songs = sorted(p.name for p in ROOT.iterdir() if (p / "mixture.wav").is_file())
n = int(SR * DUR)
print(f"{'#':>3} {'歌曲':<42}{'vocals dB':>11}{'drums dB':>10}  {'判定':<8}")
print("-" * 82)
clean = []
for i, s in enumerate(songs[:N_SCAN]):
    d = ROOT / s
    v = rms_db(d / "vocals.wav", n)
    r = rms_db(d / "drums.wav", n)
    ok = (v > -50) and (r > -50)
    if ok:
        clean.append((i, s))
    print(f"{i:>3} {s:<42}{v:>11.1f}{r:>10.1f}  {'✅干净' if ok else '❌有静音':<8}")
print("-" * 82)
print(f"前 {N_SCAN} 首中干净的: {len(clean)} 首")
print("推荐 10 首（序号）:", [i for i, _ in clean[:10]])
print("推荐 10 首（名称）:")
for i, s in clean[:10]:
    print(f"  [{i}] {s}")
