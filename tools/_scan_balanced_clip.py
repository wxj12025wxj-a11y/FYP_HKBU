# -*- coding: utf-8 -*-

"""为基准挑选「各 stem 能量均衡」的评测片段（song + offset）。

动机
----
MUSDB18 很多歌的**开头**是低频引子。实测 `A Classic Education - NightOwl` 前 10 s：
bass 占 85% 能量、鼓只占 1.4%，于是**所有模型**的 bass SDR 一起虚高到 20 dB 以上 ——
这种片段的数字无法外推。

本脚本对候选歌 × 候选 offset 网格，按「4 个 stem 的能量占比的最小值」打分
（越大越均衡），输出推荐片段。

用法: python tools/_scan_balanced_clip.py [--dur 10]
"""
from __future__ import annotations

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(_paths.MUSDB18_ROOT / "train")
SR = 44100
STEMS = ["vocals", "drums", "bass", "other"]

CANDS = [
    "A Classic Education - NightOwl",
    "ANiMAL - Clinic A",
    "Actions - One Minute Smile",
    "Creepoid - OldTree",
    "BigTroubles - Phantom",
    "Dark Ride - Burning Bridges",
    "Alexander Ross - Goodbye Bolero",
]


def load_stems(song: str):
    d = ROOT / song
    out = {}
    for st in STEMS:
        y, _ = sf.read(str(d / f"{st}.wav"), dtype="float32", always_2d=True)
        out[st] = y.mean(axis=1)          # mono
    return out


def shares(seg: dict[str, np.ndarray]) -> dict[str, float]:
    """各 stem 的功率占比（注意用 10**(dB/10)，不要写 10**(x)**2 —— 右结合会算错）。"""
    p = {k: float(np.mean(v ** 2)) + 1e-20 for k, v in seg.items()}
    tot = sum(p.values())
    return {k: 100.0 * v / tot for k, v in p.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=15.0)
    args = ap.parse_args()

    n = int(SR * args.dur)
    rows = []
    for song in CANDS:
        try:
            stems = load_stems(song)
        except Exception as e:
            print(f"  跳过 {song}: {e}")
            continue
        total = min(len(v) for v in stems.values())
        print(f"\n=== {song}  (可用 {total/SR:.0f}s)")
        for off in np.arange(0, max(1.0, total / SR - args.dur), args.step):
            s0 = int(off * SR)
            seg = {k: v[s0:s0 + n] for k, v in stems.items()}
            if any(len(v) < n for v in seg.values()):
                continue
            sh = shares(seg)
            mn = min(sh.values())
            rows.append((mn, song, float(off), sh))
            print(f"    offset {off:6.0f}s  最小占比 {mn:5.1f}%   "
                  + "  ".join(f"{k[:3]}={sh[k]:5.1f}%" for k in STEMS))

    rows.sort(reverse=True)
    print("\n" + "=" * 78)
    print("推荐片段（按均衡度排序，取 Top 6）：")
    for mn, song, off, sh in rows[:6]:
        print(f"  {mn:5.1f}%   {song:38s} offset={off:4.0f}s   "
              + "  ".join(f"{k}={sh[k]:.1f}%" for k in STEMS))
    print("\n建议评测命令：")
    for mn, song, off, sh in rows[:3]:
        print(f'  python tools/benchmark_model_universal.py --model all --song "{song}" --duration {args.dur:g} --offset {off:g}')


if __name__ == "__main__":
    sys.exit(main())
