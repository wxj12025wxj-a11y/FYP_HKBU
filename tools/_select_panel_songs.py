# -*- coding: utf-8 -*-
"""为板 1 的图表挑选「有代表性且信息量足」的 test 片段（可复现）。

动机
----
原先手选 3 首（Siren / PR - Oh No / Knockout），但实测
`PR - Oh No` 全曲 vocals 只占 **1.1%** 能量（drums 64.6%）——
对「vocals 为目标」的分解/掩码面板是**退化样本**：
参考轨近乎静音，SDR 会掉到 0.45 dB、ISR 0.57 dB，
图看着"跑通了"，数字却毫无意义（本项目第 2 类陷阱：片段能量陷阱）。

挑选规则（写死，可复跑）
------------------------
1. 对每首 test 歌，在均匀候选 offset 上滑 20 s 窗；
2. 只保留**目标轨能量占比 ≥ `--target-min`** 的窗（默认 12%）；
3. 在这些窗里取「四轨占比最小值」最大的那个作为该歌的代表片段（越均衡越好）；
4. 歌曲按该分数排序，取 Top-N。

产出 : 04_reports/separation/data/comparison/panel_song_selection.json
用法 : python tools/_select_panel_songs.py --stem vocals --top 3
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths

_paths.setup_env()

import argparse
import json
import time

import numpy as np
import soundfile as sf

SR = 44100
STEMS = ["vocals", "drums", "bass", "other"]


def window_shares(d, s0, win):
    p = {}
    for s in STEMS:
        y, sr = sf.read(str(d / f"{s}.wav"), start=s0, stop=s0 + win,
                        dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{d/f'{s}.wav'} 采样率 {sr}")
        p[s] = float(np.mean(y.astype(np.float64) ** 2)) + 1e-20
    tot = sum(p.values())
    return {k: 100.0 * v / tot for k, v in p.items()}


def score_song(d, dur, n_windows, target, target_min):
    info = sf.info(str(d / "mixture.wav"))
    n = info.frames
    win = int(dur * SR)
    if n < win:
        return None
    offs = np.linspace(0, n - win, n_windows) / SR
    best_ok, best_any = None, None
    for off in offs:
        sh = window_shares(d, int(off * SR), win)
        mn = min(sh.values())
        rec = {"offset": float(round(off, 1)), "min_share_pct": float(mn),
               "shares": {k: round(v, 2) for k, v in sh.items()}}
        if best_any is None or mn > best_any["min_share_pct"]:
            best_any = rec
        if sh[target] >= target_min and (best_ok is None or mn > best_ok["min_share_pct"]):
            best_ok = rec
    return {"best_ok": best_ok, "best_any": best_any, "frames": int(n),
            "dur_s": round(n / SR, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="vocals")
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--windows", type=int, default=8, help="每首歌采几个候选窗")
    ap.add_argument("--target-min", type=float, default=12.0)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--min-dur", type=float, default=60.0, help="太短的歌排除")
    args = ap.parse_args()

    root = _paths.MUSDB18_ROOT / "test"
    songs = sorted([p.name for p in root.iterdir() if p.is_dir()])
    print(f"扫描 {len(songs)} 首 test 歌  (stem={args.stem}, dur={args.dur}s, "
          f"target_min={args.target_min}%, 每首 {args.windows} 窗)")
    t0 = time.time()

    rows = []
    for i, song in enumerate(songs, 1):
        try:
            r = score_song(root / song, args.dur, args.windows, args.stem, args.target_min)
        except Exception as e:
            print(f"  [skip] {song}: {e}")
            continue
        if r is None or r["dur_s"] < args.min_dur:
            continue
        pick = r["best_ok"] or r["best_any"]
        rows.append({"song": song, "qualified": r["best_ok"] is not None,
                     "dur_s": r["dur_s"], **pick})
        if i % 10 == 0:
            print(f"  ... {i}/{len(songs)}  已用 {time.time()-t0:.0f}s")

    qual = sorted([r for r in rows if r["qualified"]],
                  key=lambda r: -r["min_share_pct"])
    picked = qual[: args.top]

    print("\n" + "=" * 84)
    print("Top 15（合格：目标轨占比达标，按均衡度排序）")
    for r in qual[:15]:
        print("  %5.1f%%  %-42s off=%5.0fs dur=%4.0fs  目标=%5.1f%%"
              % (r["min_share_pct"], r["song"], r["offset"], r["dur_s"],
                 r["shares"][args.stem]))
    print("\n被排除的（目标轨占比从未达标）示例：")
    for r in sorted([r for r in rows if not r["qualified"]],
                    key=lambda r: -r["shares"][args.stem])[:5]:
        print("  %-42s 最好窗目标占比仅 %5.1f%%" % (r["song"], r["shares"][args.stem]))

    out = _paths.DATA_COMPARISON / "panel_song_selection.json"
    out.write_text(json.dumps({
        "criteria": {
            "stem": args.stem, "dur_s": args.dur, "windows_per_song": args.windows,
            "target_min_pct": args.target_min, "min_song_dur_s": args.min_dur,
            "rule": "目标轨占比 ≥ target_min 的窗中，取四轨占比最小值最大的窗；歌曲按该分数排序取 Top-N",
        },
        "picked": [r["song"] for r in picked],
        "picked_detail": picked,
        "ranking_top30": qual[:30],
        "scanned": len(rows), "qualified": len(qual),
        "elapsed_s": round(time.time() - t0, 1),
        "status": "PASS" if len(picked) >= args.top else "FAIL",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print("最终选歌：")
    for r in picked:
        print("  %-42s offset=%5.0fs  最小轨道=%5.1f%%  %s"
              % (r["song"], r["offset"], r["min_share_pct"],
                 " ".join("%s=%.0f%%" % (k[:3], r["shares"][k]) for k in STEMS)))
    print(f"\nstatus = {'PASS' if len(picked) >= args.top else 'FAIL'}  耗时 {time.time()-t0:.0f}s")
    print(f"[out] {out}")
    return 0 if len(picked) >= args.top else 1


if __name__ == "__main__":
    _sys.exit(main())
