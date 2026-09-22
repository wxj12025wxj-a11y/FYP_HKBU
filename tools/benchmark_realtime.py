# -*- coding: utf-8 -*-
"""
================================================================================
 实时性基准测试 (Real-Time Factor, RTF)
--------------------------------------------------------------------------------
 目的 : 对比各方法在 CPU 上处理 30 秒音频的耗时, 判断能否用于实时系统
 RTF  : 处理耗时 / 音频时长。RTF < 1 表示比实时快; > 1 表示跑不动实时
--------------------------------------------------------------------------------
 运行:
   python tools/benchmark_realtime.py
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT

sys.path.insert(0, str(_paths.TOOLS))

import numpy as np
import soundfile as sf

from compare_separation_methods import (   # noqa: E402
    DATA_ROOT, SR, list_songs, sep_ica, sep_nmf, sep_rpca, sep_hpss,
)

DURATION = 30
OUT = _paths.comparison_dir() / "realtime_benchmark.json"
ROWS: list[dict] = []


def bench(name, fn, n_rep=1, note=""):
    try:
        fn()                                     # 预热(不计时)
        t0 = time.time()
        for _ in range(n_rep):
            fn()
        t = (time.time() - t0) / n_rep
    except Exception as exc:
        print(f"{name:<34}{'  FAILED':>10}   {type(exc).__name__}: {exc}")
        return
    rtf = t / DURATION
    verdict = "OK  可实时" if rtf < 0.3 else ("WARN 勉强" if rtf < 1.0 else "NO  跑不动实时")
    print(f"{name:<34}{t:>10.3f}{rtf:>9.3f}   {verdict}")
    ROWS.append({"method": name, "seconds": round(t, 4), "rtf": round(rtf, 4),
                 "realtime_ok": rtf < 1.0, "note": note})


def main():
    song = list_songs("train")[0]
    y, _ = sf.read(str(DATA_ROOT / "train" / song / "mixture.wav"),
                   dtype="float32", always_2d=True)
    y = y[: int(SR * DURATION)]
    mono = y.mean(axis=1).astype(np.float32)
    stereo = y.T.astype(np.float32)

    print(f"音频: {song}")
    print(f"时长 {DURATION}s | sr={SR} | CPU 推理 | 指标: 处理耗时 与 RTF(=耗时/时长)")
    print(f"{'方法':<34}{'耗时(s)':>10}{'RTF':>9}   判定")
    print("-" * 74)

    bench("HPSS (percussive)", lambda: sep_hpss(mono, SR, "percussive"), 3)
    bench("ICA (FastICA)", lambda: sep_ica(stereo, SR), 3)
    bench("NMF (unsupervised, k=50)", lambda: sep_nmf(mono, SR), 1)
    bench("RPCA (Inexact ALM)", lambda: sep_rpca(mono, SR), 1)

    # Open-Unmix: 每次 separate() 都会重建 separator(含权重加载), 反映"每文件冷启动"开销
    try:
        import torch
        from openunmix import predict as umx_predict
        UMX_ORDER = ["vocals", "drums", "bass", "other"]

        def _umx_run():
            audio = torch.as_tensor(np.ascontiguousarray(stereo), dtype=torch.float32)
            return umx_predict.separate(audio=audio, rate=SR, targets=UMX_ORDER,
                                        model_str_or_path="umxhq", device="cpu",
                                        filterbank="torch")

        bench("Open-Unmix umxhq (4 轨, 冷启动)", _umx_run, 1,
              note="含每次调用的权重加载(未常驻内存)")
    except Exception:
        traceback.print_exc()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"song": song, "duration_s": DURATION, "sr": SR,
                   "note": "RTF = 耗时/音频时长; <1 表示比实时快",
                   "results": ROWS}, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 已保存 -> {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
