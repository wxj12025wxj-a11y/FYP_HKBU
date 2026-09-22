# -*- coding: utf-8 -*-
"""
================================================================================
 Open-Unmix (umxhq)  vs  Demucs (HTDemucs)  —— 同口径头对头对比
--------------------------------------------------------------------------------
 目的 : 二者已有 SDR/ΔSDR/RTF 数据, 但 Demucs 独有 PESQ/STOI/FLOPs。
        本脚本用【同一个评估器】对两者的分离音频重算全套指标, 保证严格可比。
 数据 : A Classic Education - NightOwl 前 30s (与既有实验同源)
 评估 : museval BSSEval v4 (1s 窗, 中位数) + SI-SDR + PESQ(16k wb) + STOI
 输出 : outputs/comparison/umx_vs_demucs.json
--------------------------------------------------------------------------------
 运行: python tools/_umx_vs_demucs.py
================================================================================
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
ROOT = _paths.PROJECT_ROOT

sys.path.insert(0, str(_paths.TOOLS))

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import numpy as np
import librosa
import soundfile as sf

from compare_separation_methods import SR, evaluate  # noqa: E402

SONG = "A Classic Education - NightOwl"
DUR = 30.0
GT = _paths.MUSDB18_ROOT / "train" / SONG
UMX_DIR = _paths.OUTPUTS / "Open-Unmix" / "A_Classic_Education_-_NightOwl"
DMX_DIR = _paths.OUTPUTS / "Demucs" / "stems"
OUT = _paths.comparison_dir() / "umx_vs_demucs.json"


def load_mono(p: Path, dur: float = DUR) -> np.ndarray:
    x, sr = sf.read(str(p), always_2d=True)
    x = np.asarray(x, dtype=np.float32).mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    n = int(dur * SR)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return x[:n]


def _r16(x):
    return librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=SR, target_sr=16000)


def pesq_of(ref, deg):
    try:
        from pesq import pesq as _p
        return round(float(_p(16000, _r16(ref), _r16(deg), "wb")), 4)
    except Exception as exc:
        print("  [WARN] PESQ 不可用:", exc)
        return None


def stoi_of(ref, deg):
    try:
        from pystoi import stoi as _s
        return round(float(_s(_r16(ref), _r16(deg), 16000, extended=False)), 4)
    except Exception as exc:
        print("  [WARN] STOI 不可用:", exc)
        return None


def umx_flops(seconds: float = 10.0):
    """Open-Unmix 单目标 FLOPs。口径同 Demucs: torch FlopCounterMode,
    只统计 conv/linear/matmul (复数 FFT 不计), 属保守低估。
    注: Separator 传单个 target 会拒绝前向, 故传 2 个同构模型后折算单目标。"""
    import torch
    import openunmix
    from torch.utils.flop_counter import FlopCounterMode

    targets = ["vocals", "drums"]
    m = openunmix.umxhq(targets=targets, device="cpu").eval()
    x = torch.zeros(1, 2, int(seconds * SR))
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            m(x)
    total = float(fc.get_total_flops())
    return total / len(targets), seconds, total


def pack(name, ref, est, mix):
    base = evaluate(ref, mix)
    m = evaluate(ref, est)
    return {
        "method": name,
        "SDR": round(m["sdr"], 2) if m["sdr"] is not None else None,
        "delta_SDR": round(m["sdr"] - base["sdr"], 2) if m["sdr"] is not None else None,
        "SI_SDR": round(m["si_sdr"], 2),
        "ISR": round(m["isr"], 2) if m["isr"] is not None else None,
        "SAR": round(m["sar"], 2) if m["sar"] is not None else None,
        "PESQ": pesq_of(ref, est),
        "STOI": stoi_of(ref, est),
        "baseline_SDR": round(base["sdr"], 2) if base["sdr"] is not None else None,
        "baseline_PESQ": pesq_of(ref, mix),
        "baseline_STOI": stoi_of(ref, mix),
    }


def main():
    print("=" * 78)
    print(" Open-Unmix  vs  Demucs (HTDemucs)   [同口径头对头]")
    print("=" * 78)

    mix = load_mono(GT / "mixture.wav")
    out = {
        "song": SONG, "duration_s": DUR, "sr": SR,
        "metric": "museval BSSEval v4 (1s win, median, mono) + SI-SDR + PESQ(16k wb) + STOI",
        "targets": {},
    }

    for tgt in ("vocals", "drums"):
        ref = load_mono(GT / f"{tgt}.wav")
        print(f"\n--- {tgt} ---")
        r_umx = pack(f"Open-Unmix umxhq", ref, load_mono(UMX_DIR / f"{tgt}.wav"), mix)
        r_dmx = pack(f"Demucs HTDemucs", ref, load_mono(DMX_DIR / f"{tgt}.wav"), mix)
        out["targets"][tgt] = {"Open-Unmix": r_umx, "Demucs": r_dmx}
        for r in (r_umx, r_dmx):
            print(f"  {r['method']:<18} SDR={r['SDR']:>6}  ΔSDR={r['delta_SDR']:>6}  "
                  f"SI-SDR={r['SI_SDR']:>6}  PESQ={r['PESQ']}  STOI={r['STOI']}")
        print(f"  基线(不分离)       SDR={r_umx['baseline_SDR']:>6}                     "
              f"PESQ={r_umx['baseline_PESQ']}  STOI={r_umx['baseline_STOI']}")

    print("\n--- 复杂度 (FLOPs, 口径同 Demucs) ---")
    try:
        f, sec, tot = umx_flops(10.0)
        out["complexity_umx"] = {
            "params_vocals": 8897444, "params_4targets": 35577488,
            "input_seconds": sec, "flops": f,
            "gflops": round(f / 1e9, 2), "gflops_per_s": round(f / 1e9 / sec, 2),
            "flops_2targets_total": tot,
            "impl": "torch.flop_counter (conv/linear/matmul; 复数 STFT 未计入; 输入 10s @44.1kHz)",
        }
        print(f"  Open-Unmix(单目标): FLOPs={f/1e9:.2f} GFLOPs ({sec}s) "
              f"→ {f/1e9/sec:.2f} GFLOPs/s")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        out["complexity_umx"] = {"error": repr(exc)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 已写入 {OUT}")


if __name__ == "__main__":
    main()
