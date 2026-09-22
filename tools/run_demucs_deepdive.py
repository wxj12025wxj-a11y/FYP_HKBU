# -*- coding: utf-8 -*-
"""
================================================================================
 Demucs 纵向深入分析  (htdemucs)
--------------------------------------------------------------------------------
 数据源 : 同一首歌 A Classic Education - NightOwl 的 mixture.wav (前 30 秒)
 任务   : 运行 Demucs -> 分离 4 stem -> 复刻同一套 9 张分析图
          + 计算 SDR / SI-SDR / PESQ / STOI / Params / MACs / FLOPs
 输出   : 03_outputs\\Demucs\\         stems/     分离出的 4 个 stem (vocals/drums/bass/other)
                                     metrics.json
          04_reports\\separation\\figures\\demucs\\ 同一套 9 张分析图
          04_reports\\separation\\docs\\DEMUCS_REPORT.md
--------------------------------------------------------------------------------
 运行:
   python tools/run_demucs_deepdive.py
   python tools/run_demucs_deepdive.py --duration 30 --model htdemucs
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT

sys.path.insert(0, str(_paths.DEMUCS_SRC))
sys.path.insert(0, str(_paths.TOOLS))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from compare_separation_methods import (
    SR, si_sdr, evaluate, BASELINE_NAME,
)

OUT_ROOT = _paths.OUTPUTS / "Demucs"
STEMS_DIR = OUT_ROOT / "stems"
FIG_DIR = _paths.FIGURES / "demucs"      # 图表统一放 04_reports/figures（五大模块规则）
DATA_DIR = _paths.MUSDB18_ROOT / "train"
STEMS = ["vocals", "drums", "bass", "other"]


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def save_fig(path: Path, dpi: int = 150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"[OK] 已保存 -> {path}")
    plt.close("all")


# ------------------------------------------------------------------ #
# 指标
# ------------------------------------------------------------------ #
def _resample(x, sr, target=16000):
    return librosa.resample(np.asarray(x, dtype=np.float32),
                            orig_sr=sr, target_sr=target)


def compute_pesq(ref, deg, sr):
    """返回 (值, 实现名)。PESQ 需 8k/16k。优先 C 版 pesq, 退回 pysepm。"""
    r16, d16 = _resample(ref, sr), _resample(deg, sr)
    try:
        from pesq import pesq as pesq_c
        return float(pesq_c(16000, r16, d16, "wb")), "pesq(C)"
    except Exception:
        pass
    try:
        from pysepm import pesq as pysepm_pesq
        return float(pysepm_pesq.pesq(16000, r16, d16)), "pysepm"
    except Exception:
        pass
    return None, "unavailable"


def compute_stoi(ref, deg, sr):
    try:
        from pystoi import stoi
        r16, d16 = _resample(ref, sr), _resample(deg, sr)
        return float(stoi(r16, d16, 16000, extended=False)), "pystoi"
    except Exception:
        return None, "unavailable"


def model_complexity(model, sr):
    """Params / MACs / FLOPs。

    注意: Demucs 核心是复数 STFT + Transformer, thop / ptflops 都不支持
    (会抛 NotImplementedError / TypeError)。改用 PyTorch 内置 FlopCounterMode,
    它统计 conv / linear / matmul 等常规算子的 FLOPs (复数 FFT 未计入, 属保守低估)。
    输入取 HTDemucs 的训练段长 (默认 7.8 s @44.1 kHz), 再线性外推到"每秒"。
    get_model('htdemucs') 返回的是 BagOfModels, 故取其中的子模型直接前向。
    """
    import torch
    n_params = int(sum(p.numel() for p in model.parameters()))
    subs = list(getattr(model, "models", [model]))
    sub = subs[0]
    n_ch = int(getattr(sub, "audio_channels", 2) or 2)
    sr_model = int(getattr(sub, "samplerate", sr) or sr)
    seg = float(getattr(sub, "segment", 0) or 0) or 7.8
    n_samples = int(seg * sr_model)
    dummy = torch.zeros(1, n_ch, n_samples)
    flops = macs = None
    impl = "n/a"
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                sub(dummy)
        flops = float(fc.get_total_flops())
        macs = flops / 2.0
        impl = ("torch.flop_counter (conv/linear；复数STFT未计入；"
                f"输入 {n_samples / sr_model:.2f}s, {n_samples} samples)")
    except Exception as e:                                    # noqa: BLE001
        impl = f"failed: {type(e).__name__}: {e}"
    fps = flops / seg if flops else None
    return {
        "params": n_params,
        "params_M": round(n_params / 1e6, 2),
        "n_models_in_bag": len(subs),
        "segment_s": round(seg, 2),
        "macs": macs, "flops": flops,
        "gmacs": round(macs / 1e9, 3) if macs else None,
        "gflops": round(flops / 1e9, 3) if flops else None,
        "gmacs_per_s": round(fps / 2 / 1e9, 3) if fps else None,
        "gflops_per_s": round(fps / 1e9, 3) if fps else None,
        "impl": impl,
    }


# ------------------------------------------------------------------ #
# Demucs
# ------------------------------------------------------------------ #
def run_demucs(mix_stereo, sr, model_name="htdemucs"):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model(model_name)
    model.eval()
    print(f"[Demucs] 模型 {model_name} | sources={model.sources} | "
          f"sr={model.samplerate} | segment={getattr(model,'segment',None)}")

    wav = torch.as_tensor(np.ascontiguousarray(mix_stereo), dtype=torch.float32)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)      # Demucs 标准归一化
    t0 = time.time()
    with torch.no_grad():
        out = apply_model(model, wav[None], device="cpu", progress=True)[0]
    out = out * ref.std() + ref.mean()                 # 反归一化
    dt = time.time() - t0
    stems = {s: out[i].cpu().numpy().astype(np.float32)
             for i, s in enumerate(model.sources)}
    return stems, model, dt


# ------------------------------------------------------------------ #
# 9 张分析图 (复刻 run_audio_analysis.py 的图集)
# ------------------------------------------------------------------ #
def plot_spectrogram(y, sr, title, path, n_fft=1024, hop_length=512, y_axis="log"):
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    save_fig(Path(path))


def plot_mfcc(y, sr, title, path, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC coefficient")
    save_fig(Path(path))


def make_figures(mixture, demucs_vocals, gt_vocals, sr, duration):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # fig1 波形: mixture vs Demucs-vocals (对照真值)
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1); librosa.display.waveshow(mixture, sr=sr)
    plt.title("Mixture Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.subplot(3, 1, 2); librosa.display.waveshow(demucs_vocals, sr=sr)
    plt.title("Demucs-separated Vocals Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.subplot(3, 1, 3); librosa.display.waveshow(gt_vocals, sr=sr)
    plt.title("Ground-truth Vocals Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    save_fig(FIG_DIR / "fig1_waveform_mixture_vocals.png")

    # fig2 采样率对比 (对 Demucs 输出)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for ax, sr_new in zip(axes, [44100, 22050, 16000]):
        y = demucs_vocals if sr_new == sr else librosa.resample(
            demucs_vocals, orig_sr=sr, target_sr=sr_new)
        librosa.display.waveshow(y, sr=sr_new, ax=ax)
        ax.set_title(f"Demucs Vocals resampled to sr={sr_new} Hz")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    save_fig(FIG_DIR / "fig2_waveform_sr_compare.png")

    # fig3 频谱图
    plot_spectrogram(mixture, sr, "Mixture Spectrogram (log-freq, n_fft=1024)",
                     FIG_DIR / "fig3_spectrogram_mixture.png")
    plot_spectrogram(demucs_vocals, sr, "Demucs Vocals Spectrogram (log-freq, n_fft=1024)",
                     FIG_DIR / "fig3_spectrogram_vocals.png")

    # fig4 窗长对比
    for win in (128, 1024):
        hop = win // 2
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(demucs_vocals, n_fft=win, hop_length=hop)), ref=np.max)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(D, sr=sr, hop_length=hop, x_axis="time", y_axis="hz")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Demucs Vocals Spectrogram, win_length={win}")
        plt.xlabel("Time (s)")
        save_fig(FIG_DIR / f"fig4_spectrogram_win{win}.png")

    # fig5 trim
    trimmed, index = librosa.effects.trim(demucs_vocals, top_db=30)
    print(f"[fig5] trim index={index}  {len(demucs_vocals)} -> {len(trimmed)}")
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1); librosa.display.waveshow(demucs_vocals, sr=sr)
    plt.title("Demucs Vocals (Original)"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.subplot(2, 1, 2); librosa.display.waveshow(trimmed, sr=sr)
    plt.title("Demucs Vocals (Trimmed)"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    save_fig(FIG_DIR / "fig5_trim_before_after.png")

    # fig6 MFCC
    plot_mfcc(mixture, sr, "MFCC - Mixture", FIG_DIR / "fig6_mfcc_mixture.png")
    plot_mfcc(demucs_vocals, sr, "MFCC - Demucs Vocals", FIG_DIR / "fig6_mfcc_vocals.png")


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="A Classic Education - NightOwl")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--model", default="htdemucs")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STEMS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    d = DATA_DIR / args.song
    n = int(SR * args.duration)

    def rd(name, stereo=False):
        y, _ = sf.read(str(d / f"{name}.wav"), dtype="float32", always_2d=True)
        y = y[:n]
        return y.T.astype(np.float32) if stereo else y.mean(axis=1).astype(np.float32)

    banner(f"数据源: {args.song}  前 {args.duration}s  sr={SR}")
    mix_st = rd("mixture", stereo=True)
    mixture = rd("mixture")
    gt = {"vocals": rd("vocals"), "drums": rd("drums")}
    print("mixture(stereo):", mix_st.shape, "| mixture(mono):", mixture.shape)

    banner("运行 Demucs")
    stems, model, dt = run_demucs(mix_st, SR, args.model)
    rtf = dt / args.duration
    print(f"[Demucs] 耗时 {dt:.2f}s  RTF={rtf:.3f}  {'可实时' if rtf<1 else '跑不动实时'}")

    for s, y in stems.items():
        sf.write(str(STEMS_DIR / f"{s}.wav"), y.T, SR, subtype="PCM_16")
    print(f"[OK] 4 个 stem 已保存 -> {STEMS_DIR}")

    dm = {s: y.mean(axis=0) if y.ndim == 2 else y for s, y in stems.items()}

    banner("指标计算")
    # 基线: 直接拿 mixture 当估计, 逐目标标定
    base = {t: evaluate(gt[t], mixture) for t in ("vocals", "drums")}

    metrics = {"song": args.song, "duration_s": args.duration, "sr": SR,
               "model": args.model, "demucs_seconds": round(dt, 2),
               "rtf": round(rtf, 3), "targets": {}}

    for tgt in ("vocals", "drums"):
        est = dm.get(tgt)
        if est is None:
            continue
        m = evaluate(gt[tgt], est)
        bdr = base[tgt]["sdr"]
        m["delta_SDR"] = (round(m["sdr"] - bdr, 2)
                          if m["sdr"] is not None and bdr is not None else None)
        pesq_v, pesq_impl = compute_pesq(gt[tgt], est, SR)
        stoi_v, stoi_impl = compute_stoi(gt[tgt], est, SR)
        base_pesq, _ = compute_pesq(gt[tgt], mixture, SR)
        m["PESQ"] = round(pesq_v, 4) if pesq_v is not None else None
        m["PESQ_impl"] = pesq_impl
        m["PESQ_baseline"] = round(base_pesq, 4) if base_pesq is not None else None
        m["STOI"] = round(stoi_v, 4) if stoi_v is not None else None
        m["STOI_impl"] = stoi_impl
        m["baseline_SDR"] = round(bdr, 2) if bdr is not None else None
        metrics["targets"][tgt] = m
        print(f"  [{tgt}] SDR={m['sdr']} ΔSDR={m['delta_SDR']} "
              f"SI-SDR={round(m['si_sdr'],2)} PESQ={m['PESQ']} STOI={m['STOI']}")

    metrics["baseline_vocals_sdr"] = round(base["vocals"]["sdr"], 2)
    metrics["baseline_drums_sdr"] = round(base["drums"]["sdr"], 2)

    banner("模型复杂度 Params / MACs / FLOPs")
    cx = model_complexity(model, SR)
    metrics["complexity"] = cx
    print(f"  Params = {cx['params_M']} M ({cx['params']:,})")
    print(f"  MACs   = {cx['gmacs']} G   FLOPs = {cx['gflops']} G   [{cx['impl']}]")

    banner("生成 9 张分析图")
    make_figures(mixture, dm["vocals"], gt["vocals"], SR, args.duration)

    with open(OUT_ROOT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] metrics.json -> {OUT_ROOT / 'metrics.json'}")
    print(f"[OK] 全部产出 -> {OUT_ROOT}")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
