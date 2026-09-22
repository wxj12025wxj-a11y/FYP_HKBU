# -*- coding: utf-8 -*-
"""
================================================================================
 运行时长基准:  Open-Unmix (umxhq)  vs  Demucs (htdemucs)
--------------------------------------------------------------------------------
 数据集 : MUSDB18-HQ train 前 10 首 (A Classic Education - NightOwl, ...)
         每首取前 30 秒, 立体声, 44.1 kHz
 目的   : 量出两个模型在【同一组 10 首歌】上的真实墙钟运行时长
          - 模型加载时间 (单独计)
          - 预热时间 (单独计, 不计入推理)
          - 逐曲推理墙钟时间 + RTF
          - 10 首歌总时长 / 平均时长
          - 同时算 SDR / SI-SDR, 便于"时间-精度"联合分析
 公平性 : 先预热一次排除冷启动; 每首只跑一次完整 4 轨分离; CPU 单进程
 输出   : outputs/comparison/bench_10songs.json
          outputs/comparison/bench_10songs.csv
--------------------------------------------------------------------------------
 运行:
   python tools/benchmark_models_10songs.py
   python tools/benchmark_models_10songs.py --duration 30 --songs 10
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
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

from compare_separation_methods import (
    SR, UMX_ORDER, evaluate, load_song, list_songs, si_sdr, COMPARISON_DIR,
)

WARMUP_SECONDS = 3.0   # 预热片段长度

# 与既有实验完全一致的 10 首歌顺序
SONGS10 = [
    "A Classic Education - NightOwl",
    "ANiMAL - Clinic A",
    "ANiMAL - Easy Tiger",
    "ANiMAL - Rockshow",
    "Actions - Devil's Words",
    "Actions - One Minute Smile",
    "Actions - South Of The Water",
    "Aimee Norwich - Child",
    "Alexander Ross - Goodbye Bolero",
    "Alexander Ross - Velvet Curtain",
]


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ #
# Open-Unmix
# ------------------------------------------------------------------ #
def load_umx():
    """加载 umxhq 4 目标 Separator, 返回 (callable, load_seconds, sep)"""
    import torch
    import openunmix
    from openunmix import predict as umx_predict

    t0 = time.time()
    sep = openunmix.umxhq(targets=UMX_ORDER, device="cpu")  # 真正加载 4 个 .pth
    if hasattr(sep, "eval"):
        sep.eval()
    dt = time.time() - t0
    print(f"[Open-Unmix] umxhq loaded | targets={UMX_ORDER}")

    def run(mix_stereo):
        audio = torch.as_tensor(np.ascontiguousarray(mix_stereo), dtype=torch.float32)
        t = time.time()
        est = umx_predict.separate(audio=audio, rate=SR, targets=UMX_ORDER,
                                   separator=sep, device="cpu", filterbank="torch")
        dt_inf = time.time() - t
        out = {}
        src = est
        if isinstance(src, dict):
            for k, v in src.items():
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()
                v = np.asarray(v, dtype=np.float32)
                out[k] = v if v.ndim == 1 else v.mean(axis=0)
        else:
            if hasattr(src, "detach"):
                src = src.detach().cpu().numpy()
            arr = np.asarray(src, dtype=np.float32)
            for i, t_ in enumerate(UMX_ORDER):
                v = arr[i]
                out[t_] = v if v.ndim == 1 else v.mean(axis=0)
        return out, dt_inf

    return run, dt, sep


# ------------------------------------------------------------------ #
# Demucs
# ------------------------------------------------------------------ #
def load_demucs(model_name="htdemucs"):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    t0 = time.time()
    model = get_model(model_name)
    model.eval()
    dt = time.time() - t0
    print(f"[Demucs] {model_name} | sources={model.sources} | "
          f"sr={model.samplerate} | segment={getattr(model, 'segment', None)}")

    def run(mix_stereo):
        wav = torch.as_tensor(np.ascontiguousarray(mix_stereo), dtype=torch.float32)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        t = time.time()
        with torch.no_grad():
            out = apply_model(model, wav[None], device="cpu", progress=False)[0]
        dt_inf = time.time() - t
        out = out * ref.std() + ref.mean()
        stems = {s: out[i].cpu().numpy().astype(np.float32)
                 for i, s in enumerate(model.sources)}
        return stems, dt_inf

    return run, dt, model


# ------------------------------------------------------------------ #
# 基准主体
# ------------------------------------------------------------------ #
def bench_model(name, loader, songs, subset, duration, stereo_for=None):
    """对 10 首歌跑一个模型, 返回结果 dict"""
    banner(f"基准: {name}  ({len(songs)} 首 x {duration}s)")
    run_fn, load_s, model = loader()

    # ---- 预热 (3s 片段, 排除冷启动) ----
    _, _, _, warm_stereo = load_song(songs[0], subset, WARMUP_SECONDS)
    t0 = time.time()
    run_fn(warm_stereo)
    warm_s = time.time() - t0
    print(f"  [预热] {WARMUP_SECONDS:.0f}s 片段耗时 {warm_s:.2f}s (不计入推理)")
    print(f"  [加载] 模型加载耗时 {load_s:.2f}s")
    print("-" * 78)
    print(f"  {'歌曲':<34}{'时长s':>8}{'推理s':>9}{'RTF':>8}{'SDR-v':>8}{'SDR-d':>8}")

    per_song = []
    for i, song in enumerate(songs, 1):
        mix_mono, voc, drum, mix_stereo = load_song(song, subset, duration)
        try:
            stems, dt_inf = run_fn(mix_stereo)

            def _mono(y):
                y = np.asarray(y, dtype=np.float32)
                return y if y.ndim == 1 else y.mean(axis=0)

            sdr_v = evaluate(voc, _mono(stems["vocals"]))["sdr"] if "vocals" in stems else None
            sdr_d = evaluate(drum, _mono(stems["drums"]))["sdr"] if "drums" in stems else None
            note = ""
        except Exception as exc:
            dt_inf = float("nan")
            sdr_v = sdr_d = None
            note = f"{type(exc).__name__}: {exc}"
            print(f"  [FAIL] {song}: {note}")
        rtf = dt_inf / duration if duration else float("nan")
        per_song.append({
            "song": song, "index": i, "audio_seconds": duration,
            "inference_seconds": round(dt_inf, 3), "rtf": round(rtf, 4),
            "sdr_vocals": round(sdr_v, 3) if sdr_v is not None else None,
            "sdr_drums": round(sdr_d, 3) if sdr_d is not None else None,
            "note": note,
        })
        sv = f"{sdr_v:8.2f}" if sdr_v is not None else "     n/a"
        sd = f"{sdr_d:8.2f}" if sdr_d is not None else "     n/a"
        print(f"  {song[:34]:<34}{duration:>8}{dt_inf:>9.2f}{rtf:>8.3f}{sv}{sd}")

    times = [r["inference_seconds"] for r in per_song
             if not (isinstance(r["inference_seconds"], float) and np.isnan(r["inference_seconds"]))]
    rtfs = [r["rtf"] for r in per_song
            if not (isinstance(r["rtf"], float) and np.isnan(r["rtf"]))]
    if not times:
        raise RuntimeError(f"{name}: 没有任何成功样本, 检查上面的 FAIL 信息")
    total = float(np.sum(times))
    audio_total = duration * len(songs)
    summary = {
        "model": name,
        "model_load_seconds": round(load_s, 2),
        "warmup_seconds": round(warm_s, 2),
        "n_songs": len(songs),
        "audio_seconds_per_song": duration,
        "audio_seconds_total": audio_total,
        "inference_total_seconds": round(total, 2),
        "inference_mean_seconds": round(float(np.mean(times)), 3),
        "inference_median_seconds": round(float(np.median(times)), 3),
        "inference_min_seconds": round(float(np.min(times)), 3),
        "inference_max_seconds": round(float(np.max(times)), 3),
        "inference_std_seconds": round(float(np.std(times)), 3),
        "rtf_mean": round(float(np.mean(rtfs)), 4),
        "rtf_median": round(float(np.median(rtfs)), 4),
        "real_time_capable": bool(float(np.median(rtfs)) < 1.0),
        "wall_clock_including_load_seconds": round(load_s + warm_s + total, 2),
        "per_song": per_song,
    }
    print("-" * 78)
    print(f"  合计推理 {total:.2f}s | 平均 {np.mean(times):.2f}s/首 | "
          f"中位 RTF {np.median(rtfs):.3f} | 实时={'是' if summary['real_time_capable'] else '否'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--songs", type=int, default=10)
    ap.add_argument("--subset", default="train")
    args = ap.parse_args()

    all_songs = list_songs(args.subset)
    # 取与既有实验一致的 10 首 (按固定名单, 缺失则回退到前 N)
    songs = [s for s in SONGS10 if (_paths.MUSDB18_ROOT / args.subset / s).is_dir()]
    if len(songs) < args.songs:
        songs = all_songs[: args.songs]
    songs = songs[: args.songs]

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    banner("开始: 10 首歌运行时长基准 (CPU 单进程)")
    print(f"  数据集: {args.subset} | 歌曲数: {len(songs)} | 每首 {args.duration}s "
          f"| 总音频 {len(songs) * args.duration}s")
    print(f"  开始时间: {_dt.datetime.now().isoformat(timespec='seconds')}")

    results = {"meta": {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "subset": args.subset, "songs": songs, "duration": args.duration,
        "device": "cpu", "sr": SR,
        "warmup_seconds": WARMUP_SECONDS,
        "note": "模型加载/预热单独计; 每首仅跑一次完整 4 轨分离; 墙钟时间",
    }}

    t_start = time.time()
    results["openunmix"] = bench_model("Open-Unmix (umxhq, 4 targets)",
                                       load_umx, songs, args.subset, args.duration)
    results["demucs"] = bench_model("Demucs (htdemucs, 4 stems)",
                                    load_demucs, songs, args.subset, args.duration)
    results["meta"]["grand_total_seconds"] = round(time.time() - t_start, 2)

    # ---- 保存 JSON ----
    jpath = COMPARISON_DIR / "bench_10songs.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 保存 CSV (逐曲) ----
    cpath = COMPARISON_DIR / "bench_10songs.csv"
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "index", "song", "audio_seconds",
                    "inference_seconds", "rtf", "sdr_vocals", "sdr_drums"])
        for key, label in (("openunmix", "Open-Unmix"), ("demucs", "Demucs")):
            for r in results[key]["per_song"]:
                w.writerow([label, r["index"], r["song"], r["audio_seconds"],
                            r["inference_seconds"], r["rtf"],
                            r["sdr_vocals"], r["sdr_drums"]])

    banner("完成")
    print(f"  总墙钟: {results['meta']['grand_total_seconds']}s")
    print(f"  JSON: {jpath}")
    print(f"  CSV : {cpath}")


if __name__ == "__main__":
    main()
