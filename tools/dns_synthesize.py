#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由 DNS-Challenge 原料合成 (noisy, clean) 训练/测试对。

DNS 官方只给三样原料：**干净语音 + 噪声 + 房间冲激响应（RIR）**。
带噪语音要自己合成，公式（官方 noisyspeech_synthesizer 同口径）：

    clean_rev = clean  ⊛  rir          （可选，加混响）
    noisy     = clean_rev + α · noise  （α 由目标 SNR 决定）

本脚本用同样的原理，但只依赖 numpy / soundfile / scipy（本机 SAC 环境下可用）。

输出结构
--------
    02_databases/DNS-Challenge/synthetic/
    ├── train/{clean,noisy}/<id>.wav
    ├── test/{clean,noisy}/<id>.wav
    └── manifest.json          每对的 SNR、时长、来源文件

用法
----
    # 生成 200 对训练集（SNR 从 {-5,0,5,10,15} 随机取）+ 20 对测试集
    python tools/dns_synthesize.py --n-train 200 --n-test 20

    # 指定 SNR、片段时长、是否加混响
    python tools/dns_synthesize.py --snr -5,0,5,10 --dur 8 --rir-prob 0.5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths                                          # noqa: E402
_paths.setup_env()

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

try:
    from scipy.signal import fftconvolve, resample_poly         # noqa: E402
    _HAVE_SCIPY = True
except Exception:                                               # noqa: BLE001
    _HAVE_SCIPY = False

SR = 48000                      # DNS4 fullband
DNS = _paths.DNS_ROOT
OUT = DNS / "synthetic"


# ------------------------------------------------------------------ 工具
def scan(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def load_mono(p: Path, want_sr: int = SR) -> np.ndarray | None:
    try:
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:                                           # noqa: BLE001
        return None
    y = y.mean(axis=1)
    if sr != want_sr:
        if not _HAVE_SCIPY:
            return None
        from math import gcd
        g = gcd(int(sr), int(want_sr))
        y = resample_poly(y, want_sr // g, sr // g).astype(np.float32)
    return y


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """把信号裁/铺到 n 个采样。"""
    if len(x) >= n:
        return x[:n]
    reps = -(-n // max(len(x), 1))
    return np.tile(x, reps)[:n]


def convolve_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY:
        y = fftconvolve(x, rir)[: len(x)]
    else:
        y = np.convolve(x, rir)[: len(x)]
    return (y / (rms(y) + 1e-12) * rms(x)).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按目标 SNR 混合。返回与 clean 等长的 noisy。"""
    c_r, n_r = rms(clean), rms(noise)
    target_n = c_r / (10 ** (snr_db / 20.0))
    noise = noise * (target_n / (n_r + 1e-12))
    return (clean + noise).astype(np.float32)


def safe_write(p: Path, y: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 0.99:                                  # 防削波：整体回退增益
        y = y * (0.99 / peak)
    sf.write(str(p), y.astype(np.float32), SR, subtype="PCM_16")


# ------------------------------------------------------------------ 主流程
def build(split: str, n: int, dur: float, snrs: list[float], rir_prob: float,
          cleans: list[Path], noises: list[Path], rirs: list[Path],
          rng: random.Random) -> list[dict]:
    folder = OUT / split
    manifest: list[dict] = []
    nsamp = int(SR * dur)
    ok = 0
    for i in range(n):
        cpath = rng.choice(cleans)
        npath = rng.choice(noises)
        clean = load_mono(cpath)
        noise = load_mono(npath)
        if clean is None or noise is None or len(clean) < nsamp:
            continue
        clean = fit(clean, nsamp)
        noise = fit(noise, nsamp)

        rir_used = None
        if rirs and rng.random() < rir_prob:
            rir = load_mono(rng.choice(rirs))
            if rir is not None and len(rir) > 8:
                clean = convolve_rir(clean, rir)
                rir_used = True

        snr = rng.choice(snrs)
        noisy = mix_at_snr(clean, noise, snr)

        name = f"{split}_{i:05d}"
        safe_write(folder / "clean" / f"{name}.wav", clean)
        safe_write(folder / "noisy" / f"{name}.wav", noisy)
        manifest.append(dict(id=name, snr_db=snr, dur_s=dur,
                             clean_src=str(cpath.relative_to(DNS)),
                             noise_src=str(npath.relative_to(DNS)),
                             rir=rir_used))
        ok += 1
        if (i + 1) % 25 == 0:
            print(f"    {split}: {ok}/{n}", flush=True)
    print(f"  ✅ {split}: 生成 {ok} 对 → {folder.relative_to(_paths.PROJECT_ROOT)}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--dur", type=float, default=8.0, help="每对时长（秒）")
    ap.add_argument("--snr", default="-5,0,5,10,15", help="候选 SNR，逗号分隔（dB）")
    ap.add_argument("--rir-prob", type=float, default=0.3, help="加混响的概率")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cleans = scan(DNS / "clean")
    noises = scan(DNS / "noise")
    rirs = scan(DNS / "impulse_responses")

    print(f"DNS 原料：clean={len(cleans)}  noise={len(noises)}  rir={len(rirs)}")
    if not cleans or not noises:
        print("❌ 原料不足，请先运行： python tools/dns_fetch.py --stage all")
        return 1
    if not _HAVE_SCIPY:
        print("⚠ scipy 不可用：将跳过 RIR 卷积与重采样（仅用 48 kHz 素材）")

    snrs = [float(x) for x in args.snr.split(",") if x.strip()]
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    man = {"train": build("train", args.n_train, args.dur, snrs, args.rir_prob,
                          cleans, noises, rirs, rng),
           "test": build("test", args.n_test, args.dur, snrs, args.rir_prob,
                         cleans, noises, rirs, rng)}
    meta = dict(sr=SR, dur_s=args.dur, snr_choices=snrs, rir_prob=args.rir_prob,
                seed=args.seed, n_train=len(man["train"]), n_test=len(man["test"]),
                formula="noisy = (clean ⊛ rir) + α·noise,  α 由目标 SNR 决定")
    (OUT / "manifest.json").write_text(
        json.dumps({"meta": meta, "pairs": man}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n清单 → {(OUT / 'manifest.json').relative_to(_paths.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
