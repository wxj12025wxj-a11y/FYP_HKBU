# -*- coding: utf-8 -*-

"""
================================================================================
 Smoke-test fixture generator  —  造一个「格式合法」的 MUSDB18 stem.mp4
--------------------------------------------------------------------------------
 用途:
   在真正下载 4.4GB 的 MUSDB18 之前，先验证
   [ musdb + stempeg + FFmpeg + 绘图链路 ] 是否全部正常。
   生成的样本可以跑通 run_audio_analysis.py 并产出全部 9 张图。

 它生成的是【合成的假音频】，不是真实音乐，只用于环境自检。
 真实分析请把下载好的 MUSDB18 放到 datasets/musdb18。

 输出结构 (与 MUSDB18 完全一致):
   <out>/train/00 - Synthetic Test Song.stem.mp4   (5 条 AAC 音轨)

 运行:
   python tools/make_synthetic_musdb.py
================================================================================
"""

from __future__ import annotations

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
DURATION = 35.0          # 比 30 秒多一点，方便观察 trim
SILENCE_HEAD = 1.0       # 开头 1 秒静音 -> 用来演示 trim

OUT_ROOT = Path(_paths.MUSDB18_ROOT / "_smoketest")
TRACK_NAME = "00 - Synthetic Test Song"


def _adsr(n: int, sr: int, attack=0.05, release=0.2) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a = int(attack * sr)
    r = int(release * sr)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)
    return env


def make_vocals(sr: int, n: int) -> np.ndarray:
    """类人声: 基频 + 谐波 + 颤音，且带明显的开/停段落（体现间歇性）。"""
    t = np.arange(n) / sr
    vib = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t)
    f0 = 220.0 * vib
    y = np.zeros(n, dtype=np.float32)
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.18, 0.1], start=1):
        y += amp * np.sin(2 * np.pi * f0 * k * t)
    y /= 5.0

    # 段落包络: 唱 3s / 停 2s 交替，制造人声的间歇性
    gate = np.zeros(n, dtype=np.float32)
    period = 5.0
    sung = 3.0
    seg = int(period * sr)
    for start in range(0, n, seg):
        end = min(start + int(sung * sr), n)
        gate[start:end] = _adsr(end - start, sr)
    return (y * gate * 0.6).astype(np.float32)


def make_bass(sr: int, n: int) -> np.ndarray:
    """低频贝斯: 60~90Hz 级进，能量集中在低频。"""
    t = np.arange(n) / sr
    notes = np.array([55.0, 65.4, 73.4, 82.4])
    step = int(sr * 1.5)
    y = np.zeros(n, dtype=np.float32)
    for i, start in enumerate(range(0, n, step)):
        end = min(start + step, n)
        f = notes[i % len(notes)]
        seg = t[: end - start]
        env = _adsr(end - start, sr, 0.01, 0.1)
        y[start:end] = (np.sin(2 * np.pi * f * seg) + 0.3 * np.sin(2 * np.pi * 2 * f * seg)) * env
    return (y * 0.5).astype(np.float32)


def make_drums(sr: int, n: int) -> np.ndarray:
    """鼓组: 每 0.5s 一个噪声瞬态 + 低频 kick，体现宽带瞬态。"""
    rng = np.random.default_rng(0)
    y = np.zeros(n, dtype=np.float32)
    step = int(sr * 0.5)
    click_len = int(sr * 0.05)
    for start in range(0, n, step):
        end = min(start + click_len, n)
        ln = end - start
        noise = rng.standard_normal(ln).astype(np.float32)
        env = np.exp(-np.linspace(0, 8, ln))
        y[start:end] += noise * env * 0.7
        # kick
        klen = min(start + int(sr * 0.15), n) - start
        tt = np.arange(klen) / sr
        y[start:start + klen] += (np.sin(2 * np.pi * 55 * tt) *
                                  np.exp(-np.linspace(0, 12, klen)) * 0.8)
    return (y * 0.35).astype(np.float32)


def make_other(sr: int, n: int) -> np.ndarray:
    """伴奏和声垫: 中频三和弦，持续能量。"""
    t = np.arange(n) / sr
    chords = [(261.6, 329.6, 392.0), (220.0, 261.6, 329.6)]
    step = int(sr * 3.5)
    y = np.zeros(n, dtype=np.float32)
    for i, start in enumerate(range(0, n, step)):
        end = min(start + step, n)
        env = _adsr(end - start, sr, 0.3, 0.5)
        seg = t[: end - start]
        chord = sum(np.sin(2 * np.pi * f * seg) for f in chords[i % len(chords)])
        y[start:end] = (chord / 3.0) * env
    return (y * 0.35).astype(np.float32)


def to_stereo(y: np.ndarray) -> np.ndarray:
    """单声道 -> 轻微差异化的立体声。"""
    left = y
    right = np.roll(y, 37) * 0.98
    return np.stack([left, right], axis=-1).astype(np.float32)


def main() -> int:
    n = int(SR * DURATION)
    print("生成合成 stem 素材 ...")
    stems = {
        "drums": make_drums(SR, n),
        "bass": make_bass(SR, n),
        "other": make_other(SR, n),
        "vocals": make_vocals(SR, n),
    }

    mixture = np.zeros(n, dtype=np.float32)
    for v in stems.values():
        mixture += v

    # 头/尾加静音，方便演示 trim + 让 30 秒截取落在有内容区间
    head = int(SILENCE_HEAD * SR)
    def pad(y):
        return np.concatenate([np.zeros(head, dtype=np.float32), y])[:n]

    mixture = pad(mixture)
    for k in stems:
        stems[k] = pad(stems[k])

    train_dir = OUT_ROOT / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = train_dir / f"{TRACK_NAME}.stem.mp4"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        paths = {}
        # stream 0 = mixture
        allitems = [("mixture", mixture)] + list(stems.items())
        for name, y in allitems:
            p = td / f"{name}.wav"
            sf.write(str(p), to_stereo(y), SR, subtype="PCM_16")
            paths[name] = p

        # MUSDB18 音轨顺序: 0=mixture, 1=drums, 2=bass, 3=other, 4=vocals
        order = ["mixture", "drums", "bass", "other", "vocals"]
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for name in order:
            cmd += ["-i", str(paths[name])]
        for i in range(len(order)):
            cmd += ["-map", f"{i}:a:0"]
        cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", str(SR), "-ac", "2",
                str(out_mp4)]

        print("调用 FFmpeg 封装为 stem.mp4 ...")
        subprocess.run(cmd, check=True)

    size_mb = out_mp4.stat().st_size / 1024 / 1024
    print(f"[OK] 已生成: {out_mp4}  ({size_mb:.2f} MB)")
    print("\n自检命令:")
    print(f'  python run_audio_analysis.py --root "{OUT_ROOT}"')
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print("[ERROR] FFmpeg 执行失败，请确认 ffmpeg 已在 PATH 中。", exc)
        sys.exit(1)
    except Exception:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
