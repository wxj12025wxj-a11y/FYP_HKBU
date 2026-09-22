# -*- coding: utf-8 -*-
"""
================================================================================
 FYP Audio Analysis Script  —  MUSDB18 / MUSDB18-HQ (自动识别格式)
--------------------------------------------------------------------------------
 数据集 : 两种都支持，脚本会自动识别
          (A) 原始 MUSDB18   ——  train/xxx.stem.mp4          ( is_wav=False )
          (B) MUSDB18-HQ     ——  train/<歌名>/mixture.wav 等 5 个 wav ( is_wav=True )
 任务   : Progress Report 的音频基础分析 (Task 1 ~ Task 4)
 输出   : 9 张 PNG 图 ( dpi=150 ) + 终端形状/采样率/trim/MFCC 打印
 注意   : 本脚本只做音频基础分析。
          不涉及实时系统、深度学习、Demucs、Web 前端、模型训练、GPU 加速。
--------------------------------------------------------------------------------
 运行方式 ( Anaconda Prompt / 任意终端 ):
     cd E:\\FYP_HKBU
     python run_audio_analysis.py
 可选参数:
     python run_audio_analysis.py --duration 10        # 内存不足时改短
     python run_audio_analysis.py --song 1             # 换 train 下的第 2 首
     python run_audio_analysis.py --root "D:\\...\\datasets"   # 自定义数据集路径
     python run_audio_analysis.py --format hq          # 强制按 HQ(wav) 解析
     python run_audio_analysis.py --show               # 出图后弹窗显示
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# ------------------------------------------------------------------ #
# 让 Windows 控制台能正确打印中文 (默认 GBK 会造成乱码)
# ------------------------------------------------------------------ #
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib

# 无窗口后端: 保证在终端里跑不会因为 9 个弹窗而卡住 / 阻塞
# ( 想看图请加 --show 参数 )
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import librosa
import librosa.display
import musdb


# ================================================================== #
# 0. 配置区
# ================================================================== #
# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT
MUSDB_ROOT_CANDIDATES = [
    _paths.MUSDB18_ROOT / "musdb18",       # 原始 MUSDB18 (stem.mp4)
    _paths.MUSDB18_ROOT / "musdb18hq",     # MUSDB18-HQ
    _paths.MUSDB18_ROOT / "musdb18" / "musdb18",
    _paths.MUSDB18_ROOT / "musdb18hq" / "musdb18hq",
    _paths.MUSDB18_ROOT,                   # 直接解压到 datasets/ 下 (你当前的情况)
]
FIG_DIR = _paths.FIGURES

# MUSDB18-HQ 每首歌目录内的 5 个 wav
HQ_TARGET_FILES = ["mixture.wav", "vocals.wav", "drums.wav", "bass.wav", "other.wav"]

# 未来扩展: 本阶段固定单首歌、单时长；后续可批量/多模型对比
DEFAULT_DURATION = 30      # 秒
DEFAULT_SONG_INDEX = 0     # train 目录下第 1 首 (0-based)
TARGET_SR_OPTIONS = [44100, 22050, 16000]   # 重采样对比用

ALL_FIGURES = [
    "fig1_waveform_mixture_vocals.png",
    "fig2_waveform_sr_compare.png",
    "fig3_spectrogram_mixture.png",
    "fig3_spectrogram_vocals.png",
    "fig4_spectrogram_win128.png",
    "fig4_spectrogram_win1024.png",
    "fig5_trim_before_after.png",
    "fig6_mfcc_mixture.png",
    "fig6_mfcc_vocals.png",
]


# ================================================================== #
# 1. 工具函数
# ================================================================== #
def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def save_fig(path: Path, dpi: int = 150, show: bool = False) -> None:
    """统一保存图像并释放内存。"""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"[OK] 已保存 -> {path}")
    if show:
        plt.show()
    plt.close("all")


def _looks_like_dataset(p: Path) -> str | None:
    """
    判断目录 p 是不是一个可用的 MUSDB 根目录，并返回格式：
      "hq"   —— MUSDB18-HQ 布局: p/train/<歌名>/*.wav
      "stem" —— 原始 MUSDB18 布局: p/train/*.stem.mp4
      None   —— 不是
    """
    for subset in ("train", "test"):
        d = p / subset
        if not d.is_dir():
            continue
        # 原始: 目录下直接有 *.stem.mp4
        if next(d.glob("*.stem.mp4"), None) is not None:
            return "stem"
        # HQ: 目录下是歌曲子文件夹，且含 mixture.wav
        for child in d.iterdir():
            if child.is_dir() and (child / "mixture.wav").is_file():
                return "hq"
    return None


def detect_dataset(explicit_root: Path | None, forced_format: str) -> tuple[Path | None, str]:
    """
    自动定位 MUSDB 根目录并识别格式 ( HQ wav / 原始 stem.mp4 )。
    返回 (root, fmt)，fmt ∈ {"hq", "stem"}；找不到时 root 为 None。
    forced_format: "auto" | "hq" | "stem"
    """
    candidates = [explicit_root] if explicit_root else MUSDB_ROOT_CANDIDATES
    for cand in candidates:
        if cand is None or not cand.exists():
            continue
        fmt = _looks_like_dataset(cand)
        if fmt is None:
            # 兼容“解压多套了一层目录”的情况
            for child in sorted(x for x in cand.glob("*") if x.is_dir()):
                fmt = _looks_like_dataset(child)
                if fmt is not None:
                    cand = child
                    break
        if fmt is None:
            continue
        if forced_format != "auto" and fmt != forced_format:
            continue
        return cand, fmt
    return None, ""


def list_songs(root: Path, subset: str, fmt: str) -> list[str]:
    """按名称排序返回某子集下的歌曲标识 ( musdb 内部同样是排序后取用 )。"""
    d = root / subset
    if not d.is_dir():
        return []
    if fmt == "stem":
        return sorted(f.name[: -len(".stem.mp4")] for f in d.glob("*.stem.mp4"))
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / "mixture.wav").is_file())


def load_track_audio(track, sr: int, duration: int):
    """
    只读取前 `duration` 秒，避免把整首歌加载进内存。
    musdb 的 Track 支持 chunk_start / chunk_duration，设置后 .audio 只解码该片段。
    若该版本不支持，则回退到“整轨加载 + 切片”。
    """
    try:
        track.chunk_start = 0.0
        track.chunk_duration = float(duration)
        mix = track.audio                      # (samples, channels)
        voc = track.targets["vocals"].audio     # (samples, channels)
        if mix is not None and len(mix) > 0:
            print(f"[INFO] 按 chunk 只解码前 {duration} 秒 (省内存模式)")
            return mix, voc
    except Exception as exc:                    # noqa: BLE001
        print(f"[WARN] chunk 读取失败({exc})，回退为整轨加载后切片")

    track.chunk_start = None
    track.chunk_duration = None
    mix = track.audio
    voc = track.targets["vocals"].audio
    n = int(sr * duration)
    return mix[:n], voc[:n]


# ================================================================== #
# 2. 主流程
# ================================================================== #
def main() -> int:
    parser = argparse.ArgumentParser(description="MUSDB18 audio basic analysis")
    parser.add_argument("--root", type=str, default=None, help="MUSDB18 数据集根目录")
    parser.add_argument("--song", type=int, default=DEFAULT_SONG_INDEX,
                        help="train 目录下的歌曲序号 (0-based)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="截取时长(秒)，内存不足时改小，例如 10")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "hq", "stem"],
                        help="数据集格式: auto=自动识别 / hq=MUSDB18-HQ(wav) / stem=原始(stem.mp4)")
    parser.add_argument("--show", action="store_true", help="出图后弹窗显示")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 2.0 定位数据集 + 识别格式
    # ----------------------------------------------------------------
    banner("STEP 0 / 数据集定位与格式识别")
    musdb_root, fmt = detect_dataset(
        Path(args.root) if args.root else None, args.format)
    print("图像输出目录 :", FIG_DIR)

    if musdb_root is None:
        print("\n[ERROR] 找不到可用的数据集。请确认存在下面任意一种结构：")
        print("  (A) 原始 MUSDB18  : <root>/train/xxx.stem.mp4")
        print("  (B) MUSDB18-HQ    : <root>/train/<歌名>/mixture.wav (另有 4 个 wav)")
        print("下载地址: https://zenodo.org/record/3338373")
        print("建议解压到:", MUSDB_ROOT_CANDIDATES[0], " 或 ", MUSDB_ROOT_CANDIDATES[1])
        return 1

    is_wav = (fmt == "hq")
    print("数据集根目录 :", musdb_root)
    print("识别到的格式 :", "MUSDB18-HQ (wav, 44.1kHz)" if is_wav
          else "原始 MUSDB18 (stem.mp4)")

    songs = list_songs(musdb_root, args.subset, fmt)
    if not songs:
        print(f"\n[ERROR] {musdb_root / args.subset} 下没找到歌曲。")
        return 1

    if args.song >= len(songs):
        print(f"[WARN] --song {args.song} 超出范围，回退到第 1 首")
        args.song = 0
    song_name = songs[args.song]
    print(f"该子集共 {len(songs)} 首，本次使用第 {args.song + 1} 首")

    # ----------------------------------------------------------------
    # 2.1 用 musdb 读取 (HQ 走 wav；原始走 stem.mp4 + stempeg + FFmpeg)
    # ----------------------------------------------------------------
    banner("STEP 1 / 加载数据集")
    print("正在加载数据集 ...")
    mus = musdb.DB(root=str(musdb_root), is_wav=is_wav, subsets=args.subset)

    track = next((t for t in mus.tracks if t.name == song_name), mus.tracks[args.song])

    print("使用歌曲：", track.name)

    # musdb 的 track.rate 在未加载前可能为 None，HQ 与原始数据都是 44100
    sr = int(getattr(track, "rate", 44100) or 44100)

    mixture_stereo, vocals_stereo = load_track_audio(track, sr, args.duration)

    mixture_full = librosa.to_mono(np.asarray(mixture_stereo).T)
    vocals_full = librosa.to_mono(np.asarray(vocals_stereo).T)

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 注意：此处截取前 30 秒音频，如需修改时长请改 duration (或 --duration)  ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    duration = args.duration
    mixture = mixture_full[: sr * duration]
    vocals = vocals_full[: sr * duration]

    print("mixture shape:", mixture.shape, "sr:", sr)
    print("vocals shape:", vocals.shape)
    print(f"(已截取前 {duration} 秒 => {len(mixture)} 采样点)")

    # ----------------------------------------------------------------
    # Task 1-a: 波形图  mixture vs vocals
    # ----------------------------------------------------------------
    banner("Task 1-a / 波形图 mixture vs vocals")
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(mixture, sr=sr)
    plt.title("MUSDB18 Mixture Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(vocals, sr=sr)
    plt.title("MUSDB18 Vocals Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    save_fig(FIG_DIR / "fig1_waveform_mixture_vocals.png", show=args.show)

    # ----------------------------------------------------------------
    # Task 1-b: 重采样对比 44100 / 22050 / 16000
    # ----------------------------------------------------------------
    banner("Task 1-b / 采样率对比")
    fig, axes = plt.subplots(len(TARGET_SR_OPTIONS), 1, figsize=(14, 9))
    for ax, sr_new in zip(axes, TARGET_SR_OPTIONS):
        if sr_new != sr:
            y_res = librosa.resample(mixture, orig_sr=sr, target_sr=sr_new)
        else:
            y_res = mixture
        print(f"  sr={sr_new:>6} Hz -> samples = {len(y_res)}")
        librosa.display.waveshow(y_res, sr=sr_new, ax=ax)
        ax.set_title(f"Mixture resampled to sr={sr_new} Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    save_fig(FIG_DIR / "fig2_waveform_sr_compare.png", show=args.show)

    # ----------------------------------------------------------------
    # Task 2: 频谱图
    # ----------------------------------------------------------------
    def plot_spectrogram(y, sr_, title, filename, n_fft=1024,
                         hop_length=512, y_axis="log", show=False):
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max,
        )
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(
            D, sr=sr_, hop_length=hop_length, x_axis="time", y_axis=y_axis,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.xlabel("Time (s)")
        save_fig(Path(filename), show=show)

    banner("Task 2-a / 频谱图 (log 频率轴)")
    plot_spectrogram(mixture, sr, "Mixture Spectrogram (log-freq, n_fft=1024)",
                     FIG_DIR / "fig3_spectrogram_mixture.png", show=args.show)
    plot_spectrogram(vocals, sr, "Vocals Spectrogram (log-freq, n_fft=1024)",
                     FIG_DIR / "fig3_spectrogram_vocals.png", show=args.show)

    banner("Task 2-b / STFT 窗长对比 win=128 vs 1024")
    for win in (128, 1024):
        hop = win // 2
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(mixture, n_fft=win, hop_length=hop)), ref=np.max,
        )
        print(f"  win_length={win:>4}  hop={hop:>4}  frame={D.shape}")
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(D, sr=sr, hop_length=hop,
                                 x_axis="time", y_axis="hz")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mixture Spectrogram, win_length={win}")
        plt.xlabel("Time (s)")
        save_fig(FIG_DIR / f"fig4_spectrogram_win{win}.png", show=args.show)

    # ----------------------------------------------------------------
    # Task 3: Trim (静音裁剪)
    # ----------------------------------------------------------------
    banner("Task 3 / Trim 前后对比")
    trimmed, index = librosa.effects.trim(mixture, top_db=30)

    print("original shape:", mixture.shape)
    print("trimmed shape :", trimmed.shape)
    print("trim index    :", index)
    print(f"trimmed 部分时长 ≈ {len(trimmed) / sr:.3f} s")

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(mixture, sr=sr)
    plt.title("Original Mixture")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(trimmed, sr=sr)
    plt.title("Trimmed Mixture")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    save_fig(FIG_DIR / "fig5_trim_before_after.png", show=args.show)

    # ----------------------------------------------------------------
    # Task 4: MFCC (13 维)
    # ----------------------------------------------------------------
    def plot_mfcc(y, sr_, title, filename, n_mfcc=13, show=False):
        mfccs = librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=n_mfcc)
        print(f"{title} MFCC shape: {mfccs.shape}")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr_, x_axis="time")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("MFCC coefficient")
        save_fig(Path(filename), show=show)

    banner("Task 4 / 13 维 MFCC")
    plot_mfcc(mixture, sr, "MFCC - Mixture",
              FIG_DIR / "fig6_mfcc_mixture.png", show=args.show)
    plot_mfcc(vocals, sr, "MFCC - Vocals",
              FIG_DIR / "fig6_mfcc_vocals.png", show=args.show)

    # ----------------------------------------------------------------
    # 收尾自检
    # ----------------------------------------------------------------
    banner("DONE / 产物自检")
    missing = []
    for name in ALL_FIGURES:
        f = FIG_DIR / name
        ok = f.exists() and f.stat().st_size > 0
        print(f"  [{'OK ' if ok else 'MISS'}] {name:<38} "
              f"{f.stat().st_size // 1024 if f.exists() else 0:>5} KB")
        if not ok:
            missing.append(name)

    print("\n全部图表已生成，保存在：", FIG_DIR)
    if missing:
        print("[WARN] 以下图缺失：", missing)
        return 2
    print(f"[SUCCESS] 9/9 张图全部生成完毕 (歌曲: {track.name}, 时长: {duration}s)")
    return 0


# ================================================================== #
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[ABORT] 用户中断")
        sys.exit(130)
    except Exception:                       # noqa: BLE001
        print("\n[FATAL] 运行出错，完整堆栈如下（请整段发给 AI 排查）:\n")
        traceback.print_exc()
        sys.exit(1)
