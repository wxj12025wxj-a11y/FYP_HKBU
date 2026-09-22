# -*- coding: utf-8 -*-
"""
================================================================================
 试听包生成: mixture + Open-Unmix(umxhq) + Demucs(htdemucs)  -> Testset
--------------------------------------------------------------------------------
 目的 : 把同一组 10 首歌的 mixture 复制出来, 分别用两个模型做全曲分离,
        把三份结果各自打包成 zip, 供人工试听对比分离效果。
 数据集: MUSDB18-HQ train 前 10 首 (与 bench_10songs 完全一致)
 输出   : C:\\Users\\jerry\\Downloads\\Testset\\
           DATA.zip                  <- DATA/<song>/mixture.wav
           Open-Unmix_umxhq.zip      <- Open-Unmix_umxhq/<song>/{vocals,drums,bass,other}.wav
           Demucs_htdemucs.zip       <- Demucs_htdemucs/<song>/{vocals,drums,bass,other}.wav
           README.txt
 特性   : 断点续跑 (已存在的 wav 跳过); 逐首日志; wav 16-bit 立体声 44.1k
--------------------------------------------------------------------------------
 运行:
   python tools/prepare_listening_testset.py             # 全流程
   python tools/prepare_listening_testset.py --stage data
   python tools/prepare_listening_testset.py --stage umx
   python tools/prepare_listening_testset.py --stage demucs
   python tools/prepare_listening_testset.py --stage zip
================================================================================
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
import time
import zipfile
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

import numpy as np
import soundfile as sf

TRAIN_ROOT = _paths.MUSDB18_ROOT / "train"
SUBSET = "train"
SR = 44100

DEST = Path(r"C:\Users\jerry\Downloads\Testset")
DATA_DIR = DEST / "DATA"
UMX_DIR = DEST / "Open-Unmix_umxhq"
DEMUCS_DIR = DEST / "Demucs_htdemucs"

UMX_ORDER = ["vocals", "drums", "bass", "other"]
WARMUP_SECONDS = 3.0
LOG = DEST / "_run.log"

# 与 bench_10songs.json 完全一致的 10 首
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


def banner(t: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def log(msg: str) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def songs_existing() -> list[str]:
    out = [s for s in SONGS10 if (TRAIN_ROOT / s / "mixture.wav").is_file()]
    missing = [s for s in SONGS10 if s not in out]
    if missing:
        log(f"[警告] 缺失 {len(missing)} 首: {missing}")
    return out


def read_mix(song: str) -> np.ndarray:
    """读取全曲 mixture, 返回 (N,2) float32"""
    y, _ = sf.read(str(TRAIN_ROOT / song / "mixture.wav"), dtype="float32", always_2d=True)
    return y.astype(np.float32)


def to_stereo_np(v) -> np.ndarray:
    """把任意形状的分离输出规范成 (N,2)"""
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=np.float32)
    v = np.squeeze(v)
    if v.ndim == 1:
        return np.stack([v, v], axis=-1)
    if v.ndim == 2:
        # 用尺寸判断声道轴 (N 远大于 2)
        if v.shape[1] == 2:
            return v
        if v.shape[0] == 2:
            return v.T
        return np.stack([v.mean(1), v.mean(1)], axis=-1)
    # (2,N) 被 squeeze 失败等极端情况
    v = v.reshape(2, -1)
    return v.T


# ------------------------------------------------------------------ #
# Stage 1: 复制 mixture
# ------------------------------------------------------------------ #
def stage_data(songs: list[str]) -> None:
    banner(f"Stage 1: 复制 mixture ({len(songs)} 首) -> {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(songs, 1):
        dst = DATA_DIR / s / "mixture.wav"
        if dst.is_file() and dst.stat().st_size > 0:
            log(f"  [{i}/{len(songs)}] 已存在, 跳过  {s}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TRAIN_ROOT / s / "mixture.wav", dst)
        log(f"  [{i}/{len(songs)}] copied  {s}  ({dst.stat().st_size/1e6:.1f} MB)")
    log(f"Stage 1 完成: {DATA_DIR}")


# ------------------------------------------------------------------ #
# 模型加载
# ------------------------------------------------------------------ #
def load_umx():
    import torch
    import openunmix
    from openunmix import predict as umx_predict

    t0 = time.time()
    sep = openunmix.umxhq(targets=UMX_ORDER, device="cpu")
    if hasattr(sep, "eval"):
        sep.eval()
    log(f"[Open-Unmix] umxhq loaded in {time.time()-t0:.2f}s | targets={UMX_ORDER}")

    def run(mix_n2):
        import torch as T
        audio = T.as_tensor(np.ascontiguousarray(mix_n2.T), dtype=T.float32)  # (2,N)
        est = umx_predict.separate(audio=audio, rate=SR, targets=UMX_ORDER,
                                   separator=sep, device="cpu", filterbank="torch")
        out = {}
        src = est
        if isinstance(src, dict):
            for k, v in src.items():
                out[k] = to_stereo_np(v)
        else:
            arr = src.detach().cpu().numpy() if hasattr(src, "detach") else np.asarray(src)
            for i, name in enumerate(UMX_ORDER):
                out[name] = to_stereo_np(arr[i])
        return out

    return run, sep


def load_demucs(model_name="htdemucs"):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    t0 = time.time()
    model = get_model(model_name)
    model.eval()
    log(f"[Demucs] {model_name} loaded in {time.time()-t0:.2f}s | "
        f"sources={model.sources} | sr={model.samplerate}")

    def run(mix_n2):
        wav = torch.as_tensor(np.ascontiguousarray(mix_n2.T), dtype=torch.float32)  # (2,N)
        ref = wav.mean(0)
        wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            out = apply_model(model, wav_n[None], device="cpu", progress=False)[0]
        out = out * ref.std() + ref.mean()
        stems = {}
        for i, name in enumerate(model.sources):
            stems[name] = out[i].cpu().numpy().T.astype(np.float32)  # (N,2)
        return stems

    return run, model


# ------------------------------------------------------------------ #
# Stage 2/3: 跑模型
# ------------------------------------------------------------------ #
def stage_model(tag: str, out_root: Path, loader, songs: list[str]) -> dict:
    banner(f"{tag}: 分离 {len(songs)} 首 (全曲) -> {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    run_fn, _model = loader()

    # 预热
    if songs:
        warm = read_mix(songs[0])[: int(SR * WARMUP_SECONDS)]
        t0 = time.time()
        run_fn(warm)
        log(f"  [预热] {WARMUP_SECONDS:.0f}s 片段 {time.time()-t0:.2f}s (不计入统计)")

    manifest = []
    t_start = time.time()
    total_audio = 0.0
    total_inf = 0.0
    for i, s in enumerate(songs, 1):
        dst_dir = out_root / s
        expected = [dst_dir / f"{t}.wav" for t in UMX_ORDER]
        if all(p.is_file() and p.stat().st_size > 0 for p in expected):
            log(f"  [{i}/{len(songs)}] 已存在, 跳过  {s}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        mix = read_mix(s)
        dur = len(mix) / SR
        t0 = time.time()
        stems = run_fn(mix)
        dt = time.time() - t0
        for name, arr in stems.items():
            sf.write(str(dst_dir / f"{name}.wav"), arr, SR, subtype="PCM_16")
        total_audio += dur
        total_inf += dt
        manifest.append({"song": s, "duration_s": round(dur, 2),
                         "inference_s": round(dt, 3), "rtf": round(dt / dur, 4)})
        log(f"  [{i}/{len(songs)}] {s[:40]:<42} {dur:6.1f}s  infer {dt:7.2f}s  RTF {dt/dur:.3f}")

    grand = time.time() - t_start
    log(f"{tag} 完成: 总音频 {total_audio/60:.1f} min | 推理合计 {total_inf:.1f}s | "
        f"墙钟 {grand:.1f}s | 整体 RTF {total_inf/total_audio:.3f}" if total_audio else f"{tag} 完成")
    summary = {"model": tag, "n_songs": len(songs), "total_audio_s": round(total_audio, 1),
               "total_inference_s": round(total_inf, 1), "wall_clock_s": round(grand, 1),
               "overall_rtf": round(total_inf / total_audio, 4) if total_audio else None,
               "per_song": manifest}
    (out_root / "_timing.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ #
# Stage 4: 打包
# ------------------------------------------------------------------ #
def zip_dir(src_dir: Path, zip_path: Path, arc_root: str) -> int:
    files = sorted(p for p in src_dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, arcname=str(Path(arc_root) / p.relative_to(src_dir)))
    return len(files)


def stage_zip() -> None:
    banner("Stage 4: 打包")
    targets = [(DATA_DIR, DEST / "DATA.zip", "DATA"),
               (UMX_DIR, DEST / "Open-Unmix_umxhq.zip", "Open-Unmix_umxhq"),
               (DEMUCS_DIR, DEST / "Demucs_htdemucs.zip", "Demucs_htdemucs")]
    for src, zp, arc in targets:
        if not src.is_dir():
            log(f"  [跳过] {src} 不存在")
            continue
        t0 = time.time()
        n = zip_dir(src, zp, arc)
        log(f"  {zp.name:<28} {n:>4} files  {zp.stat().st_size/1e6:8.1f} MB  "
            f"({time.time()-t0:.1f}s)")


def write_readme() -> None:
    txt = f"""试听包说明 (Mixture / Open-Unmix / Demucs)
生成时间: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据集  : MUSDB18-HQ train 前 10 首 (与 bench_10songs 完全一致)
采样率  : 44100 Hz, 立体声, 16-bit PCM WAV

目录
  DATA.zip               原曲混合音 (mixture)   DATA/<歌曲>/mixture.wav
  Open-Unmix_umxhq.zip   传统深度基线分离结果   Open-Unmix_umxhq/<歌曲>/{{vocals,drums,bass,other}}.wav
  Demucs_htdemucs.zip    当前精度上限分离结果   Demucs_htdemucs/<歌曲>/{{vocals,drums,bass,other}}.wav

试听建议
  1) 同一首歌的四个 stem 是"相加还原原曲"的相对分离, 单独听会有串音属正常。
  2) 重点对比 vocals.wav: Demucs 人声更干净但更慢, Open-Unmix 更快更轻。
     (全曲 CPU 实测整体 RTF: Open-Unmix 0.428; Demucs 0.28~0.67 波动较大。
      两者都仍快于实时, 但"快几倍"的倍率受机器负载影响大, 勿引用单次数字。)
  3) 鼓声(PESQ 不适用)以主观听感 + SDR 为准。
"""
    (DEST / "README.txt").write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "umx", "demucs", "zip"])
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    songs = songs_existing()
    banner("试听包生成")
    log(f"目标目录: {DEST}")
    log(f"歌曲数  : {len(songs)}/{len(SONGS10)}")
    # 统计总时长
    tot = 0.0
    for s in songs:
        tot += sf.info(str(TRAIN_ROOT / s / "mixture.wav")).duration
    log(f"总音频  : {tot/60:.1f} min")

    if args.stage in ("all", "data"):
        stage_data(songs)
    if args.stage in ("all", "umx"):
        stage_model("Open-Unmix (umxhq)", UMX_DIR, load_umx, songs)
    if args.stage in ("all", "demucs"):
        stage_model("Demucs (htdemucs)", DEMUCS_DIR, load_demucs, songs)
    if args.stage in ("all", "zip"):
        write_readme()
        stage_zip()

    banner("全部完成")
    log(f"输出目录: {DEST}")


if __name__ == "__main__":
    main()
