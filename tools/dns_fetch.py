#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNS Challenge (ICASSP 2022, DNS4) 数据集子集下载 + 解压。

用途：FYP 在 MUSDB18-HQ 之外引入「降噪 / 语音增强」数据集。

为什么只下子集：DNS4 完整集解压后 ~892 GB（clean_fullband 827G / noise 58G），
单机不可行。这里按「降噪实验最小可用集」挑选 5 个官方分片：

  dev_testset_000            官方带噪测试集  → 评测基准（唯一带 reference 的官方测试集）
  noise.freesound_001        Freesound 环境噪声 → 噪声源
  clean.VocalSet_48kHz_mono  独唱人声（48k）  → 干净源，且与音乐 FYP 的 vocals 语义最近
  clean.emotional_speech_000 情感语音        → 干净源补充
  impulse_responses_000      RIR 房间冲激响应 → 加混响（可选）

下载方式：多线程分段（Range）+ 断点续传 + 每段独立文件。
  ⚠️ 不要用 `curl -C -` 配合 `--retry`：服务器忽略 Range 时会导致文件越界或截断。

用法：
  python tools/dns_fetch.py --list                # 只列清单与体积
  python tools/dns_fetch.py --stage dl            # 只下载
  python tools/dns_fetch.py --stage extract       # 只解压
  python tools/dns_fetch.py --stage all           # 下载 + 解压（默认）
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

BASE = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/"

# (本地短名, BLOB 路径, 目标解压子目录, 说明)
PACKS = [
    ("impulse_responses", "datasets_fullband.impulse_responses_000.tar.bz2",
     "impulse_responses", "RIR 房间冲激响应（加混响）"),
    ("clean_vocalset", "clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2",
     "clean/VocalSet_48kHz_mono", "独唱人声 48kHz（与音乐 vocals 语义最近）"),
    ("noise_freesound", "noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2",
     "noise/freesound", "Freesound 环境噪声"),
    ("dev_testset", "datasets_fullband.dev_testset_000.tar.bz2",
     "dev_testset", "官方带噪测试集（评测基准）"),
    ("clean_emotional", "clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2",
     "clean/emotional_speech", "情感语音（干净源补充）"),
]

NSEG = 6           # 每文件分段数（并发）
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def root() -> Path:
    return Path(__file__).resolve().parent.parent


DL_DIR = root() / "02_databases" / "DNS-Challenge" / "_download"
DS_DIR = root() / "02_databases" / "DNS-Challenge"
LOG = DL_DIR / "_dns_fetch.txt"
_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _lock:
        print(line, flush=True)
        try:
            DL_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def fetch_seg(url: str, start: int, end: int, dest: Path, retries: int = 6) -> None:
    """下载 [start, end] 闭区间到 dest。"""
    if dest.is_file() and dest.stat().st_size == end - start + 1:
        return
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
            if dest.stat().st_size == end - start + 1:
                return
            last = f"size {dest.stat().st_size} != {end-start+1}"
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:80]}"
        time.sleep(2 + 3 * i)
    raise RuntimeError(f"分段 {dest.name} 失败：{last}")


def download_one(name: str, blob: str) -> Path:
    url = BASE + blob
    final = DL_DIR / f"{name}.tar.bz2"
    if final.is_file() and final.stat().st_size > 0:
        log(f"  [{name}] 已存在，跳过（{final.stat().st_size/1e6:.1f} MB）")
        return final
    size = head_size(url)
    log(f"  [{name}] 目标 {size/1e6:.1f} MB，{NSEG} 段并发下载")
    step = -(-size // NSEG)
    segs = [(i * step, min(size - 1, (i + 1) * step - 1)) for i in range(NSEG)]
    segs = [(a, b) for a, b in segs if a <= b]

    t0 = time.time()
    errs: list[Exception] = []

    def worker(a: int, b: int, idx: int) -> None:
        try:
            fetch_seg(url, a, b, DL_DIR / f"{name}.seg{idx}")
        except Exception as e:                        # noqa: BLE001
            errs.append(e)

    ths = [threading.Thread(target=worker, args=(a, b, i), daemon=True)
           for i, (a, b) in enumerate(segs)]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(15)
        got = sum((DL_DIR / f"{name}.seg{i}").stat().st_size
                  for i in range(len(segs)) if (DL_DIR / f"{name}.seg{i}").is_file())
        el = max(time.time() - t0, 1e-6)
        log(f"    ... {got/1e6:7.1f}/{size/1e6:.1f} MB "
            f"({100*got/size:4.1f}%)  {got/1e6/el:.2f} MB/s")
    for t in ths:
        t.join()
    if errs:
        raise errs[0]

    log(f"  [{name}] 拼接 ...")
    with open(final, "wb") as out:
        for i in range(len(segs)):
            part = DL_DIR / f"{name}.seg{i}"
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, length=1 << 22)
            part.unlink()
    got = final.stat().st_size
    if got != size:
        raise RuntimeError(f"{name} 拼接后大小 {got} != {size}")
    log(f"  [{name}] ✅ 完成 {got/1e6:.1f} MB  用时 {time.time()-t0:.0f}s")
    return final


def extract_one(tar_path: Path, subdir: str) -> None:
    dest = DS_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    log(f"  [{tar_path.stem}] 解压 → {dest.relative_to(root())}")
    t0 = time.time()
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(dest)
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    log(f"  [{tar_path.stem}] ✅ 解压完成 {n} 个文件  用时 {time.time()-t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["list", "dl", "extract", "all"], default="all")
    args = ap.parse_args()

    if args.stage == "list":
        tot = 0
        for name, blob, sub, desc in PACKS:
            try:
                n = head_size(BASE + blob)
            except Exception as e:                    # noqa: BLE001
                n = 0
                desc += f"  (HEAD 失败 {e})"
            tot += n
            print(f"{name:20s} {n/1e6:8.1f} MB → {sub:26s} {desc}")
        print(f"{'合计':20s} {tot/1e6:8.1f} MB")
        return 0

    DL_DIR.mkdir(parents=True, exist_ok=True)
    (DS_DIR / "clean").mkdir(parents=True, exist_ok=True)
    (DS_DIR / "noise").mkdir(parents=True, exist_ok=True)
    log(f"===== DNS 子集任务开始 stage={args.stage} =====")

    if args.stage in ("dl", "all"):
        for name, blob, sub, desc in PACKS:
            if (DS_DIR / sub).is_dir() and any((DS_DIR / sub).rglob("*.wav")):
                log(f"  [{name}] 已解压就绪，跳过下载")
                continue
            log(f"下载 {name} — {desc}")
            download_one(name, blob)

    if args.stage in ("extract", "all"):
        for name, blob, sub, desc in PACKS:
            final = DL_DIR / f"{name}.tar.bz2"
            if not final.is_file():
                log(f"  [{name}] 缺压缩包，跳过解压")
                continue
            if (DS_DIR / sub).is_dir() and any((DS_DIR / sub).rglob("*.wav")):
                log(f"  [{name}] 已解压，跳过")
                continue
            extract_one(final, sub)

    log("===== DNS 子集任务结束 =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
