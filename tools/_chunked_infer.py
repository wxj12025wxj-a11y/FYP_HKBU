# -*- coding: utf-8 -*-
"""整曲分块推理 —— 让整张 MUSDB18 test 集能在 8 GiB 笔记本上跑得动。

为什么必须分块（2026-09-19 实测，8 GiB RTX 4070 Laptop）
------------------------------------------------------
整曲一次性前向会遇到**两种**失败，而且第二种最阴险：

1. **真 OOM**：>240 s 的曲目直接 `OutOfMemoryError`。
2. 🔴 **CUDA sysmem fallback（不报错！）**：某些长度下不报 OOM，但显存溢出到系统内存 ——
   实测 200 s 整曲 `max_memory_allocated = 21.4 GB`（卡只有 8 GiB），
   耗时被拉到 RTF 0.302。**这是「跑通了但慢 13 倍」的静默陷阱**。

分块后（同机、同权重）：

| 方式 | RTF | 峰值显存 |
|---|---|---|
| 整曲 200 s | 0.302 | **21.40 GB**（溢出到系统内存） |
| 分块 60 s | **0.0204** | **1.87 GB** |

→ 分块后 **快约 15×、显存降 11×**，而且永不触碰 sysmem fallback。

⚠️ 另一个必须避开的坑：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
（DPRNN 训练必需，见 `paths.setup_env()`）会让 `torch.istft` 在长序列上抛
`RuntimeError: CUDA driver error: device not ready` / `!handles_.at(i) INTERNAL ASSERT FAILED`。
这是 torch 2.14 + 驱动 616.56 的可复现 bug。**推理时不要设这个变量** ——
本模块的 `inference_alloc_guard()` 负责在进程内把它摘掉。

用法
----
    from _chunked_infer import chunked_separate, inference_alloc_guard

    with inference_alloc_guard():
        stems = chunked_separate(run_fn, mix, sr=44100, chunk_s=60, overlap_s=2)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable

import numpy as np

DEFAULT_CHUNK_S = 60.0
DEFAULT_OVERLAP_S = 2.0


@contextmanager
def inference_alloc_guard():
    """推理期摘掉 `expandable_segments`。

    必须在 **import torch 之前**生效才可靠；若 torch 已加载且 CUDA 已初始化，
    退化为尽力而为（打印提示），调用方应确保子进程环境干净。
    """
    key = "PYTORCH_CUDA_ALLOC_CONF"
    prev = os.environ.get(key)
    if prev and "expandable_segments" in prev:
        parts = [p for p in prev.split(",") if "expandable_segments" not in p]
        if parts:
            os.environ[key] = ",".join(parts)
        else:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _fade(n: int, sr: int, fade_s: float) -> np.ndarray:
    """生成 (n,) 的升余弦淡入淡出窗，用于接缝处交叉淡化。"""
    w = np.ones(n, dtype=np.float32)
    k = min(int(fade_s * sr), n // 2)
    if k > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k, dtype=np.float32))
        w[:k] = ramp
        w[-k:] = ramp[::-1]
    return w


def _as2d(y: np.ndarray) -> np.ndarray:
    """把任意 (n,) / (n,1) / (n,C) 归一到 (n, C)。

    🔴 为什么必须有这一步（2026-09-21 实测的静默崩溃）：
    原实现先按 `stems[k]` 的**原始维度**分配 `acc`，之后才在写入循环里把 1-D
    提升成 2-D。于是 **单声道模型（如 DPRNN，11.025 kHz 单声道）** 会走成：
    `acc[k]` 分配为 (n_total,) 一维，而写入时 `y` 已被 stack 成 (m, 2) →
    `acc[name][s:e] += y * w[:, None]` 抛
    `ValueError: operands could not be broadcast together with shapes
     (2646000,) (2646000,2) (2646000,)`。
    后果不只是报错：**整个 (模型, 歌曲) 直接 FAIL**，且若上层用子进程返回码判成败
    会被误记成 PASS（见 bench_musdb_test_songmajor.run_combo 的修复）。
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return y


def _match_channels(y: np.ndarray, ch: int) -> np.ndarray:
    """让 `y` 的声道数与已分配的 `acc` 对齐（跨块声道数不一致时的兜底）。"""
    if y.shape[1] == ch:
        return y
    if y.shape[1] == 1:
        return np.repeat(y, ch, axis=1)
    if ch == 1:
        return y.mean(axis=1, keepdims=True)
    return y[:, :ch]


def chunked_separate(
    run_fn: Callable[[np.ndarray], dict],
    mix: np.ndarray,
    sr: int = 44100,
    chunk_s: float = DEFAULT_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """把整曲切成 `chunk_s` 的块逐块分离，再用交叠淡入淡出拼回。

    run_fn(mix_stereo:(N,2)) -> {stem: (N,2)} ；分块调用时 N = 块长。

    返回 {stem: (N,2)}，长度与输入一致。
    """
    n_total = len(mix)
    step = max(1, int(chunk_s * sr))
    ov = max(0, int(overlap_s * sr))
    if ov >= step // 2:
        ov = step // 4

    acc: dict[str, np.ndarray] = {}
    wsum: np.ndarray | None = None

    pos = 0
    idx = 0
    total = (n_total + step - 1) // step
    while pos < n_total:
        # 块范围：[pos, end)，向前后各扩 ov 用于交叉淡化
        s = pos
        e = min(n_total, pos + step)
        idx += 1
        seg = mix[s:e]
        if progress:
            progress(f"    chunk {idx}/{total}  {s/sr:.0f}–{e/sr:.0f}s")
        # ⚠️ 必须先归一到 2-D 再分配 acc —— 否则单声道模型会踩上面 _as2d 说的坑
        stems = {k: _as2d(v) for k, v in run_fn(seg).items()}

        m = len(seg)
        w = _fade(m, sr, overlap_s)
        if wsum is None:
            wsum = np.zeros(n_total, dtype=np.float32)
            for k, y in stems.items():
                acc[k] = np.zeros((n_total, y.shape[1]), dtype=np.float32)

        # 首块不做左侧淡化、末块不做右侧淡化（避免边缘被压低）
        if s == 0:
            k = min(int(overlap_s * sr), m // 2)
            if k > 1:
                w[:k] = 1.0
        if e == n_total:
            k = min(int(overlap_s * sr), m // 2)
            if k > 1:
                w[-k:] = 1.0

        for name, y in stems.items():
            y = _match_channels(np.asarray(y)[:m], acc[name].shape[1])
            if len(y) < m:
                y = np.pad(y, ((0, m - len(y)), (0, 0)))
            acc[name][s:e] += y * w[:, None]
        wsum[s:e] += w
        pos = e

    if wsum is not None:
        wsum[wsum < 1e-8] = 1.0
        for k in acc:
            acc[k] /= wsum[:, None]
    return acc
