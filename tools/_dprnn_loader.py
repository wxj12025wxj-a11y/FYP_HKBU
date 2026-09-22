#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPRNN 适配器（本机**自训**权重，MUSDB18-HQ 4-stem）。

为什么单独一个模块
------------------
1. `benchmark_model_universal.py`（评测）与 `train_dprnn_musdb.py`（训练）
   必须用**完全同一套**模型构造逻辑，否则 ckpt 能加载但结构悄悄错位。
2. 第三方仓库 `Dual-Path-RNN-Pytorch` 的 `model/__init__.py` 会 `from .model import *`
   连带拉进 Conv-TasNet 那份实现，且包内文件用 `sys.path.append('../')` 找 `utils`，
   走包导入容易被路径问题绊住 → 这里用 importlib **按文件路径**加载，最干净。

⚠️ 多输出模型按位置对齐陷阱
--------------------------
ckpt 里显式存了 `targets`（stem 顺序）。加载端**必须按 ckpt 记的顺序**构造 / 映射，
否则 `load_state_dict(strict=False)` 不会报错、能出声、wav 正常落盘，
但 4 个 mask 分支静默互换 → 指标串味。见 `MODEL_REGISTRY.md` §4。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import paths as _paths

_paths.setup_env()

REPO = _paths.THIRD_PARTY / "Dual-Path-RNN-Pytorch"
CKPT_DIR = _paths.WEIGHTS / "dprnn_musdb"
SR_TRAIN = 11025                      # 模型原生采样率
TARGETS = ("vocals", "drums", "bass", "other")

_MOD = None


def load_model_cls():
    """加载并缓存 `Dual_RNN_model` 类（按文件路径导入，绕过包 `__init__`）。

    🔴 第三方仓库同名包互撞（2026-09-17 GPU 批量基准实测，两地报错）
    --------------------------------------------------------------
    `model/model_rnn.py` 第 7 行是**绝对导入** `from utils.util import check_parameters`，
    所以它需要「顶层 `utils` 包」+「`utils.util.check_parameters`」。而本机同时存在多个
    自带 `utils/` 的仓库（tky823 的 DNN-based_source_separation、ZFTurbo 的
    Music-Source-Separation-Training……），两个方向的遮蔽都会炸：

      * 把 DPRNN 仓库塞进 `sys.path` → 它的 `utils/` 抢到顶层 `utils`，
        后面 `bsroformer_*` 就报 `No module named 'utils.audio_utils'`；
      * 若 tky823 的 `utils` 已先被缓存进 `sys.modules`，
        DPRNN 自己就报 `No module named 'utils.util'`。

    做法：导入期间让 DPRNN **独占** `utils*` 命名空间（先把现有 `utils*` 摘出来，
    导完原样放回），并且不把仓库路径留在 `sys.path` 上。
    副作用清理：`model_rnn.py` 第 2 行会 `sys.path.append('../')`，一并由整体还原抹掉。
    """
    global _MOD
    if _MOD is not None:
        return _MOD
    path = REPO / "model" / "model_rnn.py"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 DPRNN 模型定义: {path}")

    saved_path = list(sys.path)
    saved_utils = {k: v for k, v in sys.modules.items()
                   if k == "utils" or k.startswith("utils.")}
    for k in saved_utils:
        del sys.modules[k]
    sys.path.insert(0, str(REPO))
    try:
        spec = importlib.util.spec_from_file_location("dprnn_model_rnn", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dprnn_model_rnn"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for k in [k for k in list(sys.modules)
                  if k == "utils" or k.startswith("utils.")]:
            del sys.modules[k]
        sys.modules.update(saved_utils)        # 现场还原，别人家的 utils 原封不动
    _MOD = mod.Dual_RNN_model
    return _MOD


def ckpt_path(name: str = "best.pt") -> Path:
    p = CKPT_DIR / name
    if p.is_file():
        return p
    alt = CKPT_DIR / ("last.pt" if name == "best.pt" else "best.pt")
    if alt.is_file():
        return alt
    raise FileNotFoundError(
        f"没有自训 ckpt：{CKPT_DIR}（先跑 tools/train_dprnn_musdb.py）")


def build(which: str = "best.pt", device: str | None = None):
    """返回 (model, meta)。meta 含 cfg / targets / sr / step。"""
    import torch
    Model = load_model_cls()
    path = ckpt_path(which)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    targets = tuple(ck.get("targets") or TARGETS)
    # 按 ckpt 记录重建：num_spks 必须与 targets 长度一致
    cfg["num_spks"] = len(targets)
    model = Model(**cfg)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict 不匹配（missing={len(missing)}, unexpected={len(unexpected)}）"
            f"，不许 strict=False 蒙过去：{list(missing)[:5]} / {list(unexpected)[:5]}")
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    meta = {"ckpt": str(path), "cfg": cfg, "targets": list(targets),
            "sr": int(ck.get("sr", SR_TRAIN)), "step": ck.get("step"),
            "val": ck.get("val"), "device": device}
    return model, meta


__all__ = ["load_model_cls", "build", "ckpt_path", "ckpt_paths", "REPO", "CKPT_DIR",
           "SR_TRAIN", "TARGETS"]


def ckpt_paths() -> dict:
    return {n: (CKPT_DIR / f"{n}.pt") for n in ("best", "last")
            if (CKPT_DIR / f"{n}.pt").is_file()}


if __name__ == "__main__":
    print(f"REPO     = {REPO}  ({'存在' if REPO.is_dir() else '缺失'})")
    print(f"CKPT_DIR = {CKPT_DIR}")
    found = ckpt_paths()
    print(f"ckpt     = {found or '（还没有自训权重）'}")
