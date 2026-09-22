#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FYP_HKBU 项目结构与资产自检。

一条命令回答三个问题：
  1. 五大模块目录是否齐备？
  2. 12 个模型的**代码 / 权重**是否都在本机？
  3. 数据集（MUSDB18 + DNS）是否到位？

用法：
    python tools/selfcheck_project.py
    python tools/selfcheck_project.py --md > 04_reports/_shared/data/STRUCTURE_SELFCHECK.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths                                          # noqa: E402

OK, BAD, WARN = "✅", "❌", "⚠️"

# (模型名, 代码候选路径列表, 权重候选路径列表, 状态, 说明)
# 状态：run=可运行 / blocked=受阻（可解） / dead=不可得
MODELS: list[tuple[str, list[Path], list[Path], str, str]] = [
    ("RPCA", [], [], "run",
     "自研 → `tools/compare_separation_methods.py::sep_rpca`（解析法，无需权重）"),
    ("RPCA+DRNN", [], [], "blocked",
     "论文未公开代码，需按 Lai & Wang 2022 自研复现"),
    ("Conv-TasNet",
     [_paths.THIRD_PARTY / "Conv-TasNet",
      _paths.THIRD_PARTY / "DNN-based_source_separation"],
     [_paths.WEIGHTS / "DNN-based_source_separation" / "ConvTasNet"], "run", ""),
    ("DPRNN", [_paths.THIRD_PARTY / "Dual-Path-RNN-Pytorch"], [], "blocked",
     "代码在位，但官方只发布语音权重，**缺 MUSDB18 权重** → 只能自训"),
    ("MMDenseLSTM", [_paths.THIRD_PARTY / "DNN-based_source_separation"],
     [_paths.WEIGHTS / "DNN-based_source_separation" / "MMDenseLSTM"], "run", ""),
    ("BS-RoFormer L12", [_paths.THIRD_PARTY / "Music-Source-Separation-Training"],
     [_paths.WEIGHTS / "BS-RoFormer"], "run", "单目标：vocals (ep_317)"),
    ("BS-RoFormer L6", [_paths.THIRD_PARTY / "Music-Source-Separation-Training"],
     [_paths.WEIGHTS / "BS-RoFormer"], "run", "单目标：vocals+other (ep_937)"),
    ("BSRNN", [_paths.THIRD_PARTY / "bsrnn"], [_paths.WEIGHTS / "BSRNN"], "run",
     "3 变体：oBSRNN / large / SIMO"),
    ("IRM/IBM Oracle", [_paths.THIRD_PARTY / "sigsep-mus-oracle"], [], "run",
     "解析 oracle（理论上界），无需权重"),
    ("MDX-Net", [_paths.THIRD_PARTY / "mdx-net", _paths.THIRD_PARTY / "mdx-net-submission"],
     [_paths.WEIGHTS / "MDX-Net"], "run", ""),
    ("Pac-HuBERT-SEP", [], [], "dead", "MERL 未开源代码与权重，无解"),
    ("Demucs", [_paths.THIRD_PARTY / "demucs"],
     [_paths.TORCH_CACHE / "hub" / "checkpoints"], "run", "htdemucs"),
]

STATUS_LABEL = {"run": f"{OK} 可运行", "blocked": "⛔ 受阻（可解）",
                "dead": "❌ 不可得（无解）"}

MODULES = [
    ("01_models", _paths.MODELS,
     [("_third_party", _paths.THIRD_PARTY), ("_weights", _paths.WEIGHTS),
      ("_selfimpl", _paths.SELF_IMPL), ("MODEL_REGISTRY.md", _paths.MODELS / "MODEL_REGISTRY.md"),
      ("<模型>/weights (抽样)", _paths.model_weights("BS-RoFormer-L12"))]),
    ("02_databases", _paths.DATABASES,
     [("MUSDB18-HQ", _paths.MUSDB18_ROOT), ("DNS-Challenge", _paths.DNS_ROOT),
      ("Valentini", _paths.VALENTINI_ROOT)]),
    ("03_outputs", _paths.OUTPUTS,
     [("<模型>/test (抽样)", _paths.out_test("BS-RoFormer-L12")),
      ("<模型>/stage0_gate (抽样)", _paths.out_gate("denoiser-dns48"))]),
    ("04_reports", _paths.REPORTS,
     [("separation", _paths.SEPARATION), ("denoiser", _paths.DENOISER),
      ("cascade", _paths.CASCADE), ("_shared", _paths.SHARED),
      ("slides", _paths.SLIDES)]),
    ("tools", _paths.TOOLS,
     [("_logs", _paths.LOGS), ("_scratch", _paths.SCRATCH), ("_bat", _paths.BAT_DIR)]),
]

DATASETS = [
    ("MUSDB18-HQ train", _paths.MUSDB18_ROOT / "train", 100),
    ("MUSDB18-HQ test", _paths.MUSDB18_ROOT / "test", 50),
    ("DNS dev_testset", _paths.DNS_ROOT / "dev_testset", None),
    ("DNS noise", _paths.DNS_ROOT / "noise", None),
    ("DNS clean", _paths.DNS_ROOT / "clean", None),
    ("DNS impulse_responses", _paths.DNS_ROOT / "impulse_responses", None),
]


def count(p: Path, pattern: str = "*.wav") -> int:
    return sum(1 for _ in p.rglob(pattern)) if p.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="输出 Markdown")
    args = ap.parse_args()
    out: list[str] = []
    w = out.append

    w("# FYP_HKBU 项目自检\n")
    w(f"> 根目录：`{_paths.PROJECT_ROOT}`\n")

    # ---------- 模块 ----------
    w("## ① 五大模块\n")
    w("| 模块 | 状态 | 子项 |")
    w("|---|---|---|")
    for name, root, subs in MODULES:
        if not root.is_dir():
            w(f"| `{name}/` | {BAD} 缺失 | — |")
            continue
        bits = []
        for label, p in subs:
            if label.endswith(".md") or label.endswith("已迁出"):
                bits.append(f"{OK if p.exists() else BAD} {label}")
            else:
                n = count(p) if "databases" in str(root) else None
                mark = OK if p.is_dir() else BAD
                bits.append(f"{mark} {label}" + (f"({n} wav)" if n else ""))
        w(f"| `{name}/` | {OK} | " + " · ".join(bits) + " |")
    w("")

    # ---------- 模型 ----------
    w("## ② 12 个模型（代码 / 权重在位）\n")
    w("| # | 模型 | 代码 | 权重 | 判定 | 说明 |")
    w("|---|---|---|---|---|---|")
    n_code = n_wt = n_run = 0
    for i, (name, code_c, wt_c, status, note) in enumerate(MODELS, 1):
        has_code = bool(code_c) and any(p.is_dir() for p in code_c)
        has_wt = bool(wt_c) and any(p.exists() for p in wt_c)
        cel = OK if has_code else "—"
        wel = OK if has_wt else ("—" if not wt_c else BAD)
        n_code += has_code
        n_wt += has_wt
        if status == "run":
            n_run += 1
        w(f"| {i} | **{name}** | {cel} | {wel} | {STATUS_LABEL[status]} | {note} |")
    w("")
    n_block = sum(1 for m in MODELS if m[3] == "blocked")
    n_dead = sum(1 for m in MODELS if m[3] == "dead")
    w(f"**汇总：共 12 个 —— 可运行 {n_run} · 受阻（可解）{n_block} · 不可得 {n_dead}**")
    w(f"（其中「11 个指定模型」= 上表去掉 Demucs：可运行 {n_run - 1} / 受阻 {n_block} / 不可得 {n_dead}）\n")

    # ---------- 数据集 ----------
    w("## ③ 数据集\n")
    w("| 数据集 | 状态 | 文件数 |")
    w("|---|---|---|")
    for label, p, expect in DATASETS:
        n = count(p)
        if not p.is_dir():
            w(f"| {label} | {BAD} 缺失 | 0 |")
        elif n == 0:
            w(f"| {label} | {WARN} 目录空（未解压或未下载）| 0 |")
        else:
            extra = ""
            if expect:
                extra = f" （曲目 {len([d for d in p.iterdir() if d.is_dir()])}/{expect}）"
            w(f"| {label} | {OK} | {n}{extra} |")
    w("")

    # ---------- 产物 ----------
    w("## ④ 产物目录（03_outputs）\n")
    if _paths.OUTPUTS.is_dir():
        dirs = sorted(d.name for d in _paths.OUTPUTS.iterdir() if d.is_dir())
        w(f"共 {len(dirs)} 个模型目录：" + "、".join(f"`{d}`" for d in dirs))
    else:
        w(f"{BAD} 缺失")
    w("")

    text = "\n".join(out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
