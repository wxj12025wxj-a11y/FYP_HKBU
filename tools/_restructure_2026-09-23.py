#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FYP_HKBU 目录重构（2026-09-23）—— 只移动 / 新建，**不删除任何文件**。

目标结构（顶层收敛为 5 项）
--------------------------------
    01_models/     模型 —— 每个模型一个独立文件夹（含模型卡 + 权重链接）
    02_databases/  数据
    03_outputs/    产物 —— 每个模型一个独立文件夹（test/ 与 stage0_gate/）
    04_reports/    报告 —— 按板块：separation / denoiser / cascade / _shared
    tools/         工具 —— 脚本 + _logs + _scratch(含 torch·hf 缓存)

移出的无关内容落在 `E:\\FYP_HKBU_evicted\\`（由 bash 段完成，见 MIGRATION 报告）。

设计约束
--------
* 权重**不做拷贝**：同盘用 NTFS 硬链接（文件）与目录联接（目录），零额外占用。
  因此 `01_models/_weights/` 仍是唯一的物理落点，脚本与 `paths.py` 无需改所有权重路径。
* 每一步都写 `_restructure_manifest.json`，可逐条逆向还原。
* 全部 print 使用 ASCII 标记（GBK 控制台下中文会崩）。

用法
----
    python tools/_restructure_2026-09-23.py --step misc|models|outputs|reports|all
    python tools/_restructure_2026-09-23.py --step misc --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths  # noqa: E402

ROOT = _paths.PROJECT_ROOT
MANIFEST = ROOT / "04_reports" / "docs" / "_restructure_manifest.json"
PROFILE_CSV = ROOT / "04_reports" / "data" / "model_analysis" / "model_profile.csv"

_ops: list[dict] = []


def log(tag: str, msg: str) -> None:
    print("[%-5s] %s" % (tag, msg))


def record(kind: str, src: str, dst: str) -> None:
    _ops.append({"kind": kind, "src": str(src), "dst": str(dst)})


def mkdir(p: Path, dry: bool) -> None:
    if p.exists():
        return
    if not dry:
        p.mkdir(parents=True, exist_ok=True)
    record("mkdir", "", str(p))


def move(src: Path, dst: Path, dry: bool) -> bool:
    """同盘 mv（os.rename），秒级完成。目标已存在则跳过。"""
    if not src.exists():
        log("skip", "%s (not present)" % src.relative_to(ROOT) if ROOT in src.parents else src)
        return False
    if dst.exists():
        log("skip", "dst exists: %s" % dst)
        return False
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(str(src), str(dst))
    record("move", str(src), str(dst))
    return True


def link_file(link: Path, target: Path, dry: bool) -> bool:
    """NTFS 硬链接：同盘零拷贝，原文件删除后链接仍持有数据。"""
    if link.exists():
        return False
    if not target.exists():
        log("warn", "target missing: %s" % target)
        return False
    if dry:
        record("hardlink", str(target), str(link))
        return True
    r = subprocess.run(["cmd", "/c", "mklink", "/H", str(link), str(target)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        log("ERR", "hardlink failed %s -> %s | %s" % (link.name, target.name,
                                                      (r.stdout + r.stderr).strip()[:120]))
        return False
    record("hardlink", str(target), str(link))
    return True


def link_dir(link: Path, target: Path, dry: bool) -> bool:
    """目录联接（junction）：不需管理员权限，指向目录。"""
    if link.exists():
        return False
    if not target.exists():
        log("warn", "target missing: %s" % target)
        return False
    if dry:
        record("junction", str(target), str(link))
        return True
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        log("ERR", "junction failed %s -> %s | %s" % (link.name, target.name,
                                                      (r.stdout + r.stderr).strip()[:120]))
        return False
    record("junction", str(target), str(link))
    return True


# ===========================================================================
# 步骤 1：05_misc 并进 tools/
# ===========================================================================
def step_misc(dry: bool) -> None:
    print("\n" + "=" * 78)
    print("STEP misc :: 05_misc -> tools/_logs, tools/_scratch")
    print("=" * 78)
    misc = ROOT / "05_misc"
    tools = ROOT / "tools"
    mkdir(tools / "_logs", dry)

    # scratch 目录整体改名（不要预先 mkdir 目标，否则 rename 会因目标已存在而跳过）
    if (misc / "scratch").is_dir() and not (tools / "_scratch").exists():
        if not dry:
            os.rename(str(misc / "scratch"), str(tools / "_scratch"))
        record("move", str(misc / "scratch"), str(tools / "_scratch"))
        log("ok", "05_misc/scratch -> tools/_scratch")

    for item in sorted(misc.iterdir()) if misc.exists() else []:
        if item.name == "logs":
            for sub in sorted(item.iterdir()):
                move(sub, tools / "_logs" / sub.name, dry)
            if not dry and item.exists() and not any(item.iterdir()):
                item.rmdir()
            continue
        if item.name == "README.md":
            # 原文整体搬到 tools/_logs/ 留档（保留字节，不重写）
            move(item, tools / "_logs" / "_FROM_05_misc_README_ORIGINAL.md", dry)
            continue
        if item.name in ("scratch", "_scratch_tmpmove"):
            continue
        # 三个 Task Scheduler 启动器 -> tools/_bat/
        mkdir(tools / "_bat", dry)
        move(item, tools / "_bat" / item.name, dry)

    # 残留空目录
    if not dry and misc.exists():
        left = [p for p in misc.rglob("*")]
        if not left:
            misc.rmdir()
            record("rmdir-empty", str(misc), "")
            log("ok", "05_misc removed (was empty)")
        else:
            log("warn", "05_misc still has %d item(s)" % len(left))


# ===========================================================================
# 步骤 2：01_models 每模型独立文件夹
# ===========================================================================
CITE = {
    "oracle": "Rafii, Liutkus, Stoter, Mimilakis, Bittner. MUSDB18 - a corpus for music "
              "separation. 2017 (sigsep-mus-oracle); IRM 定义见 Wang & Wang, IEEE/ACM TASLP 2014",
    "rpca": "Huang, Chen, Smaragdis, Hasegawa-Johnson. Singing-voice separation from "
            "monaural recordings using robust principal component analysis. ICASSP 2012; "
            "求解器 Lin, Chen, Ma. The Augmented Lagrange Multiplier Method (2010)",
    "convtasnet": "Luo, Mesgarani. Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude "
                  "Masking for Speech Separation. IEEE/ACM TASLP 27(8), 2019",
    "dprnn": "Luo, Chen, Mesgarani. Dual-Path RNN: Efficient Long Sequence Modeling for "
             "Time-Domain Single-Channel Speech Separation. ICASSP 2020",
    "mmdenselstm": "Takahashi, Matsubara, Uehara. Multi-scale Multi-band DenseLSTM for "
                   "Audio Source Separation. IEEE Signal Processing Letters 27, 2020",
    "umx": "Stoter, Uhlich, Liutkus, Mitsufuji. Open-Unmix - A Reference Implementation "
           "for Music Source Separation. JOSS 4(41):1667, 2019",
    "demucs": "Rouard, Massa, Defossez. Hybrid Transformers for Music Source Separation. "
              "ICASSP 2023",
    "mdx": "Kim, Choi, Chung, Lee. KUIELab-MDX-Net: A Two-Stream Neural Network for Music "
           "Demixing. MDX Workshop 2021; 评测框架 Mitsufuji et al., Frontiers in Signal "
           "Processing 2022",
    "bsroformer": "Lu, Wang, Jiang, Zhang. Music Source Separation with Band-Split RoPE "
                  "Transformer. 2024 (arXiv:2309.02612)",
    "bsrnn": "Luo, Yu. Music Source Separation with Band-split RNN. IEEE/ACM TASLP 2023 "
             "(arXiv:2209.15174)",
    "denoiser": "Defossez, Synnaeve, Adi. Real Time Speech Enhancement in the Waveform "
                "Domain. Interspeech 2020 (facebookresearch/denoiser)",
    "melroformer": "架构源自 Lu et al. Mel-Band RoFormer for Music Source Separation "
                   "(arXiv:2310.01809)",
}

COMMUNITY = {
    "Mel-RoFormer-Denoise": "aufr33",
    "Mel-RoFormer-Dereverb": "anvuew",
    "Mel-RoFormer-Dereverb-Echo": "Sucial",
}

MSS = "01_models/_third_party/Music-Source-Separation-Training"
CACHE = "tools/_scratch/.cache/torch/hub/checkpoints"

# folder, keys, board, third_party, weights[(kind, relpath)], citation_key, note
SPECS = [
    ("BS-RoFormer-L12", ["bsroformer_l12"], "separation", MSS,
     [("file", "01_models/_weights/BS-RoFormer/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
      ("file", "01_models/_weights/BS-RoFormer/configs/model_bs_roformer_ep_317_sdr_12.9755.yaml")],
     "bsroformer", "单目标（只出 vocals），L12 = 12 层 Transformer，本机全部条目里 FLOPs 最高"),
    ("BS-RoFormer-L6", ["bsroformer_l6"], "separation", MSS,
     [("file", "01_models/_weights/BS-RoFormer/model_bs_roformer_ep_937_sdr_10.5309.ckpt"),
      ("file", "01_models/_weights/BS-RoFormer/configs/model_bs_roformer_ep_937_sdr_10.5309.yaml")],
     "bsroformer", "输出为 mixture-drums-bass（vocals+other 残差），**不是** yaml 里写的 other"),
    ("BSRNN-opt", ["bsrnn", "bsrnn_all"], "separation", "01_models/_third_party/bsrnn",
     [("dir", "01_models/_weights/BSRNN/bsrnn-opt")],
     "bsrnn", "同一批权重服务两个条目：单目标 vocals 与四轨 4-stem（4 ckpt）"),
    ("BSRNN-large", ["bsrnn_large", "bsrnn_large_all"], "separation", "01_models/_third_party/bsrnn",
     [("dir", "01_models/_weights/BSRNN/bsrnn-large")],
     "bsrnn", "同上，large 配置；单首墙钟最快（46.7 s）"),
    ("BSRNN-SIMO", ["bsrnn_simo"], "separation", "01_models/_third_party/bsrnn",
     [("dir", "01_models/_weights/BSRNN/simo-bsrnn-opt")],
     "bsrnn", "SIMO 架构，精度第一（四轨均值 9.27 dB），但 50 首只跑完 14 首"),
    ("Demucs", ["demucs"], "separation", "01_models/_third_party/demucs",
     [("file", "%s/955717e8-8726e21a.th" % CACHE)],
     "demucs", "htdemucs 混合时域+频域架构；参考表里 128 M 指原版 v1，本机是 41.98 M"),
    ("MDX-Net", ["mdx"], "separation", "01_models/_third_party/mdx-net",
     [("dir", "01_models/_weights/MDX-Net/mdx_extra"),
      ("file", "01_models/_weights/MDX-Net/mixer.ckpt")],
     "mdx", "mdx_extra 由 4 个 submission 权重组成；全场参数最多（334.5 M）"),
    ("Open-Unmix", ["umx"], "separation", "01_models/_third_party/DNN-based_source_separation",
     [("file", "%s/vocals-b62c91ce.pth" % CACHE), ("file", "%s/drums-9619578f.pth" % CACHE),
      ("file", "%s/bass-8d85a5bd.pth" % CACHE), ("file", "%s/other-b52fbbf7.pth" % CACHE)],
     "umx", "四个目标各一套网络（各占 25% 参数），基线参考实现"),
    ("Conv-TasNet", ["convtasnet"], "separation", "01_models/_third_party/Conv-TasNet",
     [("dir", "01_models/_weights/DNN-based_source_separation/ConvTasNet")],
     "convtasnet", "纯时域；other 轨 SDR 仅 1.66 dB，是最弱的一项"),
    ("MMDenseLSTM", ["mmdenselstm"], "separation",
     "01_models/_third_party/DNN-based_source_separation",
     [("dir", "01_models/_weights/DNN-based_source_separation/MMDenseLSTM")],
     "mmdenselstm", "参数最少（5.49 M），但时延比同量级模型高"),
    ("DPRNN", ["dprnn"], "separation", "01_models/_third_party/Dual-Path-RNN-Pytorch",
     [("dir", "01_models/_weights/dprnn_musdb")],
     "dprnn", "**唯一自训模型**（3.69 M，11.025 kHz, 4 s chunk）；SI-SDR 为负 → 只作负面证据"),
    ("RPCA", ["rpca"], "separation", "01_models/_selfimpl",
     [], "rpca", "解析算法，零参数；单次 10 s 需 82.5 s（纯 CPU 矩阵迭代）；稀疏分量无天然 stem 归属"),
    ("Oracle-IRM", ["oracle"], "separation", "01_models/_third_party/sigsep-mus-oracle",
     [], "oracle", "理论上界（IRM/IBM），零参数，用来给所有模型划天花板"),
    ("denoiser-dns48", ["denoiser_dns48"], "denoiser", "01_models/_third_party/denoiser",
     [("file", "01_models/_weights/denoiser/dns48-11decc9d8e3f0998.th")],
     "denoiser", "板 2；16 kHz 因果波形域；参数量最小（18.87 M），门禁 SI-SDR 改善最高（+2.15 dB）"),
    ("denoiser-dns64", ["denoiser_dns64"], "denoiser", "01_models/_third_party/denoiser",
     [("file", "01_models/_weights/denoiser/dns64-a7761ff99a7d5bb6.th")],
     "denoiser", "板 2；16 kHz；与 master64 同参数量（33.53 M），训练集不含 Valentini"),
    ("denoiser-master64", ["denoiser_master64"], "denoiser", "01_models/_third_party/denoiser",
     [("file", "01_models/_weights/denoiser/master64-8a5dfb4bb92753dd.th")],
     "denoiser", "板 2；**训练集含 Valentini → 测 Valentini 属 in-domain，必须报 dns48/dns64 对照**"),
    ("Mel-RoFormer-Denoise", ["mel_roformer_denoise", "mel_roformer_denoise_aggr"], "denoiser",
     MSS, [("dir", "01_models/_weights/mel_roformer_denoise")],
     "melroformer", "板 2；44.1 kHz 立体声非因果；掩码估计头独占 88.3% 参数"),
    ("Mel-RoFormer-Dereverb", ["mel_roformer_dereverb"], "denoiser", MSS,
     [("dir", "01_models/_weights/mel_roformer_dereverb")],
     "melroformer", "板 2 去混响；社区权重，非论文官方"),
    ("Mel-RoFormer-Dereverb-Echo", ["mel_roformer_dereverb_echo"], "denoiser", MSS,
     [("dir", "01_models/_weights/mel_roformer_dereverb_echo")],
     "melroformer", "板 2 去混响+去回声；社区权重（Sucial），SDR 10.0169"),
]


def load_profile() -> dict:
    with open(PROFILE_CSV, encoding="utf-8-sig") as f:
        return {r["model_key"]: r for r in csv.DictReader(f)}


def step_models(dry: bool) -> None:
    print("\n" + "=" * 78)
    print("STEP models :: 01_models/<MODEL>/  (model card + weight links)")
    print("=" * 78)
    prof = load_profile()
    base = ROOT / "01_models"
    n_card = n_link = 0

    for folder, keys, board, tp, weights, cite_key, note in SPECS:
        d = base / folder
        mkdir(d, dry)
        # ---- 权重链接 ----
        wdir = d / "weights"
        need_w = bool(weights)
        if need_w:
            mkdir(wdir, dry)
        made = []
        for kind, rel in weights:
            tgt = ROOT / rel
            if kind == "dir":
                dest = wdir / Path(rel).name
                ok = link_dir(dest, tgt, dry)
            else:
                dest = wdir / Path(rel).name
                ok = link_file(dest, tgt, dry)
            if ok:
                made.append(Path(rel).name)
                n_link += 1
        if need_w and not dry and not any(wdir.iterdir()):
            wdir.rmdir()

        # ---- 模型卡 ----
        rows = [prof[k] for k in keys if k in prof]
        L = []
        L.append("# %s" % folder)
        L.append("")
        L.append("> 板块：**%s** ｜ 目录：`01_models/%s/`" % (
            "板 1 分离" if board == "separation" else "板 2 降噪 / 去混响", folder))
        L.append("")
        L.append(note)
        L.append("")
        L.append("## 论文 / 出处")
        L.append("")
        L.append("- %s" % CITE[cite_key])
        if folder in COMMUNITY:
            L.append("- ⚠ 权重由 **%s** 训练并经 ZFTurbo 框架分发，属**社区模型**，非论文官方权重，"
                     "论文引用时须注明。" % COMMUNITY[folder])
        L.append("")
        L.append("## 代码位置")
        L.append("")
        L.append("- `%s`" % tp)
        L.append("- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`"
                 "（板 2 见 `tools/_denoise_registry.py`）")
        L.append("")
        L.append("## 本机权重（真实落点）")
        L.append("")
        if weights:
            L.append("本目录下 `weights/` 是指向**同一份物理文件**的硬链接 / 目录联接，不占额外空间。")
            L.append("")
            L.append("| 权重文件 | 本机真实路径 |")
            L.append("|---|---|")
            for _, rel in weights:
                L.append("| `%s` | `%s` |" % (Path(rel).name, rel))
            L.append("")
            L.append("> 物理落点唯一：`01_models/_weights/`。改路径只改 `tools/paths.py`。")
        else:
            L.append("**本模型无权重文件**——它是解析算法（零参数），公式即模型。")
            L.append("")
            L.append("实现见 `01_models/_selfimpl/README.md` 指向的 "
                     "`tools/compare_separation_methods.py`。")
        L.append("")
        L.append("## 本机实测画像")
        L.append("")
        if rows:
            L.append("| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |")
            L.append("|---|---|---|---|---|---|---|")
            for r in rows:
                nt = r["n_targets"] or "-"
                if r["model_key"] == "oracle":
                    nt = "4"
                L.append("| `%s` | %s | %s M | %s G | %s s | %s MB | %s Hz / %s ch |" % (
                    r["model_key"], nt, r["params_M"], r["flops_G"], r["latency_s"],
                    r["peak_vram_mb"], r["native_sr"], r["native_ch"]))
            L.append("")
            L.append("口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 "
                     "`torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合"
                     "注意力不计入），跨架构比较必须配合时延看。")
            L.append("")
            L.append("数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`")
        else:
            L.append("（画像数据待补）")
        L.append("")
        L.append("## 本模型在项目里的位置")
        L.append("")
        b1 = "board-1" if board == "separation" else "board-2"
        L.append("- 产物目录：`03_outputs/%s/`" % folder)
        L.append("- 架构原理图：`04_reports/_shared/figures/model_arch/`")
        L.append("- 主结果表：`04_reports/%s/docs/`" % (
            "separation" if board == "separation" else "denoiser"))
        L.append("- 板块标签：`%s`" % b1)
        L.append("")
        if not dry:
            (d / "README.md").write_text("\n".join(L), encoding="utf-8")
        n_card += 1

    log("ok", "model folders=%d  model cards=%d  weight links=%d" % (len(SPECS), n_card, n_link))
    if not dry:
        write_models_index(SPECS, prof)
        log("ok", "wrote 01_models/README.md index")


def write_models_index(specs, prof) -> None:
    L = ["# ① 模型模块", "",
         "每个模型一个独立文件夹。**权重不做拷贝**：`<模型>/weights/` 是指向 "
         "`01_models/_weights/` 同一份物理文件的硬链接 / 目录联接，零额外占用。", "",
         "## 板 1 · 分离模型", "",
         "| 文件夹 | 条目 key | 输出轨数 | 参数 | FLOPs | 时延(10 s) | 权重 |",
         "|---|---|---|---|---|---|---|"]
    for folder, keys, board, tp, weights, ck, note in specs:
        if board != "separation":
            continue
        rows = [prof[k] for k in keys if k in prof]
        nt = "/".join(sorted({(r["n_targets"] or "-") for r in rows}))
        p = "/".join(sorted({r["params_M"] for r in rows}))
        fl = "/".join(sorted({r["flops_G"] for r in rows}))
        la = "/".join(sorted({r["latency_s"] for r in rows}))
        w = "有（%d 项）" % len(weights) if weights else "无（解析）"
        L.append("| [`%s/`](%s/) | %s | %s | %s M | %s G | %s s | %s |"
                 % (folder, folder, ", ".join("`%s`" % k for k in keys), nt, p, fl, la, w))
    L += ["", "## 板 2 · 降噪 / 去混响模型", "",
          "| 文件夹 | 条目 key | 参数 | FLOPs | 时延(10 s) | 原生率 |",
          "|---|---|---|---|---|---|"]
    for folder, keys, board, tp, weights, ck, note in specs:
        if board != "denoiser":
            continue
        rows = [prof[k] for k in keys if k in prof]
        p = "/".join(sorted({r["params_M"] for r in rows}))
        fl = "/".join(sorted({r["flops_G"] for r in rows}))
        la = "/".join(sorted({r["latency_s"] for r in rows}))
        sr = "/".join(sorted({"%.0f" % float(r["native_sr"]) for r in rows if r["native_sr"]}))
        L.append("| [`%s/`](%s/) | %s | %s M | %s G | %s s | %s Hz |"
                 % (folder, folder, ", ".join("`%s`" % k for k in keys), p, fl, la, sr))
    L += ["", "## 共享层", "",
          "| 目录 | 内容 |", "|---|---|",
          "| `_weights/` | **权重的唯一物理落点**（15 GB）。不能删，各模型文件夹的 `weights/` 都指向它 |",
          "| `_third_party/` | 13 个官方/社区仓库（743 MB）。同一仓库被多个模型复用 |",
          "| `_selfimpl/` | 自研算法索引（RPCA / NMF / ICA / HPSS / Oracle 掩码） |",
          "| `MODEL_REGISTRY.md` | 全部模型在位状态 / 可运行性 / 精度总表 |", ""]
    (ROOT / "01_models" / "README.md").write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
# 步骤 3：03_outputs 每模型独立文件夹
# ===========================================================================
def step_outputs(dry: bool) -> None:
    print("\n" + "=" * 78)
    print("STEP outputs :: 03_outputs/<MODEL>/{test, stage0_gate}")
    print("=" * 78)
    out = ROOT / "03_outputs"
    tr = out / "_test_run"
    gate = out / "_stage0_gate"

    if tr.exists():
        for sub in sorted(tr.iterdir()):
            if not sub.is_dir():
                move(sub, out / ("_" + sub.name), dry)
                continue
            dest = out / sub.name
            mkdir(dest, dry)
            move(sub, dest / "test", dry)
            log("ok", "%s -> %s/test" % (sub.name, sub.name))
        if not dry and tr.exists() and not any(tr.iterdir()):
            tr.rmdir()
            record("rmdir-empty", str(tr), "")

    # _stage0_gate/<模型 key> -> 对应模型文件夹/stage0_gate
    KEY2FOLDER = {
        "denoiser_dns48": "denoiser-dns48", "denoiser_dns64": "denoiser-dns64",
        "denoiser_master64": "denoiser-master64",
        "mel_roformer_denoise": "Mel-RoFormer-Denoise",
        "mel_roformer_denoise_aggr": "Mel-RoFormer-Denoise",
        "mel_roformer_dereverb": "Mel-RoFormer-Dereverb",
        "mel_roformer_dereverb_echo": "Mel-RoFormer-Dereverb-Echo",
    }
    if gate.exists():
        for sub in sorted(gate.iterdir()):
            if not sub.is_dir():
                continue
            folder = KEY2FOLDER.get(sub.name)
            if not folder:
                key = sub.name
                for f, keys, *_rest in SPECS:
                    if key in keys:
                        folder = f
                        break
            if not folder:
                log("warn", "no model folder for gate entry %s" % sub.name)
                continue
            dest = out / folder / "stage0_gate"
            mkdir(dest, dry)
            if (dest / sub.name).exists():
                continue
            move(sub, dest / sub.name, dry)
        if not dry and gate.exists() and not any(gate.iterdir()):
            gate.rmdir()
            record("rmdir-empty", str(gate), "")
            log("ok", "_stage0_gate removed (was empty)")

    write_outputs_index()


def write_outputs_index() -> None:
    out = ROOT / "03_outputs"
    dirs = sorted([d for d in out.iterdir() if d.is_dir()])
    L = ["# ③ 产物模块", "",
         "**每个模型一个独立文件夹**，模型自己的输出不再和其它模型混在一起。", "",
         "约定：", "",
         "```",
         "03_outputs/",
         "└── <模型名>/",
         "    ├── test/          MUSDB18 test 50 首整曲分离结果（每首一个子目录）",
         "    │   └── <歌曲名>/  vocals.flac / drums.flac / bass.flac / other.flac",
         "    └── stage0_gate/   阶段 0 门禁的单文件输出（out.wav，仅板 2 模型有）",
         "```", "",
         "| 模型文件夹 | 歌曲数 | test | stage0_gate |", "|---|---|---|---|"]
    for d in dirs:
        t = d / "test"
        g = d / "stage0_gate"
        nsong = len([x for x in t.iterdir() if x.is_dir()]) if t.is_dir() else 0
        ng = len([x for x in g.iterdir()]) if g.is_dir() else 0
        L.append("| `%s/` | %d | %s | %s |"
                 % (d.name, nsong, "有" if nsong else "—", "%d 项" % ng if ng else "—"))
    L += ["", "> 跨模型汇总、图表、指标 JSON **不放这里**，统一进 `04_reports/`。", ""]
    (out / "README.md").write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
# 步骤 4：04_reports 按板块重排
# ===========================================================================
BOARD_DIRS = ["separation", "denoiser", "cascade", "_shared"]
SUB_DIRS = ["docs", "figures", "data", "html"]

# (旧相对路径, 归属板块, 目标子目录)
DOC_MAP = {
    "DEMUCS_REPORT.md": "separation",
    "METHOD_COMPARISON_REPORT.md": "separation",
    "MUSDB18_TEST_SWEEP_2026-09-22.md": "separation",
    "MEDIAN_TABLE.md": "separation",
    "MEDIAN_TABLE_TEST.md": "separation",
    "MODEL_PROVISIONING_REPORT.md": "separation",
    "MODEL_RULE_AUDIT_2026-09-19.md": "separation",
    "PROJECT_AUDIT_2026-09-19.md": "separation",
    "CODE_REVIEW_REPORT.md": "separation",
    "FYP_PLAN_2026-09-22.md": "_shared",
    "FYP_ROADMAP_2026-09-22.md": "_shared",
    "FYP_RETROSPECTIVE_2026-09-22.md": "_shared",
    "MODEL_PROFILE_2026-09-22.md": "_shared",
    "FYP_Audio_Setup_Guide.md": "_shared",
}
FIG_DIR_MAP = {
    "comparison": "separation", "demucs": "separation", "test_run": "separation",
    "figs_time": "separation", "audio_analysis": "separation", "spectrograms": "separation",
    "dprnn": "separation", "reverb": "separation",
    "denoise": "denoiser",
    "cascade": "cascade",
    "model_arch": "_shared", "_smoketest": "_shared",
}
DATA_DIR_MAP = {
    "comparison": "separation",
    "denoise": "denoiser",
    "reverb": "separation",
    "cascade": "cascade",
    "model_analysis": "_shared",
}
HTML_MAP = {
    "METHOD_VISUAL_COMPARISON_1.3.html": "separation",
    "RUNTIME_BENCH_PPT.html": "separation",
    "test_run_dashboard.html": "separation",
    "listen_compare.html": "separation",
    "_qa_audio_probe.html": "_shared",
}
HTML_SUPERSEDED = ["METHOD_VISUAL_COMPARISON_1.0.html", "METHOD_VISUAL_COMPARISON_1.2.html"]


def step_reports(dry: bool) -> None:
    print("\n" + "=" * 78)
    print("STEP reports :: 04_reports/{separation,denoiser,cascade,_shared}/{docs,figures,data,html}")
    print("=" * 78)
    rep = ROOT / "04_reports"
    for b in BOARD_DIRS:
        for s in SUB_DIRS:
            mkdir(rep / b / s, dry)

    # ---- docs ----
    for name, board in DOC_MAP.items():
        src = rep / "docs" / name
        move(src, rep / board / "docs" / name, dry)
    if not dry and (rep / "docs").exists():
        left = sorted(p.name for p in (rep / "docs").iterdir())
        if not left:
            (rep / "docs").rmdir()
            record("rmdir-empty", str(rep / "docs"), "")
        else:
            log("info", "04_reports/docs 残留: %s" % left)

    # ---- figures ----
    for name, board in FIG_DIR_MAP.items():
        src = rep / "figures" / name
        if src.is_dir():
            tgt = rep / board / "figures" / name
            mkdir(tgt.parent, dry)
            move(src, tgt, dry)
    for extra in ["README.md"]:
        src = rep / "figures" / extra
        if src.is_file():
            move(src, rep / "_shared" / "figures" / extra, dry)
    if not dry and (rep / "figures").exists():
        left = sorted(p.name for p in (rep / "figures").iterdir())
        if not left:
            (rep / "figures").rmdir()
            record("rmdir-empty", str(rep / "figures"), "")
        else:
            log("info", "04_reports/figures 残留: %s" % left)

    # ---- data ----
    for name, board in DATA_DIR_MAP.items():
        src = rep / "data" / name
        if src.is_dir():
            tgt = rep / board / "data" / name
            mkdir(tgt.parent, dry)
            move(src, tgt, dry)
    for p in sorted((rep / "data").glob("*.json")) if (rep / "data").exists() else []:
        # dprnn_training*.json 是自训记录 -> separation
        move(p, rep / "separation" / "data" / p.name, dry)
    if not dry and (rep / "data").exists():
        left = sorted(p.name for p in (rep / "data").iterdir())
        if not left:
            (rep / "data").rmdir()
            record("rmdir-empty", str(rep / "data"), "")
        else:
            log("info", "04_reports/data 残留: %s" % left)

    # ---- html ----
    for name, board in HTML_MAP.items():
        move(rep / "html" / name, rep / board / "html" / name, dry)
    mkdir(rep / "_shared" / "html" / "_superseded", dry)
    for name in HTML_SUPERSEDED:
        move(rep / "html" / name, rep / "_shared" / "html" / "_superseded" / name, dry)
    if not dry and (rep / "html").exists():
        left = sorted(p.name for p in (rep / "html").iterdir())
        if not left:
            (rep / "html").rmdir()
            record("rmdir-empty", str(rep / "html"), "")
        else:
            log("info", "04_reports/html 残留: %s" % left)

    write_reports_index(rep)


def write_reports_index(rep: Path) -> None:
    L = ["# ④ 报告模块", "",
         "按**板块**分文件夹，每个板块下再分 `docs / figures / data / html` 四类。", "",
         "| 文件夹 | 归属 | 计划阶段 |", "|---|---|---|",
         "| `separation/` | **板 1 · 分离** | 阶段 1-4（模型画像、架构图、频谱面板、混响鲁棒性）|",
         "| `denoiser/` | **板 2 · 降噪** | 阶段 5-7（降噪评测、去混响/去回声）|",
         "| `cascade/` | **级联**（板 2 收尾）| 阶段 6-7（分离→降噪串联）|",
         "| `_shared/` | 跨板块共用 | 总纲文档、模型画像、架构图、门禁数据 |",
         "| `slides/` | 答辩材料 | 阶段 8 |",
         "", "## 各板块内容", ""]
    for b, desc in [("separation", "板 1：模型本体画像、14 张架构原理图、频谱面板、混响/去混响实验"),
                    ("denoiser", "板 2：降噪双口径评测表、去混响与去回声模型结果"),
                    ("cascade", "级联实验：分离输出 → 降噪输入 的端到端结果"),
                    ("_shared", "总纲（计划/路线图/复盘）、模型画像表、架构图数据、阶段 0 门禁")]:
        L.append("### `%s/`" % b)
        L.append("")
        L.append(desc)
        L.append("")
        for s in SUB_DIRS:
            d = rep / b / s
            if not d.is_dir():
                continue
            items = sorted(x.name for x in d.iterdir())
            if not items:
                L.append("- `%s/` —— （空，待阶段产出）" % s)
            else:
                shown = ", ".join("`%s`" % i for i in items[:12])
                more = "" if len(items) <= 12 else " …共 %d 项" % len(items)
                L.append("- `%s/` —— %s%s" % (s, shown, more))
        L.append("")
    (rep / "README.md").write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
STEPS = {"misc": step_misc, "models": step_models,
         "outputs": step_outputs, "reports": step_reports}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    help="all | misc | models | outputs | reports")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    which = list(STEPS) if args.step == "all" else [args.step]
    for s in which:
        if s not in STEPS:
            print("unknown step: %s" % s)
            return 2
        STEPS[s](args.dry_run)

    if not args.dry_run:
        old = []
        if MANIFEST.exists():
            try:
                old = json.loads(MANIFEST.read_text(encoding="utf-8")).get("ops", [])
            except Exception:  # noqa: BLE001
                old = []
        MANIFEST.write_text(json.dumps(
            {"date": "2026-09-23", "project_root": str(ROOT),
             "note": "只移动/新建，未删除任何文件。逆向：按 dst->src 反向 mv；"
                     "硬链接直接删除即可（原文件不受影响）。",
             "ops": old + _ops}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n[ok   ] manifest -> %s  (%d ops)" % (MANIFEST, len(old) + len(_ops)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
