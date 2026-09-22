# -*- coding: utf-8 -*-
"""12 模型精度 / 速度总览图（数据源：model_runs_median.json）。

产图（→ 04_reports/separation/figures/comparison/）
----------------------------------------
  fig_median_sdr_bars.png    四轨均值 SDR 横向条形（单目标模型单独标注）
  fig_median_pareto.png      精度 vs RTF 散点（对数横轴，标出 RTF=1 实时线）
  fig_median_per_stem.png    分轨（人声/鼓/贝斯/其他）分组条形

口径说明
--------
数据一律来自 `tools/median_over_clips.py` 产出的多片段中位数表，
即 3 个能量均衡片段各 10 s 取中位数 —— **不要用单片段数字画图**。
RTF 受机器负载影响，图上只标「是否 < 1」，精确值需在空闲机器上重测。

运行：
    python tools/plot_median_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MEDIAN = _paths.comparison_dir() / "model_runs_median.json"
OUT = _paths.FIGURES_COMPARISON

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

STEMS = ["vocals", "drums", "bass", "other"]
STEM_ZH = {"vocals": "人声", "drums": "鼓", "bass": "贝斯", "other": "其他"}
STEM_COLOR = {"vocals": "#4C78A8", "drums": "#F58518", "bass": "#54A24B", "other": "#B279A2"}

# 手工换行，避免长标签互相压
SHORT = {
    "oBSRNN-SIMO": "oBSRNN-SIMO",
    "oBSRNN (4 ckpt)": "oBSRNN (4×ckpt)",
    "MDX-Net": "MDX-Net",
    "BSRNN large (4 ckpt)": "BSRNN large (4×ckpt)",
    "IRM/IBM Oracle": "Oracle (上界)",
    "Demucs (htdemucs)": "Demucs (htdemucs)",
    "Open-Unmix": "Open-Unmix",
    "MMDenseLSTM": "MMDenseLSTM",
    "Conv-TasNet": "Conv-TasNet",
    "RPCA": "RPCA",
}


def load():
    data = json.loads(MEDIAN.read_text(encoding="utf-8"))
    four, single, norec = [], [], []
    for k, e in data.items():
        if e.get("single_target"):
            single.append((k, e))
        elif e.get("sdr_avg4") is None:
            # RPCA：稀疏分量没有天然 stem 归属，不出四轨均值（见报告 §3.1）
            norec.append((k, e))
        else:
            four.append((k, e))
    four.sort(key=lambda kv: -(kv[1].get("sdr_avg4") or -99))
    return data, four, single, norec


def fig_bars(four, single, norec):
    names = [SHORT.get(e["label"], e["label"]) for _, e in four]
    vals = [e["sdr_avg4"] for _, e in four]
    rtf = [e.get("rtf") or 0 for _, e in four]
    best = max(vals)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#2E7D32" if v == best else "#4C78A8" for v in vals]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.62)
    for b, v, r in zip(bars, vals[::-1], rtf[::-1]):
        tag = "（实时）" if 0 < r < 1 else ""
        ax.text(v + 0.12, b.get_y() + b.get_height() / 2,
                f"{v:.2f}{tag}", va="center", fontsize=9)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("四轨均值 museval SDR (dB) —— 3 个均衡片段 × 10 s 中位数")
    ax.set_title("12 模型：四轨均值 SDR（绿 = 本表最高）", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.20)
    ax.grid(axis="x", ls=":", alpha=0.45)
    ax.set_axisbelow(True)

    lines = []
    if single:
        lines.append("单目标模型（不产出 4 stem，不进均值排名）：")
        lines += [f"· {SHORT.get(e['label'], e['label'])}（`{e['single_target']['name']}`）"
                  f" SDR {e['single_target']['sdr_median']:.2f}" for _, e in single]
    if norec:
        lines.append("不计均值：" + "、".join(
            f"{SHORT.get(e['label'], e['label'])}（分量无 stem 归属）" for _, e in norec))
    if lines:
        ax.text(0.985, 0.02, "\n".join(lines),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
                bbox=dict(boxstyle="round,pad=0.45", fc="#fff8e1", ec="#e0c060"))
    fig.savefig(OUT / "fig_median_sdr_bars.png")
    plt.close(fig)


def fig_pareto(four):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for _, e in four:
        x, y = e.get("rtf"), e.get("sdr_avg4")
        if x is None or y is None:
            continue
        real = x < 1
        ax.scatter(x, y, s=64, color="#2E7D32" if real else "#C0392B",
                   edgecolor="white", linewidth=0.9, zorder=3)
        ax.annotate(SHORT.get(e["label"], e["label"]), (x, y),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.6)
    ax.axvline(1.0, color="#555", ls="--", lw=1.1)
    lo, hi = min((e["sdr_avg4"] for _, e in four)) * 0.9, max((e["sdr_avg4"] for _, e in four)) * 1.06
    ax.set_ylim(lo, hi)
    ax.text(1.05, lo + (hi - lo) * 0.06, "RTF = 1（实时边界）", fontsize=8.6, color="#555")
    ax.set_xscale("log")
    ax.set_xlabel("RTF（对数轴）—— 越大越慢")
    ax.set_ylabel("四轨均值 museval SDR (dB)")
    ax.set_title("精度 vs 速度：12 模型（绿 = 可实时）", fontsize=12, fontweight="bold")
    ax.grid(ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig_median_pareto.png")
    plt.close(fig)


def fig_per_stem(four):
    labels = [SHORT.get(e["label"], e["label"]) for _, e in four]
    xs = range(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for i, st in enumerate(STEMS):
        vals = []
        for _, e in four:
            d = (e.get("sdr") or {}).get(st)
            vals.append((d or {}).get("median") if d else None)
        vals = [v if v is not None else 0 for v in vals]
        ax.bar([x + (i - 1.5) * w for x in xs], vals, width=w,
               label=STEM_ZH[st], color=STEM_COLOR[st])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8.8)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_ylabel("museval SDR (dB)")
    ax.set_title("分轨精度对照（3 均衡片段中位数）", fontsize=12, fontweight="bold")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig_median_per_stem.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, four, single, norec = load()
    print(f"[plot] 4-stem 模型 {len(four)} 个 / 单目标 {len(single)} 个 / 不计均值 {len(norec)} 个 -> {OUT}")
    fig_bars(four, single, norec)
    fig_pareto(four)
    fig_per_stem(four)
    for n in ("fig_median_sdr_bars.png", "fig_median_pareto.png", "fig_median_per_stem.png"):
        p = OUT / n
        print(f"  [OK] {p}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
