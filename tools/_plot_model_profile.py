# -*- coding: utf-8 -*-
"""Render the model-profile figures (stage 1 deliverable).

Reads  : 04_reports/_shared/data/model_analysis/model_profile.csv
Writes : 04_reports/_shared/figures/model_arch/profile_params_flops.png
        04_reports/_shared/figures/model_arch/profile_pareto.png

Protocol is fixed by `_measure_model_profile.py`: 44.1 kHz stereo, 10 s input,
FLOPs = 2 x MACs, latency = median of 5 runs after 2 warm-ups.

Two honesty notes carried into the figures themselves:
  * FLOPs are a *lower bound* for models whose cost lives in STFT or fused
    attention (`FlopCounterMode` only sees dispatched conv/linear/matmul), so the
    caption says so rather than implying a like-for-like comparison.
  * oracle / rpca have no parameters; they are analytic.  They are shown in the
    bar chart for completeness and excluded from the parameter/latency bubble,
    where a zero would be meaningless.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["axes.axisbelow"] = True

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, "04_reports", "_shared", "data", "model_analysis", "model_profile.csv")
FIG = os.path.join(_ROOT, "04_reports", "_shared", "figures", "model_arch")
os.makedirs(FIG, exist_ok=True)

INK = "#1f2937"
MUTED = "#6b7280"

GROUP_STYLE = {
    "4-stem":    ("四轨分离", "#2563eb"),
    "1-target":  ("单目标", "#d97706"),
    "2-target":  ("双目标", "#7c3aed"),
    "denoise":   ("降噪", "#059669"),
    "dereverb":  ("去混响", "#0891b2"),
    "analytic":  ("解析/无参", "#9ca3af"),
}
GORDER = ["4-stem", "1-target", "2-target", "denoise", "dereverb", "analytic"]


def fnum(x, default=None):
    try:
        s = str(x).strip()
        return default if s == "" else float(s)
    except (TypeError, ValueError):
        return default


def load():
    rows = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("model_key"):
                continue
            rows.append({
                "key": r["model_key"],
                "label": (r.get("label") or r["model_key"]).strip(),
                "group": r.get("group") or "other",
                "params": fnum(r.get("params_M"), 0.0),
                "flops": fnum(r.get("macs_G"), 0.0),
                "lat": fnum(r.get("latency_s")),
                "vram": fnum(r.get("peak_vram_mb"), 0.0),
                "ntr": r.get("n_targets") or "",
            })
    return rows


def color_of(g):
    return GROUP_STYLE.get(g, ("其他", "#6b7280"))[1]


# ------------------------------------------------------------------ figure 1
def fig_params_flops(rows):
    rs = sorted(rows, key=lambda r: (r["flops"], r["params"]))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 8.4), sharey=True)
    y = np.arange(len(rs))
    cols = [color_of(r["group"]) for r in rs]
    labs = [r["label"] for r in rs]

    ax = axes[0]
    ax.barh(y, [r["params"] for r in rs], color=cols, height=0.66)
    for i, r in enumerate(rs):
        v = r["params"]
        ax.text(v + 4, i, ("%.1f M" % v) if v else "0（解析）", va="center",
                fontsize=7.8, color=INK)
    ax.set_xlim(0, max((r["params"] for r in rs), default=1) * 1.28)
    ax.set_xlabel("参数量 (M，单位：百万)")
    ax.set_title("a) 模型本体大小", fontsize=12, fontweight="bold", color=INK)

    ax = axes[1]
    vals = [max(r["flops"], 0.05) for r in rs]
    ax.barh(y, vals, color=cols, height=0.66)
    ax.set_xscale("log")
    for i, r in enumerate(rs):
        v = r["flops"]
        ax.text(max(v, 0.05) * 1.18, i, ("%.1f T" % (v / 1000)) if v >= 1000
                else ("%.0f G" % v), va="center", fontsize=7.8, color=INK)
    ax.set_xlim(0.03, max(vals) * 6)
    ax.set_xlabel("FLOPs（对数轴，10 s 立体声输入，按 2×MACs）")
    ax.set_title("b) 计算量", fontsize=12, fontweight="bold", color=INK)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labs, fontsize=8.6)
    axes[0].set_ylim(-0.7, len(rs) - 0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_STYLE[g][1]) for g in GORDER
               if any(r["group"] == g for r in rs)]
    labels = [GROUP_STYLE[g][0] for g in GORDER if any(r["group"] == g for r in rs)]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.028))
    fig.suptitle("图 模型本体画像：参数量与计算量（本机实测，22 个模型/权重）",
                 fontsize=13.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.012,
             "口径：44.1 kHz 立体声 / 10 s / FLOPs=2×MACs。"
             "[!] FlopCounterMode 只统计被调度的 conv·linear·matmul，"
             "STFT 与融合注意力未计入 → 该值对 RoFormer 类为下界。",
             ha="center", fontsize=7.6, color=MUTED)
    fig.savefig(os.path.join(FIG, "profile_params_flops.png"))
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_pareto(rows):
    rs = [r for r in rows if r["params"] > 0 and r["lat"]]
    fig, ax = plt.subplots(figsize=(11.6, 7.4))

    for r in rs:
        c = color_of(r["group"])
        s = 40 + 620 * (r["vram"] / max(x["vram"] for x in rs))
        ax.scatter(r["lat"], r["params"], s=s, color=c, alpha=0.72,
                   edgecolors="white", linewidths=1.1, zorder=3)

    # place labels with a simple repellish offset so dense clusters stay readable
    # The three Mel-RoFormer detune/denoise weights share one architecture, so
    # their (latency, params) pairs coincide to within 1%.  That coincidence is
    # real information -- it is called out in a note rather than jittered away.
    OFF = {
        "bsroformer_l12": (-12, 14, "right"), "bsroformer_l6": (0, -20, "center"),
        "mdx": (0, 18, "center"), "bsrnn_all": (14, 6, "left"),
        "bsrnn_large_all": (0, -20, "center"), "bsrnn_simo": (0, 16, "center"),
        "bsrnn": (14, 6, "left"), "bsrnn_large": (0, -21, "center"),
        "umx": (0, 17, "center"), "demucs": (14, 3, "left"),
        "convtasnet": (14, -6, "left"), "mmdenselstm": (14, -14, "left"),
        "dprnn": (16, -11, "left"),
        "mel_roformer_denoise": (0, 24, "center"),
        "mel_roformer_denoise_aggr": (54, 4, "left"),
        "mel_roformer_dereverb": (54, -14, "left"),
        "mel_roformer_dereverb_echo": (16, -18, "left"),
        "denoiser_dns48": (10, 8, "left"), "denoiser_dns64": (12, 10, "left"),
        "denoiser_master64": (12, -14, "left"),
    }
    for r in rs:
        dx, dy, ha = OFF.get(r["key"], (0, 13, "center"))
        ax.annotate(r["label"], (r["lat"], r["params"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.0, color=INK, zorder=4,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("单次前向时延（秒，对数轴；10 s 输入，本机 RTX 4070 Laptop 8 GiB）")
    ax.set_ylabel("参数量 (M)")
    ax.set_title("图 成本二维：越靠左下越「省」，气泡大小 = 峰值显存",
                 fontsize=13, fontweight="bold", color=INK)

    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_STYLE[g][1],
                          markersize=9) for g in GORDER
               if any(r["group"] == g for r in rs)]
    labels = [GROUP_STYLE[g][0] for g in GORDER if any(r["group"] == g for r in rs)]
    ax.legend(handles, labels, fontsize=9, frameon=True, framealpha=0.94,
              loc="upper right")
    fig.subplots_adjust(bottom=0.17)
    fig.text(0.5, 0.052,
             "[!] 时延含 44.1 kHz 与原生率之间的重采样（denoiser 为 16 kHz）；"
             "RPCA 是解析算法（82.5 s），已超出本图范围。",
             ha="center", fontsize=7.6, color=MUTED)
    fig.text(0.5, 0.020,
             "Mel-RoFormer 系四个权重同架构（参数 208.9~228.2 M、时延 0.97~1.02 s），"
             "因此点位几乎重合，已错开标注。",
             ha="center", fontsize=7.6, color=MUTED)
    fig.savefig(os.path.join(FIG, "profile_pareto.png"))
    plt.close(fig)


if __name__ == "__main__":
    rows = load()
    fig_params_flops(rows)
    fig_pareto(rows)
    print("rows:", len(rows))
    for f in sorted(os.listdir(FIG)):
        p = os.path.join(FIG, f)
        print("  %-34s %6d KB" % (f, os.path.getsize(p) // 1024))
