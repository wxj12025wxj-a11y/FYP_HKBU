# -*- coding: utf-8 -*-
"""
================================================================================
 10 首歌运行时长基准 -> 可视化图表集 + 对比图
--------------------------------------------------------------------------------
 输入 : outputs/comparison/bench_10songs.json
 输出 : outputs/comparison/figs_time/
          fig1_time_per_song.png        逐曲推理时长 (分组柱状)
          fig2_time_cumulative.png      累计耗时曲线
          fig3_rtf_per_song.png         逐曲 RTF (+ RTF=1 阈值线)
          fig4_time_distribution.png    时长分布箱线图
          fig5_time_vs_accuracy.png     时长-精度散点 (vocals/drums)
          fig6_compare_panels.png       四面板对比
          fig7_pareto_speed_accuracy.png 速度-精度 Pareto
--------------------------------------------------------------------------------
 运行: python tools/plot_bench_10songs.py
================================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT
COMPARISON_DIR = _paths.comparison_dir()
FIG_DIR = COMPARISON_DIR / "figs_time"
JSON = COMPARISON_DIR / "bench_10songs.json"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

C_UMX = "#2f6f9f"    # 蓝
C_DEM = "#d95f02"    # 橙
SHORT = {
    "A Classic Education - NightOwl": "A Classic Education",
    "ANiMAL - Clinic A": "ANiMAL · Clinic A",
    "ANiMAL - Easy Tiger": "ANiMAL · Easy Tiger",
    "ANiMAL - Rockshow": "ANiMAL · Rockshow",
    "Actions - Devil's Words": "Actions · Devil's Words",
    "Actions - One Minute Smile": "Actions · One Min. Smile",
    "Actions - South Of The Water": "Actions · South Of Water",
    "Aimee Norwich - Child": "Aimee Norwich · Child",
    "Alexander Ross - Goodbye Bolero": "A. Ross · Goodbye Bolero",
    "Alexander Ross - Velvet Curtain": "A. Ross · Velvet Curtain",
}
SHORT_LIST = list(SHORT.values())


def L(name):
    return SHORT.get(name, name[:22])


DEGENERATE_ABS = 30.0   # |SDR| 超过此值视为退化样本(参考轨过稀疏/静音), 聚合时剔除


def stat_sdr(dd, key):
    """返回 (median, mean, n) —— 剔除退化样本后的稳健统计"""
    v = [r[key] for r in dd["per_song"]
         if r.get(key) is not None and abs(r[key]) < DEGENERATE_ABS]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.median(v)), float(np.mean(v)), len(v)


def load():
    d = json.load(open(JSON, encoding="utf-8"))
    return d


def bar_labels(ax, bars, fmt="{:.2f}", dy=0.02, fs=8):
    for b in bars:
        h = b.get_height()
        if h != h:  # nan
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs)


# ---------------------------------------------------------------- #
def fig1_per_song(d):
    umx = d["openunmix"]["per_song"]
    dem = d["demucs"]["per_song"]
    x = np.arange(len(umx))
    w = 0.38
    tu = [r["inference_seconds"] for r in umx]
    td = [r["inference_seconds"] for r in dem]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    b1 = ax.bar(x - w / 2, tu, w, label="Open-Unmix (umxhq)", color=C_UMX)
    b2 = ax.bar(x + w / 2, td, w, label="Demucs (htdemucs)", color=C_DEM)
    bar_labels(ax, b1, "{:.2f}", fs=7.5)
    bar_labels(ax, b2, "{:.2f}", fs=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([L(r["song"]) for r in umx], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("推理墙钟时间 (秒 / 30s 音频)", fontsize=11)
    ax.set_title("图1  逐曲运行时长对比 — 同一组 10 首歌 (各 30s, CPU 单进程)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "fig1_time_per_song.png")
    plt.close(fig)


def fig2_cumulative(d):
    umx = d["openunmix"]["per_song"]
    dem = d["demucs"]["per_song"]
    n = np.arange(1, len(umx) + 1)
    cu = np.cumsum([r["inference_seconds"] for r in umx])
    cd = np.cumsum([r["inference_seconds"] for r in dem])

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(n, cu, "-o", color=C_UMX, lw=2.4, ms=7, label="Open-Unmix (umxhq)")
    ax.plot(n, cd, "-s", color=C_DEM, lw=2.4, ms=7, label="Demucs (htdemucs)")
    ax.fill_between(n, cu, cd, color="gray", alpha=0.12)
    for i in range(len(n)):
        ax.annotate(f"{cu[i]:.1f}", (n[i], cu[i]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.5, color=C_UMX)
        ax.annotate(f"{cd[i]:.1f}", (n[i], cd[i]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=C_DEM)
    ax.set_xticks(n)
    ax.set_xticklabels([f"{i}\n{L(umx[i-1]['song'])[:14]}" for i in range(1, len(n) + 1)],
                       fontsize=8)
    ax.set_xlabel("歌曲序号 (累积)", fontsize=11)
    ax.set_ylabel("累计推理时间 (秒)", fontsize=11)
    ax.set_title("图2  累计运行时长曲线 — 跑完 10 首歌各需多久", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig2_time_cumulative.png")
    plt.close(fig)


def fig3_rtf(d):
    umx = d["openunmix"]["per_song"]
    dem = d["demucs"]["per_song"]
    x = np.arange(len(umx))
    w = 0.38
    ru = [r["rtf"] for r in umx]
    rd = [r["rtf"] for r in dem]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(x - w / 2, ru, w, label="Open-Unmix (umxhq)", color=C_UMX)
    ax.bar(x + w / 2, rd, w, label="Demucs (htdemucs)", color=C_DEM)
    ax.axhline(1.0, color="red", ls="--", lw=2, label="RTF = 1 实时阈值")
    ax.set_xticks(x)
    ax.set_xticklabels([L(r["song"]) for r in umx], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("RTF  (推理时间 / 音频时长)", fontsize=11)
    ax.set_title("图3  逐曲实时因子 RTF — 低于红线即可实时 (越小越快)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "fig3_rtf_per_song.png")
    plt.close(fig)


def fig4_dist(d):
    tu = [r["inference_seconds"] for r in d["openunmix"]["per_song"]]
    td = [r["inference_seconds"] for r in d["demucs"]["per_song"]]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bp = ax.boxplot([tu, td], tick_labels=["Open-Unmix\n(umxhq)", "Demucs\n(htdemucs)"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=7))
    for box, c in zip(bp["boxes"], (C_UMX, C_DEM)):
        box.set_facecolor(c)
        box.set_alpha(0.65)
    for i, data in enumerate([tu, td], 1):
        ax.scatter(np.full(len(data), i) + np.random.uniform(-0.06, 0.06, len(data)),
                   data, color="black", s=28, zorder=3, alpha=0.7)
    ax.set_ylabel("推理时间 (秒 / 30s)", fontsize=11)
    ax.set_title("图4  运行时长分布 — 10 首歌的离散程度", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "fig4_time_distribution.png")
    plt.close(fig)


def fig5_time_vs_acc(d):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, tgt, key in ((axes[0], "人声 (vocals)", "sdr_vocals"),
                         (axes[1], "鼓声 (drums)", "sdr_drums")):
        for model, c, mk, lab in (("openunmix", C_UMX, "o", "Open-Unmix"),
                                  ("demucs", C_DEM, "s", "Demucs")):
            ps = d[model]["per_song"]
            xs = [r["inference_seconds"] for r in ps if r.get(key) is not None]
            ys = [r[key] for r in ps if r.get(key) is not None]
            ax.scatter(xs, ys, color=c, marker=mk, s=70, alpha=0.8, label=lab)
        ax.set_xlabel("推理时间 (秒 / 30s)", fontsize=11)
        ax.set_ylabel(f"{tgt}  SDR (dB)", fontsize=11)
        ax.set_title(f"{tgt}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("图5  时间 — 精度散点 (每点一首歌)：越靠左上越「又快又准」",
                 fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "fig5_time_vs_accuracy.png")
    plt.close(fig)


def fig6_panels(d):
    u, m = d["openunmix"], d["demucs"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 总时长
    ax = axes[0, 0]
    vals = [u["inference_total_seconds"], m["inference_total_seconds"]]
    bars = ax.bar(["Open-Unmix", "Demucs"], vals, color=[C_UMX, C_DEM], width=0.55)
    bar_labels(ax, bars, "{:.1f}s", fs=10)
    ax.set_title("(a) 10 首歌推理总时长", fontsize=12, fontweight="bold")
    ax.set_ylabel("秒", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # (b) 平均每首
    ax = axes[0, 1]
    vals = [u["inference_mean_seconds"], m["inference_mean_seconds"]]
    bars = ax.bar(["Open-Unmix", "Demucs"], vals, color=[C_UMX, C_DEM], width=0.55)
    bar_labels(ax, bars, "{:.2f}s", fs=10)
    ax.set_title("(b) 平均每首耗时", fontsize=12, fontweight="bold")
    ax.set_ylabel("秒 / 首", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # (c) 中位 RTF
    ax = axes[1, 0]
    vals = [u["rtf_median"], m["rtf_median"]]
    bars = ax.bar(["Open-Unmix", "Demucs"], vals, color=[C_UMX, C_DEM], width=0.55)
    bar_labels(ax, bars, "{:.3f}", fs=10)
    ax.axhline(1.0, color="red", ls="--", lw=1.8, label="RTF=1")
    ax.set_title("(c) 中位 RTF (越小越快)", fontsize=12, fontweight="bold")
    ax.set_ylabel("RTF", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (d) 中位人声 SDR (剔除退化样本)
    ax = axes[1, 1]
    mv, _, nv = stat_sdr(u, "sdr_vocals")
    md, _, nd = stat_sdr(m, "sdr_vocals")
    bars = ax.bar(["Open-Unmix", "Demucs"], [mv, md], color=[C_UMX, C_DEM], width=0.55)
    bar_labels(ax, bars, "{:.2f} dB", fs=10)
    ax.set_title(f"(d) 中位人声 SDR (剔除退化样本, n={nv}/{nd})", fontsize=12, fontweight="bold")
    ax.set_ylabel("dB", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    ratio_t = m["inference_total_seconds"] / u["inference_total_seconds"]
    fig.suptitle(f"图6  Open-Unmix vs Demucs 四维对比  (Demucs 耗时约为 Open-Unmix 的 {ratio_t:.1f}×)",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(FIG_DIR / "fig6_compare_panels.png")
    plt.close(fig)


def fig7_pareto(d):
    u, m = d["openunmix"], d["demucs"]
    fig, ax = plt.subplots(figsize=(11, 7.5))

    mv, _, _ = stat_sdr(u, "sdr_vocals")
    md, _, _ = stat_sdr(m, "sdr_vocals")
    pts = {
        "Open-Unmix": (u["inference_mean_seconds"], mv, C_UMX, "o"),
        "Demucs": (m["inference_mean_seconds"], md, C_DEM, "s"),
    }
    for name, (x, y, c, mk) in pts.items():
        ax.scatter(x, y, s=420, color=c, marker=mk, alpha=0.85, edgecolor="black",
                   zorder=3, label=name)
        ax.annotate(f"{name}\n({x:.2f}s, {y:.2f}dB)", (x, y),
                    textcoords="offset points", xytext=(14, 8), fontsize=10)
    ax.set_xlabel("平均每首推理时间 (秒, 越左越快)", fontsize=11)
    ax.set_ylabel("中位人声 SDR (dB, 越上越准)", fontsize=11)
    ax.set_title("图7  速度-精度 Pareto：理想解在左上角",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    fig.savefig(FIG_DIR / "fig7_pareto_speed_accuracy.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = load()
    fig1_per_song(d); print("  [OK] fig1_time_per_song.png")
    fig2_cumulative(d); print("  [OK] fig2_time_cumulative.png")
    fig3_rtf(d); print("  [OK] fig3_rtf_per_song.png")
    fig4_dist(d); print("  [OK] fig4_time_distribution.png")
    fig5_time_vs_acc(d); print("  [OK] fig5_time_vs_accuracy.png")
    fig6_panels(d); print("  [OK] fig6_compare_panels.png")
    fig7_pareto(d); print("  [OK] fig7_pareto_speed_accuracy.png")
    print(f"\n全部图表已保存 -> {FIG_DIR}")


if __name__ == "__main__":
    main()
