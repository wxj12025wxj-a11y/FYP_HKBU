#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_make_plan_figures.py —— FYP 计划书的甘特图 + 依赖关系图。

输出（04_reports/_shared/figures/plan/）：
    gantt_fyp.png        5 周排期甘特图（含 5 个里程碑与 3 个夜间 GPU 窗口）
    wbs_dependency.png   阶段依赖关系图（含关键路径）

用法：
    python tools/_make_plan_figures.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths  # noqa: E402

_paths.setup_env()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT_DIR = _paths.SHR_FIGURES / "plan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_M1 = "#2f6fb5"   # 板1
C_M2 = "#d97a1a"   # 板2
C_BASE = "#7a7a7a"  # 基建
C_TAIL = "#3d9a5f"  # 收尾
C_CRIT = "#c0392b"  # GPU 夜间窗口

TASKS = [
    # (阶段, 名称, 起, 止, 颜色, 分组)
    ("阶段0 基建", "下载 / 接入 / SAC 探测", date(2026, 9, 22), date(2026, 9, 24), C_BASE, "base"),
    ("板1 模型分析", "阶段1 模型本体画像", date(2026, 9, 24), date(2026, 9, 26), C_M1, "b1"),
    ("板1 模型分析", "阶段2 架构原理图 ×14", date(2026, 9, 25), date(2026, 9, 28), C_M1, "b1"),
    ("板1 模型分析", "阶段3 频谱面板 + 精度表", date(2026, 9, 28), date(2026, 9, 30), C_M1, "b1"),
    ("板1 模型分析", "阶段4 混响 / 去混响 / 去回声", date(2026, 9, 30), date(2026, 10, 6), C_M1, "b1"),
    ("板2 降噪", "阶段5 降噪双口径评测", date(2026, 10, 6), date(2026, 10, 13), C_M2, "b2"),
    ("板2 降噪", "阶段6 级联实验", date(2026, 10, 13), date(2026, 10, 18), C_M2, "b2"),
    ("交付收尾", "阶段7 看板 + 数据摘要", date(2026, 10, 18), date(2026, 10, 21), C_TAIL, "tail"),
    ("交付收尾", "阶段8 论文图表 + 答辩", date(2026, 10, 21), date(2026, 10, 27), C_TAIL, "tail"),
]

GPU_NIGHTS = [
    (date(2026, 10, 1), "混响 1.5 h"),
    (date(2026, 10, 8), "降噪 2 h"),
    (date(2026, 10, 14), "级联 4 h"),
]

MILESTONES = [
    (date(2026, 9, 29), "M1", "画像表 + 架构图"),
    (date(2026, 10, 6), "M2", "板1 交付"),
    (date(2026, 10, 12), "M3", "降噪表定稿"),
    (date(2026, 10, 20), "M4", "板2 交付"),
    (date(2026, 10, 27), "M5", "提交"),
]


def make_gantt() -> Path:
    fig, ax = plt.subplots(figsize=(13.2, 6.6), dpi=150)

    # 分组底色
    groups = ["base", "b1", "b2", "tail"]
    colors_bg = {"base": "#f2f2f2", "b1": "#eef4fb", "b2": "#fdf3e7", "tail": "#eef8f1"}
    idx_of = {}
    y = 0
    for t in reversed(TASKS):
        idx_of[t[1]] = y
        y += 1
    ymax = y

    ypos = 0
    for i, t in enumerate(TASKS):
        pass

    # 用从上到下的顺序：TASKS[0] 在最上
    order = list(TASKS)
    n = len(order)
    rows = []
    for i, (grp, name, s, e, col, g) in enumerate(order):
        ypos = n - 1 - i
        rows.append((ypos, grp, name, s, e, col, g))

    # 组背景
    cur_g = None
    start_y = None
    for ypos, grp, name, s, e, col, g in rows:
        if g != cur_g:
            if cur_g is not None:
                ax.axhspan(start_y - 0.5, prev_y + 0.5, color=colors_bg[cur_g], zorder=0)
            cur_g = g
            start_y = ypos
        prev_y = ypos
    ax.axhspan(start_y - 0.5, prev_y + 0.5, color=colors_bg[cur_g], zorder=0)

    for ypos, grp, name, s, e, col, g in rows:
        d0 = mdates.date2num(s)
        d1 = mdates.date2num(e)
        ax.broken_barh([(d0, d1 - d0)], (ypos - 0.32, 0.64),
                       facecolors=col, edgecolors="white", linewidth=0.8, zorder=3)

    # 夜间 GPU 窗口
    for d, label in GPU_NIGHTS:
        x = mdates.date2num(d)
        ax.axvline(x, color=C_CRIT, linestyle=":", linewidth=1.4, zorder=2)
        ax.text(x, len(order) - 0.35, "GPU " + label, rotation=90, va="top", ha="right",
                fontsize=7.5, color=C_CRIT, zorder=6)

    # 里程碑
    for d, tag, desc in MILESTONES:
        x = mdates.date2num(d)
        ax.plot([x], [-0.95], marker="D", markersize=8, color="#111111", zorder=6)
        ax.text(x, -1.45, "%s\n%s" % (tag, desc), ha="center", va="top",
                fontsize=7.6, color="#111111", zorder=6)

    # 起止日期：只在每个阶段组首行与末行标注
    x0 = mdates.date2num(date(2026, 9, 22))
    x1 = mdates.date2num(date(2026, 10, 28))
    ax.set_xlim(x0 - 1.2, x1 + 0.6)
    ax.set_ylim(-2.35, len(order) - 0.2)

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[2] for r in rows], fontsize=9)
    ax.set_xticks([mdates.date2num(date(2026, 9, 22) + __import__("datetime").timedelta(days=i * 7))
                   for i in range(6)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.set_title("FYP 排期甘特图（2026-09-22 → 2026-10-27，5 周）",
                 fontsize=13.5, pad=14, weight="bold")
    ax.set_xlabel("日期（月-日）", fontsize=10)

    legend = [
        mpatches.Patch(color=C_BASE, label="阶段0 基建"),
        mpatches.Patch(color=C_M1, label="板1 源分离"),
        mpatches.Patch(color=C_M2, label="板2 降噪"),
        mpatches.Patch(color=C_TAIL, label="交付收尾"),
        mpatches.Patch(color="#111111", label="里程碑 M1–M5"),
        mpatches.Patch(color=C_CRIT, label="夜间 GPU 窗口"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8.6, framealpha=0.95, ncol=3)

    fig.tight_layout()
    out = OUT_DIR / "gantt_fyp.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


NODES = {
    "T0":  (0.5, 3.6, "阶段0 基建\nValentini/denoiser 接入\n（唯一联网前置）", C_BASE),
    "T1":  (0.5, 2.55, "阶段1 模型画像\n参数/FLOPs/显存/时延", C_M1),
    "T2":  (0.5, 1.5, "阶段2 架构原理图 ×14", C_M1),
    "T3":  (2.0, 2.55, "阶段3 频谱面板\n+ 精度表重绘", C_M1),
    "T4":  (3.5, 2.55, "阶段4 混响 / 去混响", C_M1),
    "T5":  (5.0, 3.6, "阶段5 降噪双口径评测", C_M2),
    "T6":  (5.0, 2.55, "阶段6 级联实验", C_M2),
    "T7":  (3.5, 1.35, "阶段7 看板 + 数据摘要", C_TAIL),
    "T8":  (1.6, 0.55, "阶段8 论文图表 + 答辩", C_TAIL),
}

MILES = {
    "M1": (1.6, 3.0, "M1 09-29"),
    "M2": (4.3, 3.6, "M2 10-06"),
    "M3": (6.1, 3.6, "M3 10-12"),
    "M4": (6.1, 2.55, "M4 10-20"),
    "M5": (0.9, -0.15, "M5 10-27 提交"),
}

EDGES = [
    ("T0", "T1"), ("T0", "T5"), ("T1", "T2"), ("T1", "T3"),
    ("T2", "M1"), ("T3", "M1"), ("T3", "T4"), ("T4", "M2"),
    ("T5", "M3"), ("T5", "T6"), ("T6", "M4"), ("M2", "T7"),
    ("M4", "T7"), ("T7", "T8"), ("T8", "M5"),
]

BOX_W, BOX_H = 1.45, 0.62


def _center(kind, key):
    src = NODES if kind == "n" else MILES
    x, y, txt = src[key][0], src[key][1], src[key][2]
    return x, y


def make_dependency() -> Path:
    fig, ax = plt.subplots(figsize=(12.6, 6.4), dpi=150)
    ax.set_xlim(-0.6, 7.2)
    ax.set_ylim(-0.75, 4.35)
    ax.axis("off")

    def draw_box(x, y, txt, color, w=BOX_W, h=BOX_H, fs=7.8, bold=False, dashed=False):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.045,rounding_size=0.09",
            linewidth=1.3, edgecolor=color, facecolor="white",
            linestyle="--" if dashed else "-", zorder=4))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                color="#111111", zorder=5, weight="bold" if bold else "normal",
                linespacing=1.35)

    def draw_arrow(a, b, color="#555555", dashed=False):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11,
            linewidth=1.25, color=color, zorder=3,
            linestyle="--" if dashed else "-",
            shrinkA=2, shrinkB=5,
            connectionstyle="arc3,rad=0.0"))

    # 节点
    for k, (x, y, txt, col) in NODES.items():
        draw_box(x, y, txt, col)
    for k, (x, y, txt) in MILES.items():
        draw_box(x, y, txt, "#111111", w=1.15, h=0.46, fs=8.2, bold=True)

    for a, b in EDGES:
        ka = "n" if a in NODES else "m"
        kb = "n" if b in NODES else "m"
        xa, ya = _center(ka, a)
        xb, yb = _center(kb, b)
        dx = xb - xa
        dy = yb - ya
        wa = (MILES[a][0] if ka == "m" else NODES[a][0])
        # 让箭头从盒边缘出发
        sx = xa + (0.62 if dx > 0 else (-0.62 if dx < 0 else 0))
        sy = ya + (0.32 if dy > 0 else (-0.32 if dy < 0 else 0))
        ex = xb - (0.62 if dx > 0 else (-0.62 if dx < 0 else 0))
        ey = yb - (0.32 if dy > 0 else (-0.32 if dy < 0 else 0))
        if dx != 0 and dy != 0:
            sx, sy = xa + (0.72 * (1 if dx > 0 else -1)), ya
            ex, ey = xb - (0.60 * (1 if dx > 0 else -1)), yb
        draw_arrow((sx, sy), (ex, ey))

    # 关键路径高亮
    ax.text(0.02, 4.16, "关键路径：阶段0 → 阶段5 → 阶段6 → 阶段7 → 阶段8 → M5",
            fontsize=9.6, color=C_CRIT, weight="bold")
    ax.text(0.02, -0.60,
            "本地既有资产（564 组合音频 / DNS 子集 / RIR 60248 条）为 阶段3 / 阶段4 / 阶段6 的并行输入",
            fontsize=8.6, color="#555555")
    ax.set_title("FYP 阶段依赖关系图", fontsize=13.5, pad=12, weight="bold")

    fig.tight_layout()
    out = OUT_DIR / "wbs_dependency.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    g = make_gantt()
    d = make_dependency()
    for p in (g, d):
        print("OK  %s  (%.1f KB)" % (p, p.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
