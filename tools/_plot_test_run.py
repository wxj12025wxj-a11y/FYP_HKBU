# -*- coding: utf-8 -*-
"""Render the multi-dimensional figures for the 50-song MUSDB18 test sweep.

Reads  : 04_reports/separation/data/comparison/test_run_analysis.json, test_sweep_ledger.json,
         test_run_per_song.csv, musdb_test_manifest.json
Writes : 04_reports/separation/figures/test_run/*.png
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["axes.axisbelow"] = True

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMP = os.path.join(_ROOT, "04_reports", "separation", "data", "comparison")
FIG = os.path.join(_ROOT, "04_reports", "separation", "figures", "test_run")
os.makedirs(FIG, exist_ok=True)

TIMEOUT = 180.0

# manual label offsets for figure 5 (points / dx, dy, ha)
OFF = {
    "bsroformer_l12": (0, 15, "center"),
    "bsrnn": (-4, 15, "center"),
    "bsrnn_large": (-46, 5, "center"),
    "bsroformer_l6": (30, 17, "left"),
    "mdx": (-6, 15, "center"),
    "oracle": (-18, 6, "right"),
    "demucs": (18, -16, "left"),
    "bsrnn_simo": (0, 15, "center"),
    "umx": (-10, -19, "center"),
    "mmdenselstm": (12, -12, "left"),
    "convtasnet": (0, 15, "center"),
    "dprnn": (0, 15, "center"),
}

C_FULL = "#2563eb"     # 4-track models
C_VOC = "#d97706"      # vocals-only
C_VO = "#7c3aed"       # vocals+other
C_BAD = "#dc2626"
INK = "#1f2937"
MUTED = "#6b7280"


def jload(name):
    with open(os.path.join(CMP, name), encoding="utf-8") as f:
        return json.load(f)


AN = jload("test_run_analysis.json")
LED = jload("test_sweep_ledger.json")
MODELS = AN["models"]
for _m, _r in MODELS.items():                      # JSON turns int keys into str
    _r["n_tracks"] = {int(k): v for k, v in _r["n_tracks"].items()}
SCOPE = ["oracle", "demucs", "bsrnn_simo", "umx", "mdx", "convtasnet", "mmdenselstm",
         "bsroformer_l12", "bsroformer_l6", "bsrnn", "bsrnn_large", "dprnn"]
SONGS = list(LED["attempts"]["oracle"].keys())          # sweep order
SIDX = {s: i + 1 for i, s in enumerate(SONGS)}

# extra per-song detail (song duration, per-model wall)
ROWS = []
with open(os.path.join(CMP, "test_run_per_song.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        for k in ("song_dur_s", "wall_s", "infer_s", "overhead_s", "rtf", "sdr_mean",
                  "sdr_vocals", "sdr_drums", "sdr_bass", "sdr_other", "out_mb", "n_tracks"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
        r["idx"] = SIDX.get(r["song"])
        ROWS.append(r)


def wall_series(m):
    """Per-song wall clock, using the reconciled PASS/FAIL status (see _analyze_test_run)."""
    return [r["wall_s"] for r in ROWS if r["model_key"] == m and r["status"] == "PASS"
            and r["wall_s"] is not None]


def model_color(m):
    n = MODELS[m]["n_tracks"]
    if not n:
        return C_BAD
    return {4: C_FULL, 1: C_VOC}.get(max(n), C_VO)


def kind(m):
    n = MODELS[m]["n_tracks"]
    if not n:
        return "未产出", C_BAD
    if 4 in n:
        return "四轨全分离", C_FULL
    if m == "bsroformer_l6":
        return "人声+伴奏(1轨)", C_VO
    return "人声单轨", C_VOC


def barh_labels(ax, bars, fmt="{:.0f}", dx=0.0, fs=8, color=INK):
    for b in bars:
        w = b.get_width()
        ax.text(w + dx, b.get_y() + b.get_height() / 2, fmt.format(w),
                va="center", ha="left", fontsize=fs, color=color)


# =====================================================================
# 1. wall-clock time range per model
# =====================================================================
def fig_time_range():
    ms = [m for m in SCOPE if MODELS[m]["wall_s"]["median"]]
    ms.sort(key=lambda m: MODELS[m]["wall_s"]["median"])
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for i, m in enumerate(ms):
        w = np.array(wall_series(m))
        r = MODELS[m]
        c = kind(m)[1]
        # P25-P75 band
        ax.plot([np.percentile(w, 25), np.percentile(w, 75)], [i, i],
                lw=9, color=c, alpha=0.30, solid_capstyle="round", zorder=2)
        # min-max whisker
        ax.plot([w.min(), w.max()], [i, i], lw=1.6, color=c, alpha=0.55, zorder=1)
        ax.plot(w.min(), i, "|", ms=9, color=c)
        ax.plot(w.max(), i, "|", ms=9, color=c)
        ax.plot(np.median(w), i, "o", ms=8, color=c, zorder=4, mec="white", mew=1.2)
        ax.plot(w.mean(), i, "D", ms=4.5, color="white", mec=c, mew=1.3, zorder=5)
        ax.text(np.median(w) + 2.5, i + 0.30, f"{np.median(w):.0f}s", fontsize=8.5,
                color=INK, fontweight="bold")
        ax.text(w.max() + 3, i, f"n={len(w)}", fontsize=7.5, color=MUTED, va="center")
    ax.axvline(TIMEOUT, color=C_BAD, lw=1.6, ls="--", zorder=0)
    ax.text(TIMEOUT - 2, len(ms) - 0.35, "180s 超时上限", color=C_BAD, fontsize=8.5,
            ha="right", fontweight="bold")
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([MODELS[m]["label"] for m in ms], fontsize=9)
    ax.set_xlabel("单首歌曲 墙钟处理耗时 (秒)  —  点=中位数  菱形=均值  粗带=P25~P75  细线=min~max")
    ax.set_title("图1  每首歌的处理耗时分布（按中位数排序，50 首 MUSDB18 test）",
                 fontsize=12, fontweight="bold", color=INK)
    ax.set_xlim(0, max(200, max(np.percentile(np.array(wall_series(m)), 75) for m in ms) * 1.18))
    handles = [Line2D([], [], marker="o", ls="", color=C_FULL, label="四轨全分离 (4 stems)"),
               Line2D([], [], marker="o", ls="", color=C_VOC, label="人声单轨"),
               Line2D([], [], marker="o", ls="", color=C_VO, label="人声+伴奏")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.savefig(os.path.join(FIG, "01_time_range.png"))
    plt.close(fig)


# =====================================================================
# 2. inference vs fixed overhead (stacked)
# =====================================================================
def fig_inference_vs_overhead():
    ms = [m for m in SCOPE if MODELS[m]["infer_s"]["total"]]
    ms.sort(key=lambda m: MODELS[m]["infer_s"]["total"])
    fig, ax = plt.subplots(figsize=(10.5, 6))
    y = np.arange(len(ms))
    inf = np.array([MODELS[m]["infer_s"]["mean"] for m in ms])
    ovh = np.array([MODELS[m]["overhead_s_total"] / MODELS[m]["wall_s"]["n"] for m in ms])
    b1 = ax.barh(y, inf, color="#1d4ed8", label="推理本体 (infer_s)", height=0.62)
    b2 = ax.barh(y, ovh, left=inf, color="#f59e0b", label="固定开销 (解释器+导入+评测+读写)",
                 height=0.62)
    for i, m in enumerate(ms):
        tot = inf[i] + ovh[i]
        sh = 100.0 * inf[i] / tot
        ax.text(inf[i] + ovh[i] / 2, i, f"{ovh[i]:.0f}s", ha="center", va="center",
                fontsize=8, color="#7c2d12", fontweight="bold")
        ax.text(tot + 1.6, i, f"合计 {tot:.0f}s/首   推理占 {sh:.0f}%", va="center",
                fontsize=8.2, color=INK)
        if inf[i] > 6:
            ax.text(inf[i] / 2, i, f"{inf[i]:.0f}s", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{MODELS[m]['label']}  (n={MODELS[m]['wall_s']['n']})" for m in ms],
                       fontsize=9)
    ax.set_xlabel("平均单首耗时拆分 (秒) —— 推理 + 固定开销 = 每首实付时间")
    ax.set_xlim(0, (inf + ovh).max() * 1.30)
    ax.set_title("图2  时间都花在哪：推理本体 vs 固定开销（每首歌都要重付一次）",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.savefig(os.path.join(FIG, "02_inference_vs_overhead.png"))
    plt.close(fig)


# =====================================================================
# 3. does song length drive cost?  (scatter + throughput)
# =====================================================================
def fig_time_vs_songdur():
    figs, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                              gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    xs, ys = [], []
    for m in SCOPE:
        pts = [(r["song_dur_s"], r["wall_s"]) for r in ROWS
               if r["model_key"] == m and r["status"] == "PASS"
               and r["song_dur_s"] and r["wall_s"]]
        if not pts:
            continue
        x = np.array([p[0] for p in pts]); yv = np.array([p[1] for p in pts])
        if m in ("oracle", "umx", "mdx", "demucs", "convtasnet", "mmdenselstm",
                 "bsroformer_l12", "bsroformer_l6", "bsrnn", "bsrnn_large", "dprnn",
                 "bsrnn_simo"):
            ax.scatter(x, yv, s=16, alpha=0.5, color=kind(m)[1], label=MODELS[m]["label"],
                       edgecolors="none")
        xs += list(x); ys += list(yv)
    xs = np.array(xs); ys = np.array(ys)
    k, b = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min(), xs.max(), 20)
    ax.plot(xr, k * xr + b, color=INK, lw=2, ls="--",
            label=f"全体拟合  y={k:.3f}x+{b:.0f}")
    r = np.corrcoef(xs, ys)[0, 1]
    ax.text(0.02, 0.97, f"Pearson r = {r:.3f}\nn = {len(xs)} 组 (model,song)",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK,
            bbox=dict(fc="white", ec="#e5e7eb", alpha=0.9))
    ax.set_xlabel("歌曲时长 (秒)")
    ax.set_ylabel("墙钟处理耗时 (秒)")
    ax.set_title("图4-a  歌曲越长耗时越长？", fontsize=11.5, fontweight="bold", color=INK)
    ax.legend(fontsize=6.8, ncol=2, loc="lower right", framealpha=0.9)

    ax = axes[1]
    ms = [m for m in SCOPE if MODELS[m]["wall_s"]["median"]]
    tp = []
    for m in ms:
        v = [r["wall_s"] / r["song_dur_s"] for r in ROWS
             if r["model_key"] == m and r["status"] == "PASS"
             and r["song_dur_s"] and r["wall_s"]]
        tp.append((m, np.median(v) if v else np.nan))
    tp.sort(key=lambda t: t[1])
    bars = ax.barh([t[0] for t in tp], [t[1] for t in tp],
                   color=[kind(t[0])[1] for t in tp], height=0.62)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1.2)
    for b_, (m, v) in zip(bars, tp):
        ax.text(v + 0.02, b_.get_y() + b_.get_height() / 2,
                f"{v:.2f}×  ({60 / v:.0f} 首/小时)" if v > 0 else "",
                va="center", fontsize=8.1, color=INK)
    ax.set_yticks(range(len(tp)))
    ax.set_yticklabels([MODELS[m]["label"] for m, _ in tp], fontsize=8.5)
    ax.set_xlabel("墙钟实时倍率 = 处理耗时 ÷ 歌曲时长（越小越快）")
    ax.set_xlim(0, max(t[1] for t in tp) * 1.42)
    ax.set_title("图4-b  吞吐：处理 1 秒音频要花几秒", fontsize=11.5, fontweight="bold", color=INK)
    figs.suptitle("歌曲时长与耗时的关系", fontsize=13, fontweight="bold", color=INK, y=1.01)
    figs.savefig(os.path.join(FIG, "04_time_vs_songdur.png"))
    plt.close(figs)


# =====================================================================
# 4. drift over the sweep
# =====================================================================
def fig_sequence_drift():
    """Drift must be measured per unit of audio, otherwise short early songs fake a trend."""
    fig, ax = plt.subplots(figsize=(12, 5.6))
    allx, ally = [], []
    for m in SCOPE:
        seq = []
        for s, v in LED["attempts"][m].items():
            if v.get("status") != "PASS" or not SIDX.get(s):
                continue
            d = [r["song_dur_s"] for r in ROWS if r["model_key"] == m and r["song"] == s]
            if v.get("seconds") and d and d[0]:
                seq.append((SIDX[s], v["seconds"] / d[0]))
        if len(seq) < 5:
            continue
        seq.sort()
        idx = np.array([p[0] for p in seq]); u = np.array([p[1] for p in seq])
        med = np.median(u)
        ax.plot(idx, u / med, lw=1.1, alpha=0.55, color=kind(m)[1], label=MODELS[m]["label"])
        allx += list(idx); ally += list(u / med)
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    k, b = np.polyfit(np.array(allx), np.array(ally), 1)
    ax.plot([1, max(allx)], [k + b, k * max(allx) + b], color=C_BAD, lw=2.4,
            label="全体趋势")
    ax.set_xlabel("扫描顺序（第几首）")
    ax.set_ylabel("单位音频耗时 ÷ 该模型自身中位数\n(已消除歌曲长短影响；=1 表示等于自身中位数)")
    ax.set_title("图5  耗时是否随扫描进程漂移（>1 表示同样长度的音频越跑越慢）",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.9)
    ax.text(0.985, 0.04, f"趋势斜率 {k:+.5f} /首  ->  50 首累计 {k * 50:+.1%}\n"
                         f"(n={len(allx)} 组 model×song；斜率≈0 说明无系统性劣化)",
            transform=ax.transAxes, ha="right", fontsize=9.2, color=C_BAD,
            bbox=dict(fc="white", ec="#e5e7eb", alpha=0.92))
    fig.savefig(os.path.join(FIG, "05_sequence_drift.png"))
    plt.close(fig)


# =====================================================================
# 5. speed vs accuracy vs disk
# =====================================================================
def fig_speed_accuracy():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    pts = []
    for m in SCOPE:
        r = MODELS[m]
        if not (r["wall_s"]["median"] and r["sdr_mean"]["median"]):
            continue
        pts.append((m, r["wall_s"]["median"], r["sdr_mean"]["median"], r["out_mb_total"]))
    for m, x, y, mb in pts:
        ax.scatter(x, y, s=max(40, mb * 0.55), alpha=0.55, color=kind(m)[1],
                   edgecolors=kind(m)[1], linewidths=1.4, zorder=3)
        dx, dy, ha = OFF.get(m, (0, 14, "center"))
        ax.annotate(MODELS[m]["label"], (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.4, color=INK, zorder=4,
                    bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.9))
    # pareto front: minimal time, maximal sdr
    srt = sorted(pts, key=lambda p: p[1])
    best, front = -1e9, []
    for p in srt:
        if p[2] > best:
            front.append(p); best = p[2]
    if len(front) > 1:
        ax.plot([p[1] for p in front], [p[2] for p in front], color=C_BAD, lw=1.6,
                ls="--", alpha=0.8, zorder=2, label="Pareto 前沿（快且准）")
    ax.set_xlabel("单首中位处理耗时 (秒) —— 越左越快")
    ax.set_ylabel("中位平均 SDR (dB) —— 越上越准\n(单轨模型只有 1 条轨参与均值)")
    ax.set_title("图6  速度 × 质量 × 产物体积（气泡=磁盘占用；单轨模型 SDR 不可与四轨模型直接比较）",
                 fontsize=11.5, fontweight="bold", color=INK)
    ax.axvspan(0, 70, color="#22c55e", alpha=0.06)
    ax.text(2, ax.get_ylim()[1] * 0.995, "  ≤70s/首 区间", fontsize=8, color="#15803d",
            va="top")
    handles = [Line2D([], [], marker="o", ls="", color=C_FULL, label="四轨全分离"),
               Line2D([], [], marker="o", ls="", color=C_VOC, label="人声单轨"),
               Line2D([], [], marker="o", ls="", color=C_VO, label="人声+伴奏"),
               Line2D([], [], ls="--", color=C_BAD, label="Pareto 前沿")]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", framealpha=0.9)
    fig.savefig(os.path.join(FIG, "06_speed_accuracy.png"))
    plt.close(fig)


# =====================================================================
# 6. output inventory
# =====================================================================
def fig_output_inventory():
    ms = [m for m in SCOPE if MODELS[m]["out_mb_total"]]
    ms.sort(key=lambda m: MODELS[m]["out_mb_total"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    y = np.arange(len(ms))
    mb = np.array([MODELS[m]["out_mb_total"] for m in ms])
    bars = ax.barh(y, mb, color=[kind(m)[1] for m in ms], height=0.62)
    for b_, m in zip(bars, ms):
        ax.text(b_.get_width() + 40, b_.get_y() + b_.get_height() / 2,
                f"{MODELS[m]['out_mb_total']:.0f} MB", va="center", fontsize=8.4, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([MODELS[m]["label"] for m in ms], fontsize=8.8)
    ax.set_xlabel("输出音频总体积 (MB, FLAC 无损)")
    ax.set_xlim(0, mb.max() * 1.28)
    ax.set_title("图7-a  谁最占硬盘", fontsize=11.5, fontweight="bold", color=INK)

    ax = axes[1]
    rows = []
    for m in ms:
        ntr = max(MODELS[m]["n_tracks"])
        nson = MODELS[m]["n_pass"]
        rows.append((MODELS[m]["label"], ntr, nson,
                     MODELS[m]["out_mb_total"] / max(1, nson), m))
    rows.sort(key=lambda r: (r[1], -r[2]))
    yy = np.arange(len(rows))
    ax.barh(yy, [r[2] for r in rows], color=[kind(r[4])[1] for r in rows], height=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r[0]}  ({r[1]}轨)" for r in rows], fontsize=8.8)
    for i, r in enumerate(rows):
        ax.text(r[2] + 1.4, i, f"{r[2]}/50 首  ·  均 {r[3]:.1f} MB/首", va="center",
                fontsize=7.9, color=INK)
    ax.set_xlim(0, 104)
    ax.set_xlabel("成功产出歌曲数 (目标 50；BSRNN-SIMO 目录内有 17 首但仅 14 首有效)")
    ax.set_title("图7-b  产出完整度与音轨数", fontsize=11.5, fontweight="bold", color=INK)
    fig.suptitle("输出产物盘点", fontsize=13, fontweight="bold", color=INK, y=1.01)
    fig.savefig(os.path.join(FIG, "07_output_inventory.png"))
    plt.close(fig)


# =====================================================================
# 7. per-song ranking
# =====================================================================
def fig_song_ranking():
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for m in SCOPE:
        for s, v in LED["attempts"][m].items():
            if v.get("status") == "PASS" and v.get("seconds"):
                tot[s] += v["seconds"]; cnt[s] += 1
    avg = {s: tot[s] / cnt[s] for s in tot}
    order = sorted(avg, key=lambda s: -avg[s])
    pick = order[:15] + order[-10:]
    labels = [f"{s}  (n={cnt[s]})" for s in pick]
    vals = [avg[s] for s in pick]
    colors = [C_BAD if i < 15 else "#16a34a" for i in range(len(pick))]
    fig, ax = plt.subplots(figsize=(11, 8.6))
    y = np.arange(len(pick))[::-1]
    bars = ax.barh(y, vals, color=colors, height=0.68, alpha=0.85)
    for b_, v, s in zip(bars, vals, pick):
        ax.text(v + 1.2, b_.get_y() + b_.get_height() / 2,
                f"{v:.0f}s · 累计{tot[s] / 60:.0f}min", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.axhline(9.5, color=MUTED, lw=1, ls="--")
    ax.text(ax.get_xlim()[1] * 0.99, 9.5, "Top15 最慢 ↑   /   ↓ Bottom10 最快",
            ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xlabel("该曲在 12 个模型上的平均单首耗时 (秒)")
    ax.set_title("图8  歌与歌之间的差距：哪首歌最“难跑”", fontsize=12,
                 fontweight="bold", color=INK)
    fig.savefig(os.path.join(FIG, "08_song_ranking.png"))
    plt.close(fig)


# =====================================================================
# 8. event timeline
# =====================================================================
def fig_events():
    ev = []
    for m in LED["attempts"]:
        for s, v in LED["attempts"][m].items():
            if v.get("status") in ("TIMEOUT", "FAIL"):
                ev.append((dt.datetime.strptime(v["at"], "%Y-%m-%d %H:%M:%S"),
                           MODELS[m]["label"], v["status"], s))
    ev.sort()
    if not ev:
        return
    # rows ordered by the model's first attempt time
    firsts = {}
    for m, ss in LED["attempts"].items():
        ts = [dt.datetime.strptime(v["at"], "%Y-%m-%d %H:%M:%S") for v in ss.values() if v.get("at")]
        if ts:
            firsts[MODELS[m]["label"]] = min(ts)
    labs = sorted(firsts, key=lambda l: firsts[l])
    fig, ax = plt.subplots(figsize=(13, 6.2))
    for lab in labs:
        ax.plot([firsts[lab], dt.datetime.fromisoformat("2026-09-22T10:05")], [labs.index(lab)] * 2,
                color="#e2e8f0", lw=9, solid_capstyle="butt", zorder=1)
    seen = defaultdict(int)
    for t, lab, stt, s in ev:
        yy = labs.index(lab)
        k = seen[lab]
        seen[lab] += 1
        ax.scatter(t, yy, s=150, marker="X" if stt == "TIMEOUT" else "P",
                   color=C_BAD if stt == "TIMEOUT" else "#7c3aed", zorder=4,
                   edgecolors="white", linewidths=1.0)
        ax.annotate(s.split(" - ")[0][:20], (t, yy), textcoords="offset points",
                    xytext=(0, 13 if k % 2 == 0 else -20), ha="center",
                    fontsize=7.2, color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.7), zorder=5)
    for lab in labs:
        n_to = sum(1 for e in ev if e[1] == lab and e[2] == "TIMEOUT")
        n_fa = sum(1 for e in ev if e[1] == lab and e[2] == "FAIL")
        tag = []
        if n_to:
            tag.append(f"超时×{n_to}")
        if n_fa:
            tag.append(f"失败×{n_fa}")
        if tag:
            ax.text(dt.datetime.fromisoformat("2026-09-21T00:00"), labs.index(lab),
                    "  " + " · ".join(tag), fontsize=7.6, color=C_BAD, va="center",
                    fontweight="bold", zorder=5)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_ylim(-0.7, len(labs) - 0.3)
    ax.set_xlabel("发生时间（灰条 = 该模型的运行区间）")
    ax.set_title("图9  超时/失败事件时间线（X = 180s 超时 · ● = 运行失败）",
                 fontsize=12, fontweight="bold", color=INK, pad=14)
    fig.autofmt_xdate(rotation=20)
    fig.savefig(os.path.join(FIG, "09_events_timeline.png"))
    plt.close(fig)


# =====================================================================
# 9. per-track SDR heatmap
# =====================================================================
def fig_sdr_heatmap():
    ms = [m for m in SCOPE if MODELS[m]["sdr_mean"]["median"]]
    cells, labels = [], []
    for m in ms:
        vals = {t: [] for t in ("vocals", "drums", "bass", "other")}
        for r in ROWS:
            if r["model_key"] != m:
                continue
            for t in vals:
                if r.get("sdr_" + t) is not None:
                    vals[t].append(r["sdr_" + t])
        if not any(vals.values()):
            continue
        labels.append(f"{MODELS[m]['label']}  ({max(MODELS[m]['n_tracks'])}轨)")
        cells.append([np.median(vals[t]) if vals[t] else np.nan
                      for t in ("vocals", "drums", "bass", "other")])
    arr = np.array(cells, dtype=float)
    ms_ = [m for m in ms]
    order = np.argsort(-np.nanmean(arr, axis=1))
    arr = arr[order]
    labels = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=15)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["vocals", "drums", "bass", "other"], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.6)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.1f}", ha="center", va="center",
                    fontsize=8.6, color=INK if (np.isnan(v) or v < 11) else "white",
                    fontweight="bold")
    ax.set_title("图10  各轨中位 SDR (dB) — 空值=该模型不产出此轨\n（单轨模型的 SDR 分母口径与四轨模型不同，跨列比较需谨慎）",
                 fontsize=10.5, fontweight="bold", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="SDR (dB)")
    fig.savefig(os.path.join(FIG, "10_sdr_heatmap.png"))
    plt.close(fig)


# =====================================================================
# 10. where the 4-track extra cost comes from
# =====================================================================
# (a) white-box probe, tools/_probe_overhead_split.py, song "Al James - Schoolboy
#     Facination" (200.5 s), measured 2026-09-22 on the same machine.
PROBE_SONG_S = 200.53
PROBE = {
    "4轨": {"解释器+框架导入": 4.4, "GT/mixture 解码": 0.14,
            "museval 评测": 19.50, "FLAC 写盘": 2.02},
    "1轨": {"解释器+框架导入": 4.4, "GT/mixture 解码": 0.14,
            "museval 评测": 6.53, "FLAC 写盘": 0.51},
}


def fig_overhead_split():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    # ---- (a) white-box stack
    ax = axes[0]
    cols = {"解释器+框架导入": "#64748b", "GT/mixture 解码": "#94a3b8",
            "museval 评测": "#dc2626", "FLAC 写盘": "#f59e0b"}
    keys = list(PROBE["4轨"])
    for i, grp in enumerate(["4轨", "1轨"]):
        bot = 0.0
        for k in keys:
            v = PROBE[grp][k]
            ax.bar(i, v, bottom=bot, color=cols[k], width=0.55,
                   label=k if i == 0 else None)
            if v >= 1.2:
                ax.text(i, bot + v / 2, f"{v:.1f}s", ha="center", va="center",
                        fontsize=8.4, color="white", fontweight="bold")
            bot += v
        ax.text(i, bot + 1.1, f"合计 {bot:.1f}s", ha="center", fontsize=9.5,
                color=INK, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["输出 4 轨", "输出 1 轨"], fontsize=11)
    ax.set_ylabel("单首固定开销 (秒)")
    ax.set_ylim(0, 33)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(f"图3-a  白盒实测拆解（{PROBE_SONG_S:.0f}s 的曲目）",
                 fontsize=11.5, fontweight="bold", color=INK)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.95)

    # ---- (b) black-box paired delta vs song length
    ax = axes[1]
    P4 = {"Oracle-IRM", "Demucs", "Open-Unmix", "MDX-Net", "Conv-TasNet",
          "MMDenseLSTM", "DPRNN", "BSRNN-SIMO"}
    ov, du = defaultdict(lambda: defaultdict(list)), {}
    for r in ROWS:
        if r["status"] != "PASS" or r["overhead_s"] is None or not r["song_dur_s"]:
            continue
        du[r["song"]] = r["song_dur_s"]
        ov[r["song"]]["4" if r["model"] in P4 else "1"].append(r["overhead_s"])
    pairs = [(du[s], np.mean(d["4"]), np.mean(d["1"]))
             for s, d in ov.items() if d.get("4") and d.get("1")]
    bins = [(0, 150), (150, 220), (220, 300), (300, 400), (400, 999)]
    xs, ys, ns = [], [], []
    for lo, hi in bins:
        sel = [p[1] - p[2] for p in pairs if lo <= p[0] < hi]
        if sel:
            xs.append(f"{lo}–{hi if hi < 999 else 430}s")
            ys.append(float(np.median(sel))); ns.append(len(sel))
    bars = ax.bar(xs, ys, color="#dc2626", alpha=0.85, width=0.6)
    for b_, v, n_ in zip(bars, ys, ns):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.9, f"{v:.1f}s\n(n={n_})",
                ha="center", fontsize=8.4, color=INK)
    ax.set_xlabel("歌曲时长分箱")
    ax.set_ylabel("同曲配对开销差：4 轨模型 − 1 轨模型 (秒)")
    ax.set_ylim(0, max(ys) * 1.35)
    ax.set_title("图3-b  黑盒配对：开销差随歌长增长",
                 fontsize=11.5, fontweight="bold", color=INK)
    ax.text(0.985, 0.955, "共 50 首配对；越长的歌\n多出越多（museval 随歌长放大）",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6, color=MUTED,
            bbox=dict(fc="white", ec="#e5e7eb", alpha=0.9))
    fig.suptitle("为什么输出 4 轨的模型每首要多花约 30 秒", fontsize=13,
                 fontweight="bold", color=INK, y=1.02)
    fig.savefig(os.path.join(FIG, "03_overhead_split.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_time_range()
    fig_inference_vs_overhead()
    fig_time_vs_songdur()
    fig_sequence_drift()
    fig_speed_accuracy()
    fig_output_inventory()
    fig_song_ranking()
    fig_events()
    fig_sdr_heatmap()
    fig_overhead_split()
    print("figures ->", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f, os.path.getsize(os.path.join(FIG, f)) // 1024, "KB")
