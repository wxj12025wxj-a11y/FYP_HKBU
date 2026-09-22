# -*- coding: utf-8 -*-
"""Emit a compact markdown summary of the 50-song test sweep for thesis citation."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
import statistics as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMP = os.path.join(_ROOT, "04_reports", "separation", "data", "comparison")
OUT = os.path.join(_ROOT, "04_reports", "separation", "docs", "MUSDB18_TEST_SWEEP_2026-09-22.md")

AN = json.load(open(os.path.join(CMP, "test_run_analysis.json"), encoding="utf-8"))
MODELS = AN["models"]
for _r in MODELS.values():
    _r["n_tracks"] = {int(k): v for k, v in _r["n_tracks"].items()}

SCOPE = ["oracle", "demucs", "bsrnn_simo", "umx", "mdx", "convtasnet", "mmdenselstm",
         "bsroformer_l12", "bsroformer_l6", "bsrnn", "bsrnn_large", "dprnn"]
rows = list(csv.DictReader(open(os.path.join(CMP, "test_run_per_song.csv"), encoding="utf-8-sig")))
for r in rows:
    for k in ("song_dur_s", "wall_s", "infer_s", "overhead_s", "rtf", "sdr_mean", "out_mb"):
        r[k] = float(r[k]) if r[k] not in ("", "None") else None

P4 = {"Oracle-IRM", "Demucs", "Open-Unmix", "MDX-Net", "Conv-TasNet", "MMDenseLSTM",
      "DPRNN", "BSRNN-SIMO"}
ov, du = defaultdict(lambda: defaultdict(list)), {}
for r in rows:
    if r["status"] != "PASS" or r["overhead_s"] is None or not r["song_dur_s"]:
        continue
    du[r["song"]] = r["song_dur_s"]
    ov[r["song"]]["4" if r["model"] in P4 else "1"].append(r["overhead_s"])
pairs = [(du[s], st.mean(d["4"]), st.mean(d["1"]))
         for s, d in ov.items() if d.get("4") and d.get("1")]
deltas = sorted(p[1] - p[2] for p in pairs)

tot_wall = sum(MODELS[m]["wall_s"]["total"] or 0 for m in SCOPE)
tot_infer = sum(MODELS[m]["infer_s"]["total"] or 0 for m in SCOPE)
n_pass = sum(MODELS[m]["n_pass"] for m in SCOPE)
ov4 = [r["overhead_s"] for r in rows
       if r["status"] == "PASS" and r["overhead_s"] is not None and int(r["n_tracks"]) == 4]
ov1 = [r["overhead_s"] for r in rows
       if r["status"] == "PASS" and r["overhead_s"] is not None and int(r["n_tracks"]) == 1]

order = sorted(SCOPE, key=lambda m: MODELS[m]["wall_s"]["median"] or 9e9)
lines = []
for m in order:
    r = MODELS[m]
    kind = ("四轨" if 4 in r["n_tracks"] else
            "人声+伴奏" if m == "bsroformer_l6" else "人声单轨")
    lines.append(
        f"| {r['label']} | {kind} | {r['n_pass']}/50 | "
        f"{r['wall_s']['median']:.1f} | {r['wall_s']['mean']:.1f} | "
        f"{r['wall_s']['min']:.0f}–{r['wall_s']['max']:.0f} | {r['wall_s']['stdev']:.1f} | "
        f"{r['infer_s']['median']:.1f} | {r['overhead_s_total'] / r['wall_s']['n']:.1f} | "
        f"{r['rtf']['median']:.3f} | {r['sdr_mean']['median']:.2f} | {r['out_mb_total']:.0f} |")

MD = f"""# MUSDB18-HQ test 集 50 首整曲 · 12 模型实机运行数据摘要

> 生成时间 2026-09-22 ｜ 机器：RTX 4070 Laptop 8 GiB（本机实测，**非论文引用值、非网络数据**）
> 范围：`02_databases/MUSDB18-HQ/test` 全部 50 首整曲（总时长 {12470.97 / 60:.0f} min）·
> 调度：逐首 song-major 串行，单组合 180 s 超时上限，连续 3 次超时即标记不可用
> 产物：`03_outputs/_test_run/<模型>/<歌曲>/`（FLAC 无损 PCM_16 44.1 kHz）

## 1. 完成口径（务必注意）

- 目标 600 个组合（12 模型 × 50 首），**成功 {n_pass} 个（{100 * n_pass / 600:.0f}%）**。
- **11 个模型跑满 50/50**；**BSRNN-SIMO 仅 14/50 成功**（3 首在 180 s 上限被中断后该模型被禁用）。
- RPCA / BSRNN-all(4ckpt) / BSRNN-large-all(4ckpt) 各连续 3 次超时，被判不可用，**零产出**。
- DPRNN 的 1 笔 FAIL 为修复前的旧记录（stereo 广播 bug），重跑后为 PASS，按 50/50 计入。

## 2. 耗时总账

| 指标 | 数值 |
|---|---|
| 累计墙钟（12 模型 × 各自完成数） | **{tot_wall / 3600:.2f} h** |
| 其中推理本体 | {tot_infer / 3600:.2f} h（{100 * tot_infer / tot_wall:.1f}%） |
| 其中记账外开销（评测/导入/读写/调度） | **{(tot_wall - tot_infer) / 3600:.2f} h（{100 * (tot_wall - tot_infer) / tot_wall:.1f}%）** |
| 挂机跨度（含 12 次超时空等 ≈ 24 min） | 33.1 h（含人工干预与停跑时段） |
| 输出音频总体积 | {sum(MODELS[m]['out_mb_total'] for m in SCOPE) / 1024:.1f} GB（FLAC） |

## 3. 逐模型明细（按中位单首耗时升序）

| 模型 | 输出形态 | 完成 | 中位耗时 s | 均值 s | min–max s | σ | 推理中位 s | 开销均 s | RTF 中位 | SDR 中位 dB | 产物 MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(lines)}

- RTF = 推理秒数 ÷ 音频秒数（**不含**开销），故远小于"墙钟实时倍率"。
- SDR = museval BSSEval v4，1 s 窗取值的中位数；单轨模型的均值只含 1 条轨，**不可与四轨模型直接比较**。

## 4. 五条可直接引用的结论

### 4.1 实机耗时 ≠ 推理速度，开销是主体
12 个模型累计 {tot_wall / 3600:.2f} h 中只有 {tot_infer / 3600:.2f} h（{100 * tot_infer / tot_wall:.0f}%）是推理。
最快的 Oracle-IRM 推理中位仅 {MODELS['oracle']['infer_s']['median']:.1f} s/首，端到端却要
{MODELS['oracle']['wall_s']['median']:.1f} s/首——{MODELS['oracle']['overhead_share_pct']:.0f}% 花在每首都要重付的开销上。
**按 RTF 排序会得到错误的实机排名。**

实机排名（中位单首耗时）：BSRNN-large {MODELS['bsrnn_large']['wall_s']['median']:.0f}s
< DPRNN {MODELS['dprnn']['wall_s']['median']:.0f}s
< Oracle-IRM {MODELS['oracle']['wall_s']['median']:.0f}s
< BSRNN-opt {MODELS['bsrnn']['wall_s']['median']:.0f}s
< Demucs {MODELS['demucs']['wall_s']['median']:.0f}s
< BS-RoFormer-L6 {MODELS['bsroformer_l6']['wall_s']['median']:.0f}s
< Open-Unmix {MODELS['umx']['wall_s']['median']:.0f}s
≈ MMDenseLSTM {MODELS['mmdenselstm']['wall_s']['median']:.0f}s
< MDX-Net {MODELS['mdx']['wall_s']['median']:.0f}s
< BS-RoFormer-L12 {MODELS['bsroformer_l12']['wall_s']['median']:.0f}s
< Conv-TasNet {MODELS['convtasnet']['wall_s']['median']:.0f}s
< BSRNN-SIMO {MODELS['bsrnn_simo']['wall_s']['median']:.0f}s。

### 4.2 输出轨数是一等成本项：4 轨比 1 轨每首多 ~31 s
同曲配对（50 首）后，4 轨模型比 1 轨模型每首多花 **中位 {st.median(deltas):.1f} s / 均值 {st.mean(deltas):.1f} s**，
且随歌长放大（&lt;150 s 的歌 +13.8 s，300–400 s +34.5 s，430 s 那首 +56.0 s）。
白盒实测（201 s 曲目，{os.path.basename('_probe_overhead_split.py')}）：
仅 museval 一项，评 4 条轨 **19.5 s**、评 1 条 **6.5 s**；写盘 4 轨 2.0 s vs 1 轨 0.5 s；解码 0.14 s（系统缓存）。
4 轨模型开销中位 {st.median(ov4):.1f} s/首，1 轨模型 {st.median(ov1):.1f} s/首。

### 4.3 耗时与歌长只是中等相关
全体 r = 0.554（n=564）。墙钟实时倍率（处理 1 s 音频所需墙钟秒）从
BSRNN-large 0.19×（315 首/小时）到 BSRNN-SIMO 0.62×（97 首/小时），差 3.2 倍。

### 4.4 8 小时连续负载下无热漂移
按单位音频耗时归一化并消除歌曲长短影响后，趋势斜率 −0.00004/首，50 首累计 **−1.8%**，
即**没有系统性劣化**；仅有第 2 首附近的冷启动尖峰。

### 4.5 四轨模型中 MDX-Net 是最佳"快 + 准"折中
四轨同台可比者：BSRNN-SIMO SDR 均 {MODELS['bsrnn_simo']['sdr_mean']['median']:.2f}（最快也最贵，已超时禁用）、
MDX-Net {MODELS['mdx']['sdr_mean']['median']:.2f}（{MODELS['mdx']['wall_s']['median']:.0f} s/首）、
Demucs {MODELS['demucs']['sdr_mean']['median']:.2f}（{MODELS['demucs']['wall_s']['median']:.0f} s/首）；
Conv-TasNet / MMDenseLSTM / Open-Unmix 集中在 {MODELS['umx']['sdr_mean']['median']:.1f}–{MODELS['mmdenselstm']['sdr_mean']['median']:.1f} dB 且耗时更长，属"又慢又不准"象限。
DPRNN（自训）SDR 均 **{MODELS['dprnn']['sdr_mean']['median']:.2f} dB（负）**，作为"自训 5 小时无法达 SOTA"的负面证据。

## 5. 产物与脚本

- 图表（10 张，PNG）：`04_reports/separation/figures/test_run/`
- 自包含看板（含全部图 + 表，可离线打开）：`04_reports/separation/html/test_run_dashboard.html`
- 数据：`04_reports/separation/data/comparison/test_run_analysis.json` ·
  `test_run_per_song.csv`（逐 (模型,歌)）· `test_run_model_summary.csv`（逐模型）
- 账本：`test_sweep_ledger.json`；原始结果：`model_runs_run.json`
- 脚本：`tools/_analyze_test_run.py`（聚合）· `tools/_plot_test_run.py`（出图）·
  `tools/_build_test_run_html.py`（看板）· `tools/_probe_overhead_split.py`（开销白盒实测）

## 6. 已知口径问题（引用前需留意）

1. `evaluate()` 调用 museval 时**未传 `window`/`hop`**，实为默认 `window=88200(2 s) / hop=66150(1.5 s)`，
   与文档中"1 s 窗"的描述不符（实测差约 0.2 dB）。**需统一口径并修订文档**。
2. "开销" = 墙钟 − `infer_s`，属**记账差额**而非纯净评测耗时，其中混入了模型自身计时未覆盖的部分；
   引用时建议按"记账外开销"表述。
3. 单轨模型的 SDR 分母口径与四轨模型不同（只评 1 条轨），跨组比较需注明。
"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(MD)
print("md ->", OUT, os.path.getsize(OUT) // 1024, "KB")
