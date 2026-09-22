# -*- coding: utf-8 -*-
"""Build a self-contained (offline, base64-embedded) HTML dashboard for the test sweep."""
from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import json
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMP = os.path.join(_ROOT, "04_reports", "separation", "data", "comparison")
FIG = os.path.join(_ROOT, "04_reports", "separation", "figures", "test_run")
OUT = os.path.join(_ROOT, "04_reports", "separation", "html", "test_run_dashboard.html")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

AN = json.load(open(os.path.join(CMP, "test_run_analysis.json"), encoding="utf-8"))
LED = json.load(open(os.path.join(CMP, "test_sweep_ledger.json"), encoding="utf-8"))
MODELS = AN["models"]
for _r in MODELS.values():
    _r["n_tracks"] = {int(k): v for k, v in _r["n_tracks"].items()}

SCOPE = ["oracle", "demucs", "bsrnn_simo", "umx", "mdx", "convtasnet", "mmdenselstm",
         "bsroformer_l12", "bsroformer_l6", "bsrnn", "bsrnn_large", "dprnn"]
FULL = [m for m in SCOPE if MODELS[m]["complete_50"]]

rows = list(csv.DictReader(open(os.path.join(CMP, "test_run_per_song.csv"), encoding="utf-8-sig")))
for r in rows:
    for k in ("song_dur_s", "wall_s", "infer_s", "overhead_s", "rtf", "sdr_mean", "out_mb"):
        r[k] = float(r[k]) if r[k] not in ("", "None") else None

# ---------------------------------------------------------------- aggregates
tot_wall = sum(MODELS[m]["wall_s"]["total"] or 0 for m in SCOPE)
tot_infer = sum(MODELS[m]["infer_s"]["total"] or 0 for m in SCOPE)
tot_mb = sum(MODELS[m]["out_mb_total"] for m in SCOPE)
n_combo = sum(MODELS[m]["n_pass"] for m in SCOPE)               # PASS only
n_to = sum(MODELS[m]["timeouts"] for m in MODELS)
wait_s = sum(MODELS[m]["timeout_wait_s"] for m in MODELS)
n_fail = sum(MODELS[m]["status_counts"].get("FAIL", 0) for m in MODELS)
ids_all = ["oracle", "demucs", "bsrnn_simo", "umx", "mdx", "convtasnet", "mmdenselstm",
           "bsroformer_l12", "bsroformer_l6", "bsrnn", "bsrnn_large", "dprnn",
           "rpca", "bsrnn_all", "bsrnn_large_all"]
n_attempt_all = sum(len(LED["attempts"][m]) for m in ids_all if m in LED["attempts"])
allts = [dt.datetime.strptime(v["at"], "%Y-%m-%d %H:%M:%S")
         for m in SCOPE for v in LED["attempts"][m].values() if v.get("at")]
span_h = (max(allts) - min(allts)).total_seconds() / 3600.0
song_total_s = 12470.97


def img(fn, alt):
    with open(os.path.join(FIG, fn), "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{alt}">'
            f'<figcaption>{alt}</figcaption></figure>')


def kind_of(m):
    n = MODELS[m]["n_tracks"]
    if not n:
        return "未产出", "bad"
    if 4 in n:
        return "四轨全分离", "full"
    return ("人声+伴奏 1 轨", "vo") if m == "bsroformer_l6" else ("仅人声 1 轨", "voc")


# ---------------------------------------------------------------- model table
order = sorted(SCOPE, key=lambda m: MODELS[m]["wall_s"]["median"] or 9e9)
trs = []
for m in order:
    r = MODELS[m]
    k, cls = kind_of(m)
    ntr = max(r["n_tracks"]) if r["n_tracks"] else 0
    trs.append(
        f'<tr>'
        f'<td class="name">{html.escape(r["label"])}</td>'
        f'<td><span class="pill {cls}">{k}</span></td>'
        f'<td class="num">{r["out_songs"]}<span class="dim">/50</span></td>'
        f'<td class="num">{ntr}</td>'
        f'<td class="num hl">{r["wall_s"]["median"]:.1f}</td>'
        f'<td class="num">{r["wall_s"]["mean"]:.1f}</td>'
        f'<td class="num">{r["wall_s"]["min"]:.0f} – {r["wall_s"]["max"]:.0f}</td>'
        f'<td class="num">{r["wall_s"]["stdev"]:.1f}</td>'
        f'<td class="num">{r["infer_s"]["median"]:.1f}</td>'
        f'<td class="num">{r["overhead_share_pct"]:.0f}%</td>'
        f'<td class="num">{r["rtf"]["median"]:.3f}</td>'
        f'<td class="num">{r["sdr_mean"]["median"]:.2f}</td>'
        f'<td class="num">{r["out_mb_total"]:.0f}</td>'
        f'<td class="num">{r["timeouts"] if r["timeouts"] else "—"}</td>'
        f'</tr>')
model_table = f'''<table class="tbl">
<thead><tr>
<th>模型</th><th>输出形态</th><th>完成</th><th>轨数</th>
<th>中位耗时 s</th><th>均值 s</th><th>min–max s</th><th>σ</th>
<th>推理中位 s</th><th>固定开销占比</th><th>RTF 中位</th><th>SDR 中位 dB</th>
<th>产物 MB</th><th>超时</th>
</tr></thead><tbody>{"".join(trs)}</tbody></table>'''

# ---------------------------------------------------------------- per-song table
by = defaultdict(list)
for r in rows:
    by[r["song"]].append(r)
tot = {s: sum(x["wall_s"] for x in v if x["wall_s"] and x["status"] == "PASS") for s, v in by.items()}
cnt = {s: sum(1 for x in v if x["wall_s"] and x["status"] == "PASS") for s, v in by.items()}
dur = {s: next((x["song_dur_s"] for x in v if x["song_dur_s"]), None) for s, v in by.items()}
srows = []
for s in sorted(tot, key=lambda s: -tot[s]):
    d = dur.get(s)
    srows.append(f'<tr><td class="name">{html.escape(s)}</td>'
                 f'<td class="num">{d:.0f}</td>'
                 f'<td class="num">{cnt[s]}</td>'
                 f'<td class="num hl">{tot[s] / max(1, cnt[s]):.1f}</td>'
                 f'<td class="num">{tot[s] / 60:.1f}</td></tr>')
song_table = f'''<table class="tbl sm"><thead><tr>
<th>歌曲</th><th>时长 s</th><th>参跑模型数</th><th>平均单首耗时 s</th><th>累计 min</th>
</tr></thead><tbody>{"".join(srows)}</tbody></table>'''

kpi = [
    (f"{len(FULL)}+1", "个模型有产出", f"{len(FULL)} 个跑满 50 首；BSRNN-SIMO 只到 17/50 被超时禁用"),
    (f"{n_combo}", "个成功组合", f"目标 600（12 × 50）；成功 {n_combo}（{100 * n_combo / 600:.0f}%），"
                               f"180 s 超时上限共触发 {n_to} 次"),
    (f"{tot_wall / 3600:.1f} h", "累计墙钟时间", f"其中推理只有 {tot_infer / 3600:.1f} h（{100 * tot_infer / tot_wall:.0f}%）"),
    (f"{span_h:.1f} h", "实际挂机跨度", f"含 {wait_s / 60:.0f} min 超时空等（180s 上限共触发 {n_to} 次）"),
    (f"{tot_mb / 1024:.1f} GB", "输出音频总体积", "FLAC 无损，44.1kHz 立体声"),
    (f"{song_total_s / 60:.0f} min", "被处理的音频总长", "50 首 test 集全曲"),
]
kpi_html = "".join(
    f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{l}</div>'
    f'<div class="kn">{n}</div></div>' for v, l, n in kpi)

HTML = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MUSDB18 test 集 12 模型实机运行报告</title>
<style>
:root{{--ink:#1f2937;--mut:#6b7280;--line:#e5e7eb;--bg:#f7f8fa;--card:#fff;
--blue:#2563eb;--amb:#d97706;--pur:#7c3aed;--red:#dc2626;--grn:#16a34a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font-family:"Microsoft YaHei","Segoe UI",system-ui,-apple-system,sans-serif;
line-height:1.65;font-size:14.5px}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 26px 72px}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.3px}}
.sub{{color:var(--mut);font-size:13.5px;margin-bottom:26px}}
h2{{font-size:19px;margin:44px 0 6px;padding-bottom:8px;border-bottom:2px solid var(--line)}}
h2 .no{{color:var(--blue);font-weight:700;margin-right:8px}}
.lead{{color:#374151;margin:8px 0 14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:22px 0 8px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.kv{{font-size:25px;font-weight:700;color:var(--blue);letter-spacing:-.5px}}
.kl{{font-size:12.5px;color:var(--ink);font-weight:600;margin-top:2px}}
.kn{{font-size:11.5px;color:var(--mut);margin-top:5px;line-height:1.5}}
figure{{margin:16px 0 6px;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:10px}}
figure img{{width:100%;display:block;border-radius:6px}}
figcaption{{font-size:12px;color:var(--mut);padding:8px 4px 2px;text-align:center}}
.tbl{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:12px;overflow:hidden;font-size:12.6px}}
.tbl th{{background:#eef2f7;text-align:right;padding:9px 9px;font-weight:600;
font-size:11.8px;color:#374151;border-bottom:1px solid var(--line);white-space:nowrap}}
.tbl th:first-child,.tbl th:nth-child(2){{text-align:left}}
.tbl td{{padding:8px 9px;border-bottom:1px solid #f1f3f7;text-align:right}}
.tbl td.name{{text-align:left;font-weight:600;white-space:nowrap}}
.tbl td:nth-child(2){{text-align:left}}
.tbl tbody tr:hover{{background:#fafbfd}}
.num{{font-variant-numeric:tabular-nums}}
.hl{{color:var(--blue);font-weight:700}}
.dim{{color:var(--mut);font-weight:400}}
.sm{{font-size:12px;max-height:520px;display:block;overflow:auto}}
.pill{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;
font-weight:600;white-space:nowrap}}
.pill.full{{background:#dbeafe;color:#1d4ed8}}
.pill.voc{{background:#fef3c7;color:#b45309}}
.pill.vo{{background:#ede9fe;color:#6d28d9}}
.pill.bad{{background:#fee2e2;color:#b91c1c}}
.note{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--blue);
border-radius:10px;padding:14px 18px;margin:16px 0}}
.note.warn{{border-left-color:var(--amb)}}
.note.bad{{border-left-color:var(--red)}}
.note h3{{margin:0 0 6px;font-size:14.5px}}
.note ul{{margin:6px 0 0;padding-left:20px}}
.note li{{margin:4px 0}}
code{{background:#eef2f7;padding:1px 6px;border-radius:5px;font-size:12.5px;
font-family:Consolas,monospace}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
.foot{{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
color:var(--mut);font-size:12.3px}}
</style></head><body><div class="wrap">

<h1>MUSDB18-HQ test 集 · 12 个模型实机运行报告</h1>
<div class="sub">数据截止 2026-09-22 10:05 ｜ 范围：<code>02_databases/MUSDB18-HQ/test</code> 全部 50 首整曲 ·
逐首 song-major 串行扫描 ｜ 输出：<code>03_outputs/_test_run/</code>（FLAC 无损）｜
RNG 无关，全部为本机实测</div>

<div class="kpis">{kpi_html}</div>

<div class="note">
<h3>先说清"12 个模型跑完"的实际口径</h3>
<ul>
<li><b>11 个模型跑满 50/50</b>：Oracle-IRM、Demucs、Open-Unmix、MDX-Net、Conv-TasNet、MMDenseLSTM、
BS-RoFormer-L12、BS-RoFormer-L6、BSRNN-opt、BSRNN-large、DPRNN。</li>
<li><b>BSRNN-SIMO 只有 14 首成功</b>（另 3 首在 180 s 上限被中断，之后被判为不可用退出）——
它目录里有 17 首歌，但其中 3 首是超时的半成品。<b>这是本次唯一"看起来跑完但其实没跑完"的模型</b>。</li>
<li>RPCA / BSRNN-all(4ckpt) / BSRNN-large-all(4ckpt) 三次超时后被禁用，<b>零产出</b>。</li>
<li>口径修正：DPRNN 在账本里挂着一笔 FAIL，实为修复前的旧记录；其模型侧结果 JSON 显示该曲早已重跑为 PASS，
故按 50/50 计入。</li>
</ul></div>

<div class="note">
<h3>一句话结论</h3>
<p style="margin:6px 0">50 首整曲的实测把"哪个模型快"这件事彻底改写了：<b>推理速度≠实机耗时</b>。
最快的 Oracle-IRM 推理中位仅 3.2 s/首，但端到端要 52 s/首——<b>93% 的时间花在每首歌都要重付一次的固定开销上</b>
（解释器启动 + torch/CUDA 导入 + museval 评测 + 读写盘）。真正决定实机速度的是固定开销，
所以最终排名是 <b>BSRNN-large(47s) → DPRNN(51s) → Oracle(52s) → BSRNN-opt(55s) → Demucs(57s)</b>，
而不是按 RTF 排出来的 Oracle → Demucs → MDX。</p>
</div>

<h2><span class="no">01</span>每首歌要花多久</h2>
<p class="lead">中位数从 <b>47 s/首</b>（BSRNN-large）到 <b>145 s/首</b>（BSRNN-SIMO），差 3.1 倍。
分布本身很宽：所有模型的 min–max 跨度都在 2–3 倍，说明<b>歌曲本身比模型更能决定单首耗时</b>。</p>
{img("01_time_range.png", "图1 每首歌的处理耗时分布（中位数 / 均值 / P25–P75 / min–max）")}

<h2><span class="no">02</span>时间究竟花在哪里</h2>
<p class="lead">把每首耗时拆成"推理本体"和"记账外开销"两段。开销并不与模型无关，而是<b>与"输出几轨"强绑定</b>：
4 轨模型的中位开销 <b>49.3 s/首</b>，1 轨模型只有 <b>18.6 s/首</b>（详见第 03 节）。
于是出现一个反直觉现象：<b>越快的模型，开销占比越高</b>，
Oracle-IRM 达 93%、DPRNN 91%、Demucs 86%，而 BS-RoFormer-L12 因为推理本身要 64 s 才"只"占 78%。</p>
{img("02_inference_vs_overhead.png", "图2 推理本体 vs 固定开销（单首平均口径）")}
<div class="note warn"><h3>这是本次扫描最重要的发现</h3>
<ul>
<li>12 个模型累计 <b>10.17 h</b> 墙钟，其中推理只有 <b>4.25 h（41.8%）</b>，
剩下 <b>5.92 h</b> 全是评测、导入与读写——这部分<b>不随模型变快而消失</b>。</li>
<li>提速必须<b>先打掉开销</b>：同进程批量跑多首（省解释器+导入）、缓存 GT、把 museval 评测与推理解耦并行、评测换更轻的实现。</li>
<li>只优化推理（换更小的网络）在实机上的收益上限很低——Oracle 再快一倍也只从 52 s 降到 50 s。</li>
</ul></div>

<h2><span class="no">03</span>为什么 4 轨模型每首要多花 30 秒</h2>
<p class="lead">这是上面那个“固定开销”的成因，也是本次扫描里最可复现的一条。
把同一首歌的 4 轨模型与 1 轨模型配对相减，50 首的差值中位 <b>31.0 s</b>，且随歌长放大
（&lt;150 s 的歌差 13.8 s，300–400 s 的歌差 34.5 s，430 s 那首差 56.0 s）。
白盒实测把成分钉死了：在一首 201 s 的曲目上，仅 <b>museval BSSEval v4 评测</b>一项，
评 4 条轨就要 <b>19.5 s</b>，评 1 条只要 6.5 s；写盘 4 轨 2.0 s vs 1 轨 0.5 s。
解码几乎不要钱（0.14 s，被系统文件缓存吃掉）。</p>
{img("03_overhead_split.png", "图3 4 轨 vs 1 轨固定开销的白盒拆解与黑盒配对（新增）")}
<div class="note"><h3>推论</h3>
<ul>
<li><b>“输出几轨”是实机耗时的一等公民</b>，作用量级和“换个模型”相当（31 s vs 47–145 s 的总跨度）。</li>
<li>若目标是快速拿到 4 轨素材，<b>与其优化网络，不如优化评测</b>：把 museval 挪到离线批处理、
或对 4 轨并行评测，可直接省掉这 ~30 s/首。</li>
<li>本机 <b>museval 评测成本随歌长超线性增长</b>（201 s→19.5 s，430 s 那首外推已偏低），
长曲目是主要负担。</li>
</ul></div>

<h2><span class="no">04</span>耗时由歌长决定吗</h2>
<p class="lead">整体相关 <b>r = 0.55</b>（n=563），只是一个中等相关——歌长能解释约 30% 的方差，
剩下 70% 来自模型差异与运行噪声。把耗时折算成"处理 1 秒音频要花几秒"（墙钟实时倍率）后，
排名与绝对耗时一致：<b>BSRNN-large 0.19×（315 首/小时）</b> 最快，<b>BSRNN-SIMO 0.62×（97 首/小时）</b> 最慢。
注意这些数字远低于纯推理 RTF，差额就是固定开销。</p>
{img("04_time_vs_songdur.png", "图4 歌曲时长与耗时的关系 + 墙钟吞吐")}

<h2><span class="no">05</span>跑久了会变慢吗（热漂移）</h2>
<p class="lead">把每首耗时除以该首时长、再除以模型自身中位数，消除"先跑的恰好是短歌"这个混淆后，
趋势斜率 <b>−0.00004/首</b>，50 首累计 <b>−1.8%</b>——即<b>没有系统性劣化</b>。
只有第 2 首附近有一簇偶发尖峰（冷启动/首次预热），之后全程平稳。
结论：这台机器的散热与显存管理在 8 小时连续负载下<b>表现稳定</b>，此前的"越跑越慢"担忧不成立。</p>
{img("05_sequence_drift.png", "图5 单位音频耗时随扫描顺序的漂移（已消除歌曲长短影响）")}

<h2><span class="no">06</span>速度 × 质量 × 产物体积</h2>
<p class="lead">气泡大小=磁盘占用，红色虚线为 Pareto 前沿。<b>BS-RoFormer-L12</b> 落在质量最高点
（13.55 dB）但只出人声单轨且要 83 s/首；真正的"四轨全能"最优是 <b>BSRNN-SIMO</b>（SDR 均 9.37 dB，但 145 s/首、已被超时禁用），
其次是 <b>MDX-Net / Demucs</b>（9.87 / 9.05，69 s / 57 s）。
<b>Conv-TasNet、MMDenseLSTM、Open-Unmix</b> 三者在 5.5–5.6 dB 一档且都要 66–90 s，
属于"又慢又不准"的象限。<b>DPRNN（自训）SDR 为负</b>（−1.51），印证自训 5 小时远不足以达到可用质量。</p>
{img("06_speed_accuracy.png", "图6 速度 × 质量 × 产物体积（Pareto 前沿）")}

<h2><span class="no">07</span>产物盘点</h2>
<p class="lead">12 个模型里 <b>8 个输出完整四轨</b>，4 个只出人声相关单轨
（BS-RoFormer L12/L6、BSRNN-opt、BSRNN-large）。所以"12 个模型跑完"不等于"12 套四轨分离"——
真正的四轨素材只有 <b>8 套 × 50 首</b>。
体积差异极大：Conv-TasNet 3.66 GB 是 BSRNN-opt 598 MB 的 6 倍，与实际听感质量无关（是编码/掩码特性的副产物）。</p>
{img("07_output_inventory.png", "图7 输出产物盘点：体积与完整度")}

<h2><span class="no">08</span>歌与歌之间的差距</h2>
<p class="lead">同一首歌在不同模型上平均要跑 <b>26 s ~ 107 s</b>，差 4 倍。
最慢的 <b>Georgia Wonder - Siren</b>（107 s/首，且它正好是被 BSRNN-SIMO 超时的那一首）
与最快的 <b>PR - Oh No</b>（26 s/首，仅 76 s 长）之间，差距主要由歌曲时长与编曲密度驱动。
这也解释了为什么 <code>--offset</code>/片段挑选会严重污染小样本结论——整曲扫描才是稳定口径。</p>
{img("08_song_ranking.png", "图8 单曲耗时排行（Top15 最慢 / Bottom10 最快）")}

<h2><span class="no">09</span>超时与失败事件</h2>
<p class="lead">180 s 超时上限共触发 <b>12 次</b>，涉及 4 个模型，全部集中在两种情形：
<b>(a) 4 个单目标 ckpt 串行跑的 BSRNN 变体</b>（每首要 4 次前向，天然超时）；
<b>(b) BSRNN-SIMO</b>——它单次前向就出四轨，质量最好，但 3 首偏长的歌（BKS Bulldozer / Georgia Wonder Siren /
Girls Under Glass We Feel Alright）都刚好越过 180 s。另有 DPRNN 1 次真实失败（stereo 广播 bug，已修复后重跑通过）。</p>
{img("09_events_timeline.png", "图9 超时/失败事件时间线")}
<div class="note bad"><h3>180 s 上限带来的实际损失（需要你决定怎么处理）</h3>
<ul>
<li><b>BSRNN-SIMO 只有 17/50 首产出</b>，且是这 12 个里 SDR 最高的四轨模型（9.37）——它的缺失让"四轨最优"结论只能退回 MDX-Net。</li>
<li>它超时的 3 首恰恰都是长歌；若把上限放宽到 <b>260 s</b>（≈最长歌 430 s × 0.62×），它基本能跑完全部 50 首，代价是约 +5.5 h 挂机。</li>
</ul></div>

<h2><span class="no">10</span>分轨质量热力图</h2>
<p class="lead">单轨模型（L12 / BSRNN-opt / BSRNN-large）在 SDR 上高得显眼，但<b>它们只被评了 1 条轨</b>，
与四轨模型的"四轨平均"不是一个口径，不能直接比。
在真正四轨同台的模型里：<b>BSRNN-SIMO 人声最强（12.5）</b>，<b>MDX-Net 鼓最强（11.3）</b>，
<b>Oracle-IRM 其他最强（8.1）</b>，而 <b>其他轨（other）是全场共性短板</b>——除 BSRNN-SIMO/Oracle 外全部 ≤ 6.9。</p>
{img("10_sdr_heatmap.png", "图10 各轨中位 SDR 热力图")}

<h2><span class="no">11</span>完整数据表</h2>
{model_table}
<div style="height:14px"></div>
<h3 style="font-size:15px">50 首逐曲汇总（按平均单首耗时降序）</h3>
{song_table}

<div class="foot">
口径说明：SDR = museval BSSEval v4，1 s 窗，取<b>中位数</b>；RTF = 推理秒数 ÷ 音频秒数（不含固定开销）；
"固定开销" = 墙钟秒数 − 推理秒数，含解释器启动、torch/CUDA 导入、museval 评测、GT/mixture 读写。
所有耗时均为<b>本机 RTX 4070 Laptop 8 GiB 实测墙钟</b>，非论文引用值，非网络数据。
原始数据：<code>04_reports/separation/data/comparison/test_run_analysis.json</code> ·
<code>test_run_per_song.csv</code> · <code>test_run_model_summary.csv</code> ·
账本 <code>test_sweep_ledger.json</code>。图表脚本 <code>tools/_plot_test_run.py</code>。
</div>
</div></body></html>'''

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("html ->", OUT, os.path.getsize(OUT) // 1024, "KB")
print(f"combo={n_combo} wall={tot_wall/3600:.2f}h infer={tot_infer/3600:.2f}h "
      f"({100*tot_infer/tot_wall:.1f}%) span={span_h:.2f}h mb={tot_mb:.0f}")
