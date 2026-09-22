# -*- coding: utf-8 -*-
"""
================================================================================
 运行时长基准 -> HTML 汇报 PPT (自包含, 图片 base64 内联)
--------------------------------------------------------------------------------
 输入 : 04_reports/separation/data/comparison/bench_10songs.json + figs_time/*.png
 输出 : 04_reports/separation/html/RUNTIME_BENCH_PPT.html
--------------------------------------------------------------------------------
 运行: python tools/make_bench_ppt.py
================================================================================
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

from PIL import Image
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
OUT_HTML = _paths.HTML / "RUNTIME_BENCH_PPT.html"
JSON = COMPARISON_DIR / "bench_10songs.json"

SHORT = {
    "A Classic Education - NightOwl": "A Classic Education · NightOwl",
    "ANiMAL - Clinic A": "ANiMAL · Clinic A",
    "ANiMAL - Easy Tiger": "ANiMAL · Easy Tiger",
    "ANiMAL - Rockshow": "ANiMAL · Rockshow",
    "Actions - Devil's Words": "Actions · Devil's Words",
    "Actions - One Minute Smile": "Actions · One Minute Smile",
    "Actions - South Of The Water": "Actions · South Of The Water",
    "Aimee Norwich - Child": "Aimee Norwich · Child",
    "Alexander Ross - Goodbye Bolero": "Alexander Ross · Goodbye Bolero",
    "Alexander Ross - Velvet Curtain": "Alexander Ross · Velvet Curtain",
}


def img64(p: Path, max_w=1200, q=82) -> str:
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    u, m = d["openunmix"], d["demucs"]

    imgs = {p.stem: img64(p) for p in sorted(FIG_DIR.glob("fig*.png"))}

    rt = m["inference_total_seconds"] / u["inference_total_seconds"]
    rtp = m["inference_mean_seconds"] / u["inference_mean_seconds"]

    def med(dd, key):
        v = [r[key] for r in dd["per_song"]
             if r.get(key) is not None and abs(r[key]) < 30.0]
        return (float(np.median(v)) if v else float("nan")), len(v)

    umx_med, umx_n = med(u, "sdr_vocals")
    dem_med, dem_n = med(m, "sdr_vocals")

    # 逐曲表
    rows = ""
    for ru, rd in zip(u["per_song"], m["per_song"]):
        def s(v):
            return "n/a" if v is None else f"{v:.2f}"
        rows += (f"<tr><td>{SHORT.get(ru['song'], ru['song'])}</td>"
                 f"<td class='n'>{ru['inference_seconds']:.2f}</td>"
                 f"<td class='n'>{ru['rtf']:.3f}</td>"
                 f"<td class='n'>{s(ru['sdr_vocals'])}</td>"
                 f"<td class='n dem'>{rd['inference_seconds']:.2f}</td>"
                 f"<td class='n dem'>{rd['rtf']:.3f}</td>"
                 f"<td class='n dem'>{s(rd['sdr_vocals'])}</td></tr>")

    run_date = d["meta"]["created"].replace("T", " ")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open-Unmix vs Demucs · 运行时长基准汇报</title>
<style>
  :root{{
    --umx:#2f6f9f; --dem:#d95f02; --ink:#1a2230; --muted:#5b6879;
    --line:#e2e8f0; --bg:#eef2f7; --card:#ffffff; --accent:#0f766e;
  }}
  *{{box-sizing:border-box;}}
  body{{margin:0;font-family:"Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;
    background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased;}}
  .deck{{max-width:1180px;margin:0 auto;padding:28px 18px 80px;}}
  .slide{{background:var(--card);border-radius:18px;padding:38px 46px;margin:26px 0;
    box-shadow:0 6px 26px rgba(20,40,70,.09);scroll-margin-top:20px;}}
  .kicker{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;
    color:var(--accent);font-weight:700;margin-bottom:10px;}}
  h1{{font-size:38px;line-height:1.2;margin:6px 0 14px;}}
  h2{{font-size:26px;margin:0 0 18px;padding-bottom:12px;border-bottom:3px solid var(--line);}}
  h3{{font-size:17px;margin:22px 0 10px;}}
  p{{font-size:16px;line-height:1.75;color:#28313f;margin:10px 0;}}
  .sub{{font-size:17px;color:var(--muted);}}
  .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0;}}
  .kpi{{background:#f8fafc;border:1px solid var(--line);border-radius:14px;padding:18px;}}
  .kpi .lab{{font-size:13px;color:var(--muted);font-weight:600;}}
  .kpi .val{{font-size:26px;font-weight:800;margin-top:6px;line-height:1.15;}}
  .kpi .val small{{font-size:14px;font-weight:600;color:var(--muted);}}
  .umx{{color:var(--umx);}} .dem{{color:var(--dem);}}
  table{{width:100%;border-collapse:collapse;font-size:14.5px;margin:14px 0;}}
  th,td{{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;}}
  th{{background:#f1f5f9;font-weight:700;font-size:13.5px;color:#334155;}}
  td.n{{text-align:right;font-variant-numeric:tabular-nums;font-family:Consolas,monospace;}}
  tr:hover td{{background:#f8fafc;}}
  figure{{margin:18px 0 8px;}}
  figure img{{width:100%;border:1px solid var(--line);border-radius:12px;display:block;}}
  figcaption{{font-size:13.5px;color:var(--muted);margin-top:9px;text-align:center;}}
  .callout{{border-left:5px solid var(--accent);background:#f0fdfa;border-radius:0 12px 12px 0;
    padding:14px 18px;margin:16px 0;font-size:15.5px;line-height:1.7;}}
  .warn{{border-left-color:#d97706;background:#fffbeb;}}
  .good{{border-left-color:#16a34a;background:#f0fdf4;}}
  .bad{{border-left-color:#dc2626;background:#fef2f2;}}
  ul{{font-size:16px;line-height:1.85;padding-left:22px;}}
  li{{margin:6px 0;}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:26px;}}
  .idx{{position:fixed;right:18px;top:50%;transform:translateY(-50%);display:flex;
    flex-direction:column;gap:7px;z-index:50;}}
  .idx a{{width:11px;height:11px;border-radius:50%;background:#c3cdda;display:block;
    transition:.2s;}}
  .idx a:hover{{background:var(--umx);transform:scale(1.35);}}
  .pill{{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12.5px;
    font-weight:700;margin-right:6px;}}
  .pill.u{{background:#e0edf7;color:var(--umx);}} .pill.d{{background:#fbe8d8;color:var(--dem);}}
  .foot{{font-size:12.5px;color:#93a1b3;text-align:center;margin-top:22px;}}
  @media(max-width:860px){{.kpis{{grid-template-columns:repeat(2,1fr);}}.two{{grid-template-columns:1fr;}}}}
  @media print{{body{{background:#fff;}}.slide{{box-shadow:none;page-break-after:always;}}.idx{{display:none;}}}}
</style>
</head>
<body>
<div class="idx">
  <a href="#s1" title="封面"></a><a href="#s2" title="结论速览"></a>
  <a href="#s3" title="实验设置"></a><a href="#s4" title="逐曲时长"></a>
  <a href="#s5" title="累计耗时"></a><a href="#s6" title="实时性"></a>
  <a href="#s7" title="分布"></a><a href="#s8" title="时间-精度"></a>
  <a href="#s9" title="四维对比"></a><a href="#s10" title="Pareto"></a>
  <a href="#s11" title="逐曲数据"></a><a href="#s12" title="口径校正"></a>
  <a href="#s13" title="结论"></a>
</div>
<div class="deck">

  <!-- 1 封面 -->
  <section class="slide" id="s1">
    <div class="kicker">Final Year Project · 音乐源分离</div>
    <h1>Open-Unmix vs Demucs<br>同一组 10 首歌的运行时长基准</h1>
    <p class="sub">数据集：MUSDB18-HQ train 前 10 首（A Classic Education – NightOwl 等）·
      每首前 30 s · 立体声 44.1 kHz · CPU 单进程</p>
    <p class="sub">生成时间：{run_date}</p>
    <div class="callout good">
      <b>一句话结论：</b>同一组 10 首歌上，<span class="umx"><b>Open-Unmix 总耗时 {u['inference_total_seconds']:.1f} s</b></span>，
      <span class="dem"><b>Demucs 总耗时 {m['inference_total_seconds']:.1f} s</b></span>，
      Demucs 约为 Open-Unmix 的 <b>{rt:.2f}×</b>。两者 RTF 均 &lt; 1，都满足实时。
    </div>
  </section>

  <!-- 2 结论速览 -->
  <section class="slide" id="s2">
    <h2>一、结论速览</h2>
    <div class="kpis">
      <div class="kpi"><div class="lab">10 首歌推理总时长</div>
        <div class="val"><span class="umx">{u['inference_total_seconds']:.1f}s</span>
        <small> / </small><span class="dem">{m['inference_total_seconds']:.1f}s</span></div></div>
      <div class="kpi"><div class="lab">平均每首（30 s 音频）</div>
        <div class="val"><span class="umx">{u['inference_mean_seconds']:.2f}s</span>
        <small> / </small><span class="dem">{m['inference_mean_seconds']:.2f}s</span></div></div>
      <div class="kpi"><div class="lab">中位 RTF（越小越快）</div>
        <div class="val"><span class="umx">{u['rtf_median']:.3f}</span>
        <small> / </small><span class="dem">{m['rtf_median']:.3f}</span></div></div>
      <div class="kpi"><div class="lab">Demucs 相对慢</div>
        <div class="val dem">{rt:.2f}×</div></div>
    </div>
    <div class="callout">
      <b>读法：</b>RTF = 推理时间 ÷ 音频时长。<b>RTF &lt; 1 即可实时</b>。两模型 RTF 都远低于 1，
      说明「实时」不是二选一的门槛——<b>选型的关键是精度与算力预算</b>，而非能否实时。
    </div>
    <div class="callout warn">
      <b>与既有认知的差异：</b>此前基于单曲估算的「Demucs 慢约 3.3×」，
      在 <b>10 首歌实测</b>下修正为 <b>{rt:.2f}×</b>（见「口径校正」页）。
    </div>
  </section>

  <!-- 3 实验设置 -->
  <section class="slide" id="s3">
    <h2>二、实验设置与公平性</h2>
    <table>
      <tr><th>项目</th><th>设置</th></tr>
      <tr><td>数据集</td><td>MUSDB18-HQ · train 前 10 首（与既有横向对比完全同一组）</td></tr>
      <tr><td>片段</td><td>每首前 30 s，立体声 44.1 kHz（共 300 s 音频）</td></tr>
      <tr><td>模型 A</td><td>Open-Unmix（umxhq）· 4 目标一次分离（vocals/drums/bass/other）</td></tr>
      <tr><td>模型 B</td><td>Demucs（htdemucs）· 4 stem 一次分离</td></tr>
      <tr><td>硬件</td><td>CPU 单进程（无 GPU 加速）</td></tr>
      <tr><td>计时口径</td><td><b>墙钟时间</b>；模型加载 / 预热 <b>单独计</b>，不计入推理</td></tr>
      <tr><td>预热</td><td>先跑一次 3 s 片段排除冷启动（UMX {u['warmup_seconds']}s / Demucs {m['warmup_seconds']}s）</td></tr>
      <tr><td>模型加载</td><td>UMX {u['model_load_seconds']}s / Demucs {m['model_load_seconds']}s（均为缓存命中）</td></tr>
    </table>
    <div class="callout">
      <b>为什么这样设计：</b>「公平」= 同一组音频、同一次运行、同样的计时边界。
      把<b>加载/预热剥离</b>，量到的才是纯推理成本；否则第一首会被冷启动严重污染
      （旧数据里首曲耗时是后续的 5 倍，正是这个原因）。
    </div>
  </section>

  <!-- 4 逐曲时长 -->
  <section class="slide" id="s4">
    <h2>三、逐曲运行时长</h2>
    <figure><img src="{imgs['fig1_time_per_song']}" alt="逐曲时长">
      <figcaption>图1 每首歌（30 s）的推理墙钟时间。蓝＝Open-Unmix，橙＝Demucs。</figcaption></figure>
    <div class="callout">
      <b>要点：</b>逐曲结论高度一致——UMX 稳定在 <b>3.8–4.8 s</b>，Demucs 稳定在 <b>10.3–11.8 s</b>。
      两模型对歌曲长度以外的内容差异<b>不敏感</b>，耗时主要由音频时长决定。
    </div>
  </section>

  <!-- 5 累计耗时 -->
  <section class="slide" id="s5">
    <h2>四、累计耗时曲线</h2>
    <figure><img src="{imgs['fig2_time_cumulative']}" alt="累计耗时">
      <figcaption>图2 跑完整组 10 首歌的累计时间。两线之间的灰色区域即 Demucs 的额外代价。</figcaption></figure>
    <div class="callout">
      <b>要点：</b>两条曲线<b>近似平行</b>，说明速度比在全过程稳定，不存在「跑久了变慢」的衰减。
      跑完 10 首歌的净差额约 <b>{m['inference_total_seconds']-u['inference_total_seconds']:.1f} s</b>。
    </div>
  </section>

  <!-- 6 实时性 -->
  <section class="slide" id="s6">
    <h2>五、实时性（RTF）</h2>
    <figure><img src="{imgs['fig3_rtf_per_song']}" alt="RTF">
      <figcaption>图3 逐曲 RTF 与实时阈值（红线 RTF=1）。</figcaption></figure>
    <div class="callout good">
      <b>要点：</b>UMX 中位 RTF <b>{u['rtf_median']:.3f}</b>，Demucs 中位 RTF <b>{m['rtf_median']:.3f}</b>，
      均<b>远低于 1</b>。即使是无 GPU 的 CPU，两者都能实时运行——<b>「能不能实时」对两者都不是问题</b>。
    </div>
  </section>

  <!-- 7 分布 -->
  <section class="slide" id="s7">
    <h2>六、运行时长分布</h2>
    <figure><img src="{imgs['fig4_time_distribution']}" alt="分布">
      <figcaption>图4 10 首歌耗时的箱线图（白点＝均值，黑点＝逐曲样本）。</figcaption></figure>
    <div class="callout">
      <b>要点：</b>两个模型的分箱都很<b>紧凑</b>（UMX 标准差 {u['inference_std_seconds']:.2f}s、
      Demucs {m['inference_std_seconds']:.2f}s），说明单首数字即可代表整体，无需担心巨大波动。
    </div>
  </section>

  <!-- 8 时间-精度 -->
  <section class="slide" id="s8">
    <h2>七、时间 — 精度散点</h2>
    <figure><img src="{imgs['fig5_time_vs_accuracy']}" alt="时间-精度">
      <figcaption>图5 每点一首歌：横轴为耗时，纵轴为该曲 SDR（左＝人声，右＝鼓声）。</figcaption></figure>
    <div class="callout">
      <b>要点：</b>Demucs 的点整体<b>位于右侧且更高</b>——它用更多时间换更高精度。
      注意到有离群点（极低 SDR）来自<b>参考轨过稀疏的退化样本</b>，聚合时应剔除或取中位数。
    </div>
  </section>

  <!-- 9 四维对比 -->
  <section class="slide" id="s9">
    <h2>八、四维对比（核心对比图）</h2>
    <figure><img src="{imgs['fig6_compare_panels']}" alt="四维对比">
      <figcaption>图6 总时长 / 平均每首 / 中位 RTF / 中位人声 SDR 四个维度头对头。</figcaption></figure>
    <div class="two">
      <div class="callout"><b><span class="pill u">Open-Unmix</span></b>
        快、轻、CPU 友好；中位人声 SDR <b>{umx_med:.2f} dB</b>（n={umx_n}），已足够可用。</div>
      <div class="callout"><b><span class="pill d">Demucs</span></b>
        精度更高，中位人声 SDR <b>{dem_med:.2f} dB</b>（n={dem_n}）；
        代价是约 {rt:.2f}× 的耗时。</div>
    </div>
  </section>

  <!-- 10 Pareto -->
  <section class="slide" id="s10">
    <h2>九、速度 — 精度 Pareto</h2>
    <figure><img src="{imgs['fig7_pareto_speed_accuracy']}" alt="Pareto">
      <figcaption>图7 横轴为平均每首耗时（越左越快），纵轴为中位人声 SDR（越上越准）。理想解在左上角。</figcaption></figure>
    <div class="callout good">
      <b>选型建议：</b>若 FYP 强调<b>实时/低延迟/边缘部署</b> → 选 Open-Unmix；
      若强调<b>分离质量/可听感指标</b> → 选 Demucs；两者都满足实时约束，属「同一能力档位内的取舍」。
    </div>
  </section>

  <!-- 11 逐曲数据 -->
  <section class="slide" id="s11">
    <h2>十、逐曲实测数据</h2>
    <table>
      <tr><th rowspan="2">歌曲</th>
        <th colspan="3" style="text-align:center;color:var(--umx)">Open-Unmix (umxhq)</th>
        <th colspan="3" style="text-align:center;color:var(--dem)">Demucs (htdemucs)</th></tr>
      <tr><th class="n">耗时 s</th><th class="n">RTF</th><th class="n">SDR 人声</th>
        <th class="n">耗时 s</th><th class="n">RTF</th><th class="n">SDR 人声</th></tr>
      {rows}
      <tr style="font-weight:800;background:#f8fafc">
        <td>合计 / 中位</td>
        <td class="n umx">{u['inference_total_seconds']:.2f}</td><td class="n umx">{u['rtf_median']:.3f}</td><td class="n">—</td>
        <td class="n dem">{m['inference_total_seconds']:.2f}</td><td class="n dem">{m['rtf_median']:.3f}</td><td class="n">—</td>
      </tr>
    </table>
    <p class="sub">口径：museval BSSEval v4，1 s 窗，中位数，单声道。n/a ＝ 参考轨静音无法评估。</p>
  </section>

  <!-- 12 口径校正 -->
  <section class="slide" id="s12">
    <h2>十一、口径校正与注意事项（请务必看）</h2>
    <table>
      <tr><th>事项</th><th>旧数字</th><th>本次实测</th><th>说明</th></tr>
      <tr><td>Demucs 相对速度</td><td>≈ 3.3×</td><td><b>{rt:.2f}×</b></td>
        <td>旧值来自<b>单曲估算</b>；本页为 10 首歌实测，更权威</td></tr>
      <tr><td>UMX 人声 SDR（A Classic）</td><td>6.55 dB</td><td><b>6.98 dB</b></td>
        <td>旧管线单声道/形状处理差异；本次双路独立复算一致（6.978）</td></tr>
      <tr><td>Demucs 人声 SDR（A Classic）</td><td>8.74 dB</td><td>8.50 dB</td>
        <td><b>Demucs 非确定性</b>：同输入两次 8.83/8.73，波动 ±0.1~0.3 dB</td></tr>
    </table>
    <div class="callout warn">
      <b>三条口径纪律（答辩必备）：</b>
      <ul>
        <li><b>绝对 SDR 不可跨数据集横比</b>，唯一稳健的跨实验量是 <b>ΔSDR</b>（相对「不分离」基线）。</li>
        <li>多歌聚合用<b>中位数</b>并<b>剔除退化样本</b>（参考轨静音/过稀疏会让 museval 失真）。</li>
        <li>Demucs 存在<b>运行间随机性</b>，报告单曲数字时应标注波动范围或取多次平均。</li>
      </ul>
    </div>
  </section>

  <!-- 13 结论 -->
  <section class="slide" id="s13">
    <h2>十二、结论与下一步</h2>
    <h3>三条结论</h3>
    <ul>
      <li><b>速度：</b>10 首歌上 Open-Unmix 合计 <b>{u['inference_total_seconds']:.1f}s</b>，
        Demucs <b>{m['inference_total_seconds']:.1f}s</b>，Demucs ≈ <b>{rt:.2f}×</b>（CPU）。</li>
      <li><b>实时性：</b>中位 RTF <b>{u['rtf_median']:.3f}</b> vs <b>{m['rtf_median']:.3f}</b>，<b>两者都实时</b>，
        「实时」不构成二选一的门槛。</li>
      <li><b>精度：</b>Demucs 中位人声 SDR 更高，代价是约 2.7× 的耗时——<b>同一档位内的速度/精度取舍</b>。</li>
    </ul>
    <h3>下一步（建议）</h3>
    <table>
      <tr><th>#</th><th>行动项</th><th>负责</th><th>截止</th></tr>
      <tr><td>1</td><td>与导师定主路线（Open-Unmix 轻量 / Demucs / 混合）</td><td>你 + 导师</td><td>2026-09-18</td></tr>
      <tr><td>2</td><td>补测 RTX 4070 上的真实延迟（GPU 口径）</td><td>我</td><td>2026-09-19</td></tr>
      <tr><td>3</td><td>加分块流式 RTF 测试，验证「实时」不只是整段批处理</td><td>我</td><td>2026-09-19</td></tr>
    </table>
    <div class="foot">数据文件：04_reports/separation/data/comparison/bench_10songs.json / .csv ·
      图表：04_reports/separation/data/comparison/figs_time/ · 生成脚本：tools/benchmark_models_10songs.py</div>
  </section>

</div>
<script>
  document.addEventListener('keydown', e=>{{
    const slides=[...document.querySelectorAll('.slide')];
    const cur=slides.findIndex(s=>s.getBoundingClientRect().top>-50);
    if(e.key==='ArrowRight'||e.key==='PageDown'){{
      e.preventDefault(); (slides[Math.min(cur+1,slides.length-1)]||slides[0]).scrollIntoView({{behavior:'smooth'}});}}
    if(e.key==='ArrowLeft'||e.key==='PageUp'){{
      e.preventDefault(); (slides[Math.max(cur-1,0)]).scrollIntoView({{behavior:'smooth'}});}}
  }});
</script>
</body>
</html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[OK] {OUT_HTML}")
    print(f"     大小 {len(html)/1024/1024:.2f} MB, 内联图片 {len(imgs)} 张")


if __name__ == "__main__":
    main()
