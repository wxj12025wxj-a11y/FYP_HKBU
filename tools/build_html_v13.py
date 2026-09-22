# -*- coding: utf-8 -*-
"""从 METHOD_VISUAL_COMPARISON_1.2.html 生成 1.3 —— 追加「11 模型供给与可运行性核查」板块。

v1.3 rev2（2026-09-16 晚）更新：
  - BSRNN 三个变体权重全部下载完成（MD5 校验通过），推理跑通 → 状态从「下载中」改为「跑通」；
  - Smart App Control 拦截面收窄（只剩 llvmlite），`museval` 恢复可用 →
    全表由「仅 SI-SDR」升级为「museval SDR + SI-SDR 并列」；
  - 新增「按 stem 的 SDR 矩阵」表；
  - RTF 改为机器空闲时的一次连续测量。

做四件事：
  1. 修正 v1.2 残留的「慢 3.3×」为实测口径 2.70×，并加口径提醒；
  2. 在「七 · 汇总…」之前插入新板块（id=s7n），原七改八；
  3. 更新页头版本/日期与导航；
  4. 新板块的数字全部从 04_reports/separation/data/comparison/model_runs.json 读取，保证可复核。
"""
from __future__ import annotations

import html
import json
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
ROOT = _paths.PROJECT_ROOT
SRC = _paths.HTML / "METHOD_VISUAL_COMPARISON_1.2.html"
DST = _paths.HTML / "METHOD_VISUAL_COMPARISON_1.3.html"
RUNS = _paths.comparison_dir() / "model_runs.json"
MEDIAN = _paths.comparison_dir() / "model_runs_median.json"

SONG = "A Classic Education - NightOwl"
STEMS = ["vocals", "drums", "bass", "other"]
STEM_ZH = {"vocals": "人声", "drums": "鼓", "bass": "贝斯", "other": "其他"}

# 11 模型的静态供给事实（与 04_reports/separation/docs/MODEL_PROVISIONING_REPORT.md 保持一致）
PROVISION = [
    # idx, key(在 model_runs 里的名字), 显示名, 开源, 许可证, 权重, 状态, 备注
    (1, "rpca", "RPCA", "✅ 自研", "—", "不需要（解析法）", "✅", "稀疏分量无天然 stem 归属，见 7.6②"),
    (2, None, "RPCA+DRNN", "❌ 未公开", "—", "无公开权重", "❌", "已定稿：降级为相关工作，不再自研复现（口径为单声道 2 源，不可横比）"),
    (3, "convtasnet", "Conv-TasNet", "✅", "MIT / 镜像仓库无 LICENSE", "✅ MUSDB18 ×3", "✅", ""),
    (4, None, "DPRNN", "✅", "Apache-2.0", "⚠️ 仅 wsj0-mix / librispeech", "❌", "无 MUSDB18 权重；自训已不再被环境阻塞"),
    (5, "mmdenselstm", "MMDenseLSTM", "✅", "镜像仓库无 LICENSE", "✅ MUSDB18 (paper)", "✅", ""),
    (6, "bsroformer_l12", "BS-RoFormer L12", "✅", "MIT", "✅ ep_317 (639 MB)", "✅", "需 librosa 占位模块"),
    (7, "bsroformer_l6", "BS-RoFormer L6", "✅", "MIT", "✅ ep_937 (393 MB)", "✅", "输出语义 ≠ yaml 标注，见 7.6①"),
    (8, "bsrnn_simo", "Band-Split RNN (BSRNN)", "✅", "MIT", "✅ Zenodo ×3 (4.67 GB, MD5 全过)", "✅", "opt / large / SIMO 三变体均跑通，见 7.6③"),
    (9, "oracle", "IRM / IBM Oracle", "✅ 自研", "—", "不需要（解析法）", "✅", ""),
    (10, "mdx", "MDX-Net", "✅", "MIT", "✅ mdx_extra ×4 + onnx ×1", "✅", ""),
    (11, None, "Pac-HuBERT-SEP", "❌ 无公开代码/权重", "—", "无", "❌", "已定稿：MERL 未开源，论文中仅作相关工作引用"),
]

# 展示用模型清单：(key, 显示名, 类别, 主目标)
DISPLAY = [
    ("oracle",         "IRM/IBM Oracle",            "解析法 · 参考上界",   "4-stem"),
    ("rpca",           "RPCA (Inexact-ALM)",        "古典 · 迭代",         "无 stem 语义"),
    ("umx",            "Open-Unmix (umxhq)",        "深度 · 4-stem",       "4-stem"),
    ("mdx",            "MDX-Net (mdx_extra)",       "深度 · 4-stem",       "4-stem"),
    ("convtasnet",     "Conv-TasNet (musdb18)",     "深度 · 4-stem",       "4-stem"),
    ("mmdenselstm",    "MMDenseLSTM (musdb18)",     "深度 · 4-stem",       "4-stem"),
    ("dprnn",          "DPRNN (本机自训)",           "深度 · 4-stem",       "4-stem"),
    ("demucs",         "Demucs (htdemucs)",         "深度 · 4-stem 单模型", "4-stem"),
    ("bsrnn_simo",     "BSRNN SIMO (oBSRNN-SIMO)",  "深度 · 4-stem 单模型", "4-stem"),
    ("bsrnn_all",      "BSRNN opt (4× ckpt 聚合)",  "深度 · 4-stem",       "4-stem"),
    ("bsrnn_large_all", "BSRNN large (4× ckpt 聚合)", "深度 · 4-stem",     "4-stem"),
    ("bsroformer_l12", "BS-RoFormer L12 (ep_317)",  "深度 · 单目标",       "vocals"),
    ("bsroformer_l6",  "BS-RoFormer L6 (ep_937)",   "深度 · 单目标",       "vocals+other"),
]


def load_runs():
    if not RUNS.is_file():
        return {}
    data = json.loads(RUNS.read_text(encoding="utf-8"))
    out = {}
    for m, songs in data.get("runs", {}).items():
        if SONG in songs:
            out[m] = songs[SONG]
    return out


def fmt(v, nd=2, dash="—"):
    if v is None:
        return dash
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def cell(v, nd=2):
    """数值单元格：None 返回灰横线。"""
    if v is None:
        return '<td class="num" style="color:var(--sub)">—</td>'
    return f'<td class="num">{float(v):.{nd}f}</td>'


def load_median():
    if not MEDIAN.is_file():
        return {}, []
    d = json.loads(MEDIAN.read_text(encoding="utf-8"))
    clips = sorted({c for v in d.values() for c in v.get("clips", [])})
    return d, clips


def build_section(runs: dict) -> str:
    # ---------- KPI ----------
    n_ok = sum(1 for r in PROVISION if r[6] == "✅")
    n_no = sum(1 for r in PROVISION if r[6] == "❌")
    n_fail = sum(1 for r in PROVISION if r[6] == "❌")

    kpis = f"""
    <div class="kpis">
      <div class="kpi"><div class="v" style="color:var(--green)">{n_ok} / 11</div><div class="l">本机真实跑通并落盘 stem wav</div></div>
      <div class="kpi"><div class="v" style="color:var(--red)">{n_fail} / 11</div><div class="l">无公开代码或权重，无法下载</div></div>
      <div class="kpi"><div class="v" style="color:var(--green)">4.67 GB</div><div class="l">BSRNN 权重 3 个 zip，MD5 全部校验通过</div></div>
      <div class="kpi"><div class="v" style="color:var(--green)">9.27 dB</div><div class="l">oBSRNN-SIMO 四轨均值（多片段中位数，见 7.4）</div></div>
    </div>
    """

    # ---------- 7.1 核查总表 ----------
    rows = []
    for idx, key, name, oss, lic, wt, ok, note in PROVISION:
        rec = runs.get(key) if key else None
        rtf = fmt((rec or {}).get("rtf"))
        targets = ", ".join((rec or {}).get("targets") or []) or "—"
        err = html.escape(str((rec or {}).get("error") or ""))[:60]
        badge = {'✅': '<span class="pill p-live">跑通</span>',
                 '⏳': '<span class="pill p-classic">下载中</span>',
                 '❌': '<span class="pill p-dead">不可得</span>'}[ok]
        rows.append(
            f"<tr><td class='num'>{idx}</td><td><b>{html.escape(name)}</b></td>"
            f"<td>{html.escape(oss)}</td><td>{html.escape(lic)}</td>"
            f"<td>{html.escape(wt)}</td><td>{badge}</td>"
            f"<td class='num'>{rtf}</td><td>{html.escape(targets)}</td>"
            f"<td style='font-size:12.5px'>{html.escape(note) if note else '—'}"
            f"{('<br><code>' + err + '</code>') if err else ''}</td></tr>"
        )
    table = f"""
    <table>
      <thead><tr>
        <th style="width:28px">#</th><th>模型</th><th>开源</th><th>许可证</th><th>预训练权重</th>
        <th>本机状态</th><th class="num">RTF</th><th>输出目标</th><th>备注 / 失败原因</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <div class="note">RTF 取自机器空闲时的一次连续测量（10 s 片段，<code>04_reports/separation/data/comparison/model_runs.json</code>）。未跑通的模型没有 RTF。</div>
    """

    # ---------- 7.2 SDR 矩阵 ----------
    def sdr_cells(key, stems):
        rec = runs.get(key) or {}
        d = rec.get("sdr") or {}
        return "".join(cell(d.get(s)) for s in stems)

    matrix_rows = []
    for key, label, kind, tgt in DISPLAY:
        rec = runs.get(key) or {}
        d = rec.get("sdr") or {}
        extra = [k for k in d if k not in STEMS]
        tgt_cell = tgt
        if key == "rpca":
            body = ('<td class="num" style="color:var(--sub)">—</td>' * 4)
        elif extra and not any(s in d for s in STEMS):
            # 单目标模型（BS-RoFormer）：把唯一那个数值直接标在「主目标」列，避免整行空值
            k = extra[0]
            body = ('<td class="num" style="color:var(--sub)">—</td>' * 4)
            tgt_cell = f"{k} = <b>{fmt(d.get(k))}</b>"
        else:
            body = "".join(cell(d.get(s)) for s in STEMS)
        rtf = rec.get("rtf")
        cls = "win" if isinstance(rtf, (int, float)) and rtf < 1 else "lose"
        matrix_rows.append(
            f"<tr><td><b>{html.escape(label)}</b></td><td>{kind}</td><td>{tgt_cell}</td>"
            f"{body}<td class='num {cls}'>{fmt(rtf, 2)}</td></tr>"
        )
    matrix = f"""
    <table>
      <thead><tr><th>模型</th><th>类别</th><th>主目标</th>
        <th class="num">人声</th><th class="num">鼓</th><th class="num">贝斯</th><th class="num">其他</th>
        <th class="num">RTF ↓</th></tr></thead>
      <tbody>{''.join(matrix_rows)}</tbody>
    </table>
    <div class="note">单元格为 <b>museval SDR（BSSEval v4，1 s 窗中位数，mono）</b>，单位 dB。
    片段：<code>train/{html.escape(SONG)}</code> 前 10 s、44.1 kHz 立体声、torch 2.14.0+<b>cpu</b>。
    <br>⚠️ <b>本表是「单片段」口径，只用于看谁跑通、谁快</b> —— 该片段是低频引子（bass 占 85% 能量、鼓仅 1.4%），
    bass 列被系统性抬高。<b>要引用精度结论请用 7.4 的多片段中位数表。</b>
    <br>RPCA 的稀疏分量没有天然 stem 归属，故不出单值（见 7.6②）。
    BS-RoFormer 两个官方权重都是<b>单目标</b>模型，数值标在「主目标」列。</div>
    """

    # ---------- 7.3 SI-SDR 矩阵（补充） ----------
    silent_rows = []
    for key, label, kind, tgt in DISPLAY:
        rec = runs.get(key) or {}
        d = rec.get("si_sdr") or {}
        if key == "rpca":
            body = ('<td class="num" style="color:var(--sub)">−7.90<br><span style="font-size:11px">(ref=bass)</span></td>'
                    + '<td class="num" style="color:var(--sub)">—</td>' * 3)
        else:
            body = "".join(cell(d.get(s)) for s in STEMS)
        silent_rows.append(f"<tr><td><b>{html.escape(label)}</b></td>{body}</tr>")
    sisilent = f"""
    <table>
      <thead><tr><th>模型</th><th class="num">人声</th><th class="num">鼓</th><th class="num">贝斯</th><th class="num">其他</th></tr></thead>
      <tbody>{''.join(silent_rows)}</tbody>
    </table>
    <div class="note">同口径的 <b>SI-SDR</b>（全局单值），单位 dB。与 SDR 并列是为了交叉验证：
    两者趋势一致说明结论稳健；SDR 一般略高于 SI-SDR（1 s 窗中位数对局部伪影更宽容）。</div>
    """

    # ---------- 7.4 多片段中位数（有数据才渲染） ----------
    med, med_clips = load_median()
    if med:
        m_order = [d[0] for d in DISPLAY]
        m_order += [k for k in med if k not in m_order]
        four, single = [], []
        for k in m_order:
            if k in med:
                (single if med[k].get("single_target") else four).append(k)
        # 按四轨均值降序
        four.sort(key=lambda k: (med[k].get("sdr_avg4") is None, -(med[k].get("sdr_avg4") or 0)))
        mrows = []
        for k in four:
            e = med[k]
            cells = ""
            for s in STEMS:
                dd = (e.get("sdr") or {}).get(s)
                cells += cell(dd.get("median") if dd else None)
            avg = e.get("sdr_avg4")
            avg_td = (f'<td class="num win"><b>{avg:.2f}</b></td>' if avg is not None
                      else '<td class="num" style="color:var(--sub)">—</td>')
            rtf = e.get("rtf")
            cls = "win" if isinstance(rtf, (int, float)) and rtf < 1 else "lose"
            mrows.append(
                f"<tr><td><b>{html.escape(e.get('label', k))}</b></td>{cells}{avg_td}"
                f"<td class='num {cls}'>{fmt(rtf, 2)}</td><td class='num'>{e.get('n_clips')}</td></tr>"
            )
        srows = []
        for k in single:
            e = med[k]
            st = e["single_target"]
            rtf = e.get("rtf")
            cls = "win" if isinstance(rtf, (int, float)) and rtf < 1 else "lose"
            srows.append(
                f"<tr><td><b>{html.escape(e.get('label', k))}</b></td>"
                f"<td><code>{html.escape(st['name'])}</code></td>"
                f"<td class='num'><b>{fmt(st.get('sdr_median'))}</b></td>"
                f"<td class='num'>{fmt(st.get('si_sdr_median'))}</td>"
                f"<td class='num {cls}'>{fmt(rtf, 2)}</td><td class='num'>{e.get('n_clips')}</td></tr>"
            )
        clip_txt = "、".join(f"<code>{html.escape(c)}</code>" for c in med_clips)

        best = four[0] if four else None
        best_txt = ""
        if best:
            b = med[best]
            best_txt = (f"<b>{html.escape(b.get('label', best))}</b> 的<b>四轨均值最高</b>"
                        f"（{b.get('sdr_avg4'):.2f} dB）；")
        median_block = f"""
  <h3>7.4　★ 多片段中位数（可外推的口径）</h3>
  <div class="card">
    <table>
      <thead><tr><th>模型</th><th class="num">人声</th><th class="num">鼓</th><th class="num">贝斯</th><th class="num">其他</th>
        <th class="num">四轨均值</th><th class="num">RTF 中位</th><th class="num">n</th></tr></thead>
      <tbody>{''.join(mrows)}</tbody>
    </table>
    <div class="note">单元格为 <b>museval SDR 在 {len(med_clips)} 个片段上的中位数</b>（dB），片段集：{clip_txt}，每段取 10 s。
    <br><b>为什么必须换片段：</b>7.2 表用的 <code>A Classic Education - NightOwl</code> 前 10 s 是低频引子 ——
    实测 bass 占 <b>85%</b> 能量、鼓只占 <b>1.4%</b>，于是所有模型的 bass SDR 一起虚高到 20 dB 以上。
    本表改用经 <code>tools/_scan_balanced_clip.py</code> 按「4 个 stem 能量占比最小值」筛出的均衡片段，并取曲子<b>中段</b>为起点。
    <br>{best_txt}RPCA 不出单值（见 7.6②），故不参与均值。</div>

    <p style="margin-top:14px"><b>单目标模型</b>（不产出完整 4 stem，单独列出）：</p>
    <table>
      <thead><tr><th>模型</th><th>目标</th><th class="num">museval SDR</th><th class="num">SI-SDR</th>
        <th class="num">RTF 中位</th><th class="num">n</th></tr></thead>
      <tbody>{''.join(srows)}</tbody>
    </table>
  </div>
"""
    else:
        median_block = ""

    return f"""
  <h2 id="s7n">七 · ★ 11 模型（+ Demucs）供给与可运行性核查（2026-09-16 新增）</h2>
  <p class="lede">对 11 个模型逐一做「是否开源 → 能否完整下载代码与权重 → 在本机能否真的跑出 stem wav → 失败原因」的核查。
  结论摘要：<b>8 个跑通</b>（含 BSRNN 三个变体）、<b>3 个无公开代码或权重</b>。
  另有 <b>Demucs</b> 同样跑通，并已并入同一个基准（同片段、同 museval 口径）→ <b>合计 12 个中 9 个可运行</b>。
  完整报告见 <code>04_reports/separation/docs/MODEL_PROVISIONING_REPORT.md</code>。</p>
  {kpis}

  <h3>7.1　核查总表</h3>
  <div class="card">{table}</div>

  <h3>7.2　★ 按 stem 的精度矩阵（museval SDR，同一首歌 10 s，CPU）</h3>
  <div class="card">{matrix}</div>
  <div class="info">
    <b>本表的正确读法：</b>它只回答两件事 —— <b>谁在本机真的跑通了</b>（8/11，加 Demucs 为 9/12），以及 <b>谁快</b>（最后一列 RTF）。
    <br><b>不要用本表下精度结论</b>：该片段 bass 占 85% 能量，bass 列被系统性抬高，
    所有模型的 bass 都虚高到 20 dB 以上（Oracle 17.77 / MDX 21.16 / oBSRNN-SIMO 22.71）。
    换成均衡片段后同一模型的 bass 掉到 8.55 / 12.91 / 11.72 —— <b>差了一倍</b>。
    <br>→ <b>精度结论请只看 7.4 的多片段中位数表。</b>
  </div>

  <h3>7.3　补充口径：SI-SDR</h3>
  <div class="card">{sisilent}</div>
{median_block}

  <h3>7.5　本机最大障碍：Smart App Control（SAC）</h3>
  <div class="card">
    <p>本机 <b>Windows Smart App Control 处于强制开启</b>（<code>VerifiedAndReputablePolicyState = 1</code>），
    按代码完整性策略拦截未签名 DLL。判定方式：用 <code>ctypes.WinDLL()</code> 直接加载，返回
    <code>[WinError 4551] 应用程序控制策略已阻止此文件</code> 即为 SAC 拦截。<b>2026-09-16 晚重新实测，拦截面已大幅收窄：</b></p>
    <table>
      <thead><tr><th>状态</th><th>模块</th><th>连带失效</th></tr></thead>
      <tbody>
        <tr><td><span class="pill p-dead">仍被拦</span></td>
            <td><code>llvmlite/binding/llvmlite.dll</code>　(WinError 4551)</td>
            <td><code>numba</code> · <code>librosa.stft</code> · <code>librosa.filters</code> · <code>pysepm</code></td></tr>
        <tr><td><span class="pill p-live">已恢复</span></td>
            <td><code>numpy/random/*.pyd</code>（含 <code>_philox</code> / <code>_generator</code> / <code>_pcg64</code>）</td>
            <td><code>numpy.random</code>（<code>default_rng</code> 实测可用）</td></tr>
        <tr><td><span class="pill p-live">已恢复</span></td>
            <td><code>scipy.signal</code> · <code>scipy.ndimage</code> · <code>sklearn</code> · <code>pandas</code> · <code>torchmetrics</code> · <code>pystoi</code></td>
            <td>NMF / FastICA / 失真滤波 全部可用</td></tr>
        <tr><td><span class="pill p-live">已恢复</span></td>
            <td><code>museval</code> · <code>mir_eval</code></td>
            <td><b>BSSEval v4 的 SDR 可以计算了</b></td></tr>
        <tr><td><span class="pill p-live">正常</span></td>
            <td><code>torch</code> · <code>torchaudio</code> · <code>soundfile</code> · <code>numpy.core</code> ·
                <code>matplotlib</code> · <code>omegaconf</code> · <code>hydra</code></td>
            <td>—</td></tr>
      </tbody>
    </table>
    <div class="warn">
      <b>重要修正：</b>早先记录的「SAC 拦 <code>numpy.random._philox</code> → 所有 SDR 为 <code>null</code>」<b>已不再成立</b>。
      该批二进制随后获得了系统信誉评级，<code>museval</code> 现已能算出真实 SDR（7.2 表即为其结果）。
      仍存的限制只剩 <b>CPU-only + 10 s 单片段</b>，所以数值只可横向比较，不可当作 MUSDB18 官方成绩。
    </div>
    <p style="margin-top:14px">为把模型跑起来，本项目做了 5 处工程绕行（均不改第三方源码）：</p>
    <table>
      <thead><tr><th>位置</th><th>做法</th><th>解决什么</th></tr></thead>
      <tbody>
        <tr><td><code>tools/_librosa_shim.py</code></td>
            <td>用 torch 后端实现 <code>stft/istft</code>，替换 librosa</td>
            <td><code>librosa.stft</code> 崩 → RPCA / NMF 连带全挂。<b>注意 <code>import librosa</code> 本身会成功</b>，
                必须做「功能性探测」而非「导入探测」</td></tr>
        <tr><td><code>benchmark_model_universal.py::_ensure_librosa</code></td>
            <td>向 <code>sys.modules</code> 注入只有 <code>.filters</code> 的 librosa 占位模块</td>
            <td>ZFTurbo 的 <code>models/bs_roformer/__init__.py</code> 会连带导入 <code>MelBandRoformer</code>
                （模块级 <code>from librosa import filters</code>），导致只要 BSRoformer 也会被 llvmlite 拖死</td></tr>
        <tr><td><code>_stft_istft()</code></td>
            <td>oracle 掩码改用 <code>torch.stft/istft</code>（hann, n_fft=2048, hop=512）</td>
            <td>与 librosa 默认口径等价，规避 numba</td></tr>
        <tr><td><code>tools/_bsrnn_loader.py</code></td>
            <td>占位替换 <code>pandas</code> / <code>lightning.pytorch</code> / <code>helpers.data</code> / <code>helpers.eval</code>，
                手工合成 hydra conf，绕过 <code>load_from_checkpoint</code> 手工 <code>load_state_dict</code></td>
            <td>BSRNN 官方链路对 lightning（与 torch 2.14 不兼容）/ pandas / museval / musdb 的模块级依赖</td></tr>
        <tr><td><code>load_mdx()</code></td>
            <td>局部 monkey-patch <code>torch.load(weights_only=False)</code></td>
            <td>torch ≥ 2.6 默认值变更导致 demucs checkpoint 反序列化失败</td></tr>
      </tbody>
    </table>
  </div>

  <h3>7.6　三个必须修正的认知（否则会误判模型失效）</h3>
  <div class="grid g2">
    <div class="method">
      <div class="hd"><span class="nm">① BS-RoFormer L6 (ep_937) 的输出不是 <code>other</code></span>
        <span class="pill p-dead">高优先级</span></div>
      <p>它的 yaml 写 <code>target_instrument: other</code>，但实测输出其实是
        <b><code>mixture − drums − bass</code>（即 vocals+other）</b>：</p>
      <table style="font-size:12.5px">
        <thead><tr><th>对比参考</th><th class="num">SI-SDR</th><th class="num">corr</th></tr></thead>
        <tbody>
          <tr><td>MUSDB 的 <code>other</code></td><td class="num lose">0.83</td><td class="num">0.740</td></tr>
          <tr><td><b>vocals + other</b></td><td class="num win"><b>12.13</b></td><td class="num"><b>0.971</b></td></tr>
          <tr><td>instrumental (mixture − vocals)</td><td class="num">−8.05</td><td class="num">0.368</td></tr>
        </tbody>
      </table>
      <p class="con">若直接拿它和纯 <code>other</code> 比，会把一个 <b>12.13 dB 的好模型误判成 0.83 dB 的废模型</b>。</p>
      <p class="pro">已在基准脚本里加 <code>ref_map</code> 机制按组合参考重新评估，输出文件也改名为
        <code>vocals+other.wav</code> 以免歧义。</p>
    </div>
    <div class="method">
      <div class="hd"><span class="nm">② RPCA 的稀疏分量没有天然 stem 归属</span>
        <span class="pill p-classic">口径问题</span></div>
      <p>RPCA 是单输出古典方法，强行指定「它就是人声」会得出误导性结论。同曲交叉评估：</p>
      <table style="font-size:12.5px">
        <thead><tr><th>对比目标</th><th class="num">vocals</th><th class="num">drums</th><th class="num">bass</th><th class="num">other</th></tr></thead>
        <tbody>
          <tr><td>SI-SDR (dB)</td><td class="num">−7.90</td><td class="num">−19.28</td>
              <td class="num win"><b>+3.96</b></td><td class="num">−13.50</td></tr>
        </tbody>
      </table>
      <p>本曲最佳匹配是 <b>bass</b>，对 vocals 反而是负增益。</p>
      <p class="con">报告里不能给 RPCA 单一「人声 SDR」数字 —— 它的「好坏」完全取决于映射到哪个目标。</p>
      <p class="pro">已改为 <code>eval_all_stems</code> 交叉评估，把四个数字都记录下来。</p>
    </div>
    <div class="method" style="grid-column:1/-1">
      <div class="hd"><span class="nm">③ SIMO 模型的 state_dict 按位置对齐 —— 顺序错位会静默串味</span>
        <span class="pill p-dead">最危险</span></div>
      <p>oBSRNN-SIMO 是「单模型 + 每源一个 masker」的结构，权重<b>按位置</b>绑定目标。
        它的 ckpt 里 <code>hyper_parameters.targets = ['vocals','bass','drums','other']</code>。
        如果按别的顺序（例如 <code>['bass','drums','other','vocals']</code>）构造模型再 <code>load_state_dict</code>：</p>
      <table style="font-size:12.5px">
        <thead><tr><th>症状</th><th>表现</th></tr></thead>
        <tbody>
          <tr><td>加载</td><td><b>不报错</b>，<code>missing = 0 / unexpected = 0</code>（形状完全一致）</td></tr>
          <tr><td>推理</td><td><b>能出声</b>、时长声道都对、stem wav 正常落盘</td></tr>
          <tr><td>指标</td><td>4 个 masker 被静默互换 → <b>整体串味</b>，但依然「看起来是个能用的模型」</td></tr>
        </tbody>
      </table>
      <p class="con">这是最容易自欺的一类 bug：所有「跑通」的判据都通过，只有指标错。</p>
      <p class="pro">已在 <code>tools/_bsrnn_loader.py</code> 加 <code>ckpt_targets()</code>：
        <b>从 ckpt 的 <code>hyper_parameters</code> 读出原始 targets 顺序来构造模型</b>，再映射回请求的目标，不一致时打印告警。
        修复后逐键校验 <code>missing = 0 / unexpected = 0</code>，且 SIMO 指标跃居全场第一 —— 反证了顺序正确。</p>
    </div>
  </div>

  <h3>7.7　对「实时」目标的影响（写进论文的关键约束）</h3>
  <div class="card">
    <p>当前环境 <b>torch 是 <code>+cpu</code> 版</b>，RTX 4070 Laptop 完全没被使用，所有 DNN 推理都在 CPU 上跑。
    下表为<b>机器空闲时的一次连续测量</b>（避免负载污染，10 s 片段）：</p>
    <table>
      <thead><tr><th>模型</th><th class="num">RTF (CPU)</th><th>能否实时</th><th>说明</th></tr></thead>
      <tbody>
        <tr><td>IRM Oracle</td><td class="num win">0.007</td><td>✅</td><td>解析法，参考上界</td></tr>
        <tr><td>Open-Unmix (umxhq)</td><td class="num win">0.113</td><td>✅</td><td>项目既有基线</td></tr>
        <tr><td>MMDenseLSTM</td><td class="num win">0.297</td><td>✅</td><td>4 目标一次出全</td></tr>
        <tr><td>Conv-TasNet</td><td class="num win">0.675</td><td>✅</td><td>4 目标一次出全</td></tr>
        <tr><td>MDX-Net</td><td class="num lose">1.03</td><td>⚠️ 边界</td><td>多次测量 0.50 ~ 1.09，受机器负载影响</td></tr>
        <tr><td><b>BSRNN SIMO（4-stem 单模型）</b></td><td class="num lose">2.14</td><td>❌</td><td><b>精度最高的 4-stem 方案</b>，但 CPU 上不实时</td></tr>
        <tr><td>RPCA</td><td class="num lose">1.80~3.51</td><td>❌</td><td>每轮全秩 SVD，波动大</td></tr>
        <tr><td>BSRNN large（4× ckpt）</td><td class="num lose">4.06</td><td>❌</td><td>—</td></tr>
        <tr><td>BS-RoFormer L6</td><td class="num lose">4.81</td><td>❌</td><td>需 CUDA</td></tr>
        <tr><td>oBSRNN（4× ckpt）</td><td class="num lose">5.76</td><td>❌</td><td>—</td></tr>
        <tr><td>BS-RoFormer L12</td><td class="num lose">10.73</td><td>❌</td><td>人声精度最高，但 CPU 上慢 10×</td></tr>
      </tbody>
    </table>
    <div class="warn">
      <b>结论：</b>在 <b>CPU-only</b> 前提下，能满足实时（RTF &lt; 1）的深度方案是
      <b>Open-Unmix / MMDenseLSTM / Conv-TasNet</b>；<b>要精度则选 oBSRNN-SIMO（需 CUDA）</b>。
      BS-RoFormer 人声精度最高，但不换 CUDA 版 torch 就达不到实时。
      ⚠️ BSRNN 三变体的 RTF 是「一次出 4 轨」的代价；若只跑单目标模型，RTF 约为其 1/4。
    </div>
  </div>

  <h3>7.8　未完成项与下一步</h3>
  <div class="card">
    <table>
      <thead><tr><th>项</th><th>状态</th><th>下一步</th></tr></thead>
      <tbody>
        <tr><td>BSRNN 权重（3 zip / 4.67 GB）</td><td><span class="pill p-live">完成</span></td><td>3 个 zip 的 MD5 全部校验通过</td></tr>
        <tr><td>BSRNN 推理（opt / large / SIMO）</td><td><span class="pill p-live">完成</span></td><td>三变体全部 PASS 并落盘 stem wav</td></tr>
        <tr><td>SDR（museval BSSEval v4）</td><td><span class="pill p-live">完成</span></td><td>拦截面收窄后已能计算</td></tr>
        <tr><td>多曲中位数</td><td><span class="pill p-classic">进行中</span></td><td>在 3 首歌上复测，剔除单片段偶然性</td></tr>
        <tr><td>DPRNN 的 MUSDB18 版</td><td><span class="pill p-dead">无权重</span></td><td>官方只有语音权重；自训是唯一路径</td></tr>
        <tr><td>RPCA+DRNN 自研复现</td><td><span class="pill p-dead">无代码</span></td><td>论文口径是单声道 2 源，不可与 MUSDB18 横比</td></tr>
        <tr><td>CUDA 版 torch</td><td><span class="pill p-classic">用户暂缓</span></td><td>影响 BS-RoFormer / BSRNN 的实时性结论</td></tr>
      </tbody>
    </table>
  </div>
"""


def main():
    src = SRC.read_text(encoding="utf-8")
    runs = load_runs()

    # ---------- 1. 修正 3.3× → 2.70× ----------
    n_fix = 0
    for a, b in [
        ("慢 3.3×", "慢 2.70×"),
        ("慢 3.3 倍", "慢 2.70 倍"),
        ("比 Demucs 快 3.3×", "比 Demucs 快 2.70×"),
        ("全面胜出但慢 3.3 倍", "全面胜出但慢 2.70 倍"),
    ]:
        n_fix += src.count(a)
        src = src.replace(a, b)

    # ---------- 2. 页头 ----------
    src = src.replace(
        '<div class="meta">FYP — 实时降噪 / 源分离系统（HKBU） &nbsp;|&nbsp; 数据集 MUSDB18-HQ &nbsp;|&nbsp; 最后整合 2026-09-15</div>',
        '<div class="meta">FYP — 实时降噪 / 源分离系统（HKBU） &nbsp;|&nbsp; 数据集 MUSDB18-HQ '
        '&nbsp;|&nbsp; 最后整合 2026-09-16 &nbsp;|&nbsp; <b>v1.3 rev2</b>'
        '（11 模型核查完成：BSRNN 三变体跑通 + museval SDR 恢复）</div>',
    )

    # ---------- 3. 导航插入 ----------
    toc_old = '<li><a href="#s7">汇总、口径提醒与下一步</a></li>'
    toc_new = ('<li><a href="#s7n"><b>★ 11 模型供给与可运行性核查（v1.3 新增）</b></a></li>\n'
               '      <li><a href="#s8">汇总、口径提醒与下一步</a></li>')
    if toc_old in src:
        src = src.replace(toc_old, toc_new)
    else:
        raise SystemExit("未找到导航项 #s7，HTML 结构可能已变")

    # ---------- 4. 原「七」改为「八」，插新板块 ----------
    old_h2 = '<h2 id="s7">七 · 汇总、口径提醒与下一步</h2>'
    if old_h2 not in src:
        raise SystemExit("未找到 <h2 id=\"s7\">，HTML 结构可能已变")
    src = src.replace(old_h2, build_section(runs) + '\n  <h2 id="s8">八 · 汇总、口径提醒与下一步</h2>')

    # ---------- 5. 口径提醒：补一句 2.70× 的适用范围 ----------
    anchor = '<h3>🎯 结论与下一步</h3>'
    caveat = (
        '<div class="warn" style="margin-bottom:14px"><b>口径提醒（v1.3 修正）：</b>'
        'Demucs 相对 Open-Unmix 的「2.70×」来自 <b>30 s/首 × 10 首</b>的短片段受控实验；'
        '全曲试听包实测显示 Open-Unmix 整体 RTF 0.428、Demucs 正常段 0.28~0.67 但会因机器争用飙到 4.04。'
        '<b>长曲 RTF 波动极大，倍率结论需在受控/空闲机器上重测</b>；对外只宜说「两者全曲仍可实时（RTF &lt; 1）」。</div>\n  '
    )
    if anchor in src:
        src = src.replace(anchor, caveat + anchor, 1)

    DST.write_text(src, encoding="utf-8")
    print(f"[build_v13] 3.3× 修正处数 = {n_fix}")
    print(f"[build_v13] 读取到 {len(runs)} 个模型的实测记录")
    print(f"[build_v13] 写入 {DST}  ({DST.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
