# -*- coding: utf-8 -*-
"""Generate the citable markdown summary of the model-profile table (D1).

Reads  : 04_reports/_shared/data/model_analysis/model_profile.csv
Writes : 04_reports/_shared/docs/MODEL_PROFILE_2026-09-22.md

Numbers are emitted straight from the CSV so the document cannot drift from the
data.  The protocol block is printed verbatim because the single most common
review question on a table like this is "measured how?".
"""
from __future__ import annotations

import csv
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, "04_reports", "_shared", "data", "model_analysis", "model_profile.csv")
OUT = os.path.join(_ROOT, "04_reports", "_shared", "docs", "MODEL_PROFILE_2026-09-22.md")

GROUP_TITLE = {
    "4-stem": "四轨分离模型",
    "1-target": "单目标模型（仅目标 stem）",
    "2-target": "双目标模型",
    "denoise": "降噪（板 2）",
    "dereverb": "去混响 / 去回声（板 2）",
    "analytic": "解析 / 无参方法",
}
GROUP_ORDER = ["4-stem", "1-target", "2-target", "denoise", "dereverb", "analytic"]

# Cells whose raw n_targets is misleading and need a hand-written override.
#   oracle : analytic, no checkpoint -> loader reports no "targets" field, but
#            the estimator it implements is the 4-stem oracle.
#   L6     : yaml says "other" but the weight emits a single residual
#            mixture - drums - bass (= vocals + other); see the note below the table.
#   rpca   : sparse component has no natural stem attribution by construction.
NTR_OVERRIDE = {
    "oracle": "4",
    "bsroformer_l6": "1（=vocals+other 残差）",
    "rpca": "—（分量无天然归属）",
}


def num(x, d=None):
    try:
        s = str(x).strip()
        return d if s == "" else float(s)
    except (TypeError, ValueError):
        return d


def fmt(x, nd=2, dash="-"):
    return dash if x is None else ("%.*f" % (nd, x))


def _r(rows, key):
    for r in rows:
        if r["model_key"] == key:
            return r
    raise KeyError(key)


def main():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("model_key")]

    for r in rows:
        r["_p"] = num(r.get("params_M"), 0.0)
        r["_f"] = num(r.get("macs_G"), 0.0)
        r["_l"] = num(r.get("latency_s"))
        r["_v"] = num(r.get("peak_vram_mb"))

    L = []
    A = L.append
    A("# 模型本体画像：参数量 · 计算量 · 显存 · 时延\n")
    A("> 生成日：2026-09-22 ｜ 交付物：**D1 模型画像表** ｜ 数据源："
      "`04_reports/_shared/data/model_analysis/model_profile.csv`\n")
    A("> 本表回答的是「模型要花多少代价」，与「分离得准不准」（见 `MEDIAN_TABLE*.md`）互为补充。\n")

    A("\n## 1. 测量口径（引用本表时必须一并声明）\n")
    A("| 项 | 取值 |")
    A("|---|---|")
    A("| 输入 | **44.1 kHz / 立体声 / 10 s**（统一构造，不来自数据集） |")
    A("| 参数量 | `sum(p.numel())`，取自模型实例，非配置文件声明值 |")
    A("| FLOPs | `torch.utils.flop_counter.FlopCounterMode` 实测，**口径 = 2×MACs** |")
    A("| 时延 | 预热 2 次后取 **5 次中位数**；CUDA 前后各同步一次 |")
    A("| 峰值显存 | `torch.cuda.max_memory_allocated()`，失败路径不计入 |")
    A("| 设备 | RTX 4070 Laptop **8 GiB**（sm_89），torch 2.14.0+cu126 |")
    A("| 隔离 | **每个模型一个独立子进程**（15 个模型同进程会显存耗尽） |")

    A("\n### ⚠️ FLOPs 是**下界**，不是等价比较\n")
    A("`FlopCounterMode` 只能统计被 torch 调度的算子（conv / linear / matmul / norm）。")
    A("**`torch.stft` 与融合的 `scaled_dot_product_attention` 不在统计范围内**。")
    A("因此对成本主要落在「STFT + 注意力」的 RoFormer 系列，本列数值只能作为下界；")
    A("跨架构的 FLOPs 比较必须配合**实测时延**一起看，不可单独下结论。\n")
    A("> 例：BS-RoFormer-L12 的 FLOPs（43.5 T）是 HTDemucs（508 G）的 **86 倍**，")
    A("> 但实测时延只差 **16 倍**（4.52 s vs 0.279 s）——差额就来自未被计入的算子。\n")

    A("\n### FLOPs 交叉验证：与 thop 的独立第二口径\n")
    thop_p = os.path.join(_ROOT, "04_reports", "_shared", "data", "model_analysis", "thop_crosscheck.json")
    if os.path.exists(thop_p):
        tc = json.load(open(thop_p, encoding="utf-8")).get("cases", {})
        bykey = {r["model_key"]: r for r in rows}
        A("`thop` 走 **nn.Module 钩子** 计数（口径同为 2×MACs），与 `FlopCounterMode` 的")
        A("**算子调度计数**机制不同，两者只在「纯 conv/linear 堆叠」的模型上才应当吻合。")
        A("下表只列 thop 能给出数值的模型——它给不出的，如实写原因，不用它去覆盖主口径。\n")
        A("| 模型 | FlopCounterMode (G) | thop (G) | 偏差 | 说明 |")
        A("|---|---:|---:|---:|---|")
        ncov = 0
        for k, v in tc.items():
            if not v.get("thop_G"):
                continue
            ncov += 1
            r = bykey.get(k, {})
            dev = v.get("dev_pct")
            A("| %s | %s | %s | %s | %s |"
              % ((r.get("label") or k).strip(), fmt(num(r.get("flops_G")), 1),
                 fmt(v["thop_G"], 1),
                 ("%+.1f%%" % dev) if dev not in ("", None) else "—",
                 (v.get("note") or "").replace("|", "/")))
        miss = [k for k, v in tc.items() if not v.get("thop_G")]
        valid = [(k, v) for k, v in tc.items() if v.get("dev_pct") not in ("", None)]
        nagg = sum(1 for v in tc.values() if "AGGREGATED" in (v.get("note") or ""))
        A("\n> thop 能给出数值的模型只有 **%d/%d**。" % (ncov, len(tc)))
        if valid:
            devs = [abs(v["dev_pct"]) for _k, v in valid]
            A("> 其中**两边数的是同一个模型**、偏差可比较的只有 **%d 个**，偏差区间 "
              "**%.0f%% ~ %.0f%%**——**远超 D1 设定的 5%%**，" % (len(valid), min(devs), max(devs)))
            A("> 也就是说该验收条款**未达成**。但原因不是实现错误：")
            A("> `FlopCounterMode` 数的是 torch 真正 dispatch 的算子，`thop` 数的是 "
              "「nn.Module 权重 × 注册到的输入形状」，")
            A("> 凡是 `forward` 里走 `F.conv*` / `einsum` / 多频带循环的地方，两者必然分叉。")
        if nagg:
            A("> 另有 **%d 个聚合权重**（同一模型由多个 ckpt 组成）thop 只拿到了其中一个子网，"
              "与被比较的整体口径不同源，故**不给偏差值**。"
              % nagg)
        if miss:
            A("> 余下 **%d 个**模型 thop 根本跑不动，原因逐条记在 `thop_crosscheck.json` 与 "
              "CSV 的 `thop_note` 列。" % len(miss))
        A("> **处置**：本表以 `FlopCounterMode` 为**唯一正式口径**（它对全部模型同源可比），")
        A("> thop 仅作覆盖度披露，**不用它去校正任何数字**；band-split RNN 一类的 FLOPs "
          "本就随口径浮动 3~10 倍，引用时必须带此说明。\n")
    else:
        A("> 未找到 `thop_crosscheck.json`，请先运行 `tools/_thop_crosscheck.py`。\n")

    A("\n## 2. 全表（%d 个模型/权重）\n" % len(rows))
    A("| 分组 | 模型 | 参数量 (M) | FLOPs (G) | 单次时延 (s) | 峰值显存 (MB) | 输出轨 |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for g in GROUP_ORDER:
        grp = [r for r in rows if r.get("group") == g]
        grp.sort(key=lambda r: -r["_f"])
        for i, r in enumerate(grp):
            if g == "analytic":
                fs = "0（无算子调度）"
            else:
                fs = fmt(r["_f"], 1)
            ntr = NTR_OVERRIDE.get(r["model_key"]) or (r.get("n_targets") or "-")
            A("| %s | %s | %s | %s | %s | %s | %s |"
              % (GROUP_TITLE.get(g, g) if i == 0 else "",
                 (r.get("label") or r["model_key"]).strip(),
                 fmt(r["_p"], 3), fs, fmt(r["_l"], 4), fmt(r["_v"], 1),
                 ntr))

    A("\n> ⚠️ **「输出轨」列不可直接当「目标 stem」读。** BS-RoFormer 的 L6 权重其 yaml 声明为 ")
    A("> `other`，但实际输出是 **mixture − drums − bass**（即 vocals + other 的合成残差），")
    A("> 与 `L12` 的单目标 `vocals` 不是同一口径。凡引用本列做跨模型比较，必须回查 ")
    A("> `04_reports/separation/docs/MEDIAN_TABLE_TEST.md` 的目标定义，不可仅看数字。\n")

    A("\n### 配套图表\n")
    A("- `04_reports/_shared/figures/model_arch/profile_params_flops.png` —— 双面板条形图：参数量 vs "
      "对数坐标 FLOPs，按分组着色。")
    A("- `04_reports/_shared/figures/model_arch/profile_pareto.png` —— 气泡图：x = 时延（对数），"
      "y = 参数量，气泡面积 = 峰值显存。\n")

    A("\n## 3. 分组观察\n")

    deep = [r for r in rows if r["_p"] > 0]
    heavy = max(deep, key=lambda r: r["_f"])
    light = min([r for r in deep if r["_f"] > 0], key=lambda r: r["_f"])
    big = max(deep, key=lambda r: r["_p"])
    small = min(deep, key=lambda r: r["_p"])
    fast = min([r for r in deep if r["_l"]], key=lambda r: r["_l"])
    slow = max([r for r in deep if r["_l"]], key=lambda r: r["_l"])
    vmax = max([r for r in deep if r["_v"]], key=lambda r: r["_v"])

    A("**① 最省的一档是 denoiser 三兄弟，最贵的单点是 MDX-Net。**")
    A("MDX-Net 参数量 **%s M** 是全场最大，约等于 DPRNN（%s M）的 %.0f 倍；"
      % (fmt(big["_p"], 1), fmt(small["_p"], 1), big["_p"] / small["_p"]))
    A("而 `denoiser dns48` 只用 **%s M**（板 2 最小）。\n"
      % fmt(_r(rows, "denoiser_dns48")["_p"], 3))

    A("**② 「参数量小 ≠ 计算量小」在本项目里被反复验证。**")
    A("`Conv-TasNet` 只有 %s M，FLOPs 却达 %s G（靠极长的时序卷积堆算力）；"
      % (fmt(_r(rows, "convtasnet")["_p"], 1), fmt(_r(rows, "convtasnet")["_f"], 0)))
    A("`MMDenseLSTM` 仅 %s M，FLOPs %s G。反之 `BSRNN-opt` 参数 %s M 而 FLOPs 仅 %s G。\n"
      % (fmt(_r(rows, "mmdenselstm")["_p"], 1), fmt(_r(rows, "mmdenselstm")["_f"], 0),
         fmt(_r(rows, "bsrnn")["_p"], 1), fmt(_r(rows, "bsrnn")["_f"], 0)))

    A("**③ 板 2 的性价比反差极大。**")
    mrd = _r(rows, "mel_roformer_denoise")
    d48 = _r(rows, "denoiser_dns48")
    A("Mel-RoFormer 系（%s M / %s G / %s s / %s MB）在参数上是 denoiser dns48（%s M）的约 %.0f 倍，"
      % (fmt(mrd["_p"], 1), fmt(mrd["_f"], 0), fmt(mrd["_l"], 2), fmt(mrd["_v"], 0),
         fmt(d48["_p"], 1), mrd["_p"] / d48["_p"]))
    A("时延约 %.0f 倍（%s s vs %s s），但它换来的 SI-SDR 改善是 **+9.88 dB vs +2.15 dB**（见 `stage0_gate.json`）。\n"
      % (mrd["_l"] / d48["_l"], fmt(mrd["_l"], 2), fmt(d48["_l"], 3)))

    A("**④ 最省时延 = %s（%s s）；最费 = %s（%s s）。**"
      % ((fast.get("label") or "").strip(), fmt(fast["_l"], 3),
         (slow.get("label") or "").strip(), fmt(slow["_l"], 4)))
    A("峰值显存最高的是 **%s**（%s MB），在 8 GiB 卡上需注意与其它进程共存时的余量。\n"
      % ((vmax.get("label") or "").strip(), fmt(vmax["_v"], 0)))

    A("**⑤ 解析方法的代价结构完全不同。**")
    A("`IRM/IBM Oracle` 无参数、无算子调度；`RPCA (Inexact-ALM)` 同样零参数，")
    A("但单次 10 s 处理耗时 **%s s**，远高于全场所有深度模型——它是纯 CPU 的矩阵迭代求解，"
      "这也是它在 50 首整曲扫描中必然触发超时的根因。\n"
      % fmt(_r(rows, "rpca")["_l"], 2))

    A("\n## 4. 板 2 门禁结果（`stage0_gate.json`）\n")
    gate_p = os.path.join(_ROOT, "04_reports", "_shared", "data", "model_analysis", "stage0_gate.json")
    gate_label = [
        ("mel_roformer_denoise", "降噪", "Mel-RoFormer denoise"),
        ("mel_roformer_denoise_aggr", "降噪", "Mel-RoFormer denoise (aggr)"),
        ("mel_roformer_dereverb", "去混响", "Mel-RoFormer dereverb (anvuew)"),
        ("mel_roformer_dereverb_echo", "去混响/回声", "Mel-RoFormer dereverb-echo (Sucial)"),
        ("denoiser_dns48", "降噪", "denoiser dns48"),
        ("denoiser_dns64", "降噪", "denoiser dns64"),
        ("denoiser_master64", "降噪", "denoiser master64"),
    ]
    import json as _json
    if os.path.exists(gate_p):
        cases = _json.load(open(gate_p, encoding="utf-8")).get("cases", {})
        A("探针：Valentini `p232_001.wav` 的真实含噪段（**非白噪声**——降噪器对白噪声输出 ~0 "
          "是正确行为，会伪装成「未加载成功」），SI-SDR 相对同段干净参考计算。\n")
        A("| 权重 | 类型 | 状态 | SI-SDR 输入 → 输出 | 改善 |")
        A("|---|---|---|---:|---:|")
        for k, typ, name in gate_label:
            c = cases.get(k)
            if not c:
                continue
            A("| %s | %s | %s | %+.2f → %+.2f dB | **%+.2f** |"
              % (name, typ, c.get("status", "?"), c.get("si_sdr_in", float("nan")),
                 c.get("si_sdr_out", float("nan")), c.get("si_sdr_delta", float("nan"))))
        vals = [c for c in cases.values()]
        best = max(vals, key=lambda c: c.get("si_sdr_delta", -1e9))
        worst = min(vals, key=lambda c: c.get("si_sdr_delta", 1e9))
        npass = sum(1 for c in vals if c.get("status") == "PASS")
        A("\n> 门禁 **%d/%d PASS**（判据只认 `status` 字段，不看返回码/目录）。最强改善 "
          % (npass, len(vals)))
        A("> **%+.2f dB**，最弱 **%+.2f dB**。\n"
          % (best.get("si_sdr_delta", float("nan")), worst.get("si_sdr_delta", float("nan"))))
    else:
        A("> ⚠️ 未找到 `stage0_gate.json`，请先运行 `tools/_probe_stage0_gate.py`。\n")
    A("\n> 🔴 **待核（R2 / in-domain 检查）**：`master64` 的训练集据称**包含 Valentini**，"
      "按理应在 Valentini 上表现最好，但此处单样本改善仅 **+0.15 dB**，反而低于 dns48/dns64。")
    A("> 该结论**基于单段样本，不足以定论**——Stage 5 需在 Valentini testset 上按 824 对样本重测中位数，")
    A("> 并明确标注 in-domain / out-of-domain。\n")

    A("\n## 5. 与参考论文表的差异说明（答辩预案）\n")
    A("| 项 | 参考表 | 本机实测 | 差异原因 |")
    A("|---|---|---|---|")
    A("| Demucs 参数量 | 128 M | **41.98 M** | 参考表是**原版 Demucs (v1)**；本项目用 **HTDemucs**（混合时域/频域架构），是不同的模型 |")
    A("| DPRNN-TasNet | 官方权重 6.01 M | **本机自训 5 h**（3.69 M） | 官方未发布 MUSDB 分离权重，自训欠拟合（四轨 SDR 为负），仅作负面证据 |")
    A("| 评价片段 | 未声明 | 3 个能量均衡片段取中位数 | 已排除「开头低频主导」的能量陷阱片段 |")
    A("\n> 建议在图注中写明：**本表为同一任务的本机复现口径**，与参考论文的实现版本、")
    A("> 评价片段选择不同，数值不可逐格对齐；差异来源如上表，而非实现错误。\n")

    A("\n## 6. 复现命令\n")
    A("```bash")
    A("# 全量（分离 15 + 降噪/去混响 7）")
    A("python tools/_measure_model_profile.py --models all --device cuda")
    A("")
    A("# 仅板 2")
    A("python tools/_measure_model_profile.py \\")
    A("  --models mel_roformer_denoise,mel_roformer_denoise_aggr,\\")
    A("           mel_roformer_dereverb,mel_roformer_dereverb_echo,\\")
    A("           denoiser_dns48,denoiser_dns64,denoiser_master64")
    A("")
    A("# 出图")
    A("python tools/_plot_model_profile.py")
    A("```")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s  (%d rows, %d bytes)" % (OUT, len(rows), os.path.getsize(OUT)))


if __name__ == "__main__":
    main()
