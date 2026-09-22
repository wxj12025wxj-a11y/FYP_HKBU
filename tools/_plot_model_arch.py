# -*- coding: utf-8 -*-
"""Render the architecture explainer figures (T2.2 / deliverable D2).

Reads  : 04_reports/_shared/data/model_analysis/graph_<key>.json   (from _extract_model_graph.py)
Writes : 04_reports/_shared/figures/model_arch/arch_<family>.png  x14

Layout of every figure
----------------------
LEFT   concept flow: *how the method works*, stage by stage.  Hand-written from
       the source paper, but every shape annotation printed on a stage is RESOLVED
       FROM THE TRACE by `_resolve_shape()` -- nothing is typed in by hand, so it
       is traceable back to `graph_<key>.json` (D2 acceptance criterion).
RIGHT  real trace, data-driven:
       (a) 100 % stacked bar of parameter composition.  The values are
           `params_attr`, i.e. each distinct parameter tensor credited to the
           deepest module that fired -- guaranteed to sum to the checkpoint total,
           which is why the shares add up to exactly 100 %.
       (b) the structural skeleton actually executed (class, in -> out shapes,
           share, call counts).

Two families (`rpca`, `oracle`) are analytic and have no `nn.Module`; their
graphs are recorded with `status = NO_MODULE`.  Those figures deliberately print
NO shapes anywhere and say so on the canvas, rather than inventing plausible ones.
"""
from __future__ import annotations

import glob
import json
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "04_reports", "_shared", "data", "model_analysis")
FIG = os.path.join(_ROOT, "04_reports", "_shared", "figures", "model_arch")
os.makedirs(FIG, exist_ok=True)

INK = "#111827"
MUTED = "#6b7280"
ACC = "#4f46e5"

# ---------------------------------------------------------------------------
# Palette for parameter-composition groups (stable per family)
# ---------------------------------------------------------------------------
C = {
    "stem": "#2563eb", "core": "#4f46e5", "attn": "#7c3aed", "rnn": "#0891b2",
    "encdec": "#0d9488", "band": "#d97706", "embed": "#db2777",
    "mask": "#16a34a", "aux": "#9ca3af", "trunk": "#f59e0b",
}


# ---------------------------------------------------------------------------
# Family specs.  14 families cover all 22 model/weight entries.
#   keys      : graph_<key>.json files this figure speaks for
#   primary   : the trace drawn in the right panel (bar + skeleton)
#   groups    : ordered prefix rules for the parameter-composition bar
#   stages    : concept flow;  "find" -> (path token, class token or None)
#               resolves the shape printed on that stage from the trace
# ---------------------------------------------------------------------------
def F(title, subtitle, keys, primary, citation, code, weights, groups, stages, notes=()):
    return dict(title=title, subtitle=subtitle, keys=keys, primary=primary,
                citation=citation, code=code, weights=weights, groups=groups,
                stages=stages, notes=list(notes))


FAMILIES = [
    # ---------------------------------------------------------------- board 1
    F("IRM / IBM Oracle", "解析上界：不用任何学习参数，直接由真值算理想掩码",
      ["oracle"], "oracle",
      "Rafii, Liutkus, Stöter, Mimilakis, Bittner. MUSDB18 — a corpus for music "
      "separation. 2017（sigsep-mus-oracle）；IRM 定义见 Wang & Wang, "
      "IEEE/ACM TASLP 2014",
      "01_models/_third_party/sigsep-mus-oracle + tools/compare_separation_methods.py::sep_oracle",
      [], [],
      [dict(n="1", name="混合信号", desc="直接取 MUSDB18 的 mixture.wav（44.1 kHz 立体声），不做任何预处理。",
            find=None, fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="STFT", desc="短时傅里叶变换，把波形搬到时频域，得到复数谱 X(t,f)。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="3", name="理想比值掩码（IRM）", desc="用 GT 四轨与估计四轨的能量逐 bin 计算 "
            "M_k = |S_k| / Σ_j|S_j|；IBM 则退化为 0/1 硬分配。",
            find=None, fc="#ecfdf5", ec="#059669"),
       dict(n="4", name="掩码作用于混合谱", desc="Ŝ_k(t,f) = M_k(t,f) · X(t,f)，逐目标重建复数谱。",
            find=None, fc="#f5f3ff", ec="#7c3aed"),
       dict(n="5", name="iSTFT 与拼接", desc="逆变换回波形，四轨相加重建 ≈ 原混合（这是 oracle 的性质，不是模型的）。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="6", name="用途：理论上界参照", desc="给所有学习型模型一个「不可能超过」的天花板，本项目实测四轨均值 8.30 dB。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["本图为解析算法：`graph_oracle.json` 记录 `status=NO_MODULE`（loader 闭包内无 nn.Module），"
       "因此整张图**不含任何 shape 标注**——没有 trace 可溯源，就不写数字。",
       "参数量为 0 是真实的，不是缺失：Oracle 不训练、不存权重。"]),

    F("RPCA (+ Inexact ALM)", "古典矩阵分解：把人声当「稀疏」、伴奏当「低秩」",
      ["rpca"], "rpca",
      "Huang, Chen, Smaragdis, Hasegawa-Johnson. Singing-voice separation from "
      "monaural recordings using robust principal component analysis. ICASSP 2012；"
      "求解器：Lin, Chen, Ma. The Augmented Lagrange Multiplier Method for Exact "
      "Recovery of Corrupted Low-Rank Matrices. 2010",
      "tools/compare_separation_methods.py::sep_rpca（自研实现）",
      [], [],
      [dict(n="1", name="混合信号", desc="44.1 kHz 立体声；本实现按声道分别处理后再平均，避免立体声相位干扰。",
            find=None, fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="STFT → 幅度谱", desc="取复数谱的幅度，得到非负矩阵作为分解对象。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="3", name="构造待分解矩阵 M", desc="把幅度谱按时间帧-频点排成矩阵 M（帧 × 频点）。",
            find=None, fc="#f0fdf4", ec="#16a34a"),
       dict(n="4", name="Inexact ALM 交替迭代", desc="min ‖L‖_* + λ‖S‖₁ s.t. M = L + S："
            "SVD 收缩算子更新低秩项 L，软阈值更新稀疏项 S，循环至收敛。",
            find=None, fc="#fef2f2", ec="#dc2626"),
       dict(n="5", name="取稀疏分量 S 当人声", desc="低秩 L ≈ 平稳伴奏，稀疏 S ≈ 人声；用 S 构造掩码 M = S/(L+S)。",
            find=None, fc="#f5f3ff", ec="#7c3aed"),
       dict(n="6", name="iSTFT", desc="乘回原相位、逆变换回波形。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="7", name="[!] 无天然 4 轨归属", desc="S 只能给「人声 vs 伴奏」两个概念，"
            "无法分成鼓/贝斯/人声/其他。本项目因此只报区间 1.8~3.5 dB，不给单一数字。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["本图为古典算法：`graph_rpca.json` 记录 `status=NO_MODULE`，整张图**不含任何 shape 标注**。",
       "实测单次 10 s 耗时 82.53 s（纯 CPU 矩阵迭代，零参数），这也是它在 50 首整曲扫描里必然超时的根因。"]),

    F("Conv-TasNet", "端到端时域分离：完全绕开 STFT，用可学习的编码基",
      ["convtasnet"], "convtasnet",
      "Luo, Mesgarani. Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude "
      "Masking for Speech Separation. IEEE/ACM TASLP 27(8), 2019",
      "01_models/_third_party/Conv-TasNet + DNN-based_source_separation（MUSDB18 权重）",
      ["01_models/_weights/DNN-based_source_separation/ConvTasNet/**/best.pth"], [
       ("分离主干 separator（TCN）", C["core"], ["separator"]),
       ("编码器 encoder", C["encdec"], ["encoder"]),
       ("解码器 decoder", C["encdec"], ["decoder"])],
      [dict(n="1", name="波形输入", desc="直接吃原始波形（44.1 kHz 立体声），不做 STFT。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="1D 卷积编码器", desc="一组学习到的基函数，把波形切成重叠帧并投影到 N 维隐空间。",
            find=("encoder", "Conv1d"), fc="#eef2ff", ec=ACC),
       dict(n="3", name="分离模块（TCN）", desc="堆叠的时域卷积块：1×1 卷积 + 空洞深度可分离卷积，"
            "每块带全局层归一化与 PReLU，用不同膨胀率扩大感受野。",
            find=("separator", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="4", name="四目标掩码", desc="末尾 1×1 卷积同时给出四个目标的掩码，作用于编码器输出。",
            find=("mask", "Conv1d"), fc="#ecfdf5", ec="#059669"),
       dict(n="5", name="转置卷积解码器", desc="把每个目标的隐表示还原成波形。",
            find=("decoder", None), fc="#eef2ff", ec=ACC),
       dict(n="6", name="输出四轨", desc="每轨与混合等长，允许直接相加重建。",
            find=None, fc="#fff7ed", ec="#d97706")]),

    F("DPRNN（本项目自训）", "双路 RNN：把长序列拆成「块内 × 块间」两级建模",
      ["dprnn"], "dprnn",
      "Luo, Chen, Mesgarani. Dual-Path RNN: Efficient Long Sequence Modeling for "
      "Time-Domain Single-Channel Speech Separation. ICASSP 2020",
      "01_models/_third_party/Dual-Path-RNN-Pytorch（仅复用模型定义）"
      "+ tools/train_dprnn_musdb.py（自写 MUSDB18 数据管线）",
      ["01_models/_weights/dprnn_musdb/best.pt", "01_models/_weights/dprnn_musdb/last.pt"], [
       ("双路分离模块 separation", C["core"], ["separation"]),
       ("编码器 encoder", C["encdec"], ["encoder"]),
       ("解码器 decoder", C["encdec"], ["decoder"])],
      [dict(n="1", name="波形输入（11.025 kHz）", desc="为压显存降到 44.1k ÷ 4，用整数抽取避免重采样伪影。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="1D 卷积编码器", desc="64 通道、kernel 16，把波形投影到隐空间。",
            find=("encoder", None), fc="#eef2ff", ec=ACC),
       dict(n="3", name="分块（K=200, hop=100）", desc="把长序列切成重叠块，重排成「块 × 块内采样」二维张量——"
            "这是双路结构能处理长音频的关键。",
            find=None, fc="#f0fdf4", ec="#16a34a"),
       dict(n="4", name="双路块 ×6", desc="每块内先做块内 BiLSTM（局部），再做块间 BiLSTM（全局），"
            "交替堆叠 6 层，兼顾局部细节与长程依赖。",
            find=("separation", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="5", name="四目标掩码 + 重叠相加", desc="掩码作用于编码器输出，逆分块后重叠相加。",
            find=("decoder", None), fc="#ecfdf5", ec="#059669"),
       dict(n="6", name="输出四轨", desc="波形域四轨，与混合等长。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["[!] 权重为**本项目自训**（官方只发语音权重）：3.69 M 参数、step 31900、best 验证 SI-SDR +2.745。"
       "引用必须标注 ckpt step。",
       "测试集四轨均值 −1.72 dB（四轨全负）→ 只作「8 GiB 单卡 + 5 h 预算复现不出论文精度」的负面证据，"
       "**不参与最快/最准排名**。"]),

    F("MMDenseLSTM", "多尺度密集连接 + LSTM：频域幅度掩码路线",
      ["mmdenselstm"], "mmdenselstm",
      "Takahashi, Matsubara, Uehara. Multi-scale Multi-band DenseLSTM for Audio "
      "Source Separation. IEEE Signal Processing Letters 27, 2020",
      "01_models/_third_party/DNN-based_source_separation（官方 MUSDB18 paper 权重）",
      ["01_models/_weights/DNN-based_source_separation/MMDenseLSTM/musdb18/sr44100/paper/model/*/best.pth"],
      [("每目标 DenseNet+LSTM 主干 base_model.net.<t>", C["core"], ["base_model"])],
      [dict(n="1", name="波形输入", desc="44.1 kHz 立体声，先做 STFT。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="幅度谱", desc="只用幅度，相位沿用混合信号（这是该路线的核心近似）。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="3", name="多分辨率 DenseNet 分支", desc="不同下采样倍率的分支并行，各分支内部密集连接："
            "每层输入 = 之前所有层输出的拼接，用同一套特征反复复用。",
            find=("base_model", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="4", name="特征融合", desc="各分支输出上采样回同一分辨率后拼接。",
            find=None, fc="#f0fdf4", ec="#16a34a"),
       dict(n="5", name="LSTM 时序建模", desc="沿时间轴做双向 LSTM，补上卷积看不到的长程上下文。"
            "在本机实现的 trace 里没有独立的 LSTM 子模块（循环层写在自定义类内部），"
            "因此本级不标 shape。",
            find=None, fc="#ecfeff", ec="#0891b2"),
       dict(n="6", name="四目标掩码 + iSTFT", desc="每个目标一套网络，输出幅度掩码后乘回混合谱、逆变换成波形。",
            find=None, fc="#fff7ed", ec="#d97706")]),

    F("Open-Unmix (umxhq)", "参考实现：每个目标一个独立的「MLP + BiLSTM」小网络",
      ["umx"], "umx",
      "Stöter, Uhlich, Liutkus, Mitsufuji. Open-Unmix — A Reference Implementation "
      "for Music Source Separation. Journal of Open Source Software 4(41):1667, 2019",
      "01_models/_third_party/open-unmix-pytorch + _weights/umxhq",
      ["tools/_scratch/.cache/torch/hub/checkpoints/vocals-*.pth",
       "tools/_scratch/.cache/torch/hub/checkpoints/drums-*.pth",
       "tools/_scratch/.cache/torch/hub/checkpoints/bass-*.pth",
       "tools/_scratch/.cache/torch/hub/checkpoints/other-*.pth"],
      [("每目标独立网络 ×4", C["core"], ["target_models"]),
       ("STFT 与幅度谱", C["band"], ["stft", "complexnorm"]),
       ("iSTFT", C["encdec"], ["istft"])],
      [dict(n="1", name="混合信号", desc="44.1 kHz 立体声。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="STFT", desc="n_fft=4096，窗长与 hop 决定时频分辨率。",
            find=("stft", None), fc="#eef2ff", ec=ACC),
       dict(n="3", name="复数取模", desc="丢掉相位，只留幅度谱——这是 UMX 的建模对象。",
            find=("complexnorm", None), fc="#eef2ff", ec=ACC),
       dict(n="4", name="每目标独立网络 ×4", desc="fc1 升维 → BatchNorm + ReLU + Dropout → 双层 BiLSTM → "
            "fc2 → ReLU → fc3 降回 1024 维（2 通道 × 512 频点）。四个目标互不共享权重。",
            find=("target_models", "OpenUnmix"), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="5", name="掩码 × 幅度谱", desc="网络输出当作掩码，乘回混合幅度谱得到各目标幅度。",
            find=None, fc="#ecfdf5", ec="#059669"),
       dict(n="6", name="iSTFT（复用混合相位）", desc="逆变换回波形，四轨相加 ≈ 混合。",
            find=("istft", None), fc="#fff7ed", ec="#d97706")]),

    F("Demucs v4 / HTDemucs", "混合域架构：时域分支 + 频域分支 + 跨域 Transformer",
      ["demucs"], "demucs",
      "Rouard, Massa, Défossez. Hybrid Transformers for Music Source Separation. "
      "ICASSP 2023",
      "01_models/_third_party/demucs（官方 htdemucs 权重）",
      ["tools/_scratch/.cache/torch/hub/checkpoints/955717e8-*.th"],
      [("跨域 Transformer crosstransformer", C["attn"], ["crosstransformer"]),
       ("时域编码器 tencoder", C["encdec"], ["tencoder"]),
       ("频域编码器 encoder", C["band"], ["encoder"]),
       ("时域解码器 tdecoder", C["encdec"], ["tdecoder"]),
       ("频域解码器 decoder", C["band"], ["decoder"])],
      [dict(n="1", name="波形输入", desc="44.1 kHz 立体声，同时走时域与频域两条路。"
            "入口是 `BagOfModels.apply_model`，它自己的 forward 不被调用，"
            "trace 里没有 `<root>` 节点，故本级不标 shape。",
            find=None, fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="时域编码器", desc="Conv1d 逐层下采样（stride 4），通道数递增，把波形压成隐表示。",
            find=("models.0.tencoder", "Conv1d"), fc="#eef2ff", ec=ACC),
       dict(n="3", name="频域编码器", desc="先 STFT，再用卷积处理复数谱，得到与频带对应的隐表示"
            "（trace 中的 `models.0.encoder`，与 `tencoder` 是两条独立分支）。",
            find=("models.0.encoder", "Conv1d"), fc="#eef2ff", ec=ACC),
       dict(n="4", name="跨域 Transformer 层 ×N", desc="每层内先在时间维做自注意力、再在频率维做自注意力，"
            "两个分支之间用交叉注意力互换信息——这是 HTDemucs 的核心创新。",
            find=("crosstransformer", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="5", name="时域/频域解码器", desc="两条路的隐表示各自还原，再相加得到四轨。",
            find=("models.0.tdecoder", None), fc="#0d9488", ec="#0d9488"),
       dict(n="6", name="输出四轨", desc="单模型直接给四轨，不需为每个目标单独存一份权重。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["HTDemucs 架构：Transformer 层内**时间注意力与频率注意力交替**，并用交叉注意力在时/频分支间传信息。"]),

    F("MDX-Net (KUIELab / mdx_extra)", "频域 U-Net：TFC-TDF 卷积块堆成的编解码器",
      ["mdx"], "mdx",
      "Kim, Choi, Chung, Lee. KUIELab-MDX-Net: A Two-Stream Neural Network for Music "
      "Demixing. MDX Workshop 2021；评测框架：Mitsufuji et al. Music Demixing "
      "Challenge 2021, Frontiers in Signal Processing 2022",
      "01_models/_third_party/mdx-net + 3 个 submission 仓库；权重 mdx_extra ×4",
      ["01_models/_weights/MDX-Net/mdx_extra/*.th"],
      [("TFC-TDF 瓶颈 dconv", C["trunk"], ["dconv"]),
       ("4 个 submission 子网 models.*（其余）", C["core"], ["models"])],
      [dict(n="1", name="波形输入", desc="44.1 kHz 立体声，先转成复数谱。"
            "入口是 `BagOfModels.apply_model`，`<root>` 未被调用，本级不标 shape。",
            find=None, fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="频谱张量（实/虚）", desc="把复数谱的实部与虚部当通道堆成输入张量。"
            "该步由函数式 STFT 完成，trace 里没有独立模块，故不标 shape。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="3", name="编码器 ×6（stride 2×1）", desc="二维卷积逐层降采样，只压频率维、保留时间维，"
            "每层带 DConv（密集连接卷积）增强特征复用。",
            find=("encoder", None), fc="#2563eb", ec="#2563eb"),
       dict(n="4", name="TFC-TDF 瓶颈 ×N", desc="时间-频率卷积块 + 时间分布式全连接块交替，"
            "把整段时间的频谱上下文混进来。",
            find=("dconv", None), fc="#4f46e5", ec="#4f46e5"),
       dict(n="5", name="解码器（转置卷积 + 跳跃连接）", desc="逐层上采样回原尺寸，"
            "每层用 rewrite 卷积把跳跃过来的编码器特征重新投影。",
            find=("decoder", None), fc="#0d9488", ec="#0d9488"),
       dict(n="6", name="四目标频谱 + iSTFT", desc="输出四组实/虚谱，逆变换得四轨波形。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["mdx_extra 由 4 个 submission 权重组成，本项目按目标融合，整条目 334.5 M —— 全场参数最多。"]),

    F("BS-RoFormer（L12 / L6）", "频带划分 + 旋转位置编码 Transformer",
      ["bsroformer_l12", "bsroformer_l6"], "bsroformer_l12",
      "Lu, Wang, Jiang, Zhang. Music Source Separation with Band-Split RoPE "
      "Transformer. 2024（arXiv:2309.02612）",
      "01_models/_third_party/Music-Source-Separation-Training（ZFTurbo)+ BS-RoFormer",
      ["01_models/_weights/BS-RoFormer/model_bs_roformer_ep_317_*.ckpt",
       "01_models/_weights/BS-RoFormer/model_bs_roformer_ep_937_*.ckpt"],
      [("频带划分 band_split", C["band"], ["band_split"]),
       ("Transformer 层 layers", C["attn"], ["layers"]),
       ("掩码估计头 mask_estimators", C["mask"], ["mask_estimators", "mask"])],
      [dict(n="1", name="波形输入", desc="44.1 kHz 立体声 → STFT。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="频带划分（band split）", desc="不按单个频点建模，而是按 mel 边界把上千个频点分组成几十个"
            "非均匀频带——低频带宽窄、高频带宽宽，贴合音乐能量分布。",
            find=("band_split", None), fc="#fff7ed", ec="#d97706"),
       dict(n="3", name="频带特征投影", desc="每个频带内做归一化 + 线性层，投到统一的 d_model 维。",
            find=("to_features", None), fc="#fdf2f8", ec="#db2777"),
       dict(n="4", name="RoPE Transformer ×L", desc="对「频带 × 时间」二维序列做自注意力，"
            "位置编码用旋转位置编码（RoPE），前馈层做通道混合。L12 = 12 层，L6 = 6 层。",
            find=("layers", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="5", name="掩码估计头", desc="逐频带输出掩码，再展开回原来的频点数。",
            find=("mask_estimators", None), fc="#ecfdf5", ec="#059669"),
       dict(n="6", name="复数谱乘掩码 + iSTFT", desc="掩码作用于混合复数谱，逆变换成波形。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["[!] 本项目 L6 权重（ep_937）的实际输出是 ** vocals + other 残差**（mixture − drums − bass），"
       "不是 yaml 里写的 other —— 引用时不能按 yaml 的字面目标描述。",
       "FLOPs 43.5 T 是 HTDemucs 的 86 倍，但实测时延只差 16 倍：差额在 FlopCounterMode 看不到的 "
       "STFT 与融合注意力上，所以 FLOPs 列只当下界。"]),

    F("BSRNN（band-split RNN）", "先切频带、再在「频带内 × 频带间」两级做 RNN",
      ["bsrnn", "bsrnn_all", "bsrnn_simo", "bsrnn_large", "bsrnn_large_all"],
      "bsrnn_all",
      "Luo, Yu. Music Source Separation with Band-split RNN. IEEE/ACM TASLP, 2023"
      "（arXiv:2209.15174）",
      "01_models/_third_party/bsrnn（Zenodo 官方权重）",
      ["01_models/_weights/BSRNN/bsrnn-opt/bsrnn-opt/*.ckpt",
       "01_models/_weights/BSRNN/bsrnn-large/bsrnn-large/*.ckpt",
       "01_models/_weights/BSRNN/simo-bsrnn-opt/simo-bsrnn-opt/*.ckpt"],
      [("频带划分与归一化 BN/stft", C["band"], ["BN", "stft"]),
       ("双路分离主干 separator", C["core"], ["separator", "maskers"])],
      [dict(n="1", name="波形输入", desc="44.1 kHz 立体声 → 复数 STFT。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="频带划分 + 子带归一化", desc="按子带边界把频点分组，每个子带内做归一化（BN），"
            "消除不同频段的能量量级差异。",
            find=("BN", None), fc="#fff7ed", ec="#d97706"),
       dict(n="3", name="频带内 RNN（band-level）", desc="在单个子带的频率轴上跑 RNN，学习带内频谱包络。",
            find=("band_net", None), fc="#ecfeff", ec="#0891b2"),
       dict(n="4", name="频带间 RNN（time-level）", desc="在时间轴上跑 RNN，跨帧建模长程动态。两者交替堆叠。",
            find=("time_net", None), fc="#ecfeff", ec="#0891b2"),
       dict(n="5", name="每目标 masker", desc="四个目标（或 SIMO 的单个共享主干 + 多目标输出）各接一个 masker，"
            "把隐表示升回频点数并生成掩码。",
            find=("maskers", None), fc="#ecfdf5", ec="#059669"),
       dict(n="6", name="复数谱乘掩码 + iSTFT", desc="逆变换成四轨波形。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["本项目含 5 个 BSRNN 条目：bsrnn-opt 单目标 ×4 聚合（164.1 M）、opt 单目标（40.8 M）、"
       "SIMO（108.7 M，单主干四输出，四轨均值最高 9.27 dB）、large（36.4 M）、large ×4 聚合（146.7 M）。",
       "[!] 聚合条目（bsrnn_all / bsrnn_large_all）在 loader 闭包内是 4 个独立网络，"
       "本图右侧展示的正是 4 个子网全量 trace（路径前缀 net0/..net3/）。"]),

    # ---------------------------------------------------------------- board 2
    F("denoiser（DEMUCS-denoise）", "16 kHz 因果波形域降噪，官方 dns48/dns64/master64",
      ["denoiser_dns48", "denoiser_dns64", "denoiser_master64"], "denoiser_dns64",
      "Défossez, Synnaeve, Adi. Real Time Speech Enhancement in the Waveform "
      "Domain. Interspeech 2020（facebookresearch/denoiser）",
      "01_models/_third_party/denoiser（官方权重，经 torch.hub 下载）",
      ["01_models/_weights/denoiser/*.th"],
      [("编码器 encoder", C["encdec"], ["encoder"]),
       ("双向/单向 LSTM", C["rnn"], ["lstm"]),
       ("解码器 decoder", C["encdec"], ["decoder"])],
      [dict(n="1", name="含噪语音（16 kHz）", desc="模型原生 16 kHz；本项目把 44.1 kHz 输入重采样进来，"
            "输出再重采样回去，时延里含这两次重采样。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="编码器（Conv1d, stride 4）", desc="逐层下采样，把波形压成隐表示。因果版靠"
            "「只看过去」的卷积实现实时，非因果版可看双向。",
            find=("encoder", None), fc="#eef2ff", ec=ACC),
       dict(n="3", name="LSTM 序列建模", desc="在隐空间沿时间建模；这是模型参数的主要去向。",
            find=("lstm", None), fc="#ecfeff", ec="#0891b2"),
       dict(n="4", name="解码器（ConvTranspose）", desc="上采样回原长度，输出干净语音。",
            find=("decoder", None), fc="#0d9488", ec="#0d9488"),
       dict(n="5", name="dry/wet 混合", desc="推理时可调干湿比：湿=模型输出，干=原输入，"
            "用于控制降噪强度（不是训练结构，是推理期旋钮）。",
            find=None, fc="#f5f3ff", ec="#7c3aed"),
       dict(n="6", name="输出增强语音", desc="单通道输出，不是四轨分离。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["[!] License **CC-BY-NC 4.0（非商用）**，论文引用时必须注明。",
       "[!] `master64` 的训练集据称**包含 Valentini**，因此在 Valentini 上评测属 in-domain，"
       "必须同时报 dns48 / dns64 作为 out-of-domain 对照。",
       "单样本探针结果出现反直觉现象：改善量 dns48 +2.15 > dns64 +1.69 > master64 **+0.15**，"
       "与「in-domain 应更好」相反 —— 已登记为待验证项（R2），单样本不足以下结论。"]),

    F("Mel-Band RoFormer — Denoise (aufr33)", "mel 频带 + RoPE Transformer，输出 dry / other 双路",
      ["mel_roformer_denoise", "mel_roformer_denoise_aggr"], "mel_roformer_denoise",
      "架构源自 Lu et al. Mel-Band RoFormer for Music Source Separation"
      "（arXiv:2310.01809）；权重由 **aufr33** 训练，经 ZFTurbo "
      "Music-Source-Separation-Training 分发 —— 属社区模型，非论文官方权重",
      "01_models/_third_party/Music-Source-Separation-Training（复用其 mel_band_roformer 实现）",
      ["01_models/_weights/mel_roformer_denoise/*.ckpt"],
      [("频带划分 + 特征投影", C["band"], ["band_split"]),
       ("Transformer 层 layers", C["attn"], ["layers"]),
       ("掩码估计头 mask_estimators", C["mask"], ["mask_estimators", "mask"])],
      [dict(n="1", name="44.1 kHz 立体声输入", desc="与板 1 的分离模型同采样率，属非因果（可看全曲）。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="mel 频带划分", desc="按 mel 刻度划子带，比线性频带更贴合听感；"
            "每个子带单独投影成特征向量。",
            find=("band_split", None), fc="#fff7ed", ec="#d97706"),
       dict(n="3", name="RoPE Transformer ×N", desc="在「频带 × 时间」上做自注意力，"
            "旋转位置编码 + 前馈网络交替。",
            find=("layers", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="4", name="双路输出头", desc="同时给出 **dry（保留信号）** 与 **other（被剥离的噪声/混响）**"
            " —— 残差面板是白送的，可直接用于误差分析。",
            find=("mask_estimators", None), fc="#ecfdf5", ec="#059669"),
       dict(n="5", name="iSTFT", desc="两路各自逆变换回波形。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="6", name="用途：板 2 主降噪模型", desc="在本项目的门禁探针上 SI-SDR 改善 +9.88 dB（aggr +9.67）。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["两版权重：`..._sdr_27.9959.ckpt`（标准）与 `..._aggr_sdr_27.9768.ckpt`（更激进），各 913 MB；"
       "两者同架构，参数与时延差异 <1%。",
       "[!] 社区权重（非论文官方），论文中必须标注来源与这一限制。"]),

    F("Mel-Band RoFormer — Dereverb (anvuew)", "同架构换任务：去混响",
      ["mel_roformer_dereverb"], "mel_roformer_dereverb",
      "架构同上（Lu et al., Mel-Band RoFormer）；权重 `dereverb_mel_band_roformer_anvuew` "
      "由 **anvuew** 训练，经 ZFTurbo 框架分发 —— 社区模型",
      "01_models/_third_party/Music-Source-Separation-Training",
      ["01_models/_weights/mel_roformer_dereverb/*.ckpt"],
      [("频带划分 + 特征投影", C["band"], ["band_split"]),
       ("Transformer 层 layers", C["attn"], ["layers"]),
       ("掩码估计头 mask_estimators", C["mask"], ["mask_estimators", "mask"])],
      [dict(n="1", name="44.1 kHz 立体声输入", desc="含混响信号（dry + 房间脉冲响应卷积的结果）。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="mel 频带划分", desc="与降噪版完全一致。",
            find=("band_split", None), fc="#fff7ed", ec="#d97706"),
       dict(n="3", name="RoPE Transformer ×N", desc="在频带 × 时间序列上建模；"
            "混响表现为同一能量在时间上的拖尾，靠注意力跨帧捕捉。",
            find=("layers", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="4", name="掩码输出", desc="输出抑制拖尾后的信号。",
            find=("mask_estimators", None), fc="#ecfdf5", ec="#059669"),
       dict(n="5", name="iSTFT", desc="逆变换回波形。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="6", name="用途：对应参考图的 Reverberation", desc="用于阶段 4 的混响对照实验"
            "（T60 扫描 + ΔSDR 退化曲线）。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["门禁探针 SI-SDR 改善 +3.59 dB —— 明显低于降噪版的 +9.88，"
       "因为去混响面对的是「信号本身也被改变」，SI-SDR 不是它的最佳指标。"]),

    F("Mel-Band RoFormer — Dereverb-Echo (Sucial)", "去混响 + 去回声，参数量略小",
      ["mel_roformer_dereverb_echo"], "mel_roformer_dereverb_echo",
      "架构同上；权重 `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt` 由 **Sucial** 训练，"
      "经 ZFTurbo 框架分发 —— 社区模型",
      "01_models/_third_party/Music-Source-Separation-Training",
      ["01_models/_weights/mel_roformer_dereverb_echo/*.ckpt"],
      [("频带划分 + 特征投影", C["band"], ["band_split"]),
       ("Transformer 层 layers", C["attn"], ["layers"]),
       ("掩码估计头 mask_estimators", C["mask"], ["mask_estimators", "mask"])],
      [dict(n="1", name="44.1 kHz 立体声输入", desc="同时含混响与回声（远场/双讲类退化）。",
            find=("<root>", None), fc="#f3f4f6", ec="#9ca3af"),
       dict(n="2", name="mel 频带划分", desc="同族一致。",
            find=("band_split", None), fc="#fff7ed", ec="#d97706"),
       dict(n="3", name="RoPE Transformer ×N", desc="层配置与降噪/去混响版不同（参数量 208.9 M vs 228.2 M）。",
            find=("layers", None), fc="#f5f3ff", ec="#7c3aed"),
       dict(n="4", name="掩码输出", desc="同时压制混响拖尾与延迟回声。",
            find=("mask_estimators", None), fc="#ecfdf5", ec="#059669"),
       dict(n="5", name="iSTFT", desc="逆变换回波形。",
            find=None, fc="#eef2ff", ec=ACC),
       dict(n="6", name="用途：对应参考图的 Target Echo", desc="阶段 4 的回声对照分支。",
            find=None, fc="#fff7ed", ec="#d97706")],
      ["门禁探针 SI-SDR 改善 +3.60 dB。",
       "同族四个权重里唯一参数量不同的一个（208.9 M）；本图右侧构成条因此与另外三张不同，属真实差异。"]),
]


# ---------------------------------------------------------------------------
def load_graph(key):
    p = os.path.join(DATA, "graph_%s.json" % key)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def noprefix(path):
    return path.split("/", 1)[1] if path.startswith("net") and "/" in path else path


def canon_nodes(g):
    """Canonical, non-alias nodes in execution order."""
    return [n for n in g.get("exec_order", []) if n.get("role") != "alias"]


def _fmt1(o):
    if isinstance(o, list):
        if not o:
            return "[]"
        if all(isinstance(x, int) for x in o):
            return "x".join(str(x) for x in o)
        return "(" + ", ".join(_fmt1(x) for x in o[:2]) + (", ..)" if len(o) > 2 else ")")
    return str(o)


def fmt_shapes(v):
    if isinstance(v, list) and v and isinstance(v[0], list):
        return "(" + ", ".join(_fmt1(x) for x in v[:2]) + (", .." if len(v) > 2 else "") + ")"
    return _fmt1(v)


def resolve_shape(g, find, in_only=False):
    """Return 'in -> out' for the first canonical node matching (path token, class token).

    `in_only` is used for the waveform-input stage: there the trace is the model
    root, whose OUTPUT is whatever the loader returns (for BSRNN a dict of
    `{'waveforms': ..., 'stfts': ...}`) -- true, but noise on a box labelled
    "waveform input".
    """
    if not find:
        return ""
    tok, cls = find
    for n in canon_nodes(g):
        p = noprefix(n["path"])
        if tok not in p and tok != "<root>":
            continue
        if tok == "<root>" and p != "<root>":
            continue
        if cls and cls.lower() not in (n.get("cls") or "").lower():
            continue
        if in_only:
            return fmt_shapes(n["in_shapes"])
        return "%s -> %s" % (fmt_shapes(n["in_shapes"]), fmt_shapes(n["out_shapes"]))
    return ""


def graph_state(g):
    """'analytic' | 'ok' | 'stale' -- never let a stale graph masquerade as analytic.

    `graph_*.json` written before the multi-root/parameter-attribution rewrite has
    no `traced_params_M`, so a renderer that only checks `if not total` would draw
    the "analytic, 0 params" panel for a perfectly ordinary neural network.  That
    is the project's recurring "ran but silently wrong" failure mode, so the two
    cases are now separated explicitly.
    """
    if not g:
        return "missing"
    if g.get("status") == "NO_MODULE":
        return "analytic"
    return "ok" if g.get("traced_params_M") else "stale"


def _prefix_hit(path, q):
    """Match `q` on a dotted-component boundary anywhere in `path`.

    Needed because group rules are written the way a human describes the model
    ("encoder", "decoder") while the traced path is nested
    ("models.0.encoder").  A plain `startswith` silently matched nothing and
    collapsed the 100 % bar into a single bucket.

    The `.` in the containment test is what keeps "encoder" from matching
    "tencoder" -- verified on the demucs trace, where both branches fire.
    """
    if path == q or path.startswith(q) or path.startswith(q + "."):
        return True
    return ("." + q) in path


def composition(g, groups):
    """Parameter composition in MILLIONS; sums to `traced_params_M`.

    Note the unit: `params_attr` in the JSON is a raw tensor count while
    `traced_params_M` is in millions.  Mixing them produced bar widths of ~1e6
    which overflowed the Agg renderer inside `savefig` (a crash far from its
    cause), so the conversion happens here, once.
    """
    tot = (g.get("traced_params_M") or 0.0)
    if not tot:
        return [], 0.0
    buckets = {}
    for n in canon_nodes(g):
        v = n.get("params_attr") or 0
        if not v:
            continue
        p = noprefix(n["path"])
        lab, col = "其他/辅助", C["aux"]
        for name, color, prefixes in groups:
            if any(_prefix_hit(p, q) for q in prefixes):
                lab, col = name, color
                break
        b = buckets.setdefault(lab, [0.0, col])
        b[0] += v / 1e6
    items = [(k, v[0], v[1]) for k, v in buckets.items() if k != "其他/辅助"]
    items.sort(key=lambda x: -x[1])
    if "其他/辅助" in buckets:
        items.append(("其他/辅助", buckets["其他/辅助"][0], buckets["其他/辅助"][1]))
    return items, tot


def norm_path(p):
    """`models.0.encoder.5.dconv` -> `models.#.encoder.#.dconv`.

    Numeric indices make sibling modules look structurally different, which
    broke the two earlier grouping attempts: keying on (parent, class) merged
    MDX-Net's four sub-nets into a single 100 % row that then had to be
    discarded, leaving the table empty.  Collapsing each index to `#` shows the
    shape of the architecture rather than of the enumeration.
    """
    return re.sub(r"\.\d+", ".#", p)


def skeleton_rows(g, max_rows=13, max_depth=4):
    """Structural skeleton with SUBTREE parameter shares.

    A module's own `params_attr` is 0 whenever it is a container, so a table
    keyed on it printed "0.0%" next to the module holding two thirds of the
    model (Demucs' `crosstransformer`).  What a reader wants is the whole
    subtree below that path, which is what is computed here.  The `net0/`,
    `net1/` ... prefixes of aggregated entries are stripped first so their
    sub-nets share one row.
    """
    nodes = canon_nodes(g)
    leaf = {}
    for n in nodes:
        v = n.get("params_attr") or 0
        if v:
            leaf[n["path"]] = leaf.get(n["path"], 0) + v
    tot = g.get("traced_params_M") or 0.0

    def subtree(raw):
        pre = raw + "."
        return sum(v for p, v in leaf.items() if p == raw or p.startswith(pre))

    seen, rows = {}, []
    for n in nodes:
        raw = n["path"]
        disp = noprefix(raw)
        if disp == "<root>" or disp.count(".") > max_depth - 1:
            continue
        label = norm_path(disp)
        key = (label, n.get("cls"))
        sub = subtree(raw) / 1e6
        if key in seen:
            # accumulate, do not just count: four sibling sub-nets each own an
            # `encoder`, and keeping only the first one's subtree made the row
            # read a quarter of its real share
            seen[key]["dups"] += 1
            seen[key]["attr"] += sub
            continue
        rec = dict(path=label, cls=n.get("cls") or "", attr=sub, dups=1,
                   io="%s -> %s" % (fmt_shapes(n["in_shapes"]), fmt_shapes(n["out_shapes"])),
                   calls=n.get("calls") or 1, role=n.get("role"))
        seen[key] = rec
        rows.append(rec)
    # Drop the row that IS the whole model (it says nothing), then walk the rest
    # in descending share and skip any that is merely a restatement of a row
    # already kept -- `separator` and `separator.0` both read "60.7 %" because
    # one contains the other, and printing both invites double counting.
    rows = [r for r in rows if not (tot and r["attr"] >= 0.999 * tot)]
    # shallower first on ties, so the container survives and the nesting restates
    # are the ones pruned
    rows.sort(key=lambda r: (-r["attr"], r["path"].count("."), r["path"]))
    kept, out = [], []
    for r in rows:
        restated = False
        for k in kept:
            # `r` is skipped only when a kept row is nested with it AND accounts
            # for essentially all of its share, i.e. `r` adds nothing.  (Comparing
            # the other way round deleted `models.0.encoder` -- 11.7 % -- merely
            # because its parent `models.0` is 25 %.)  Both directions are tested
            # because which of the pair sorts first depends on a rounding margin.
            nested = (r["path"] == k["path"]
                      or r["path"].startswith(k["path"] + ".")
                      or k["path"].startswith(r["path"] + "."))
            if nested and min(r["attr"], k["attr"]) >= 0.995 * max(r["attr"], k["attr"]):
                restated = True
                break
        if not restated:
            kept.append(r)
            out.append(r)
        if len(out) >= max_rows:
            break
    rows = out
    for r in rows:
        r["pct"] = (100.0 * r["attr"] / tot) if tot else 0.0
        if r["dups"] > 1:
            r["path"] = "%s  x%d" % (r["path"], r["dups"])
    return rows


# ------------------------------------------------------------------ panels
def draw_flow(ax, fam, g):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = fam["stages"]
    n = len(stages)
    top, bot = 0.995, 0.005
    h = (top - bot) / n
    gap = min(0.024, h * 0.30)
    unresolved = []
    for i, s in enumerate(stages):
        y1 = top - i * h - gap * 0.45
        y0 = y1 - (h - gap)
        ax.add_patch(FancyBboxPatch((0.02, y0), 0.96, (h - gap),
                                    boxstyle="round,pad=0.004,rounding_size=0.014",
                                    fc=s.get("fc", "#eef2ff"), ec=s.get("ec", ACC),
                                    lw=1.25, zorder=2))
        ax.text(0.045, y1 - gap * 0.30, "%s  %s" % (s["n"], s["name"]),
                fontsize=10.6, fontweight="bold", color=INK, va="top", zorder=3)
        body = "\n".join(textwrap.wrap(s["desc"], 52))[:300]
        ax.text(0.045, y1 - gap * 0.30 - h * 0.185, body,
                fontsize=7.9, color="#374151", va="top", zorder=3, linespacing=1.35)
        sh = resolve_shape(g, s.get("find"),
                           in_only=(s.get("find") or ("",))[0] == "<root>")
        if sh:
            ax.text(0.975, y1 - gap * 0.30, sh, fontsize=7.0, color="#6d28d9",
                    va="top", ha="right", zorder=3, family="monospace")
        elif s.get("find"):
            unresolved.append(s["name"])
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y0 - gap),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color="#9ca3af", lw=1.2, zorder=1))
    return unresolved


def draw_bar(ax, items, tot, state):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, 1.05)
    ax.axis("off")
    if state in ("analytic", "missing"):
        ax.text(0.5, 0.40, "解析算法：无 nn.Module，参数量恒为 0",
                ha="center", va="center", fontsize=10.5, color=MUTED)
        ax.text(0.5, 0.08, "graph json 记录 status = NO_MODULE",
                ha="center", va="center", fontsize=8.2, color=MUTED)
        return
    if state == "stale" or not tot:
        ax.text(0.5, 0.40, "图数据为旧格式：缺少 traced_params_M",
                ha="center", va="center", fontsize=10.5, color="#b91c1c",
                fontweight="bold")
        ax.text(0.5, 0.08, "请重跑 tools/_extract_model_graph.py 后重新渲染",
                ha="center", va="center", fontsize=8.2, color="#b91c1c")
        return
    left = 0.0
    for lab, v, col in items:
        w = min(1.0, max(0.0, v / tot))
        ax.barh(0.42, w, left=left, height=0.46, color=col,
                edgecolor="white", lw=1.0, zorder=3)
        if w >= 0.045:
            ax.text(left + w / 2, 0.42, "%.1f%%" % (100 * w), ha="center", va="center",
                    fontsize=8.3, color="white", fontweight="bold", zorder=4)
        left += w
    ax.text(0.0, 0.80, "参数构成（按深度归属，合计 = 100 %）",
            fontsize=10.2, fontweight="bold", color=INK, va="bottom")
    ax.text(1.0, 0.80, "合计 %.2f M" % tot, fontsize=9.0, color=MUTED,
            va="bottom", ha="right")
    handles = [Line2D([], [], marker="s", ls="", color=c, markersize=9) for _, _, c in items]
    labels = ["%s  %.1f%%" % (l, 100 * v / tot) for l, v, _ in items]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=False, fontsize=8.0, handletextpad=0.5, columnspacing=1.4)


def draw_table(ax, rows, g):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.985, "真实执行骨架（forward hook 抓到，按参数量排序）",
            fontsize=10.2, fontweight="bold", color=INK, va="top")
    note = "总模块 %s，本次触发 %s" % (g.get("n_modules_total"), g.get("n_modules_fired"))
    if g.get("n_roots", 1) > 1:
        note += "（%d 个并列子网 net0/..net%d/）" % (g["n_roots"], g["n_roots"] - 1)
    ax.text(1.0, 0.985, note, fontsize=7.8, color=MUTED, va="top", ha="right")

    xp, xc, xi, xa = 0.0, 0.335, 0.545, 0.885
    y = 0.905
    ax.text(xp, y, "模块路径", fontsize=7.9, fontweight="bold", color=INK)
    ax.text(xc, y, "类", fontsize=7.9, fontweight="bold", color=INK)
    ax.text(xi, y, "输入 -> 输出", fontsize=7.9, fontweight="bold", color=INK)
    ax.text(xa, y, "占比", fontsize=7.9, fontweight="bold", color=INK)
    ax.text(xa + 0.055, y, "（含子模块）", fontsize=6.8, color=MUTED)
    y -= 0.030
    ax.plot([0, 1], [y, y], color="#d1d5db", lw=0.9)
    rh = min(0.058, (y - 0.02) / max(len(rows), 1))
    for r in rows:
        y -= rh
        col = INK if r["attr"] else MUTED
        ax.text(xp, y, r["path"][:44], fontsize=7.4, color=col, va="center",
                family="monospace")
        ax.text(xc, y, (r["cls"] or "")[:16], fontsize=7.4, color=col, va="center")
        ax.text(xi, y, r["io"][:40], fontsize=6.9, color="#6d28d9", va="center",
                family="monospace")
        ax.text(xa, y, ("%.1f%%" % r["pct"]) if r["attr"] else "-", fontsize=7.4,
                color=col, va="center")
        if r["calls"] > 1:
            ax.text(xa + 0.075, y, "x%d" % r["calls"], fontsize=6.8, color=MUTED,
                    va="center")
    return


def weight_lines(patterns):
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(_ROOT, pat), recursive=True))
        for h in hits:
            rel = os.path.relpath(h, _ROOT).replace("\\", "/")
            if "$RECYCLE" in rel:
                continue
            out.append("%s  (%.1f MB)" % (rel, os.path.getsize(h) / 1e6))
    return out[:4]


def render(fam):
    g = load_graph(fam["primary"])

    fig = plt.figure(figsize=(15.6, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.32],
                          height_ratios=[0.62, 1.38], hspace=0.30, wspace=0.08,
                          left=0.035, right=0.975, top=0.895, bottom=0.145)
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_tab = fig.add_subplot(gs[1, 1])

    unresolved = draw_flow(ax_flow, fam, g)
    state = graph_state(g)
    items, tot = composition(g, fam["groups"])
    draw_bar(ax_bar, items, tot, state)
    draw_table(ax_tab, skeleton_rows(g), g)

    fig.suptitle("图 %s —— %s" % (fam["title"], fam["subtitle"]),
                 fontsize=14.2, fontweight="bold", color=INK, y=0.972)
    if len(fam["keys"]) > 1:
        variants = []
        for k in fam["keys"]:
            kk = load_graph(k)
            if kk.get("params_M"):
                variants.append("%s %.1f M" % (kk.get("label") or k, kk["params_M"]))
        fig.text(0.035, 0.936, "涵盖条目：" + "  |  ".join(variants),
                 fontsize=8.4, color="#374151")
    fig.text(0.035, 0.918, "论文引用：" + fam["citation"],
             fontsize=7.6, color=MUTED, wrap=True)

    lines = []
    wl = weight_lines(fam["weights"])
    if wl:
        lines.append("本机权重：" + " ； ".join(wl))
    else:
        lines.append("本机权重：无（解析算法，不需要权重文件）")
    lines.append("代码落点：" + fam["code"])
    TRACE_NOTE = {
        "ok": "形状可溯源：右侧所有 in -> out 均取自 04_reports/_shared/data/model_analysis/"
              "graph_%s.json（status=OK）" % fam["primary"],
        "analytic": "本图为解析算法，`graph_%s.json` 记录 status=NO_MODULE，"
                    "故整张图不标注任何 shape（没有 trace 可溯源就不写数字）" % fam["primary"],
    }
    lines.append(TRACE_NOTE.get(state, "[!] 图数据为旧格式，请重跑 tools/_extract_model_graph.py"))
    for n in fam["notes"]:
        lines.append(n)
    y = 0.118
    for ln in lines:
        for seg in textwrap.wrap(ln, 168) or [""]:
            fig.text(0.035, y, seg, fontsize=7.2, color=MUTED)
            y -= 0.0180

    fn = fam["family_file"]
    out = os.path.join(FIG, fn)
    fig.savefig(out)
    plt.close(fig)
    return out, unresolved, len(items), tot


if __name__ == "__main__":
    import re
    bad = []
    for i, fam in enumerate(FAMILIES, 1):
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", fam["title"].split("（")[0]).strip("_").lower()
        fam["family_file"] = "arch_%02d_%s.png" % (i, slug)
        out, unresolved, ng, tot = render(fam)
        flag = ("  UNRESOLVED-SHAPE:%s" % unresolved) if unresolved else ""
        print("OK  %-46s groups=%d total=%.2fM  %s%s"
              % (os.path.basename(out), ng, tot, "%d KB" % (os.path.getsize(out) // 1024), flag))
        if unresolved:
            bad.append((fam["title"], unresolved))
    print("\n[figures] %d -> %s" % (len(FAMILIES), FIG))
    if bad:
        print("[warn] 有概念流程阶段拿不到可溯源的 shape：")
        for t, u in bad:
            print("   %-40s %s" % (t, u))
