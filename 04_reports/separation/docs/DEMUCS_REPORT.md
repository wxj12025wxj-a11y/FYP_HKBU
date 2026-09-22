# Demucs 纵向深入分析报告

**模型**：Demucs v4 **HTDemucs**（Hybrid Transformer Demucs，BagOfModels，内含 1 个子模型）
**数据源**：`A Classic Education - NightOwl` 的 `mixture.wav`，前 **30 秒**（1323000 采样点 @ 44100 Hz）
**生成日期**：2026-09-15 ｜ **口径更正**：2026-09-16
**产出目录**：`03_outputs/Demucs/stems/`、`04_reports/figures/demucs/`、`04_reports/data/comparison/demucs_metrics.json`

> ### ⚠️ 口径更正（2026-09-16，务必先读）
> 本报告 §2/§5 的「Demucs 绝对精度最高」是在**单首歌、单个 30 s 片段、且对比池只有 6 个方法**的条件下得出的。
> 2026-09-16 把 Demucs 并入**同一个统一基准**后（3 个能量均衡片段、各 10 s、取中位数，对比池 12 个模型），
> 结论变为：**Demucs 四轨均值 SDR = 8.09 dB，在 12 个模型中排第 5**
> （oBSRNN-SIMO 9.27 > oBSRNN 9.16 > MDX-Net 9.13 > BSRNN-large 8.42 > **Demucs 8.09**）。
> 分轨看：鼓 8.87 / 贝斯 10.96 属第一梯队，**其他（other）5.44 偏弱**，人声 7.09 为中游。
> 权威口径以 **`04_reports/data/comparison/model_runs_median.json`** 与 `04_reports/html/METHOD_VISUAL_COMPARISON_1.3.html` §7.4 为准。
> Demucs 真正的优势在**工程侧**：单一模型（41.98 M）直接出 4 stem，**RTF ≈ 0.39**，是「单模型 4 轨」里最快的一档。
> 另注：Demucs 在 CPU 上**非确定性**（同输入两次 ±0.1~0.3 dB），本报告的单片段数字不可与其他模型逐位比较。

---

## 1. 任务概述

延续此前的**横向对比**（同数据上比较 ICA / NMF / RPCA / HPSS / Open-Unmix），本次做**纵向深入**：
对**同一首歌**运行 **Demucs**，复刻**同一套 9 张分析图**，并计算 6 项指标
—— **SDR、SI-SDR、PESQ、Params、MACs、FLOPs**。

---

## 2. 分离质量指标

> 口径与前序横向对比一致：**museval（BSSEval v4），1 s 窗、中位数聚合**；基线 = 直接拿 mixture 当估计。

| 目标 | **SDR (dB)** | **ΔSDR (dB)** | SI-SDR | ISR | SAR | STOI | **PESQ** | 基线 SDR |
|---|---|---|---|---|---|---|---|---|
| **vocals** | **+8.74** | **+17.91** | 9.49 | 18.93 | 8.52 | 0.643 | **2.158** | −9.16 |
| **drums** | **+4.57** | **+22.51** | 2.55 | 7.59 | 3.54 | 0.601 | **1.080** | −17.93 |

**关键解读：**

- **Demucs 人声 SDR 8.74 dB，ΔSDR +17.91 dB**，在本报告当时的对比池（ICA/NMF/RPCA/HPSS/Open-Unmix）中是**绝对精度最高**的
  —— ⚠️ 但这只是**单首歌单片段**结论，扩到 12 模型 × 3 均衡片段的中位数口径后 Demucs 排第 5，见文首「口径更正」。
  对比同一首歌上的 **Open-Unmix（6.55 dB / +15.71）**，Demucs **高出约 2.2 dB**。
- 鼓声 ΔSDR **+22.51 dB**（基线 −17.93 → 4.57），提升幅度甚至大于人声——符合 Demucs 在多 stem 上的均衡能力。
- **STOI**（短时客观可懂度，0~1）人声 0.643、鼓声 0.601，处于"可懂度良好"区间，与 SDR 结论一致。
- **PESQ（人声）2.16**，显著优于基线 1.07 → **+1.09**，是感知质量提升的直接证据。
  但 **PESQ（鼓声）出现"倒挂"**（基线 2.79 → 1.08），原因见 §5 —— 这是 PESQ 适用域问题，不是 Demucs 的问题。

---

## 3. 模型复杂度（Params / MACs / FLOPs）

| 指标 | 数值 | 说明 |
|---|---|---|
| **Params** | **41.98 M**（41,984,456） | 模型参数量 |
| **MACs** | **112.23 G** | 单个 7.8 s 训练段 |
| **FLOPs** | **224.47 G** | ≈ 2 × MACs |
| MACs / 秒 | 14.39 G/s | 段长线性外推 |
| FLOPs / 秒 | 28.78 G/s | 段长线性外推 |
| 30 s 外推 | ≈ 431.7 GMACs / 863.3 GFLOPs | 仅供参考（注意力对长度非线性） |

> **计数方法**：Demucs 核心是**复数 STFT + Transformer**，`thop` / `ptflops` 均不支持
> （抛 `NotImplementedError` / `TypeError`）。改用 **PyTorch 内置 `FlopCounterMode`**，
> 统计 conv / linear / matmul 等常规算子；**复数 FFT 未计入 → 属保守低估**。
> 输入取 HTDemucs 的训练段长 **7.8 s**（343980 采样点）。

**横向感受一下量级**：Params 41.98 M 与 Open-Unmix（**8.90 M / 单目标** × 4 目标 = **35.58 M**）**量级相当**（仅大 18.4%）。
> ⚠️ 本报告早期版本误写为「Open-Unmix 约 4.6 M / 单目标、Demucs 是其 9 倍」，已于 2026-09-15 用
> `openunmix` 实际加载权重核实并更正（单目标 8,897,444；四目标合计 35,577,488）。
> 两者组织方式不同：Demucs 是「**单模型出 4 stem**」，Open-Unmix 是「**4 个独立模型各出一轨**」。
> 因此 Demucs 慢 3.3 倍的代价，主要来自**架构更重**（Transformer 注意力 + 时域高采样率卷积），而非参数量。

FLOPs 按 30 s 音频算接近 **860 GFLOPs**（复数 FFT 未计入，属保守低估）。

---

## 4. 实时性

| 项目 | 数值 |
|---|---|
| 30 s 音频推理耗时 | **11.6 s** |
| **RTF**（耗时 ÷ 时长） | **0.387** ✅ 可实时（< 1） |
| 对比 Open-Unmix RTF | 0.117（更快） |

在本报告的单片段口径下 Demucs 精度最高，但**单次推理比 Open-Unmix 慢约 3.3 倍**（后续受控 10 首短片段实测为 **2.70×**）。
二者都满足实时（RTF < 1）——Demucs 的算力余量更小，若 FYP 走实时路线，需重点评估其延迟与硬件需求。
⚠️ 跨模型精度排名请以 12 模型中位数表为准（Demucs 第 5），不要引用本段的单片段结论。

---

## 5. PESQ 与感知指标

**PESQ 已成功计算**（本机原缺 C 编译器，改用 conda-forge 的 **预编译 pesq 0.0.4 (py313 win-64)** 直接装入 venv，未污染系统环境）。

| 目标 | PESQ（Demucs） | PESQ（基线） | ΔPESQ | STOI（Demucs） |
|---|---|---|---|---|
| vocals | **2.158** | 1.069 | **+1.09** | 0.643 |
| drums | **1.080** | 2.791 | **−1.71** | 0.601 |

> ⚠️ **鼓声 PESQ 的"倒挂"必须解释清楚，不能糊弄过去**：
> PESQ 是**为语音通信设计的**（ITU-T P.862），其内部模型基于**人耳对语音的感知**。
> 把它用在**打击乐（drums）**上，输出**不具物理意义**——基线（整段混音）听起来"更像鼓"，只是因为
> 混音里本就含大量鼓的能量成分，而 PESQ 的语音模型无法区分"鼓的还原度"。
> **结论：PESQ 只采信人声列（+1.09），鼓声列的 PESQ 应弃用，改以 SDR/STOI 为准。**
> 这也是文献里"感知指标不等于分离质量"的经典教训。

**口径**：PESQ 需 8 k / 16 kHz，统一重采样到 **16 kHz 宽带（wb）**；STOI 同为 16 kHz。

---

## 6. 产出清单

```
Output-01_models/_third_party/demucs/
├── stems/          4 个分离 stem: vocals.wav / drums.wav / bass.wav / other.wav
├── figures/        同一套 9 张分析图 (fig1~fig6)
├── metrics.json    全部指标 (机器可读)
├── DEMUCS_REPORT.md  本报告
└── run.log         运行日志
```

| 图 | 说明（均为 Demucs 分离结果） |
|---|---|
| `fig1_waveform_mixture_vocals.png` | mixture / Demucs-vocals / 真值 vocals 波形三对照 |
| `fig2_waveform_sr_compare.png` | Demucs 输出重采样 44100 / 22050 / 16000 |
| `fig3_spectrogram_mixture.png` / `fig3_spectrogram_vocals.png` | mixture 与 Demucs-vocals 频谱图 |
| `fig4_spectrogram_win128.png` / `fig4_spectrogram_win1024.png` | STFT 窗长对比 |
| `fig5_trim_before_after.png` | 静音裁剪前后 |
| `fig6_mfcc_mixture.png` / `fig6_mfcc_vocals.png` | 13 维 MFCC |

---

## 7. 复现

```bash
cd E:\FYP_HKBU
python tools\run_demucs_deepdive.py --duration 30
# 换模型:  --model htdemucs_ft   (bag of 4, 精度更高但更慢)
```

> 说明：Demucs 推理在 CPU 上存在**微小数值非确定性**（多线程），
> 重复运行 SDR 波动约 ±0.2 dB，属正常范围。
