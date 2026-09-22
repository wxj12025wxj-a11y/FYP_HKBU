# 模型供给与可运行性核查报告（12 模型）

> 项目：HKBU FYP — 实时音乐源分离 / 降噪
> 机器：Windows + RTX 4070 Laptop (8 GB) + Python 3.13.14
> 环境：`C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio`（torch **2.14.0+cu126**，CUDA 已启用）
> 日期：2026-09-16 建稿 ｜ **2026-09-17 重大修订**：装 CUDA、DPRNN 自训补齐、**全部模型在 GPU 上重跑 RTF**
> 关联产物：`04_reports/data/comparison/model_runs.json`、`model_runs_median.json`（§1.6 主结果表）、
> `model_runs_cpu_2026-09-16.json`（CPU 基线备份，用于 CPU↔GPU 对照）

---

## 0. 一句话结论

12 个模型里：**2 个拿不到可运行的公开代码**（RPCA+DRNN 论文无代码、Pac-HuBERT-SEP 未放出）；
**10 个在本机跑通了真实推理并落盘了 stem wav**（RPCA、Conv-TasNet、**DPRNN(自训)**、MMDenseLSTM、
BS-RoFormer L12、BS-RoFormer L6、**BSRNN 全三变体**、MDX-Net、IRM/IBM Oracle、**Demucs**）。

**DPRNN 已由「缺权重」转为「自训补齐」**：官方只发语音权重，本轮用本机 RTX 4070 自训 MUSDB18 4-stem 版
（`tools/train_dprnn_musdb.py`，3.69 M 参数）。⚠️ 它精度仍低于其余 11 个模型（算力受限的欠训练），
**只用于给出「端到端时域方法在 8 GiB 单卡 + 数小时预算下无法复现论文级精度」这个负面结论**，
不参与「最快/最准」排名。细节与 4 个坑见 `01_models/MODEL_REGISTRY.md` §6。

> 📈 **DPRNN 复训进展（2026-09-19 刷新）**：在 step 3045 的欠训练权重基础上续训到 **step 31900**
> （300 min 预算实跑 250.7 min），验证集均值 SI-SDR **+0.781 → +2.745 dB**，
> 测试集四轨均值 SDR **-4.54 → -1.72 dB**（+2.82 dB）。四轨仍为负，**排名结论不变**。


**Smart App Control**：本机唯一硬性环境限制曾是拦 `llvmlite.dll`（连带 `numba` / `librosa.stft`）。
**2026-09-17 复测：拦截面已放行，`llvmlite` / `numba.njit` / `librosa.stft` 实算全部通过**；
装 CUDA 时也确认 **SAC 没有拦截任何 CUDA DLL**（cudart / cublas / cudnn 正常加载）。
⚠️ SAC 拦截面在本项目里已经变过 **3 次** —— 引用「环境阻塞」结论必须带时间戳。

| 类别 | 数量 | 模型 |
|---|---|---|
| ✅ 本机真实跑出 stem wav | **10** | RPCA、Conv-TasNet、**DPRNN（自训）**、MMDenseLSTM、BS-RoFormer L12、BS-RoFormer L6、BSRNN（opt/large/SIMO）、MDX-Net、IRM/IBM Oracle、Demucs |
| ❌ 无公开代码 | **2** | RPCA+DRNN、Pac-HuBERT-SEP |

**精度与实时性的最终结论（12 模型，3 个均衡片段中位数，RTX 4070 8 GiB，见 §1.6）**：

- **精度最高**：**oBSRNN-SIMO**（四轨均值 **9.27 dB**，单一 108.73 M 模型）> oBSRNN 9.16 > MDX-Net 8.96 > BSRNN-large 8.42 > Oracle 8.30 > **Demucs 8.11**。
- **实时性已被 CUDA 改写**：装上 CUDA 后深度模型普遍提速 **3~22×**，**精度前三名全部实时** ——
  oBSRNN-SIMO **RTF 0.19**、oBSRNN 0.42、MDX-Net 0.13。
- **最佳折中**：**MDX-Net** —— 精度仅比 SOTA 低 0.31 dB，**RTF 0.13**（全场最快的深度 4-stem 方案）。
- **单目标最强**：**BS-RoFormer L12**，人声 SDR **9.69**（仅次于 Oracle 上界 10.03）；
  GPU 上 **RTF 0.49** —— 从 CPU 的「RTF 11 离线模型」变成**实时模型**（提速 22×）。
- ⚠️ **RTF 只报 GPU 数字是不够的**：`RPCA` 是纯 CPU 算法、完全不受 CUDA 影响，
  两次测量却在 1.6~4.6 之间跳 → 报区间，别报单点。深度模型建议 CPU/GPU 双报（见 `MODEL_REGISTRY.md` §5.1）。
- ⚠️ **Demucs 的口径更正**：早期「Demucs 精度最高（人声 8.74 dB）」来自**单曲 30 s 片段 + 6 方法对比池**；
  并入统一基准后 Demucs 四轨均值 **8.11 dB**（鼓 8.96 / 贝斯 10.87 属第一梯队，**other 5.49 偏弱**）。
  它的工程优势是 **41.98 M 单模型直接出 4 stem、GPU 上 RTF 0.03**（全场最快）。CPU 上非确定性（±0.1~0.3 dB）。

---

## 1. 核查总表

| # | 模型 | 开源 | 许可证 | 预训练权重 | 本机跑通 | 失败/受限原因 |
|---|---|---|---|---|---|---|
| 1 | RPCA | ✅ 自研 | — | 不需要（解析法） | ✅ **PASS** | 需 librosa 垫片绕 SAC（见 §4.1） |
| 2 | RPCA+DRNN | ❌ **未公开** | — | 无 | ❌ 需自研 | 论文无代码；详见 §3.2 |
| 3 | Conv-TasNet | ✅ | MIT（原仓库）/ 镜像仓库无 LICENSE | ✅ MUSDB18 ×3 | ✅ **PASS** | — |
| 4 | DPRNN | ✅ | Apache-2.0 | ⚠️ 只有 wsj0-mix / librispeech | ❌ 无 MUSDB18 权重 | 需自训；`numpy.random` 现已可用，自训不再被环境阻塞 |
| 5 | MMDenseLSTM | ✅ | 镜像仓库**无 LICENSE** | ✅ MUSDB18 (paper) | ✅ **PASS** | — |
| 6 | BS-RoFormer L12 | ✅ | MIT | ✅ ep_317 (639 MB) | ✅ **PASS** | 需 librosa 占位模块（见 §4.2） |
| 7 | BS-RoFormer L6 | ✅ | MIT | ✅ ep_937 (393 MB) | ✅ **PASS** | 同上；且**输出语义与 yaml 标注不符**（见 §3.6） |
| 8 | Band-Split RNN (BSRNN) | ✅ | MIT | ✅ Zenodo ×3 (4.67 GB, MD5 全过) | ✅ **PASS** | opt / large / SIMO 三变体均已跑通；需占位 lightning（见 §3.8） |
| 9 | IRM / IBM Oracle | ✅ 自研 | — | 不需要（解析法） | ✅ **PASS** | 需 torch.stft 替代 librosa（见 §4.3） |
| 10 | MDX-Net | ✅ | MIT | ✅ mdx_extra ×4 + onnx ×1 | ✅ **PASS** | 需 `torch.load(weights_only=False)` 补丁（见 §4.4） |
| 11 | Pac-HuBERT-SEP | ❌ **无公开代码/权重** | — | 无 | ❌ 不可得 | MERL 未放出（见 §3.11） |

---

## 1.5 评测口径与片段选取（读数字前必看）

**指标口径**
- `SDR / ISR / SAR`：**museval（BSSEval v4）**，1 s 窗、取中位数，单声道 —— MUSDB18 官方口径。
- `SI-SDR`：全局单值，作为交叉验证。两者趋势一致才说明结论稳健（SDR 一般略高于 SI-SDR）。
- ⚠️ **ΔSDR**（相对「不分离基线」的增益）才是唯一可跨数据集横比的量；**绝对 SDR 不可横比**。

**片段选取 —— 本轮发现的一个真问题**
最初所有模型都在 `A Classic Education - NightOwl` 的**前 10 s** 上评测。后来一量能量分布：

| 片段 | vocals | drums | bass | other |
|---|---|---|---|---|
| NightOwl 前 10 s | 9.9% | **1.4%** | **85.4%** | 7.6% |

这段是低频引子 —— bass 占 85% 能量、鼓只占 1.4%，
于是**所有模型**的 bass SDR 一起虚高到 20 dB 以上（MDX 21.2、oBSRNN 22.5、SIMO 22.7），
而这条数字**完全不能外推**。

因此本轮补充了 `--offset` 参数，并用 `tools/_scan_balanced_clip.py`
按「4 个 stem 能量占比的最小值」筛出均衡片段，得到**多片段中位数**口径：

| 片段 | vocals | drums | bass | other | 最小占比 |
|---|---|---|---|---|---|
| `ANiMAL - Clinic A` @140 s | 23.0% | 18.5% | 22.9% | 35.6% | **18.5%** |
| `Creepoid - OldTree` @60 s | 42.4% | 15.6% | 26.7% | 15.3% | 15.3% |
| `Dark Ride - Burning Bridges` @180 s | 20.7% | 36.9% | 28.4% | 14.0% | 14.0% |

**结论一律以「多片段中位数」为准**；单片段数字只用于「是否跑通」的判定。

**测量的非确定性（引用数字时必须标注）**
- **BS-RoFormer**：同输入两次 8.83 / 8.73（±0.1 ~ 0.3 dB）→ 报单曲数字须标注波动。**Open-Unmix / BSRNN 是确定性的。**
- **MDX-Net**：同片段重复测量有 ±0.1 dB 抖动（drums 6.145 ↔ 6.218）。
- **RTF 受机器负载影响极大**：本轮所有 RTF 均取自**机器空闲时的一次连续测量**；
  下载任务并行时同一模型的 RTF 会翻倍（例：RPCA 18 s → 35 s，BS-RoFormer L12 10.2 → 11.1）。
- 因此对外报数一律：**精度给中位数 + n**，**速度给单次受控测量 + 波动区间**。

---

## 1.6 主结果表：多片段中位数（3 个均衡片段）

数据来源 `04_reports/data/comparison/model_runs_median.json`（由 `tools/median_over_clips.py` 生成）。

配套图（`04_reports/figures/comparison/`，由 `tools/plot_median_summary.py` 生成）：

| 图 | 文件 | 看什么 |
|---|---|---|
| 四轨均值条形图 | `fig_median_sdr_bars.png` | 12 模型总体精度排序；单目标模型与 RPCA 单独标注 |
| 精度-速度散点 | `fig_median_pareto.png` | 对数 RTF 横轴，绿点 = 可实时（RTF<1）；找「快+准」折中最直观 |
| 分轨对照 | `fig_median_per_stem.png` | 人声/鼓/贝斯/其他 四轨分组，看各模型的强项与短板 |

**出全 4 个 stem 的模型**（按四轨均值降序；**RTF 主列为 GPU**，CPU 列供对照）：

| 模型 | 人声 | 鼓 | 贝斯 | 其他 | **四轨均值** | RTF(**GPU**) | RTF(CPU) | n |
|---|---|---|---|---|---|---|---|---|
| **oBSRNN-SIMO** | 8.65 | **9.66** | 11.72 | **7.03** | **9.27** | **0.19** | 1.87 | 3 |
| oBSRNN（4× 单目标 ckpt） | **9.17** | 9.19 | **11.92** | 6.37 | 9.16 | 0.42 | 5.17 | 3 |
| MDX-Net (mdx_extra) | 7.95 | 8.65 | **12.98** | 6.26 | 8.96 | **0.13** | 0.88 | 3 |
| BSRNN large（4× 单目标 ckpt） | 8.28 | 9.24 | 10.66 | 5.49 | 8.42 | 0.32 | 3.83 | 3 |
| IRM/IBM Oracle（参考上界） | 10.03 | 8.48 | 8.55 | 6.15 | 8.30 | 0.01 | 0.01 | 3 |
| **Demucs (htdemucs)** | 7.11 | 8.96 | 10.87 | 5.49 | **8.11** | **0.03** | 0.39 | 3 |
| Open-Unmix (umxhq) | 6.05 | 7.31 | 8.35 | 4.90 | 6.65 | 0.06 | 0.12 | 3 |
| MMDenseLSTM (musdb18) | 5.60 | 7.15 | 7.36 | 4.96 | 6.27 | 0.08 | 0.27 | 3 |
| Conv-TasNet (musdb18) | 5.00 | 7.78 | 9.29 | 1.66 | 5.93 | 0.18 | 0.64 | 3 |
| **DPRNN（自训）** | -2.18 | -1.86 | -0.82 | -2.02 | **-1.72** ⚠️ | 0.03 | 0.05 | 3 |
| RPCA | —（无天然归属，见 §3.1） | — | — | — | — | 4.56 | 1.61 | 3 |

> ⚠️ **DPRNN 的负分不是 bug，是欠训练**：自训 8 GiB 单卡、**300 分钟预算**（实跑 250.7 min、step 31900）
> 仍只到「个位数 epoch」量级，模型还没学会「不破坏混合信号」。它**不参与精度排名**，
> 只作为一个负面结论用证据。见 `MODEL_REGISTRY.md` §6。
>
> 📈 **本轮已刷新（2026-09-19）**：DPRNN 权重从 step 3045 续训到 **step 31900**，
> 验证集均值 SI-SDR 由 **+0.781 → +2.745 dB**（+2.0 dB）；测试集四轨均值 SDR 随之
> 由 **-4.54 → -1.72 dB**（**+2.82 dB，提升 62%**），四轨**全部收窄但仍为负**，
> 其中贝斯最好（-0.82）、人声 -2.18、鼓 -1.86、其他 -2.02。
> ⚠️ **SI-SDR 已明显转正**（人声 3.34 / 鼓 4.78 / 贝斯 5.05），只有 `other` 仍为负（-2.99）——
> 两个指标方向一致，说明模型确实学到了东西，只是 museval SDR 口径更严苛。

>
> ⚠️ **RTF 的两个陷阱**：
> ① 受机器负载影响极大（同一模型不同轮次可差 30%+）→ 对外只报「是否 < 1」，要精确值须在空闲机器上重测；
> ② `RPCA` 是**纯 CPU 算法**、完全不受 CUDA 影响，两次测量却在 1.6~4.6 之间跳 → **报区间，别报单点**。
>
> ⚠️ **MDX-Net 鼓轨异常：已复测结案（2026-09-19）**
> 现象：`ANiMAL - Clinic A` @140 s 的**鼓轨**，CPU 轮测得 **11.96**、GPU 轮测得 **6.99**
> （其余两个片段与全部轨的 CPU/GPU 差均 < 0.1 dB）。
> **复测结果：GPU 上重跑得 6.97 —— 与 6.99 一致，是稳定值而非随机抖动。**
> **结论**：该片段鼓轨在两个设备上是**真实的系统性差异**（非测量噪声），
> 成因未定，但**不影响任何对外结论** —— 三片段中位数口径下鼓轨为 **8.65 dB**，
> 该片段只是 3 个样本之一，且 MDX 的鼓轨**从来不是它的强项**（强项是贝斯 12.98）。
> **引用口径：MDX 鼓轨一律用中位数 8.65 dB，不要单独引用 ANiMAL 片段的值。**


**单目标模型**（不产出完整 4 stem）：

| 模型 | 目标 | museval SDR | SI-SDR | RTF(GPU) | RTF(CPU) | n |
|---|---|---|---|---|---|---|
| BS-RoFormer L12 (ep_317) | `vocals` | **9.69** | 9.17 | **0.49** | 10.97 | 3 |
| BS-RoFormer L6 (ep_937) | `vocals+other` | 10.06 | 8.68 | **0.23** | 4.31 | 3 |

### 从这张表能得到的硬结论

1. **oBSRNN-SIMO 是唯一一个「单模型出 4 轨、四轨均值最高」的方案（9.27 dB）**，
   且它只用 **108.73 M** 参数，比「4 个单目标 ckpt 聚合」的 164.12 M 少 34% —— 架构效率最优。
   **GPU 上 RTF 0.19，实时**。
2. **MDX-Net 是最强的「快 + 准」平衡点**：四轨均值 8.96（与 SIMO 只差 0.31 dB），
   但 **RTF 0.13 —— 全场最快的深度 4-stem 方案**（比 SIMO 快 1.5×，比 Demucs 慢一点但精度高 0.85 dB）。
   它唯一明显的短板是 `vocals`（7.95 vs SIMO 8.65）。
3. **BS-RoFormer L12 的 `vocals` 9.69 是所有可用方案里最高的**（仅次于 Oracle 上界 10.03）。
   ⚠️ **它的实时性完全取决于有没有 CUDA**：CPU 上 RTF 10.97（离线模型），GPU 上 **0.49（实时）**，提速 22×。
4. **Demucs 四轨均值 8.11 dB，但它是「精度/算力」最省事的方案**：41.98 M 单模型直接出 4 轨、**GPU 上 RTF 0.03（全场最快）**。
   分轨上鼓 8.96、贝斯 10.87 属第一梯队，**`other` 只有 5.49** 是它最弱的一环。
   ⚠️ 早期「Demucs 精度最高（8.74）」是单曲 30 s 片段 + 6 方法池的结论，**已作废**。
5. **Oracle 上界只有 8.30 均值**，说明这三个片段本身不"容易"；
   同时 Oracle 在 **vocals 上（10.03）反而是全场最高** —— 说明这三个片段的鼓/贝斯对掩码法并不友好
   （鼓是宽频瞬态，IRM 的逐 bin 掩码理论最优但受 STFT 分辨率限制）。
6. **Conv-TasNet 的 `other` 只有 1.66 dB**，是全场最弱项，与它「时域端到端、无显式谐波先验」的特性一致。
   自训的 DPRNN 同族（时域端到端）**四轨均值仍为负（-1.72）**，进一步印证这个方向对数据/算力的要求最高。
7. **单片段口径与多片段口径差异极大**（例：oBSRNN-SIMO 的 bass 从 22.71 → 11.72；
   Oracle 的 bass 从 17.77 → 8.55）。**论文里只能引用本表**，单片段数字不可用。
8. **⭐ 实时性结论（本项目主题）**：装上 CUDA 后，**精度前 3 名全部实时**
   （oBSRNN-SIMO 0.19 / oBSRNN 0.42 / MDX-Net 0.13），BS-RoFormer L12 也进入实时（0.49）。
   **「实时」与「高精度」在这台 8 GiB 笔记本上不再是取舍关系。**

---

## 2. 本机最大的坑：Smart App Control 拦截 DLL

### 2.1 现象与根因

Windows **Smart App Control (SAC)** 在本机处于**强制开启**状态：

```
注册表 HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy
  VerifiedAndReputablePolicyState = 1   ← 强制模式，用户无法关闭
```

它按「代码完整性策略」拦截未经签名/信誉不足的 PE 文件加载，报错形如：

```
OSError: Could not find/load shared object file 'llvmlite.dll' ...
```

**判定方式**：用 `ctypes.WinDLL()` 直接加载 .pyd/.dll，若返回
`OSError: [WinError 4551] 应用程序控制策略已阻止此文件。` 即为 SAC 拦截（而非缺依赖）。

### 2.2 被拦 / 未被拦矩阵（2026-09-16 重新实测）

> ⚠️ **本节结论已在 2026-09-16 晚修正。** 上午的实测里 `numpy.random._philox` / `_generator`
> 被 SAC 拦截，连带 `scipy` / `sklearn` / `pandas` / `museval` 全挂；**下午复测时这批文件已恢复加载**，
> 拦截面收窄到只剩 `llvmlite.dll` 一条。推测原因是这些二进制随后获得了系统的信誉评级。
> 下表为**当前有效**状态（`ctypes` 逐文件实测）。

| 状态 | 模块 | 连带失效的上层 |
|---|---|---|
| 🚫 **仍被拦** | `llvmlite/binding/llvmlite.dll`　（WinError **4551**） | `numba`、`librosa.stft` / `librosa.filters`、`pysepm` |
| ✅ 已恢复 | `numpy/random/*.pyd`（含 `_philox`、`_generator`、`_pcg64`） | `numpy.random`（`default_rng` 实测可用） |
| ✅ 已恢复 | `scipy.signal`、`scipy.ndimage` | 失真滤波、窗函数 |
| ✅ 已恢复 | `sklearn.decomposition` | `NMF` / `FastICA` |
| ✅ 已恢复 | `pandas`、`torchmetrics`、`pystoi` | — |
| ✅ **已恢复** | `museval`、`mir_eval` | **BSSEval v4 的 SDR 可以算了** |
| ✅ 正常 | `torch`、`torchaudio`、`soundfile`、`numpy.core`、`matplotlib`、`omegaconf`、`hydra`、`yaml` | — |
| ⚠️ 版本问题 | `lightning` | 与 torch 2.14 私有 API 不兼容（**非 SAC 问题**）|

实测环境版本：`numpy 2.5.3` / `scipy 1.18.1` / `sklearn 1.9.0` / `pandas 3.0.5` / `torch 2.14.0+cu126`（2026-09-17 起，见 §2.2b/§2.2c）。

### 2.2b ⚡ 拦截面再次变化：**已全部恢复**（2026-09-17 00:54:55 实测）

> ⚠️ **这是本项目里 SAC 拦截面第 3 次变化。** 任何「环境阻塞」结论**必须带时间戳**，
> 引用历史记录前一律先重跑功能探测。

| 项 | 2026-09-16 晚 | **2026-09-17 00:54:55** |
|---|---|---|
| `llvmlite` | 🚫 被拦（WinError 4551） | ✅ **已恢复** |
| `numba.njit` | 不可用 | ✅ **实算通过**（JIT 求和 332833500.0） |
| `librosa.stft` | 不可用 | ✅ **实算通过**（`(1025, 33)`） |
| `librosa.filters` | 不可用 | ✅ 可用 |
| `pysepm.pesq` | 受 `llvmlite` 牵连 | ⚠️ 仍不可用，但根因是**缺 `srmrpy` 普通依赖**（`ModuleNotFoundError`），**与 SAC 无关** |
| `torch` | `2.14.0+cpu` | **`2.14.0+cu126`（GPU 可用）** |

**结论**：SAC 目前**不再拦截本项目需要的任何 Python 原生扩展**。
§4 那 5 处工程改造（librosa 占位模块等）**可以保留**（它们是「能用真的就用真的」的写法，不产生副作用），
但它们**已不再是必需项**；论文里不应再把 SAC 写成当前限制。

### 2.2c GPU 启用（2026-09-17）

`torch 2.14.0+cu126` + `torchaudio 2.11.0+cu126` 安装成功（`--index-url https://download.pytorch.org/whl/cu126`，wheel 2.6 GB）。

```
torch 2.14.0+cu126 | cuda build 12.6 | avail True
dev NVIDIA GeForce RTX 4070 Laptop GPU (8, 9)  | 512×512 @ 运算实算通过
torchaudio 2.11.0+cu126 import OK, resample OK
SAC 未拦截任何 CUDA DLL（cudart / cublas / cudnn 全部正常加载）
```

> 配对关系（踩过的点）：PyPI 上 **没有** torchaudio 的 CUDA wheel，
> 必须从 `download.pytorch.org/whl/cu126` 装，且 **torch 2.14.0 ↔ torchaudio 2.11.0** 才是配套版本
> （cu126 索引里 torchaudio 最新就是 2.11.0）。
> 另外 `+cu128` 索引**没有** cp313/win_amd64 的 torch 2.14.0，只有 `+cu126` 与 `+cu130`。

**⚠️ 因此：本文所有 RTF 数字仍然是 CPU 数字**，装完 GPU 后需要**全部重跑**才谈得上「实时性」结论（见 §7）。

### 2.3 对评估口径的直接影响（重要）

**`museval`（BSSEval v4）现已可用 → 本轮 11 模型基准已能给出真实的 `SDR`，与 `SI-SDR` 并列报告。**

- `SI-SDR`（全局单值）与 `SDR`（1 s 窗中位数）**不可直接互换**：后者对局部伪影更宽容，数值通常更高。
  例：Open-Unmix 人声 `SI-SDR 6.263` vs `SDR 6.492`；MDX-Net 贝斯 `21.812` vs `21.229`。
- 与论文对齐时**以 `SDR` 为准**；`SI-SDR` 作为补充。
- **仍未解决的限制**：本机 `torch` 是 `+cpu` 版，且只有 10 s 单片段（论文是整曲 test 集），
  因此数值**只能用于「模型之间横向比较」，不能当作 MUSDB18 官方成绩**。
- `llvmlite` 仍被拦 ⇒ `librosa.stft` 不可用 ⇒ RPCA/NMF 仍走 §4.1 的 torch 垫片，
  BSRNN/MDX 仍走 §4.2 的 librosa 占位模块。**但这两处绕行现在只影响 librosa，不影响指标。**


---

## 3. 逐模型详解

### 3.1 RPCA（Inexact-ALM）

- **开源**：✅ 项目自研实现（`tools/compare_separation_methods.py::_rpca_ialm`）。
- **权重**：不需要，是解析/迭代算法。
- **运行结果**：✅ PASS，10 s 片段 **35.1 s（RTF 3.51）**，产物 `03_outputs/RPCA/<song>/vocals.wav`。
  ⚠️ RTF 波动大：同一片段早先测得 **18.0 s（RTF 1.80）**，本次空闲机器复测为 35.1 s。
  两次环境已不同（`numpy` 升到 2.5.3 / `scipy` 升到 1.18.1），且 RPCA 每轮做全秩 SVD，
  对 BLAS 线程与实现版本极敏感。**给结论时应写「RTF ≈ 1.8 ~ 3.5」，而不是单点值。**
- **关键结论**：RPCA 输出的是「稀疏分量」，**没有天然的 stem 归属**。本曲交叉评估：

  | 对比目标 | vocals | drums | bass | other |
  |---|---|---|---|---|
  | SI-SDR (dB) | −7.90 | −19.28 | **+3.96** | −13.50 |

  最佳匹配是 **bass**，对 vocals 是**有害的负增益**。所以「RPCA 好不好」完全取决于把它映射到哪个目标——报告里不能写单一数字。
  （RPCA 无固定 stem 语义，故不给 SDR 单元格，只给跨 stem 的 SI-SDR。）
- **RTF > 1** → 在本机**跑不动实时**（与项目此前结论一致）。

### 3.2 RPCA+DRNN

- **开源**：❌ **未公开代码**。
- **溯源结论**：图谱里 RPCA+DRNN 的 6.41 dB 出自 **Lai & Wang, EURASIP J. Audio Speech Music Process. 2022:4**（DOI `10.1186/s13636-022-00236-9`）Table 7。
  ⚠️ **口径不可横比**：该论文是**单声道、2 源（人声 / 伴奏）**任务，**不是** MUSDB18 的 4-stem 任务。
- **结论**：无法「下载」，只能按论文自研复现。
- **✅ 决策（2026-09-17）：不复现，降级为「相关工作」。** 三条理由：
  1. **无官方代码 → 结果无法验证**。自研实现的「是否复现到位」没有任何可对齐的基线，产出的数字不可信；
  2. **口径不可横比**（上条已述）：单声道 2 源 vs MUSDB18 4-stem，放进同一张表是方法论错误；
  3. **投入产出比差**：3–5 天工作量换一个无法验证的数字。
- **替代覆盖**：该方案的**两条思路本项目都已各自独立实现并实测** ——
  古典分解那一路是 `--model rpca`（RPCA/Inexact-ALM，产物 `03_outputs/RPCA/`），
  时域 DL 那一路是 `--model dprnn`（DPRNN，本机自训，产物 `03_outputs/DPRNN/`，见 §3.4）。
  论文里以「相关工作 + 这两条独立线索」的形式讨论即可。
- 占位说明：`03_outputs/RPCA-DRNN/README.md`（目录**保持为空**，避免被误读成复现结果）。

### 3.3 Conv-TasNet

- **开源**：✅ [tky823/DNN-based_source_separation](https://github.com/tky823/DNN-based_source_separation)（原始 Conv-TasNet 由 Luo & Mesgarani 提出，MIT）。
- **许可证**：⚠️ 原 Conv-TasNet 仓库 MIT；但**镜像仓库 `DNN-based_source_separation` 没有 LICENSE 文件**（许可风险，需在论文中谨慎表述）。
- **权重**：MUSDB18 预训练 ×3（`4sec_L20` / `8sec_L20` / `8sec_L64`），Google Drive，已下载并 `torch.load` 校验通过。
- **运行结果**：✅ PASS，**load 0.23 s / infer 6.75 s（RTF 0.675）**，参数量 **13.39 M**。

  | 目标 | vocals | drums | bass | other |
  |---|---|---|---|---|
  | museval SDR | 4.73 | 2.05 | 17.30 | 5.58 |
  | SI-SDR | 4.19 | −1.83 | 18.14 | 4.92 |

- **踩坑**：`extract_latent` 内部硬编码 `assert C_in == 1`，输入必须显式带上单通道维 → 传 `(1, 1, n_mics, T)`；按时间轴做 mean/std 标准化。

### 3.4 DPRNN（自训，MUSDB18-HQ 4-stem）

- **开源**：✅ `01_models/_third_party/Dual-Path-RNN-Pytorch`
  （**JusperLee** 的实现，Apache-2.0；⚠️ 此前本报告误记为 tky823，已更正 —— tky823 的
  `DNN-based_source_separation` 里**没有** DPRNN）。
- **权重**：⚠️ 官方只发布 **wsj0-mix / librispeech（语音分离）**权重，**没有 MUSDB18 音乐权重**。
- **结论**：❌ 拿不到现成权重，**无法满足「下载权重后跑推理」的验收标准** → 只能自训。

#### 3.4b 自训管线（2026-09-17 落地）

| 项 | 内容 |
|---|---|
| 训练脚本 | `tools/train_dprnn_musdb.py`（可续跑 `--resume`、按时间预算自动停、每 N 步验证并存 best/last） |
| 评测适配 | `tools/_dprnn_loader.py`（训练与评测**共用同一套模型构造**，ckpt 不匹配直接抛错） |
| 基准入口 | `tools/benchmark_model_universal.py --model dprnn` |
| ckpt | `01_models/_weights/dprnn_musdb/{best,last}.pt` |
| 训练曲线 | `04_reports/figures/dprnn/training_curve.png` |
| 训练历史 | `04_reports/data/dprnn_training.json` |
| 日志 | `05_misc/logs/train_dprnn_musdb.txt` |

**只复用模型定义，不复用它的数据管线**：原仓库是 WSJ0 语音 + `.scp` 格式，
本脚本另写 MUSDB18-HQ 管线（读 4 个 stem → 单声道 → 合成 mixture = Σstems）。

关键设定：

- **采样率 11025 Hz**（44.1 kHz 的 1/4，`resample_poly(x,1,4)` 整数抽取，无重采样伪影）。
  这是音乐 DPRNN 类工作的通行训练率。
  **评测时会被升回 44.1 kHz**，但 **11 kHz 以上的带宽损失会如实计入 SDR**（升回去不会凭空长出高频），
  报告里必须说明 —— 这是该模型数字偏低的结构性原因之一，不是实现 bug。
- **模型**：`kernel_size=16, in_channels(N)=64, out_channels=128, hidden_channels=128,
  LSTM, norm=ln, bidirectional, num_layers=6, K=200, num_spks=4`，**3.69 M 参数**。
  4 s chunk（44100 采样点，对齐到 stride=8）。
- **损失**：负 SI-SDR（DPRNN 原文口径），4 个 stem 取均值，**静音 stem 按参考能量自动屏蔽**。
- **混合精度**：bf16 autocast（连续 NaN 自动回退 fp32）。
- **硬件**：RTX 4070 Laptop 8 GiB，batch 4，约 **126 step/min**（≈0.48 s/step，3.9 GB 显存）。

⚠️ **两个必须记住的坑**

1. **长度必须对齐到 stride**：编码器 `Conv1d(kernel=16, stride=8)` 后解码得到
   `floor(L/8)*8`，所以输入长度**必须**是 8 的整数倍，否则 est 比 ref 短几个采样点直接崩
   （`RuntimeError: size of tensor a (44096) must match b (44100)`）。
2. **多输出模型按位置对齐**：ckpt 里显式存 `targets`，加载端必须按 ckpt 顺序构造 4 个 mask 分支，
   否则不报错、能出声、wav 正常，但**指标静默串味**（同 §4 的 BSRNN-SIMO 陷阱）。

#### 3.4c 自训结果（3 个均衡片段，GPU）

**训练规模（2026-09-19 最终版）**：batch **8**、300 min 预算、实跑 **250.7 min**（step 3404 → **31900**）、
lr 从 5e-4 退火到 1e-5、bf16 autocast、峰值显存 5.88 GiB。

验证集进展（每 100 步在 3 个均衡片段上算 SI-SDR）：

| 阶段 | step | 验证均值 SI-SDR | vocals | drums | bass | other |
|---|---|---|---|---|---|---|
| step 3045 旧 ckpt | 3045 | +0.781 | +2.47 | +2.86 | +2.26 | −4.46 |
| step 3404 | 3404 | +1.366 | +3.76 | +3.02 | +2.77 | −4.09 |
| **step 31900（最终）** | 31900 | **+2.745** | +3.72 | +5.16 | +5.09 | −3.00 |

**测试集四轨中位数 SDR（museval）**：

| 指标 | vocals | drums | bass | other | 四轨均值 |
|---|---|---|---|---|---|
| 旧 ckpt（step 3045） | −3.83 | −3.57 | −4.35 | −6.40 | **−4.54** |
| **新 ckpt（step 31900）** | −2.18 | −1.86 | −0.82 | −2.02 | **−1.72** |
| **提升** | +1.65 | +1.71 | **+3.53** | **+4.38** | **+2.82** |

- **结论：确实学到东西了，但仍远不到能用** —— 四轨全部收窄却**仍为负**（SDR<0 表示比直接用混合信号还差）。
- **两个指标方向一致**：SI-SDR 侧人声 3.34 / 鼓 4.78 / 贝斯 5.05 已明显转正，只有 `other` 仍为负（−2.99）。
  museval SDR 口径更严苛（1 s 窗逐窗算再取中位数），所以整体更低 —— 这不是矛盾，是口径差异。
- **贝斯进步最大（+3.53）**，`other` 从 −6.40 收窄到 −2.02（+4.38）—— 与验证集里 drums/bass 涨得最多吻合。
- **本表用途不变**：作为「端到端时域方法需要多少算力才能达标」的负面证据，**不参与精度排名**。

> 数值见 §1.6 主结果表；训练进程与验证曲线见上表所列产物路径。

### 3.5 MMDenseLSTM

- **开源**：✅ tky823 镜像仓库（**无 LICENSE 文件**，同 §3.3 风险）。
- **权重**：MUSDB18 (paper) 配置，Google Drive，已下载并校验（zip 内是 `model/<target>/best.pth`，4 个目标）。
- **运行结果**：✅ PASS，**load 0.36 s / infer 2.97 s（RTF 0.297）**，参数量 **5.48 M**（4 目标合计，单目标约 1.37 M）。
  配置：`n_fft=4096 / hop=1024 / sections=[380,644,1025]`，与论文一致。

  | 目标 | vocals | drums | bass | other |
  |---|---|---|---|---|
  | museval SDR | 7.02 | 2.37 | 17.13 | 7.83 |
  | SI-SDR | 6.68 | −0.99 | 17.58 | 7.59 |

- **踩坑（两个类不能混用）**：
  - 单目标 `MMDenseLSTM` 的 `forward` 签名里没有 target → 报 `RuntimeError: size of tensor a (2) must match b (2049)`；
  - 必须用 `ParallelMMDenseLSTM.build_from_pretrained(...)` + **`ParallelMMDenseLSTM`**`.TimeDomainWrapper`（单目标版 `MMDenseLSTM.TimeDomainWrapper` 输出形状 `(B,C,T)`，会 `IndexError`）。
  - 输入 `(1, 1, n_mics, T)`，输出 `(1, n_sources, n_mics, T)`。

### 3.6 BS-RoFormer L12（ep_317）与 L6（ep_937）

- **开源**：✅ 架构 [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)（MIT）+ 官方推理链路 [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)（MIT）。
- **权重**：`TRvlvr/model_repo` releases，大小精确匹配
  - `model_bs_roformer_ep_317_sdr_12.9755.ckpt` = **639,331,213 B** ✅
  - `model_bs_roformer_ep_937_sdr_10.5309.ckpt` = **393,068,365 B** ✅
- **运行结果**：

  | 变体 | dim/depth | 参数量 | 目标 | museval SDR | SI-SDR | RTF (CPU) |
  |---|---|---|---|---|---|---|
  | L12 = ep_317 | 512 / 12 | **159.76 M** | vocals | **10.00** | 11.10 | **10.73** |
  | L6 = ep_937 | 384 / 12 | **98.19 M** | vocals+other | **12.63** | 12.13 | **4.81** |

- ⚠️ **发现 1：两个官方权重都是「单目标」模型**（各自 yaml `num_stems: 1` + `training.target_instrument` 指定唯一目标），**不是 4-stem 模型**。L12 出人声，L6 出「除鼓贝斯外全部」。
- ⚠️ **发现 2（重要）：ep_937 的实际输出 ≠ yaml 里写的 `other`。**
  它的 yaml 写 `target_instrument: other`，但实测：

  | 对比参考 | MUSDB 的 `other` | **vocals+other** | instrumental(v+o) | mixture |
  |---|---|---|---|---|
  | SI-SDR (dB) | 0.83 | **12.13** | −8.05 | −7.96 |
  | 相关系数 | 0.740 | **0.971** | 0.368 | 0.372 |

  即它输出的是 **`mixture − drums − bass`（vocals+other）**，与 ZFTurbo `justfile` 注释 *"this removes drums and bass only, keeping the rest"* 完全吻合。
  → 若直接拿它和 MUSDB18 的纯 `other` 比，会**误判为模型失效（0.83 dB）**。已在基准脚本里加 `ref_map` 机制修正参考信号。
  → 另注：ZFTurbo 官方文档给该 ckpt 的 "other" SDR 是 **6.85**（Multisong Dataset），文件名里的 10.5309 并非指该目标的 SDR。
- 🎯 **性能对照（关键工程结论）**：L12 在 4 首歌上测得 SI-SDR **11.10 / 15.38 / 12.42 dB**（第 4 首 GT 人声为空 → 退化），**稳定且与标称 SDR 12.98 相符**，说明推理链路数值正确。
- ⚠️ **发现 3：BS-RoFormer 在本机 CPU 上不是实时的（RTF 10.17 / 4.18）**。要实时必须用 CUDA，而当前 torch 是 `+cpu` 版 → 这是 FYP「实时」目标上的一个硬约束，见 §5。

### 3.7 IRM / IBM Oracle

- **开源**：✅ 解析法（Ideal Ratio Mask / Ideal Binary Mask），项目自研，参考 [sigsep/mus-oracle](https://github.com/sigsep/sigsep-mus-oracle)（MIT）。
- **用户笔误确认**：原始需求中的 "IRN oracle" 应为 **IRM/IBM** oracle，由 GT 解析计算，**无需权重**。
- **运行结果**：✅ PASS，RTF **0.006**（几乎零成本）。

  | 目标 | vocals | drums | bass | other |
  |---|---|---|---|---|
  | SI-SDR (dB) | 8.16 | 0.68 | 17.97 | 8.73 |

### 3.8 Band-Split RNN (BSRNN)

- **开源**：✅ [magronp/bsrnn](https://github.com/magronp/bsrnn)（"An Open and Optimized Implementation of BSRNN"，**MIT**），
  配套论文 *The Costs of Reproducibility in Music Separation Research: a Replication of Band-Split RNN*（arXiv:2603.09187）。
- **权重（3 个 zip，共 4.67 GB）—— 全部下载完成且 MD5 逐一校验通过**：

  | 文件 | 大小 (B) | MD5 | 校验 |
  |---|---|---|---|
  | `bsrnn-opt.zip` | 1,827,060,285 | `89075aac776f82295a074a56d0428a18` | ✅ |
  | `bsrnn-large.zip` | 1,627,580,581 | `0d1e298e725bf45577a4f61e0a30c2b1` | ✅ |
  | `simo-bsrnn-opt.zip` | 1,213,413,202 | `817e8ce4bcd17c38bd3812167e97db58` | ✅ |

  ```text
  bsrnn-opt/        bass.ckpt  drums.ckpt  other.ckpt  vocals.ckpt      ← 4 个单目标模型
  bsrnn-large/      bass.ckpt  drums.ckpt  other.ckpt  vocals.ckpt      ← 4 个单目标模型
  simo-bsrnn-opt/   separator.ckpt (1.31 GB)                            ← 1 个 4 目标模型
  ```

- ⚠️ **下载踩坑（值得记录）**：`curl -C -` 与 `--retry` 组合会在重试时把偏移量重置为 0 并**截断文件**（实测倒退 320 MB → 232 MB）；
  PowerShell 5.1 读 **BOM-less `.ps1` 会把中文当 ANSI** 导致语法错误。
  最终方案 `tools/par_download.py`：**按段并行（urllib + Range）+ 每段独立落盘 + 已下载前缀自动迁移 + 拼接后 MD5 校验**，
  单连接 44 KB/s → 6 连接 ~1.6 MB/s（≈8×），且任意中断都不丢进度。

- ⚠️ **推理依赖阻塞与绕行**：
  - `lightning 2.6.6` 与 **torch 2.14.0** 私有 API 不兼容（`cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'`）；
  - `models/pl_module.py` 在**模块级** `import pandas` / `lightning.pytorch` / `helpers.data`(musdb) / `helpers.eval`(museval)；
  - `separate.py` 还用 `Model.load_from_checkpoint(...)`（Lightning 类方法）。
  → 绕行（`tools/_bsrnn_loader.py`，**不改第三方源码**）：
  占位替换 `pandas` / `lightning.pytorch`（`LightningModule` 退化为 `torch.nn.Module`）/ `helpers.data` / `helpers.eval`，
  手工合成 hydra 风格 conf，再 `torch.load(weights_only=False)` + `load_state_dict`，推理直接调 `_apply_model_to_track`。

- ✅ **运行结果（10 s 片段，CPU）—— 三种变体全部跑通并落盘 stem wav**：

  | 变体 | 结构 | 参数量 | vocals | drums | bass | other |
  |---|---|---|---|---|---|---|
  | **oBSRNN**（4× 单目标 ckpt 聚合） | `joint_bandsplit=False` + attention + TAC | 164.12 M | 8.90 / 9.42 | 5.94 / 5.01 | 22.51 / 23.25 | 9.29 / 9.64 |
  | **BSRNN-large**（4× 单目标 ckpt 聚合） | `joint_bandsplit=False`，无 attention/TAC | 146.66 M | 8.95 / 9.41 | 6.32 / 5.05 | 20.45 / 21.70 | 8.58 / 8.72 |
  | **oBSRNN-SIMO**（单模型出 4 轨） | `joint_bandsplit=True` + attention + TAC | **108.73 M** | **9.58 / 10.79** | 6.10 / 5.34 | **22.71 / 23.41** | **10.60 / 11.02** |

  单元格格式：**museval SDR（1 s 窗中位数）/ SI-SDR**，单位 dB。

- 🏆 **oBSRNN-SIMO 是本轮所有模型里最强的 4-stem 方案**：vocals / bass / other 三项均为全场最高，
  drums 仅略低于 MDX-Net（6.10 vs 6.15）；且只用一个 108.73 M 的单模型，参数比「4 个单目标模型聚合」的 164.12 M 少 34%。
  与论文结论（oBSRNN-SIMO 10.66/9.73/10.98/7.78，全场最优）方向一致。

- 🔴 **踩坑（最危险的一个，值得写进论文的「复现陷阱」）**：**SIMO 模型的 state_dict 按位置对齐**。
  ckpt 里 `hyper_parameters.targets = ['vocals','bass','drums','other']`，
  若按别的顺序（例如 `['bass','drums','other','vocals']`）构造模型，4 个 masker 会**静默互换** ——
  不报错、不丢键、能出声，但指标整体串味。已在 `_bsrnn_loader.py` 加 `ckpt_targets()`：
  **从 ckpt 的 `hyper_parameters` 读原始 targets 顺序来构造模型，再映射回请求的目标**，并在不一致时打印告警。

- ✅ **逐键校验**：三变体的每个 ckpt 都是 `missing = 0 / unexpected = 0`。
  单目标模型的参数量逐目标不同（oBSRNN：vocals 40.78 / drums 45.49 / bass 37.07 / other 40.78 M），
  这是 masker 按频带数不同造成的，属正常；参数量与 ckpt 内张量总和不符即为加载错误的信号。


### 3.9 MDX-Net

- **开源**：✅ [KINoAI/mdx-net](https://github.com/KINoAI/mdx-net)（MIT）+ AIcrowd submission（MIT）。
- **权重捷径**：不需要走 KUIELab 的 ONNX 路线 —— Demucs 自带 `mdx_extra` 模型包（4 个 `.th`，各约 167 MB，`dl.fbaipublicfiles.com`），另有 `onnx_A.zip`(110 MB) 与 `mixer.ckpt`(1.2 KB) 备查。
- **运行结果**：✅ PASS，**load 2.17 s / infer 10.32 s（RTF 1.03）**。

  | 目标 | vocals | drums | bass | other |
  |---|---|---|---|---|
  | museval SDR | 8.52 | 6.15 | 21.23 | 8.88 |
  | SI-SDR | 8.86 | 5.01 | 21.81 | 8.62 |

- **踩坑**：torch ≥ 2.6 默认 `weights_only=True`，拒绝反序列化含类定义的 demucs checkpoint → `UnpicklingError`。修法：在调用点临时 monkey-patch `torch.load` 设 `weights_only=False`（权重已由 demucs 的文件名 sha256 前缀强校验，安全性可接受）。
- ⚠️ 该模型在本机 **RTF 1.03**，正好压在实时边界上（多次测量在 0.50 ~ 1.09 之间波动，受机器负载影响）。

### 3.10 DPRNN（自训）与 3.11 Pac-HuBERT-SEP

**Pac-HuBERT-SEP**：

- 用户需求中的 "PaG-HuBERT-SEP" 经确认是 **Pac-HuBERT**（Ke Chen et al., MERL/UCSD, arXiv:2304.02160）笔误。
- ❌ **无公开代码、无公开权重**（MERL 项目页未提供下载）。无法下载，也无法运行。
- **✅ 决策（2026-09-17）：论文中仅作相关工作引用，不再投入任何工时。**
  建议表述：「Pac-HuBERT-SEP (MERL) 未公开代码与权重，本文不予复现，仅在相关工作部分讨论其
  『自监督语音表征（HuBERT）+ 分离头』的思路对源分离的启发。」
  占位说明见 `03_outputs/Pac-HuBERT-SEP/README.md`（目录保持为空）。

**DPRNN** 见 §3.4（自训部分见 §3.4b）。

---

## 4. 为绕开 SAC 做的工程改造（5 处，均已落地）

| # | 位置 | 改造 | 解决的问题 |
|---|---|---|---|
| 4.1 | `tools/_librosa_shim.py`（新增）<br>`tools/compare_separation_methods.py` | 用 **功能性探测**（试跑 `librosa.stft` 往返）判断 librosa 是否可用；不可用则替换为 torch 后端的鸭子类型垫片 | `librosa.stft` 崩 → 连带 `sep_rpca` / NMF 全挂。**注意 `import librosa` 本身会成功**（子模块惰性加载），所以「导入探测」无效，必须功能探测 |
| 4.2 | `tools/benchmark_model_universal.py::_ensure_librosa` | 向 `sys.modules` 注入一个只有 `.filters` 属性的 `librosa` 占位模块 | ZFTurbo 的 `models/bs_roformer/__init__.py` 会连带导入 `MelBandRoformer`（模块级 `from librosa import filters`），导致**只想要 BSRoformer 也会被 llvmlite 拖死** |
| 4.3 | 同上 `_stft_istft()` | oracle 掩码改用 `torch.stft/istft`（hann, n_fft=2048, hop=512），与 librosa 默认口径等价 | 回避 `librosa.stft` → numba |
| 4.4 | `tools/_bsrnn_loader.py`（新增） | 占位替换 `pandas` / `lightning.pytorch` / `helpers.data` / `helpers.eval`；手工合成 hydra conf；绕过 `load_from_checkpoint` 改手工 `load_state_dict`；**并用 `ckpt_targets()` 按 ckpt 记录的原始 targets 顺序构造 SIMO 模型** | BSRNN 官方链路对 lightning（与 torch 2.14 不兼容）/ pandas / museval / musdb 的模块级依赖；以及 SIMO 多 masker 的**位置对齐陷阱**（见 §3.8） |
| 4.5 | `load_mdx()` | 局部 monkey-patch `torch.load` 临时 `weights_only=False`；并从本地 `01_models/_weights/MDX-Net/mdx_extra/` 走 `get_model(..., repo=local_repo)` | 免下载 + 反序列化兼容 |

另外修掉的 3 个 download / 环境坑：
- `curl -C -` + `--retry` 会截断文件 → 自写分段并行下载器 `tools/par_download.py`（详见 §3.8）。
- GitHub 克隆 DNN-based 仓库时因 Windows 260 字符路径限制文件全丢 → `git config core.longpaths true` 后 `git reset --hard` 还原（263 MB）。
- **BSRNN SIMO 的目标顺序陷阱** → `tools/_bsrnn_loader.py::ckpt_targets()`；见 §3.8，这是本轮最危险的一个坑。

**重要提醒**：以上绕行**并非都还需要**。`llvmlite` 仍被拦，所以 4.1 / 4.2 / 4.3 仍必要；
但 4.4 中的 `pandas` / `helpers.eval` 两项占位**已变为可选**（`pandas` / `museval` 已恢复加载）——
之所以保留，是因为「占位只在真的不可用时才生效」的写法更稳，能在拦截面再变化时自动回退。

---

## 4.5 附：Open-Unmix（项目既有基线，不在这 11 个模型之列）

为便于横向定位，把项目一直在用的 Open-Unmix（umxhq）与上表同口径列出（同一首歌、10 s、CPU）：

| 目标 | vocals | drums | bass | other |
|---|---|---|---|---|
| museval SDR | 6.49 | 1.88 | 17.36 | 7.58 |
| SI-SDR | 6.26 | −1.82 | 17.65 | 7.13 |

**RTF 0.113**，参数量单目标 8.90 M / 四目标合计 35.58 M。
它是「CPU 上又快又能打」的代表：比 oBSRNN-SIMO 精度低约 3 dB（人声），但快约 19×。

---

## 5. 对 FYP「实时」目标的直接影响（建议重点写进论文）

1. ✅ **本节已完成 GPU 重测（2026-09-17）**。左列 CPU、右列 GPU（RTX 4070 Laptop 8 GiB）。
2. 本机实测（**3 个均衡片段各 10 s，取中位数**）：

   | 模型 | RTF (CPU) | RTF (**GPU**) | 提速 | GPU 下实时 |
   |---|---|---|---|---|
   | Oracle (IRM) | 0.007 | 0.007 | — | ✅ 解析法 |
   | Demucs (htdemucs) | 0.389 | **0.028** | 14× | ✅ |
   | Open-Unmix (umxhq) | 0.122 | **0.059** | 2× | ✅ |
   | MMDenseLSTM | 0.271 | **0.082** | 3× | ✅ |
   | MDX-Net | 0.875 | **0.126** | 7× | ✅ |
   | Conv-TasNet | 0.641 | **0.176** | 4× | ✅ |
   | **BSRNN SIMO**（4-stem 单模型） | 1.874 | **0.191** | **10×** | ✅ **精度冠军也实时** |
   | BSRNN large (4× ckpt) | 3.832 | **0.321** | 12× | ✅ |
   | oBSRNN (4× ckpt) | 5.169 | **0.418** | 12× | ✅ |
   | BS-RoFormer L6 | 4.314 | **0.227** | 19× | ✅ |
   | **BS-RoFormer L12** | 10.968 | **0.494** | **22×** | ✅ **离线 → 实时** |
   | RPCA | 1.61 | 4.56 | — | ⚠️ **纯 CPU 算法**，不受 CUDA 影响，报区间 1.6~4.6 |
   | DPRNN（自训） | — | 0.016 | — | ✅（但精度欠训练，见 §0） |

3. **最终结论（GPU 口径）**：**FYP 要求的「实时 + 高精度」在本机同时成立** ——
   精度前三名 oBSRNN-SIMO（9.27 / RTF 0.19）、oBSRNN（9.16 / 0.42）、MDX-Net（8.96 / 0.13）**全部实时**；
   BS-RoFormer L12 人声 9.69（全场最高）也从 CPU 的 RTF 10.97 降到 **0.49**。
   - ⚠️ 若必须在 **CPU-only** 环境部署，可行组合只剩 Open-Unmix / MMDenseLSTM / Conv-TasNet。
   - ⚠️ BSRNN 三变体的 RTF 是「一次出 4 轨」的代价；只跑单目标模型时约为其 1/4。
   - ⚠️ 这些数字来自 **8 GiB 笔记本 GPU（有功耗墙）**，报论文时须附硬件型号与电源模式。
4. **CUDA 版 torch 的风险已实测排除（2026-09-17）**：
   `pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.0+cu126" "torchaudio==2.11.0+cu126"` 成功，
   SAC **没有拦截任何 CUDA DLL**（cudart / cublas / cudnn 全部正常加载），
   `torch.cuda.is_available()=True`，RTX 4070 实算通过。
   → 此前「SAC 可能拦 CUDA DLL」的顾虑**已解除**。
5. ✅ **已完成**：12 个模型 × 3 个均衡片段全部在 GPU 上重跑通过（13/13 PASS）。
   `model_runs.json` 现为 **GPU 数据**，CPU 基线另存 `model_runs_cpu_2026-09-16.json` 以便对照。

### 5.1 🔴 切到 CUDA 后才暴露的 4 个问题（均已修，值得记进论文「工程实现」章）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | **BSRNN 三变体 RTF 完全不动**（SIMO 1.87 → 2.08） | `tools/_bsrnn_loader.py` 里 **三处把 `device` 写死成 `"cpu"`**（无 CUDA 时代的遗留），`cfg.eval.device` / `model.eval_device` 都是 `"cpu"` | 改为按 `torch.cuda.is_available()` 动态判定（对齐官方 `separate.py:42` 的约定）→ **提速 10~12×** |
| 2 | **BS-RoFormer 报 `No module named 'utils.audio_utils'`** | `_dprnn_loader` 把 DPRNN 仓库塞进 `sys.path`，其 `utils/` 包**遮蔽**了 BSRoFormer 的同名包；反向也一样（DPRNN 报 `No module named 'utils.util'`） | `_dprnn_loader.load_model_cls()` 改为**导入期间独占 `utils*` 命名空间**，导完还原 `sys.path` 与 `sys.modules` |
| 3 | **MMDenseLSTM 报 `cuda:0 and cpu` 张量混用** | tky823 参考实现的 `update_em()`（`frequency_mask.py:295/302/310/316`）用**裸 `torch.eye(n_channels)`** 造单位阵，默认落在 CPU | 不改第三方代码：用 `with torch.device(dev)` 上下文把内部裸创建的张量也钉到同一设备 |
| 4 | **两个基准进程并发写 `model_runs.json`** | 一条被网络中断「以为已死」的后台流水线其实还活着，训练一结束就自己启动了基准 | `bench_multisong.py` 加 **PID 单实例锁**（`05_misc/logs/_bench_multisong.lock`），重复启动直接拒绝 |

> 教训汇总：**「换个设备重跑一遍」本身就是一个测试** —— 上面 3 个是纯粹的 GPU 侧 bug，
> 在 CPU 上永远不暴露。任何「RTF 没变」都应先去查代码里有没有写死的 device。

---

## 6. 产物清单（重组后的五大模块布局）

```
E:\FYP_HKBU\
├── 01_models/                            ① 模型
│   ├── _third_party/                     12 个官方仓库（BS-RoFormer、bsrnn、demucs、mdx-net ×3、
│   │                                     DNN-based_source_separation、Conv-TasNet、DPRNN、sigsep-mus-oracle …）
│   ├── _weights/                         预训练权重
│   │   ├── BS-RoFormer/                  ep_317 (639 MB) + ep_937 (393 MB) + 精确匹配的 configs
│   │   ├── MDX-Net/mdx_extra/            4 × .th
│   │   ├── DNN-based_source_separation/  ConvTasNet + MMDenseLSTM
│   │   └── BSRNN/                        bsrnn-opt (4 ckpt) + bsrnn-large (4 ckpt)
│   │                                     + simo-bsrnn-opt (separator.ckpt 1.31 GB)
│   │                                     ← 3 个 zip 共 4.67 GB，MD5 全部校验通过
│   └── MODEL_REGISTRY.md                 ★ 12 个模型在位 / 可运行 / 精度总表
│
├── 02_databases/                         ② 数据集
│   ├── MUSDB18-HQ/{train,test,_smoketest}
│   └── DNS-Challenge/                    clean / noise / dev_testset / impulse_responses
│
├── 03_outputs/                           ③ 产物（每模型一个独立目录）
│   ├── Oracle-IRM/<song>/{vocals,drums,bass,other}.wav
│   ├── RPCA/<song>/vocals.wav
│   ├── Open-Unmix/<song>/{vocals,drums,bass,other}.wav
│   ├── MDX-Net/<song>/{vocals,drums,bass,other}.wav
│   ├── Conv-TasNet/<song>/{vocals,drums,bass,other}.wav
│   ├── MMDenseLSTM/<song>/{vocals,drums,bass,other}.wav
│   ├── BS-RoFormer-L12/<song>/vocals.wav
│   ├── BS-RoFormer-L6/<song>/vocals+other.wav   ← 文件名即语义（见 §3.6）
│   ├── BSRNN-opt|large|SIMO/<song>/{vocals,drums,bass,other}.wav
│   ├── Demucs/stems/{vocals,drums,bass,other}.wav
│   └── DPRNN/ RPCA-DRNN/ Pac-HuBERT-SEP/        ← 受阻模型占位（内含原因说明）
│
├── 04_reports/                           ④ 报告（文档 / 图表 / 数据）
│   ├── docs/MODEL_PROVISIONING_REPORT.md ← 本报告
│   ├── html/METHOD_VISUAL_COMPARISON_1.3.html
│   ├── figures/                          全部图表（含 comparison/、figs_time/、demucs/）
│   └── data/comparison/                  ★ 本报告所有数字的来源
│       ├── model_runs.json               逐模型逐曲运行记录（SDR + SI-SDR + RTF）
│       ├── model_runs_median.json        多片段中位数（§1.6 主结果表）
│       ├── model_provisioning.json/.md   开源 / 许可 / 权重供给清单
│       ├── dependency_probe.json         逐模型依赖探针（含 SAC 拦截实测）
│       └── weight_loadability.json       权重可加载性 + 参数量
│
├── 05_misc/                              ⑤ 杂项
│   ├── logs/                             每个模型一次运行的完整日志 + traceback
│   ├── _archive/                         历史归档（含重组前的 tools 备份）
│   ├── scratch/                          临时缓存（.cache / .tmp_* / __pycache__）
│   └── lyrics-game/                      独立子项目
│
└── tools/                                脚本入口
    └── paths.py                          ★ 路径中枢（唯一真理源）
        benchmark_model_universal.py  median_over_clips.py  bench_multisong.py
        _scan_balanced_clip.py  _bsrnn_loader.py  _librosa_shim.py
        par_download.py  dns_fetch.py  extract_bsrnn.py
        compare_separation_methods.py  build_html_v13.py
        _migrate_layout.py  _rewire_paths.py   （本次重组用，幂等）
```

> 目录重组说明与「旧 → 新」完整映射表见项目根 `README.md`。

---

## 7. 未完成项与后续动作

| 项 | 状态 | 下一步 |
|---|---|---|
| BSRNN 权重（3 zip / 4.67 GB） | ✅ **完成** | MD5 全部校验通过 |
| BSRNN 推理（opt / large / SIMO） | ✅ **完成** | 三变体全部 PASS 并落盘 stem wav |
| SDR（museval BSSEval v4） | ✅ **完成** | 拦截面收窄后已能计算；全表改为 SDR + SI-SDR 并列 |
| 多片段中位数（3 个均衡片段） | ✅ **完成** | §1.6 主结果表；单片段数字已降级为「跑通判据」 |
| Demucs 并入统一基准 | ✅ **完成** | `--model demucs`；3 均衡片段 + 基准片段全跑通，数据进 `model_runs.json` |
| MUSDB18 官方 test 集（50 首整曲） | ⬜ 未做 | 当前只跑了 train 的 10 s 片段；要出可发表成绩需跑 test 集 |
| **DPRNN MUSDB18 自训** | ✅ **完成（2026-09-19 收尾）** | 官方只有语音权重 → 自训。step 3045 → **31900**，验证均值 SI-SDR +0.781 → **+2.745**，测试四轨均值 −4.54 → **−1.72**（见 §3.4c）|
| RPCA+DRNN 自研 | ❌ 无代码 | 按 Lai & Wang 2022（EURASIP 2022:4）复现；口径是单声道 2 源，不可与 MUSDB18 横比 |
| CUDA 版 torch | ✅ **完成** | 已装 2.14.0+cu126；§5 实时性结论已按 GPU 口径改写 |
| **MDX-Net 鼓轨异常复测** | ✅ **已结案（2026-09-19）** | GPU 复跑得 6.97（与 6.99 一致）→ 是真实设备差异、非抖动；中位数口径不受影响（见 §1.6）|
| RPCA 性能优化 | 🟠 建议 | `_rpca_ialm` 每轮全秩 SVD ×120 → 改 `randomized_svd` 低秩截断 + `mu` 上限，RTF 有望压到 1 以下 |
| **DPRNN 训练历史元数据** | ✅ **已修（2026-09-19）** | `--resume` 原只合并 `steps`/`vals`，`batch`/`finished`/`final_step` 停在上一轮 → 已修脚本 + 回填历史文件 |


---

*本报告由 `tools/model_provisioning_inventory.py`、`tools/benchmark_model_universal.py`、`tools/probe_model_deps.py`、`tools/verify_weights_loadable.py`、`tools/summarize_model_runs.py` 的产出汇总而成，所有数字均可在 `04_reports/data/comparison/` 下复核。*
