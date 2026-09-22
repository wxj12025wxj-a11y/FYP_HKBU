# FYP 项目梳理与实施计划

> 生成：2026-09-22 ｜ 项目根：`E:\FYP_HKBU`
> 目标：完成「源分离模型分析（板 1）」与「降噪（板 2）」两大板块，产出可进论文与答辩的图、表、数据。
> 原则：**所有指标一律本机实测**，不引用网络数值；任何跨模型比较先声明口径。

---

## 0. 一页纸总览

你现在有两条并行主线：

| 板块 | 任务 | 已有基础 | 要新建的量 |
|---|---|---|---|
| **板 1** 源分离 | 讲清 12 个模型**怎么运行** + 出多维度容量/精度图表 + 频谱图面板 + 混响/去混响实验 | 12 模型整曲跑完（564 组合）、564 份 stem 音频、50 首清单 | 模型画像脚本、架构图、频谱面板、混响实验、2 个去混响模型 |
| **板 2** 降噪 | 两个降噪工具（Demucs-Denoiser / Mel-RoFormer-Denoise）在 Valentini + DNS 上评测，并做「降噪 → 分离」级联 | DNS-Challenge 子集（dev_testset 921 条带参考）、RIR 60248 条、PESQ/STOI 库已装 | 工具接入、Valentini 下载、统一评测口径、级联实验 |

**关键结论先说**：两个板块不是并列的，**级联实验是把它们缝成一篇论文的那一针**（你已确认要做）。它回答一个真问题：*把语音降噪器接在音乐分离前面，SDR 是涨还是跌？*

---

## 1. 项目现状梳理

### 1.1 已完成（可直接复用，不要重跑）

| 项 | 状态 |
|---|---|
| MUSDB18-HQ 整曲扫描 | 目标 600 组合 → **成功 564（94%）**；11 个模型 50/50 跑满 |
| 分离音频产物 | `03_outputs/_test_run/<模型>/<歌曲>/*.flac`，21 GB，样本级与源混音对齐 |
| 逐曲逐模型指标 | `04_reports/data/comparison/test_run_per_song.csv`（564 行）、`test_run_model_summary.csv` |
| 时间开销拆解 | 累计墙钟 10.17 h，推理仅 4.25 h（41.8%）；**开销与「输出几轨」强绑定** |
| 中位数精度表 | `04_reports/docs/MEDIAN_TABLE.md`（3 均衡片段）；`MEDIAN_TABLE_TEST.md`（整曲） |
| 人耳聆听页 | `04_reports/html/listen_compare.html`（3 首歌 × 37 行可播音频） |
| DNS-Challenge 子集 | dev_testset 921 条（**唯一带参考的官方测试集**）、clean 13733、noise 2308、**RIR 60248** |
| 降噪合成脚本 | `tools/dns_synthesize.py`（clean ⊛ RIR + α·noise，官方同口径） |

### 1.2 硬件与工具链（已实测可用）

- GPU：RTX 4070 Laptop，**8 GiB**（`sm_89`）｜torch 2.14.0+cu126 / torchaudio 2.11.0+cu126
- 解释器：**必须用** `C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe`
- 已装且这次要用：
  - `torch.utils.flop_counter.FlopCounterMode` ✅ → **实测 FLOPs**（比 thop 可靠，走真实前向）
  - `thop 0.1.1` / `ptflops` ✅（备用交叉验证）
  - `pystoi 0.4.1` ✅ / `pesq` ✅ → 降噪主观客观指标
  - `soundfile 0.14.0` / `librosa 1.0.0` ✅
- 磁盘：E: 剩 **62 GB**，C: 剩 **405 GB**（新增大文件建议落 C 盘或存 FLAC）
- 网络：GitHub / HuggingFace 均可达 ✅
- 已克隆框架：`01_models/_third_party/Music-Source-Separation-Training`（ZFTurbo）——**已支持 `mel_band_roformer` 模型类型**，BS-RoFormer 走的就是这条链路

### 1.3 缺口（本次要补的）

1. 没有「模型本体」画像（参数量 / FLOPs / 磁盘 / 激活显存 / 单次时延）→ 对应参考图**下半表**
2. 没有架构原理图（讲"模型怎么运行"）
3. 没有信号级频谱图面板（对应参考图的 Reverberation / Target Echo）
4. 无去混响 / 去回声模型
5. 无降噪工具（`facebookresearch/denoiser`、Mel-RoFormer-Denoise 均未接入）
6. 无 Valentini 数据集
7. **BSRNN-SIMO 仅 14/50**（你已决定暂不补跑 → 报告需显式标注截断）
8. FLOPs / 参数量**没有任何实测记录**

---

## 2. 第一板块：源分离模型分析

### 2.1 模型本体画像表（对应参考图**下半表**）

**口径（写死，全表统一）**：44.1 kHz 立体声，输入 **10 s** 片段（441,000 样本）；`FLOPs` 按 **2×MACs** 约定；显存取 `torch.cuda.max_memory_allocated()` 峰值；时延取**预热后 5 次中位数**；RTF 标注机器负载状态。

要出的列：

| 列 | 取法 |
|---|---|
| 参数量 | `sum(p.numel() for p in model.parameters())`，区分 trainable |
| **FLOPs / MACs** | `FlopCounterMode` 包住真实前向；thop 交叉验证 |
| 权重文件体积 | ckpt 字节数（BSRNN 为多 ckpt 求和，需注明） |
| 推理峰值显存 | `max_memory_allocated`（8 GiB 卡上很关键） |
| 单次前向时延 / RTF | 实测，非论文值 |
| 输出轨数 / 是否 SIMO | 1 / 4，以及是否需要多 ckpt 拼装 |
| 采样率 | 44.1 k / 11.025 k（DPRNN 特例）/ 16 k |
| 原论文 + 本机仓库路径 | 可追溯性 |

**分组呈现**（保持你既有的「跨组不可比」纪律）：

- **四轨完整组**：Demucs、MDX-Net、Open-Unmix、Conv-TasNet、MMDenseLSTM、DPRNN(自训)、BSRNN-SIMO *(14/50)*
- **单目标组**：BS-RoFormer L12(vocals)、BS-RoFormer L6(vocals+other)、BSRNN-opt(vocals)、BSRNN-large(vocals)
- **解析组（无参）**：IRM/IBM Oracle、RPCA —— 单独表，不参与容量排名
- **新增去混响组**：见 §2.5

**脚本**：`tools/_measure_model_profile.py`
- 复用 `benchmark_model_universal.py::REGISTRY` 的 loader，**不重写加载链路**（避免口径分裂）
- 输出 `04_reports/data/model_analysis/model_profile.json` + `.csv`
- 图：`fig_params_flops.png`（双轴条形）、`fig_profile_pareto.png`（参数量 × FLOPs × SDR 气泡图）

### 2.2 精度表（对应参考图**上半表**）

直接用已有数据，但要**重画成参考表的样式**：

- 主表：**Average SDR (dB)**，按四轨均值排序，附分轨（vocals / drums / bass / other）
- 数据源：`MEDIAN_TABLE.md`（3 均衡片段中位数，主表）+ `MEDIAN_TABLE_TEST.md`（整曲，副表）
- 副表：**RTF 与实机耗时**（已有的 `test_run_model_summary.csv`）
- ⚠️ 参考图上半表的数字是论文值，**与我们实测不同**，原因见 §7

### 2.3 架构原理图系列（"讲清模型怎么运行"）

每个模型一张图，**数据驱动**，不凭想象画：

```
输入 waveform
  → 变换层（STFT / mel / learned encoder）   ← 标注真实张量 shape
  → 分离核心（mask / TCN / BiLSTM / attention）← 标注参数量占比
  → 逆变换 → 输出 stems
```

**做法（关键）**：先用 forward hook 抓真实 shape，再画图。

- `tools/_extract_model_graph.py`：对每个模型跑 1 s 片段，hook 所有子模块，落盘
  `(模块类名, 输入 shape, 输出 shape, 参数量)` 的完整树 → JSON
- `tools/_plot_model_arch.py`：把 JSON 渲染成框图
  - 左：**概念流程图**（手绘矢量，解释机制）
  - 右：**真实层级条带图**（每层参数量占比 + 张量 shape），保证图与实现对得上
- 每张图角标：论文引用 + 本机实现路径

覆盖 14 个模型：RPCA(IALM)、Conv-TasNet、DPRNN、MMDenseLSTM、Open-Unmix、MDX-Net(TFC-TDF)、Demucs(HTDemucs)、BS-RoFormer(rotary band-split)、BSRNN(Band-Split RNN)、IRM/IBM Oracle，加 §2.5 的 2 个去混响模型。

**复用**：`04_reports/figures/audio_analysis/` 已有 `fig3_spectrogram_*`、`fig4_spectrogram_win*` 等基础图，作为"变换层"的直观补充，不重复造。

### 2.4 信号级频谱图面板系列（对应你给的参考图）

你参考的那张图是水下声学的 6 面板。**不要照抄形式**，要找到**本领域的理论等价物** —— 这个等价物是现成的：

> **BSS-Eval 的四分量分解**：`估计 = s_target + e_interference + e_artifacts + e_noise`

我们用的 museval 就是 BSSEval v4，所以这四分量**本来就在评测链路里**，只是没被画出来。这是最有说服力的映射。

**分离板块 · 6 面板**（同一首歌、同一时间窗，统一色标 dB）：

| 面板 | 内容 | 对应参考图 |
|---|---|---|
| ① | **Mixture** 输入混音频谱 | Received signal |
| ② | **Target** 真值 stem（GT） | **Target Echo** |
| ③ | **Estimate** 模型估出的 stem | Signal recovery |
| ④ | **Interference** 别轨泄漏进来的分量 | — |
| ⑤ | **Artifacts** 模型自己造出的伪影 | — |
| ⑥ | **多模型并排**（4–6 个模型同一 stem） | Other methods |

再加两张"机制图"：

- **掩码可视化**：模型输出的 mask 随时间-频率分布热力图（直接看到"模型怎么切的"）
- **Band-split 可视化**：BSRNN / Mel-RoFormer 的频带分组边界叠加在频谱上

**脚本**：`tools/_make_spectrogram_panels.py`
- 音频直接读 `03_outputs/_test_run/`，**不需要重新推理**
- 选曲：已有人耳页的 3 首（长 430 s / 短 76 s / 中位 270 s）→ 复用现成叙事
- 输出：`04_reports/figures/spectrograms/`（PNG，300 dpi，统一色标）

### 2.5 混响与去混响 / 去回声实验（对应参考图的 Reverberation）

你参考图的另外两个概念，在本领域有**专门的模型**，不是硬凑：

| 新增模型 | 来源 | 采样率 | 用途 |
|---|---|---|---|
| **Dereverb-Mel-RoFormer (anvuew)** | HF `anvuew/dereverb_mel_band_roformer` | 44.1 k | 去混响（Reverberation） |
| **Vocals-Dereverb/DeEcho Mel-RoFormer (SUC-DriverOld)** | HF `Sucial/Dereverb-Echo_Mel_Band_Roformer` | 44.1 k | 去混响 + 去回声（Target Echo），论文报 SDR 10.01 |

两者都走**已有的 ZFTurbo 链路**（`model_type: mel_band_roformer`），加载方式与 BS-RoFormer 一致。

**实验设计**（本机已有 **60248 条 RIR**，这是白送的资源）：

```
① 干信号 (MUSDB stem)
② 人工加混响：②a 小房间 T60≈0.6 s   ②b 大厅 T60≈1.2 s     ← clean ⊛ RIR
③ 用去混响模型恢复 → "de-reverb 输出"
④ 残差 = ② − ③（被去掉的混响尾）
```

**产出图**（`04_reports/figures/reverb/`）：
- **RIR 本体图**：冲激响应波形 + 能量衰减曲线（EDT / T30 估计）—— 这张才配叫 **Reverberation**
- **6 面板对照**：干 / 混响 / 去混响输出 / 残差 / 早期反射 vs 晚期混响分离 / 各模型在同一混响下的退化
- **退化曲线**：T60 → ΔSDR（现有 12 个模型在混响下掉多少）
- **回声图**：加人工延迟回声 → de-echo 前后

**规模**：3 首歌（长/中/短）× 2 档混响 × (12 分离 + 2 去混响) ≈ 84 组合，**约 1.5 h GPU**。

**脚本**：`tools/_reverb_experiment.py`

---

## 3. 第二板块：降噪（Denoiser）

### 3.1 工具接入

**工具 A：`facebookresearch/denoiser`（Demucs 波形域，Interspeech 2020）**

- 输出采样率 **16 kHz**，因果模型，可在 CPU 上实时
- 预训练权重 3 个：`dns48`、`dns64`、**`master64`**
- 现成 CLI：`python -m denoiser.enhance --master64 --noisy_dir=... --out_dir=...`
  评测：`python -m denoiser.evaluate --master64 --data_dir=... --matching=dns`（自带 PESQ）
- 有 **dry/wet 旋钮**（`--dry`）→ 可以直接出「降噪强度 vs 指标」曲线，很出图
- ⚠️ **License = CC-BY-NC 4.0（非商用）**，论文里必须注明
- ⚠️ 接入方式：`git clone` → `pip install -e .` 装进 venv；**先做 SAC 功能性探测**（本机 Smart App Control 会拦新 dll，历史拦截面会变，不要采信旧记录）

**工具 B：`Mel-RoFormer-Denoise-Aufr33`**

- **44.1 kHz、立体声、非因果**，原生就是音乐采样率
- 权重（各约 913 MB）：
  - `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`（标准）
  - `denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt`（激进）
  - 配置：`model_mel_band_roformer_denoise.yaml`（框架 release v.1.0.7）
- **输出两轨：`dry`（去噪后）+ `other`（被去掉的噪声）** → **直接给出"残差"面板**，不用自己算
- 走已有 ZFTurbo 链路，`get_model_from_config("mel_band_roformer", cfg)`
- ⚠️ 这是**社区模型（aufr33），非同行评议论文**，必须在报告里标注来源与性质
- ⚠️ 本地 ZFTurbo clone 是 v0.1.0（2026-09-09），需确认/升级到含 denoise 配置的版本

### 3.2 数据集

**Valentini-Botinhao（48 kHz）** —— 你给的两个来源**是同一个东西**：

> 你给的 `datashare.ed.ac.uk/items/6ed35425-...` 就是这个数据集本身。
> 标准名：*Noisy speech database for training speech enhancement algorithms and TTS models*（Valentini-Botinhao, 2017），DOI `10.7488/ds/2117`，University of Edinburgh CSTR。

| 分片 | 体积 | 决定 |
|---|---|---|
| `noisy_testset_wav.zip` | 163 MB | **必下**（824 条带噪测试） |
| `clean_testset_wav.zip` | 147 MB | **必下**（配对参考） |
| `noisy/clean_trainset_28spk_wav.zip` | 2.64 + 2.32 GB | **下**（扩评测集 / 后续微调备用） |
| `noisy/clean_trainset_56spk_wav.zip` | 5.24 + 4.44 GB | 暂不下（磁盘留给产物） |

**DNS dev_testset** —— **已在本地**，921 条，**唯一带 reference 的官方测试集** → 作跨数据集验证，零成本。

**本机 RIR 60248 条** → 混响评测（与板 1 共用）。

**落点**：`02_databases/Valentini/`（新建），与 `DNS-Challenge/` 平级。
**下载器**：复用 `tools/par_download.py` 的分段断点续传逻辑（DNS 那套已验证）。

### 3.3 评测口径（⚠️ 这一节最容易出错，必须先定死）

三个采样率撞在一起：**denoiser 只能 16 k**、**Mel-RoFormer 44.1 k**、**Valentini 48 k**。

**硬约束：PESQ 官方只支持 8 k / 16 k。在 48 kHz 上跑 PESQ 是错的。**

所以定两套口径，**分开报，绝不混在一张表里**：

| 口径 | 采样率 | 指标 | 用途 |
|---|---|---|---|
| **主口径**（跨模型可比） | 统一重采样到 **16 kHz** | PESQ-WB、STOI、SI-SDR | denoiser vs Mel-RoFormer 唯一公平比较的地方 |
| **副口径**（原生率） | denoiser 16 k / Mel-RoFormer 44.1 k | SI-SDR、LSD（对数谱距离）、多分辨率 STFT 距离、Mel 距离 | 各自发挥真实水平；PESQ/STOI 在此不适用 |

另加：
- **推理峰值显存 / RTF / 参数量 / FLOPs** —— 与板 1 同一套画像口径，直接合表
- **DNSMOS**（若可装）作无参考主观预测；不可用则跳过，不硬凑

**脚本**：`tools/_denoise_eval.py`（统一入口，同时服务工具 A/B、Valentini/DNS 两数据集）

**⚠️ 数据泄漏警示**：`master64` 是 **在 DNS + Valentini 上训练的** → 用它测 Valentini 属 in-domain。
**必须同时报 `dns48` / `dns64`（未用 Valentini 训练）作对照**，否则结论会被质疑。
这一点要在报告里显式写出来，不要等答辩被问。

### 3.4 级联实验（你已确认要做 —— 这是板 1/板 2 的缝合点）

**要回答的问题**：*把语音降噪器接在音乐源分离前面，分离质量是涨还是跌？*

**数据**：MUSDB18-HQ test 取 **10 首**（覆盖长/中/短），人工加 DNS 噪声按 SNR 混合 → 有 GT，可算 ΔSDR
- SNR 档：{-5, 0, 5, 10} dB
- 噪声源：`02_databases/DNS-Challenge/noise/`

**四条链路对照**：

| 链路 | 说明 |
|---|---|
| L0 基线 | noisy mixture → 分离模型（不降噪） |
| L1 前置降噪 | noisy mixture → **降噪** → 分离模型 |
| L2 后置降噪 | noisy mixture → 分离模型 → 对 vocals 轨**降噪** |
| L3 双端 | 前置 + 后置都降噪 |

> 分离模型取四轨完整组代表：Demucs、MDX-Net、BSRNN-opt、Open-Unmix（4 个，控制规模）

**规模**：10 首 × 4 SNR × 4 链路 × 4 模型 ≈ 640 组合，**约 4 h GPU**（可挂夜间）

**⚠️ 重要提示**：denoiser 是**语音**降噪器，用在**音乐**上属域外（out-of-domain）。
如果结果是变差，**那也是有价值的负面结论**（说明语音降噪器不可直接迁移到音乐）——
不要为了"好看"去挑数据。这个诚实度是 FYP 最该守住的东西。

**脚本**：`tools/_cascade_experiment.py`
**产出**：`fig_cascade_delta_sdr.png`（按 SNR 分箱的 ΔSDR 图）、ΔSDR 明细表

---

## 4. 目录与产物落点

遵循项目既有约定（模型→01 / 数据→02 / 产物→03 / **图表数据全放 04** / 其余→05）：

```
02_databases/Valentini/                      新建（数据）

04_reports/docs/
    FYP_ROADMAP_2026-09-22.md                ← 本文件
    MODEL_PROFILE_<date>.md                  模型画像摘要（论文引用）
    DENOISE_EVAL_<date>.md                   降噪评测摘要
    REVERB_EVAL_<date>.md                    混响实验摘要
    CASCADE_EVAL_<date>.md                   级联实验摘要

04_reports/figures/
    model_arch/          板1 架构原理图（14 张）
    spectrograms/        板1 频谱图面板系列（参考图对应）
    reverb/              板1 混响 / 回声实验
    denoise/             板2 降噪
    cascade/             板2 级联

04_reports/data/
    model_analysis/      model_profile.json / .csv / graph_*.json
    denoise/             denoise_eval.json / .csv
    reverb/              reverb_eval.json / .csv
    cascade/             cascade_eval.json / .csv

04_reports/html/         3 个自包含看板（沿用 listen_compare.html 的离线内嵌模式）
    model_analysis_dashboard.html
    denoise_dashboard.html
    cascade_dashboard.html

tools/                   新增脚本（全部 import paths as _paths 风格）
    _measure_model_profile.py
    _extract_model_graph.py
    _plot_model_arch.py
    _make_spectrogram_panels.py
    _reverb_experiment.py
    _denoise_eval.py
    _cascade_experiment.py
    _build_model_dashboards.py
```

---

## 5. 阶段计划

按「先吃掉零风险、高复用的部分」排序。每阶段都有可验收产物，不排空转。

### 阶段 0 · 基建（无 GPU 长任务）
1. 下 Valentini（testset + 28spk trainset ≈ 5.6 GB）→ `02_databases/Valentini/`
2. clone `facebookresearch/denoiser` → pip 装进 venv → **SAC 功能性探测**
3. 下 Mel-RoFormer-Denoise 两个 ckpt（≈1.8 GB）+ 2 个去混响 ckpt + 配置
4. 确认 ZFTurbo clone 版本是否需升级（要含 `mel_band_roformer` denoise 配置）
5. 建目录骨架

**验收**：4 个新模型能在本机**各自跑通一次 10 s 前向**并落盘 wav，`status=PASS`（看结果 JSON，不看返回码）

### 阶段 1 · 模型本体画像（对应参考图下半表）
`_measure_model_profile.py` → 参数量 / FLOPs / 显存 / 时延，覆盖 12 + 4 = 16 个模型

**验收**：`model_profile.csv` 有 16 行、零缺失；FLOPs 与 thop 交叉验证偏差 <5%（不一致的标注出来）

### 阶段 2 · 架构原理图（板 1 核心交付）
`_extract_model_graph.py` → `_plot_model_arch.py` → 14 张图

**验收**：每张图的 shape 标注都能在 JSON 里溯源；参数量占比之和 = 100%

### 阶段 3 · 频谱图面板 + 精度表重绘（对应参考图上半表 + 全图）
`_make_spectrogram_panels.py`（复用现成音频，**零推理**）

**验收**：3 首歌 × 6 面板，统一色标；BSS-Eval 四分量之和与原始估计做残差校验（应≈0）

### 阶段 4 · 混响 / 去混响实验
`_reverb_experiment.py`，约 1.5 h GPU（可挂夜间）

**验收**：RIR 图 + 6 面板 + T60→ΔSDR 退化曲线

### 阶段 5 · 降噪评测（板 2 主体）
`_denoise_eval.py`：2 工具 × 2–3 ckpt × (Valentini 824 + DNS 921)

**验收**：双口径表（16 k 主 / 原生率副）；**in-domain 对照（dns48/dns64 vs master64）必须在表内**

### 阶段 6 · 级联实验
`_cascade_experiment.py`，约 4 h GPU（挂夜间）

**验收**：L0/L1/L2/L3 × 4 SNR 的 ΔSDR 矩阵；每个数字标注可复现命令

### 阶段 7 · 三个看板 + 论文摘要
`_build_model_dashboards.py` + 4 份 docs

**验收**：HTML 离线可开、图片内嵌、无外部依赖；双击即可看

> GPU 挂机总增量：**约 6–8 h**（阶段 4 + 5 + 6），可拆两个夜间跑；用 `schtasks` 起（**后台任务会随会话结束被杀**，这是本项目的既有教训）。

---

## 6. 风险清单

| # | 风险 | 处理 |
|---|---|---|
| R1 | **采样率不可比**：16 k / 44.1 k / 48 k 三套并存 | §3.3 双口径，分开报 |
| R2 | **in-domain 污染**：`master64` 训练集含 Valentini | 必报 dns48/dns64 对照 |
| R3 | 两个降噪/去混响模型都是**社区模型**，非论文 | 显式标注来源与性质 |
| R4 | 本地 ZFTurbo clone 版本旧 | 先确认 `mel_band_roformer` + denoise 配置是否齐 |
| R5 | **SAC 拦新装 dll**（denoiser 首次 import） | 先功能性探测；历史记录不采信 |
| R6 | PESQ 不支持 48 kHz | 统一重采样 16 k 再算 |
| R7 | FLOPs 口径歧义（MACs vs FLOPs、序列长） | 写死输入 10 s + 2×MACs，图表脚注标明 |
| R8 | RTF 受负载影响 | 报区间 + 标注机器状态 |
| R9 | 磁盘：新增约 17 GB（数据 6 + 权重 2 + 产物 9） | E 盘剩 62 GB；产物存 FLAC；必要时落 C 盘 405 GB |
| R10 | **单目标 vs 四轨模型 SDR 不可横比** | 保持既有分组纪律，图/表都分组 |
| R11 | DPRNN 是自训权重、11.025 kHz、四轨 SDR 为负 | 只作负面证据，**引用必须注 ckpt step** |
| R12 | 级联实验可能得到"变差"的结果 | 照实报，这是有价值的域外泛化结论 |
| R13 | 长任务被会话结束杀掉 | 一律走 `schtasks`，别用后台任务 |

---

## 7. 与参考表的差异说明（答辩会被问，先准备好）

你给的参考表（Average SDR + Size/Flops）与本机实测**数字不会一样**，原因不是我们做错了：

| 项 | 参考表 | 本机 | 原因 |
|---|---|---|---|
| Demucs 参数量 | 128 M | **41.98 M** | 参考表是 **原版 Demucs(v1)**，我们用的是 **HTDemucs** |
| Demucs SDR | 6.03 dB | **8.11 dB** | 口径不同：参考表为整曲、单模型对比池；**我们为 3 均衡片段中位数、12 模型池** |
| DPRNN-TasNet | 6.01 dB | **−1.72 dB** | 我们的是**自训 5 h 权重**（官方只发语音权重），欠训练，非同一物 |
| Conv-TasNet / MMDenseLSTM / Open-Unmix | 5.7 / 6.0 / 5.3 | 5.93 / 6.27 / 6.65 | 片段选择与评价口径不同（我们排除了"能量陷阱"片段） |
| RPCA+DRNN | 5.80 dB | **无产出** | 论文未公开代码，无法复现（既有审计结论） |

**建议**：
1. 主表就用**本机实测**，脚注写清口径（这是 FYP 的正确做法）
2. 若要**逐格对齐**参考表，需补：原版 Demucs(v1)、官方 DPRNN-TasNet 权重 —— 但 DPRNN 官方只发语音权重，TasNet 版需自行训练，**成本高、收益低，不建议**
3. 图注里加一句"参考表数值来自 XX 论文，本表为同一任务的本机复现口径，数字差异源于实现版本与评价片段选择" —— 这句话能挡掉大部分提问

---

## 8. 已确认的决策

| 决策 | 结论 |
|---|---|
| 降噪板块范围 | **评测 + 级联实验**（不做微调） |
| 去混响 / 去回声模型 | **纳入**（+2 个模型，用 RIR 做三段对照） |
| BSRNN-SIMO 补跑 | **暂不补跑** → 保留 14/50，报告中显式标注截断；四轨最优结论以 MDX-Net 为可用代表 |

## 9. 开工前唯一待办

`facebookresearch/denoiser` 与两个去混响模型**尚未下载**，阶段 0 需要联网。
其余所有资产（MUSDB18 stem 音频、50 首清单、DNS 子集、RIR、指标库）**全部已在本地**，阶段 1–4 可以立刻开工。
