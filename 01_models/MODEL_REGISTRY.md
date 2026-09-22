# 模型注册表 MODEL_REGISTRY

> 更新：2026-09-17 ｜ 维护规则：每次新增 / 替换 / 跑通一个模型，都在本表登记
> 对应产物：`03_outputs/<模型名>/` ｜ 指标数据：`04_reports/separation/data/comparison/`
> 统一基准工具：`tools/benchmark_model_universal.py`（每次实测自动更新 `model_runs.json`）

> 🔄 **目录布局变更（2026-09-23 第三次重组）**：模型已按板块分类 ——
> `01_models/separation/<模型>/`（板 1，13 个）与 `01_models/denoising/<模型>/`（板 2，6 个），
> 每个模型目录内**只保留 `code/`**（指向 `_third_party` 的目录联接）。
> 逐模型说明（论文出处 / 代码落点 / 权重真实路径 / 画像）见**板块索引**：
> `01_models/separation/README.md`、`01_models/denoising/README.md`。
> 下文中出现的 `01_models/<模型>/` 仅为历史表述，实际路径请按板块加一层。

---

## 0. 一句话结论

你指定核查的 **11 个模型**中：

- ✅ **9 个：已实际跑通推理、落盘了 stem wav** —— 其中 8 个用官方权重；
  **DPRNN 官方只发语音权重，已用本机 GPU 自训补齐**
- ⛔ **2 个：受阻（均无公开代码）** —— RPCA+DRNN（Lai & Wang 2022）、Pac-HuBERT-SEP（MERL）

另外要对比的 **Demucs 也已跑通**并并入统一基准 →
**合计 12 个模型中 10 个可运行、2 个受阻。**

> ⚠️ DPRNN 是**自训**权重（非官方发布），精度受 8 GiB 显存与训练时长限制，
> 是目前唯一「跑得通但不具可比精度」的模型 —— 详情与限制见 **§6**。论文引用时必须标注「自训」。

---

## 1. 总表（12 个模型）

| # | 模型 | 类别 | 代码在本机 | 权重在本机 | 可运行 | RTF(CPU) | RTF(**GPU**) | 四轨均值 SDR |
|---|---|---|---|---|---|---|---|---|
| 1 | **RPCA** | 古典·矩阵分解 | ✅ 自研实现 | 不需要（解析法） | ✅ | 1.61 | 4.56 | 无天然 stem 归属 |
| 2 | **RPCA+DRNN** | 混合（古典+DL） | ❌ 论文未公开 | ❌ | ⛔ | — | — | — |
| 3 | **Conv-TasNet** | 时域 DL | ✅ `_third_party/Conv-TasNet`<br>✅ `_third_party/DNN-based_source_separation` | ✅ MUSDB18 | ✅ | 0.64 | **0.18** | 5.93 |
| 4 | **DPRNN** | 时域 DL | ✅ `_third_party/Dual-Path-RNN-Pytorch` | ⚠️ **自训**（见 §6） | ✅ | — | 0.03 | **-1.72** ⚠️（欠训练，不参与排名） |
| 5 | **MMDenseLSTM** | 频域 DL | ✅ `_third_party/DNN-based_source_separation` | ✅ MUSDB18 (paper) | ✅ | 0.27 | **0.08** | 6.27 |
| 6 | **BS-RoFormer L12** | 频域 Transformer | ✅ `_third_party/Music-Source-Separation-Training`<br>✅ `_third_party/BS-RoFormer` | ✅ ep_317 (639 MB) | ✅ | 10.97 | **0.49** | 9.69（vocals 单目标）|
| 7 | **BS-RoFormer L6** | 频域 Transformer | ✅ 同上 | ✅ ep_937 (393 MB) | ✅ | 4.31 | **0.23** | 10.06（vocals+other 单目标）|
| 8 | **BSRNN** | 频域 DL（Band-Split RNN）| ✅ `_third_party/bsrnn` | ✅ Zenodo 3 包 (4.67 GB) | ✅ | 1.87 / 5.17 / 3.83 | **0.19 / 0.42 / 0.32** | **9.27**(SIMO) / 9.16(4×ckpt) / 8.42(large) |
| 9 | **IRM / IBM Oracle** | 解析 oracle | ✅ `_third_party/sigsep-mus-oracle` + 自研 | 不需要 | ✅ | 0.01 | 0.01 | 8.30（理论上界）|
| 10 | **MDX-Net** | 频域 DL | ✅ `_third_party/mdx-net` + 3 个 submission 仓库 | ✅ mdx_extra ×4 | ✅ | 0.88 | **0.13** | 8.96 |
| 11 | **Pac-HuBERT-SEP** | 自监督 + DL | ❌ MERL 未放出 | ❌ | ⛔ | — | — | — |
| 12 | **Demucs** | 时域 DL（htdemucs）| ✅ `_third_party/demucs` | ✅ htdemucs (84 MB) | ✅ | 0.39 | **0.03** | 8.11（四轨均值）|

> **口径**：3 个均衡片段（`ANiMAL - Clinic A @140s`、`Creepoid - OldTree @60s`、`Dark Ride - Burning Bridges @180s`）
> 各 10 s，取**中位数**。`RTF(CPU)` = `torch 2.14.0+cpu`；`RTF(GPU)` = `torch 2.14.0+cu126`（RTX 4070 Laptop 8 GiB）。
> 单片段数字会严重误导（低频引子会让 bass SDR 虚高到 20 dB+），**论文只能引用中位数**。
>
> ⚠️ **RTF 的两个陷阱**：
> ① **受机器负载影响极大** —— `RPCA` 是纯 numpy/scipy、**完全不用 GPU**，两次测量却在 1.6~4.6 之间跳，
>    据此不要给单一数字，要报区间。② **别只报 CPU 数字**：装上 CUDA 后深度模型普遍快 **3~22×**，
>    实时性结论完全改写（见 §5 与 `04_reports/separation/docs/MEDIAN_TABLE.md`）。
>
> 🔴 **8 GiB 显存的隐形陷阱**：本机 NVIDIA 驱动开了 **CUDA sysmem fallback** —— 超过显存**不报错**，
> 只是把张量悄悄放系统内存，步耗时暴涨 10~100×（实测 DPRNN batch 16：0.37 s → 27 s/步）。
> 所以显存规划必须靠实测峰值，不能等它自己 OOM。详见 §6。
>
> ⚠️ **Demucs 的口径更正（2026-09-16）**：此前「Demucs 绝对精度最高（人声 8.74 dB）」是**单曲 30 s 片段 + 6 方法对比池**的结论。
> 并入统一基准后（3 均衡片段中位数、12 模型对比池），Demucs 四轨均值 **8.11 dB**
> （oBSRNN-SIMO 9.27 > oBSRNN 9.16 > MDX-Net 8.96 > BSRNN-large 8.42 > Oracle 8.30 > **Demucs 8.11**）。
> 分轨：鼓 8.96、贝斯 10.87 属第一梯队；人声 7.11 中游；**other 5.49 偏弱**。
> Demucs 的真正优势在工程侧 —— **单模型 41.98 M 直接出 4 stem，GPU 上 RTF ≈ 0.03**，是所有深度方案里最快的一档。
> ⚠️ CPU 上 **非确定性**（同输入两次 ±0.1~0.3 dB），报单曲数字须标注波动。

**项目里另外保留的对比模型**（不在你这 11 个清单内，但历史实验已有产物）：
`Open-Unmix`（RTF 0.11，四轨 6.65）、`Baseline`（不分离基线）、`HPSS`、`NMF`、`ICA`。

**配套图表**（`04_reports/separation/figures/comparison/`，`tools/plot_median_summary.py` 生成）：

| 图 | 文件 |
|---|---|
| 12 模型四轨均值条形图 | `fig_median_sdr_bars.png` |
| 精度 vs 速度散点（绿=可实时） | `fig_median_pareto.png` |
| 分轨对照（人声/鼓/贝斯/其他） | `fig_median_per_stem.png` |

---

## 2. 逐模型明细

### 2.1 ✅ 可运行（10 个）

| 模型 | 代码落点 | 权重落点 | 产物落点 | 复现命令 |
|---|---|---|---|---|
| RPCA | `tools/compare_separation_methods.py::sep_rpca` | — | `03_outputs/RPCA/` | `--model rpca` |
| Conv-TasNet | `01_models/_third_party/{Conv-TasNet,DNN-based_source_separation}` | `01_models/_weights/DNN-based_source_separation/ConvTasNet/` | `03_outputs/Conv-TasNet/` | `--model convtasnet` |
| **DPRNN（自训）** | `01_models/_third_party/Dual-Path-RNN-Pytorch` + `tools/_dprnn_loader.py` | ⚠️ **自训**：`01_models/_weights/dprnn_musdb/{best,last}.pt` | `03_outputs/DPRNN/` | `--model dprnn`（训练：`tools/train_dprnn_musdb.py`，见 §6） |
| MMDenseLSTM | `01_models/_third_party/DNN-based_source_separation` | `.../MMDenseLSTM/musdb18/sr44100/paper/` | `03_outputs/MMDenseLSTM/` | `--model mmdenselstm` |
| BS-RoFormer L12 | `01_models/_third_party/Music-Source-Separation-Training` | `01_models/_weights/BS-RoFormer/model_bs_roformer_ep_317_*.ckpt` | `03_outputs/BS-RoFormer-L12/` | `--model bsroformer_l12` |
| BS-RoFormer L6 | 同上 | `..._ep_937_*.ckpt` | `03_outputs/BS-RoFormer-L6/` | `--model bsroformer_l6` |
| BSRNN（3 变体）| `01_models/_third_party/bsrnn` | `01_models/_weights/BSRNN/{bsrnn-opt,bsrnn-large,simo-bsrnn-opt}` | `03_outputs/BSRNN-{opt,large,SIMO}/` | `--model bsrnn_all,bsrnn_large_all,bsrnn_simo` |
| MDX-Net | `01_models/_third_party/mdx-net` + `mdx-net-submission*` | `01_models/_weights/MDX-Net/mdx_extra/` | `03_outputs/MDX-Net/` | `--model mdx` |
| IRM/IBM Oracle | `01_models/_third_party/sigsep-mus-oracle` + 自研 | — | `03_outputs/Oracle-IRM/` | `--model oracle` |
| Demucs | `01_models/_third_party/demucs` | `01_models/_weights/Demucs/955717e8-8726e21a.th`（torch.hub 缓存 `tools/_scratch/.cache/torch/hub/checkpoints/` 的同源硬链接） | `03_outputs/Demucs/` | `--model demucs`（基准）／`tools/run_demucs_deepdive.py`（深入分析） |

### 2.2 ⛔ 受阻（2 个，均无公开代码）

| 模型 | 阻塞类型 | 技术根因 | 是否有解法 |
|---|---|---|---|
| **RPCA+DRNN** | 无代码（需自研）| Lai & Wang 2022（EURASIP JASMP 2022:4）**未公开任何代码或权重** | ⛔ 已定：降级为「相关工作」。且其口径是单声道 2 源，**不可与 MUSDB18 4-stem 横比** |
| **Pac-HuBERT-SEP** | 无代码（无解）| MERL 未放出代码与权重 | ❌ 无。只能作为相关工作引用 |

> DPRNN 已从本表移出 —— 官方虽无音乐权重，但已**自训补齐**（见 §6），现列在 §2.1。

---

## 3. 5 项决策 —— 已全部拍板并执行（2026-09-17）

> 本节由「待你决策」转为**决策记录**。5 项均按建议推进，执行结果如下。

| # | 事项 | 决策 | 执行结果 |
|---|---|---|---|
| 1 | **CUDA 版 torch** | ✅ 装 | **已完成**。`torch 2.14.0+cu126` + `torchaudio 2.11.0+cu126`，`torch.cuda.is_available()=True`，GPU 实算通过（RTX 4070 Laptop, sm_89, 8 GiB）。**SAC 没有拦 CUDA 的 DLL** |
| 2 | **DPRNN 自训** | ✅ 训 | **已完成**。`tools/train_dprnn_musdb.py`（MUSDB18-HQ 4-stem, 11.025 kHz）+ `tools/_dprnn_loader.py` + 基准 `--model dprnn`。最后一轮到 step **31900**（300 min 预算实跑 250.7 min），验证均值 SI-SDR +2.745，测试四轨均值 **−1.72**（仍为负，不参与排名）。见 §6 |
| 3 | **RPCA+DRNN 自研** | ⛔ 不复现 | **已定稿：降级为「相关工作」**。理由：无官方代码 → 自研结果无法验证；论文口径是**单声道 2 源**，与 MUSDB18 4-stem 不可横比。其两条思路本项目已分别用 `RPCA` 与 `DPRNN` 独立覆盖 |
| 4 | **SAC 拦 `llvmlite.dll`** | ⛔ 不动 | **已确认无需处理**：拦截面又变了 —— `llvmlite` / `numba` / `librosa.stft` **已全部恢复**（实测时间戳 2026-09-17 00:54:55） |
| 5 | **Pac-HuBERT-SEP** | ⛔ 无解 | **已定稿：仅作相关工作引用**。论文里一句「MERL 未公开代码与权重，本文不予复现」即可 |

### 环境状态（2026-09-17 00:54:55 实测）

| 项 | 状态 |
|---|---|
| torch | **2.14.0+cu126**（已从 `+cpu` 换成 CUDA 版） |
| GPU | RTX 4070 Laptop，8 GiB，`cap=(8,9)`，matmul 实算通过 |
| `numpy` / `scipy` / `sklearn` / `pandas` / **`museval`** / `mir_eval` | ✅ 全部可用 |
| `librosa.stft` / `librosa.filters` / **`numba.njit`** | ✅ **已恢复**（此前被 SAC 拦的 `llvmlite` 现在能实算） |
| `pysepm.pesq` | ⚠️ 不可用，但原因是**缺 `srmrpy` 这个普通依赖**（`pip install srmrpy` 可解），与 SAC 无关 |

> ⚠️ **SAC 拦截面在本项目里已经变过 3 次**。任何「环境阻塞」的结论**必须带时间戳**，
> 引用历史记录前一律先重跑探测（技能：`~/.workbuddy/skills/sac-blocked-python/`）。

---

## 4. 快速复现

```bash
PY="C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe"

# 列出全部已注册模型
$PY tools/benchmark_model_universal.py --list

# 跑单个模型（10 秒片段，写产物 + 更新指标 json）
$PY tools/benchmark_model_universal.py --model bsrnn_simo --duration 10

# 跑全部
$PY tools/benchmark_model_universal.py --model oracle,rpca,umx,mdx,convtasnet,mmdenselstm,bsroformer_l12,bsroformer_l6,bsrnn_all,bsrnn_large_all,bsrnn_simo --duration 10

# 三片段中位数聚合（可外推口径）
$PY tools/median_over_clips.py
```

---

## 5. 未完成项登记

| 项 | 状态 | 负责人 | 备注 |
|---|---|---|---|
| ~~CUDA 版 torch~~ | ✅ **已完成 2026-09-17** | 我 | `2.14.0+cu126`，GPU 实算通过；见 §3-1 |
| ~~DPRNN 自训~~ | ✅ **已完成 2026-09-17** | 我 | 见 §3-2 与 §6（含 3 个致命 bug 的修复记录） |
| ~~RPCA+DRNN 复现~~ | ✅ **已定：不复现** | 我 | 降级为相关工作，见 §3-3 |
| ~~Pac-HuBERT-SEP~~ | ✅ **已定：仅相关工作** | 我 | 见 §3-5 |
| ~~SAC 拦 `llvmlite.dll`~~ | ✅ **已自行消失** | — | 拦截面已恢复，见 §3-4 |
| ⭐ ~~装 CUDA 后重跑全部模型的 RTF~~ | ✅ **已完成 2026-09-17** | 我 | 12 模型 × 3 片段全部在 CUDA 上重跑；深度方案提速 **3~22×**，见 §1 与 §5.1 |
| MUSDB18 官方 **test 集 50 首整曲**评测 | ⏳ 未做 | 我 | 当前只用 train 的 3 个 10 s 片段；要出可发表成绩需跑 test 集 |
| 模型对比补图（精度/速度散点） | ✅ **已完成** | 我 | `04_reports/separation/figures/comparison/` 3 张图，见 §1 |

### 5.1 CUDA 带来的实时性改写（2026-09-17 实测）

CPU → GPU 的中位 RTF（3 个均衡片段）：

| 模型 | CPU | GPU | 提速 | GPU 上是否实时 |
|---|---|---|---|---|
| **BS-RoFormer L12** | 10.97 | **0.49** | **22×** | ✅ 从「不可实时」变实时 |
| **BS-RoFormer L6** | 4.31 | **0.23** | **19×** | ✅ |
| **oBSRNN (4×ckpt)** | 5.17 | **0.42** | **12×** | ✅ |
| **BSRNN large (4×ckpt)** | 3.83 | **0.32** | **12×** | ✅ |
| **oBSRNN-SIMO** | 1.87 | **0.19** | **10×** | ✅ |
| Demucs (htdemucs) | 0.39 | **0.03** | 14× | ✅ |
| MDX-Net | 0.88 | **0.13** | 7× | ✅ |
| Conv-TasNet | 0.64 | **0.18** | 4× | ✅ |
| MMDenseLSTM | 0.27 | **0.08** | 3× | ✅ |
| Open-Unmix | 0.12 | **0.06** | 2× | ✅ |
| IRM/IBM Oracle | 0.01 | 0.01 | — | ✅（解析法） |
| **RPCA** | 1.61 | 4.56 | —— | ⚠️ **纯 CPU 算法**，数字跳动与负载有关，报区间 1.6~4.6 |

**结论**：装上 CUDA 之后，**精度前三名（oBSRNN-SIMO 9.27 / oBSRNN 9.16 / MDX-Net 8.96）全部实时**，
BS-RoFormer L12 也从「RTF 11 的离线模型」变成 **RTF 0.49 的实时模型**（人声 SDR 9.69，全场最高）。
**FYP 的「实时 + 高精度」目标在 GPU 上已经成立**，不需要再牺牲精度换速度。
⚠️ 这些数字是 RTX 4070 Laptop 8 GiB（笔记本功耗墙）上的结果，报论文时须附硬件与电源模式。

---

## 6. DPRNN 自训（2026-09-17）—— 本表唯一「非官方权重」的模型

### 6.1 为什么必须自训

`01_models/_third_party/Dual-Path-RNN-Pytorch`（JusperLee）代码完整，`Dual_RNN_model` 有 `num_spks`
参数可直接做 4 输出，**但官方 `pretrained_model_ids` 只有 `wsj0-mix` / `librispeech`（语音分离）**，
没有 MUSDB18 音乐权重，且其数据管线是 WSJ0 的 `.scp` 格式。
→ 只复用 `model/model_rnn.py` 的**模型定义**，自己写 MUSDB18-HQ 数据管线从零训练。

| 项 | 设定 |
|---|---|
| 脚本 | `tools/train_dprnn_musdb.py`（训练） + `tools/_dprnn_loader.py`（训练/评测共用同一套模型构造） |
| 模型 | DPRNN-TasNet，**3.69 M 参数**（kernel 16 / 64ch enc / 128ch sep / 6 层双向 LSTM / K=200） |
| 采样率 | **11.025 kHz**（44100 ÷ 4，用 `resample_poly(x,1,4)` 整数抽取，无重采样伪影） |
| 片段 | 4 s chunk，每个 step 采 **8 个互不相同**的片段 |
| 目标 | 4 stem（vocals/drums/bass/other），ckpt 内显式存 `targets` 顺序 |
| 损失 | 负 SI-SDR（DPRNN 原文口径），4 轨取均值，**静音 stem 按参考能量屏蔽** |
| 混合 | `mixture = Σ4 stems`（不是读 `mixture.wav`），保证掩码一致性数值严格成立 |
| 权重 | `01_models/_weights/dprnn_musdb/{best,last}.pt`（各 14 MB） |

### 6.2 🔴 三个「跑得通但结果静默错」的坑（都已在代码里修掉并注释）

1. **假 batch（最严重）**：原写法 `x = torch.as_tensor(mix)[None].expand(batch, -1)` ——
   `expand` **只是把同一段音频复制 batch 份**，梯度等价于 `batch=1`。
   日志显示「129 step/min、batch=4」很漂亮，实际数据吞吐只有 4 s/step，
   22 分钟训练只见过 **约 0.5 个 MUSDB epoch** → 均值 SI-SDR 卡死在 **+0.78 dB**。
   修：真采 `batch` 个不同片段再 `np.stack`。修完 600 步就从 0.78 升到 **1.37**。
2. **`--resume` 从未可用**：两处存盘点都**没写 `opt` 状态**，`opt.load_state_dict(ck["opt"])` 必然 `KeyError`。
   修：存盘点补 `"opt"`，且 resume 时缺 `opt` 就退化为「只续权重」；
   同时把 `opt.load_state_dict` 之后**显式覆盖 lr**（它会连带恢复旧 lr，让命令行 `--lr` 静默失效）。
3. **CUDA sysmem fallback**：8 GiB 卡上 batch ≥ 12 时峰值显存 8.73 GiB，
   驱动**不报 OOM**，而是把张量放到系统内存 → 步耗时从 0.37 s 暴涨到 **3.4 s**（batch 16 时 27 s/步）。
   实测档位（bf16 autocast）：batch 4 → 3.03 GiB / 0.254 s；6 → 4.45 GiB / 0.386 s；
   **8 → 5.88 GiB / 0.374 s（选它）**；12 → 8.73 GiB ✗ 溢出。
   修：`tools/paths.py::setup_env()` 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
   训练循环加 OOM 捕获 + batch 自动减半，并把实测档位写进注释。
   复现探针：`05_misc/scratch/_dprnn_timing_probe.py`。

> ➕ **第 4 个坑（不是代码问题）**：首轮 110 分钟预算被**一次 8 小时系统睡眠**吃掉
> （Kernel-Power EventID 42 @ 02:08:33，日志从 step 3000 直接跳到「用时 555 min」）。
> 修：`train_dprnn_musdb.py::keep_awake()` 用 `SetThreadExecutionState` 做**进程级**禁睡
> （比只改 `powercfg` 可靠：电源计划被改回去就失效），并已把 AC 下 standby/hibernate/disk/video
> 超时全部置 0。

### 6.3 结果与限制（必须如实写进论文）

> ✅ **数值已定稿（2026-09-19）**：最后一轮 300 min 训练（11:08 → 15:19，实跑 250.7 min）
> 把 step 推到 **31900**，权重与全部报告已同步刷新。**当前所有 DPRNN 数字均为 step 31900。**

**训练进展（验证集 3 个均衡片段，SI-SDR 均值）**

| 阶段 | step | 均值 | vocals | drums | bass | other |
|---|---|---|---|---|---|---|
| 旧 ckpt | 3045 | +0.781 | +2.47 | +2.86 | +2.26 | −4.46 |
| 中途 | 3404 | +1.366 | +3.76 | +3.02 | +2.77 | −4.09 |
| **最终** | **31900** | **+2.745** | +3.72 | +5.16 | +5.09 | −3.00 |

**测试集四轨中位数 SDR（museval，3 均衡片段）**

| 指标 | vocals | drums | bass | other | **四轨均值** |
|---|---|---|---|---|---|
| 旧 ckpt（step 3045） | −3.83 | −3.57 | −4.35 | −6.40 | **−4.54** |
| **新 ckpt（step 31900）** | −2.18 | −1.86 | −0.82 | −2.02 | **−1.72** |
| 提升 | +1.65 | +1.71 | **+3.53** | **+4.38** | **+2.82** |

- **结论：确实学到了东西，但离能用还差得远** —— 四轨虽然全部收窄，**却仍全为负**
  （SDR < 0 = 比直接拿混合信号当估计还差）。属**算力受限的欠训练**，不代表 DPRNN 的架构上限。
- **两指标方向一致**：SI-SDR 侧 vocals 3.34 / drums 4.78 / bass 5.05 已明显转正，
  只有 `other` 仍为负（−2.99）。museval SDR 用 1 s 窗逐窗算再取中位数，口径更严 → 数值更低，非矛盾。
- **因此本表把它单独标注**：DPRNN 一栏**不参与「最快/最准」结论**，只用于说明
  「端到端时域方法在 8 GiB 单卡 + 5 小时预算下仍无法复现论文级精度」这一**负面结论**。
- **RTF 0.03 是真实的**（推理极快），短板全在训练侧。
- 若要提升：需 CUDA 级算力（非笔记本 4070）、更长的训练（数十小时级）、
  以及 MUSDB18 标准增强（source remixing / channel swap）。
- 训练曲线：`04_reports/separation/figures/dprnn/training_curve.png`；训练历史：`04_reports/separation/data/dprnn_training.json`。


### 6.4 复现

```bash
PY="C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe"

# 训练（--resume 断点续训；--batch 8 是 8 GiB 卡上的实测最优档）
$PY tools/train_dprnn_musdb.py --minutes 300 --batch 8 --resume

# 评测（统一基准，与其他 12 个模型同片段同口径）
$PY tools/benchmark_model_universal.py --model dprnn --song "Creepoid - OldTree" --duration 10 --offset 60
```

### 6.5 训练收尾链（`tools/post_dprnn_chain.py`）

**问题**：`--minutes 300` 要跑 5 小时，「训完那一刻」才是真正该动手的时刻，
但人在旁边守着不现实。而「挂一个后台进程等着」本项目已经出过事故：

> 🔴 2026-09-17 事故：一条**以为已被网络中断杀掉**的后台流水线其实还活着，训练一结束它
> 就自己启动基准，与手动启动的实例**同时写 `model_runs.json`**。
> `bench_multisong.py` 的 PID 单实例锁就是这次事故后加的。

**做法**：`tools/post_dprnn_chain.py` 一条命令串完全程，四步都复用**已测试过**的脚本：

| 步 | 动作 | 命令 |
|---|---|---|
| ① | 重跑 DPRNN 三片段基准 | `bench_multisong.py --models dprnn` |
| ② | 聚合中位数 + 刷新 md | `median_over_clips.py --write docs/MEDIAN_TABLE.md` |
| ③ | 重绘对比图 | `plot_median_summary.py` |
| ④ | 重建 HTML | `build_html_v13.py` |

设计要点（都是踩过坑才加的）：

1. **「训练结束」判据（2026-09-19 修补）** —— 原设计只认日志里新增的 `训练结束:` 行，
   但 09-17 那轮训练是**被外部中断**的（预算 300 min 只跑到 250.7 min），
   收尾代码根本没执行到 → **日志里永远不会有这一行**，链条会一路空等到超时。
   现改为**双路判据**，满足其一即通过：
   - **正常路径**：日志出现 `训练结束:` + `last.pt` 稳定 ≥90 s + 无训练进程残留；
   - **兜底路径**：无结束标记，但**训练进程已退 且 `last.pt` 已静止 ≥270 s（3× stable）**
     → 判定为「疑似外部中断」，继续收尾并在日志里显式标注 `[兜底判据]`。
   兜底阈值取 3 倍是为了不把「验证间隙」误判成「训练结束」。
2. **进程探测的假阳性**：最初只按 `CommandLine -like '*train_dprnn_musdb*'` 过滤，
   一次命中 **7 个**进程（`WorkBuddy.exe`、3 个 bash 包装壳、**查询自身的 powershell**）→
   会死等到超时。修法是加 `Name -like 'python*'` 过滤并排除自身。
   实测真正的训练是**两个** python：`Scripts\python.exe`（启动器壳 ~5 MB，父）
   与它 spawn 的真实解释器（~2.1 GB，子），命令行完全相同，退出时一起退。
3. **两层锁**：本脚本自己的 PID 锁 + 强制走 `bench_multisong.py`（它也有锁）。
   宁可拒绝启动，也不并发写 `model_runs.json`。
4. **动手前备份** `model_runs.json` → `model_runs_pre-dprnn-<时间戳>.json`。
   本项目的 `mv -f` 覆盖事故（78 KB 被覆盖成 2.9 KB）说明必须有退路。
5. `--dry-run` 只校验路径与命令，不执行；`--force` 跳过等待（会在欠训练权重上评测，慎用）；
   `--verify-mdx` 额外复测 MDX-Net 鼓轨异常（会改写 MDX 记录，故默认关闭）。

```bash
$PY tools/post_dprnn_chain.py --dry-run              # 先校验
$PY tools/post_dprnn_chain.py --verify-mdx           # 等到训练结束再自动收尾（顺带复测 MDX）
```

**实际执行记录（2026-09-19 11:17）**：6 步全通过，耗时 3.5 min。

| 步 | 结果 |
|---|---|
| ① 重跑 DPRNN 三片段 | ✅ DPRNN 四轨均值 **−4.54 → −1.72** |
| ② 聚合中位数 | ✅ 刷新 `model_runs_median.json` + `MEDIAN_TABLE.md` |
| ③ 重绘对比图 | ✅ 3 张 png 更新 |
| ④ 重建 HTML | ✅ `METHOD_VISUAL_COMPARISON_1.3.html` 更新 |
| ⑤ 复测 MDX 鼓轨 | ✅ 得 **6.97**（与旧 GPU 值 6.99 一致）→ 结案 |
| ⑥ 再聚合 | ✅ 纳入 MDX 复测结果 |

> ⚠️ 链条只负责**重跑与重绘**，**不改文档正文**。

> 跑完后仍需人工把新数字同步进 `MODEL_REGISTRY.md` §1/§6.3 与
> `MODEL_PROVISIONING_REPORT.md` §0/§1.6/§3.4 —— 这是故意的，
> 「文档里写什么结论」不该由脚本静默决定。
