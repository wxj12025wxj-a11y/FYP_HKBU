# 板 1 · 分离（separation）

> 13 个模型 ｜ 目录：`01_models/separation/<模型>/`
>
> **本板块的模型目录只保留 `code/`**（指向 `01_models/_third_party/<repo>` 的目录联接，零额外占用）。
> 模型卡 `README.md` 与 `weights/` 联接层已于 2026-09-23 精简移除：
> 权重的**唯一物理落点**是 `01_models/_weights/`，全部脚本（含 `tools/paths.py`）都从那里直接加载。

## 索引

| 模型 | 条目 key | 参数量 | FLOPs | 时延(10 s) | 一句话定位 |
|---|---|---|---|---|---|
| [`BS-RoFormer-L12/`](BS-RoFormer-L12/) | `bsroformer_l12` | 159.758 M | 43480.878 G | 4.5208 s | 单目标（只出 vocals），L12 = 12 层 Transformer，本机全部条目里 FLOPs 最高 |
| [`BS-RoFormer-L6/`](BS-RoFormer-L6/) | `bsroformer_l6` | 98.195 M | 14975.185 G | 2.1702 s | 输出为 mixture-drums-bass（vocals+other 残差），**不是** yaml 里写的 other |
| [`BSRNN-large/`](BSRNN-large/) | `bsrnn_large`, `bsrnn_large_all` | 36.417 M | 276.819 G | 0.6294 s | 同上，large 配置；单首墙钟最快（46.7 s） |
| [`BSRNN-opt/`](BSRNN-opt/) | `bsrnn`, `bsrnn_all` | 40.777 M | 955.494 G | 1.0814 s | 同一批权重服务两个条目：单目标 vocals 与四轨 4-stem（4 ckpt） |
| [`BSRNN-SIMO/`](BSRNN-SIMO/) | `bsrnn_simo` | 108.728 M | 1540.109 G | 1.2105 s | SIMO 架构，精度第一（四轨均值 9.27 dB），但 50 首只跑完 14 首 |
| [`Conv-TasNet/`](Conv-TasNet/) | `convtasnet` | 13.395 M | 1171.744 G | 1.5221 s | 纯时域；other 轨 SDR 仅 1.66 dB，是最弱的一项 |
| [`Demucs/`](Demucs/) | `demucs` | 41.984 M | 508.127 G | 0.2749 s | htdemucs 混合时域+频域架构；参考表里 128 M 指原版 v1，本机是 41.98 M |
| [`DPRNN/`](DPRNN/) | `dprnn` | 3.686 M | 30.573 G | 0.103 s | **唯一自训模型**（3.69 M，11.025 kHz, 4 s chunk）；SI-SDR 为负 → 只作负面证据 |
| [`MDX-Net/`](MDX-Net/) | `mdx` | 334.544 M | 777.218 G | 0.7373 s | mdx_extra 由 4 个 submission 权重组成；全场参数最多（334.5 M） |
| [`MMDenseLSTM/`](MMDenseLSTM/) | `mmdenselstm` | 5.487 M | 294.469 G | 0.3331 s | 参数最少（5.49 M），但时延比同量级模型高 |
| [`Open-Unmix/`](Open-Unmix/) | `umx` | 35.577 M | 14.292 G | 0.2 s | 四个目标各一套网络（各占 25% 参数），基线参考实现 |
| [`Oracle-IRM/`](Oracle-IRM/) | `oracle` | 0.0 M | 0.0 G | 0.0 s | 理论上界（IRM/IBM），零参数，用来给所有模型划天花板 |
| [`RPCA/`](RPCA/) | `rpca` | 0.0 M | 0.0 G | 82.5301 s | 解析算法，零参数；单次 10 s 需 82.5 s（纯 CPU 矩阵迭代）；稀疏分量无天然 stem 归属 |

## 逐模型明细（原模型卡内容）

### BS-RoFormer-L12

> 单目标（只出 vocals），L12 = 12 层 Transformer，本机全部条目里 FLOPs 最高

**论文 / 出处**

- Lu, Wang, Jiang, Zhang. Music Source Separation with Band-Split RoPE Transformer. 2024 (arXiv:2309.02612)

**代码位置**

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | `01_models/_weights/BS-RoFormer/model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| `model_bs_roformer_ep_317_sdr_12.9755.yaml` | `01_models/_weights/BS-RoFormer/configs/model_bs_roformer_ep_317_sdr_12.9755.yaml` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsroformer_l12` | 1 | 159.758 M | 43480.878 G | 4.5208 s | 3394.3 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/BS-RoFormer-L12/`（`code/` 为目录联接）
- 产物：`03_outputs/BS-RoFormer-L12/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### BS-RoFormer-L6

> 输出为 mixture-drums-bass（vocals+other 残差），**不是** yaml 里写的 other

**论文 / 出处**

- Lu, Wang, Jiang, Zhang. Music Source Separation with Band-Split RoPE Transformer. 2024 (arXiv:2309.02612)

**代码位置**

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `model_bs_roformer_ep_937_sdr_10.5309.ckpt` | `01_models/_weights/BS-RoFormer/model_bs_roformer_ep_937_sdr_10.5309.ckpt` |
| `model_bs_roformer_ep_937_sdr_10.5309.yaml` | `01_models/_weights/BS-RoFormer/configs/model_bs_roformer_ep_937_sdr_10.5309.yaml` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsroformer_l6` | 1 | 98.195 M | 14975.185 G | 2.1702 s | 2087.4 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/BS-RoFormer-L6/`（`code/` 为目录联接）
- 产物：`03_outputs/BS-RoFormer-L6/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### BSRNN-large

> 同上，large 配置；单首墙钟最快（46.7 s）

**论文 / 出处**

- Luo, Yu. Music Source Separation with Band-split RNN. IEEE/ACM TASLP 2023 (arXiv:2209.15174)

**代码位置**

- `01_models/_third_party/bsrnn`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `bsrnn-large` | `01_models/_weights/BSRNN/bsrnn-large` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsrnn_large` | 1 | 36.417 M | 276.819 G | 0.6294 s | 1657.4 MB | 44100 Hz / 2 ch |
| `bsrnn_large_all` | 4 | 146.656 M | 1126.935 G | 2.9494 s | 2567.9 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/BSRNN-large/`（`code/` 为目录联接）
- 产物：`03_outputs/BSRNN-large/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### BSRNN-opt

> 同一批权重服务两个条目：单目标 vocals 与四轨 4-stem（4 ckpt）

**论文 / 出处**

- Luo, Yu. Music Source Separation with Band-split RNN. IEEE/ACM TASLP 2023 (arXiv:2209.15174)

**代码位置**

- `01_models/_third_party/bsrnn`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `bsrnn-opt` | `01_models/_weights/BSRNN/bsrnn-opt` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsrnn` | 1 | 40.777 M | 955.494 G | 1.0814 s | 1675.0 MB | 44100 Hz / 2 ch |
| `bsrnn_all` | 4 | 164.119 M | 3891.294 G | 5.7328 s | 2637.4 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/BSRNN-opt/`（`code/` 为目录联接）
- 产物：`03_outputs/BSRNN-opt/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### BSRNN-SIMO

> SIMO 架构，精度第一（四轨均值 9.27 dB），但 50 首只跑完 14 首

**论文 / 出处**

- Luo, Yu. Music Source Separation with Band-split RNN. IEEE/ACM TASLP 2023 (arXiv:2209.15174)

**代码位置**

- `01_models/_third_party/bsrnn`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `simo-bsrnn-opt` | `01_models/_weights/BSRNN/simo-bsrnn-opt` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsrnn_simo` | 4 | 108.728 M | 1540.109 G | 1.2105 s | 2483.0 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/BSRNN-SIMO/`（`code/` 为目录联接）
- 产物：`03_outputs/BSRNN-SIMO/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### Conv-TasNet

> 纯时域；other 轨 SDR 仅 1.66 dB，是最弱的一项

**论文 / 出处**

- Luo, Mesgarani. Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation. IEEE/ACM TASLP 27(8), 2019

**代码位置**

- `01_models/_third_party/Conv-TasNet`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `ConvTasNet` | `01_models/_weights/DNN-based_source_separation/ConvTasNet` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `convtasnet` | 4 | 13.395 M | 1171.744 G | 1.5221 s | 606.6 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/Conv-TasNet/`（`code/` 为目录联接）
- 产物：`03_outputs/Conv-TasNet/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### Demucs

> htdemucs 混合时域+频域架构；参考表里 128 M 指原版 v1，本机是 41.98 M

**论文 / 出处**

- Rouard, Massa, Defossez. Hybrid Transformers for Music Source Separation. ICASSP 2023

**代码位置**

- `01_models/_third_party/demucs`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `955717e8-8726e21a.th` | `tools/_scratch/.cache/torch/hub/checkpoints/955717e8-8726e21a.th` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `demucs` | 4 | 41.984 M | 508.127 G | 0.2749 s | 599.7 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/Demucs/`（`code/` 为目录联接）
- 产物：`03_outputs/Demucs/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### DPRNN

> **唯一自训模型**（3.69 M，11.025 kHz, 4 s chunk）；SI-SDR 为负 → 只作负面证据

**论文 / 出处**

- Luo, Chen, Mesgarani. Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation. ICASSP 2020

**代码位置**

- `01_models/_third_party/Dual-Path-RNN-Pytorch`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `dprnn_musdb` | `01_models/_weights/dprnn_musdb` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `dprnn` | 4 | 3.686 M | 30.573 G | 0.103 s | 579.7 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/DPRNN/`（`code/` 为目录联接）
- 产物：`03_outputs/DPRNN/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### MDX-Net

> mdx_extra 由 4 个 submission 权重组成；全场参数最多（334.5 M）

**论文 / 出处**

- Kim, Choi, Chung, Lee. KUIELab-MDX-Net: A Two-Stream Neural Network for Music Demixing. MDX Workshop 2021; 评测框架 Mitsufuji et al., Frontiers in Signal Processing 2022

**代码位置**

- `01_models/_third_party/mdx-net`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `mdx_extra` | `01_models/_weights/MDX-Net/mdx_extra` |
| `mixer.ckpt` | `01_models/_weights/MDX-Net/mixer.ckpt` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `mdx` | 4 | 334.544 M | 777.218 G | 0.7373 s | 1912.8 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/MDX-Net/`（`code/` 为目录联接）
- 产物：`03_outputs/MDX-Net/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### MMDenseLSTM

> 参数最少（5.49 M），但时延比同量级模型高

**论文 / 出处**

- Takahashi, Matsubara, Uehara. Multi-scale Multi-band DenseLSTM for Audio Source Separation. IEEE Signal Processing Letters 27, 2020

**代码位置**

- `01_models/_third_party/DNN-based_source_separation`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `MMDenseLSTM` | `01_models/_weights/DNN-based_source_separation/MMDenseLSTM` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `mmdenselstm` | 4 | 5.487 M | 294.469 G | 0.3331 s | 643.0 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/MMDenseLSTM/`（`code/` 为目录联接）
- 产物：`03_outputs/MMDenseLSTM/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### Open-Unmix

> 四个目标各一套网络（各占 25% 参数），基线参考实现

**论文 / 出处**

- Stoter, Uhlich, Liutkus, Mitsufuji. Open-Unmix - A Reference Implementation for Music Source Separation. JOSS 4(41):1667, 2019

**代码位置**

- `01_models/_third_party/DNN-based_source_separation`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `vocals-b62c91ce.pth` | `tools/_scratch/.cache/torch/hub/checkpoints/vocals-b62c91ce.pth` |
| `drums-9619578f.pth` | `tools/_scratch/.cache/torch/hub/checkpoints/drums-9619578f.pth` |
| `bass-8d85a5bd.pth` | `tools/_scratch/.cache/torch/hub/checkpoints/bass-8d85a5bd.pth` |
| `other-b52fbbf7.pth` | `tools/_scratch/.cache/torch/hub/checkpoints/other-b52fbbf7.pth` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `umx` | 4 | 35.577 M | 14.292 G | 0.2 s | 1227.9 MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/Open-Unmix/`（`code/` 为目录联接）
- 产物：`03_outputs/Open-Unmix/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### Oracle-IRM

> 理论上界（IRM/IBM），零参数，用来给所有模型划天花板

**论文 / 出处**

- Rafii, Liutkus, Stoter, Mimilakis, Bittner. MUSDB18 - a corpus for music separation. 2017 (sigsep-mus-oracle); IRM 定义见 Wang & Wang, IEEE/ACM TASLP 2014

**代码位置**

- `01_models/_third_party/sigsep-mus-oracle`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重**：无（解析算法，零参数）。实现见 `01_models/_selfimpl/README.md`。

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `oracle` | 4 | 0.0 M | 0.0 G | 0.0 s | MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/Oracle-IRM/`（`code/` 为目录联接）
- 产物：`03_outputs/Oracle-IRM/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### RPCA

> 解析算法，零参数；单次 10 s 需 82.5 s（纯 CPU 矩阵迭代）；稀疏分量无天然 stem 归属

**论文 / 出处**

- Huang, Chen, Smaragdis, Hasegawa-Johnson. Singing-voice separation from monaural recordings using robust principal component analysis. ICASSP 2012; 求解器 Lin, Chen, Ma. The Augmented Lagrange Multiplier Method (2010)

**代码位置**

- `01_models/_selfimpl`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重**：无（解析算法，零参数）。实现见 `01_models/_selfimpl/README.md`。

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `rpca` | 1 | 0.0 M | 0.0 G | 82.5301 s | MB | 44100 Hz / 2 ch |

- 目录：`01_models/separation/RPCA/`（`code/` 为目录联接）
- 产物：`03_outputs/RPCA/`
- 主结果表：`04_reports/separation/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
