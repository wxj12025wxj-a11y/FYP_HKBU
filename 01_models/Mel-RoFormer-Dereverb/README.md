# Mel-RoFormer-Dereverb

> 板块：**板 2 降噪 / 去混响** ｜ 目录：`01_models/Mel-RoFormer-Dereverb/`

板 2 去混响；社区权重，非论文官方

## 论文 / 出处

- 架构源自 Lu et al. Mel-Band RoFormer for Music Source Separation (arXiv:2310.01809)
- ⚠ 权重由 **anvuew** 训练并经 ZFTurbo 框架分发，属**社区模型**，非论文官方权重，论文引用时须注明。

## 代码位置

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

## 本机权重（真实落点）

本目录下 `weights/` 是指向**同一份物理文件**的硬链接 / 目录联接，不占额外空间。

| 权重文件 | 本机真实路径 |
|---|---|
| `mel_roformer_dereverb` | `01_models/_weights/mel_roformer_dereverb` |

> 物理落点唯一：`01_models/_weights/`。改路径只改 `tools/paths.py`。

## 本机实测画像

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `mel_roformer_dereverb` | 1 | 228.203 M | 4030.469 G | 0.9809 s | 2011.2 MB | 44100 Hz / 2 ch |

口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合注意力不计入），跨架构比较必须配合时延看。

数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

## 本模型在项目里的位置

- 产物目录：`03_outputs/Mel-RoFormer-Dereverb/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
- 主结果表：`04_reports/denoiser/docs/`
- 板块标签：`board-2`
