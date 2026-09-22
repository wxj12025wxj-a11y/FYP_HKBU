# Demucs

> 板块：**板 1 分离** ｜ 目录：`01_models/Demucs/`

htdemucs 混合时域+频域架构；参考表里 128 M 指原版 v1，本机是 41.98 M

## 论文 / 出处

- Rouard, Massa, Defossez. Hybrid Transformers for Music Source Separation. ICASSP 2023

## 代码位置

- `01_models/_third_party/demucs`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

## 本机权重（真实落点）

本目录下 `weights/` 是指向**同一份物理文件**的硬链接 / 目录联接，不占额外空间。

| 权重文件 | 本机真实路径 |
|---|---|
| `955717e8-8726e21a.th` | `tools/_scratch/.cache/torch/hub/checkpoints/955717e8-8726e21a.th` |

> 物理落点唯一：`01_models/_weights/`。改路径只改 `tools/paths.py`。

## 本机实测画像

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `demucs` | 4 | 41.984 M | 508.127 G | 0.2749 s | 599.7 MB | 44100 Hz / 2 ch |

口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合注意力不计入），跨架构比较必须配合时延看。

数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

## 本模型在项目里的位置

- 产物目录：`03_outputs/Demucs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
- 主结果表：`04_reports/separation/docs/`
- 板块标签：`board-1`
