# BS-RoFormer-L12

> 板块：**板 1 分离** ｜ 目录：`01_models/BS-RoFormer-L12/`

单目标（只出 vocals），L12 = 12 层 Transformer，本机全部条目里 FLOPs 最高

## 论文 / 出处

- Lu, Wang, Jiang, Zhang. Music Source Separation with Band-Split RoPE Transformer. 2024 (arXiv:2309.02612)

## 代码位置

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

## 本机权重（真实落点）

本目录下 `weights/` 是指向**同一份物理文件**的硬链接 / 目录联接，不占额外空间。

| 权重文件 | 本机真实路径 |
|---|---|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | `01_models/_weights/BS-RoFormer/model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| `model_bs_roformer_ep_317_sdr_12.9755.yaml` | `01_models/_weights/BS-RoFormer/configs/model_bs_roformer_ep_317_sdr_12.9755.yaml` |

> 物理落点唯一：`01_models/_weights/`。改路径只改 `tools/paths.py`。

## 本机实测画像

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsroformer_l12` | 1 | 159.758 M | 43480.878 G | 4.5208 s | 3394.3 MB | 44100 Hz / 2 ch |

口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合注意力不计入），跨架构比较必须配合时延看。

数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

## 本模型在项目里的位置

- 产物目录：`03_outputs/BS-RoFormer-L12/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
- 主结果表：`04_reports/separation/docs/`
- 板块标签：`board-1`
