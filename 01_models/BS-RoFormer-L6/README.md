# BS-RoFormer-L6

> 板块：**板 1 分离** ｜ 目录：`01_models/BS-RoFormer-L6/`

输出为 mixture-drums-bass（vocals+other 残差），**不是** yaml 里写的 other

## 论文 / 出处

- Lu, Wang, Jiang, Zhang. Music Source Separation with Band-Split RoPE Transformer. 2024 (arXiv:2309.02612)

## 代码位置

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

## 本机权重（真实落点）

本目录下 `weights/` 是指向**同一份物理文件**的硬链接 / 目录联接，不占额外空间。

| 权重文件 | 本机真实路径 |
|---|---|
| `model_bs_roformer_ep_937_sdr_10.5309.ckpt` | `01_models/_weights/BS-RoFormer/model_bs_roformer_ep_937_sdr_10.5309.ckpt` |
| `model_bs_roformer_ep_937_sdr_10.5309.yaml` | `01_models/_weights/BS-RoFormer/configs/model_bs_roformer_ep_937_sdr_10.5309.yaml` |

> 物理落点唯一：`01_models/_weights/`。改路径只改 `tools/paths.py`。

## 本机实测画像

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsroformer_l6` | 1 | 98.195 M | 14975.185 G | 2.1702 s | 2087.4 MB | 44100 Hz / 2 ch |

口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合注意力不计入），跨架构比较必须配合时延看。

数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

## 本模型在项目里的位置

- 产物目录：`03_outputs/BS-RoFormer-L6/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
- 主结果表：`04_reports/separation/docs/`
- 板块标签：`board-1`
