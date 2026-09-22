# BSRNN-opt

> 板块：**板 1 分离** ｜ 目录：`01_models/BSRNN-opt/`

同一批权重服务两个条目：单目标 vocals 与四轨 4-stem（4 ckpt）

## 论文 / 出处

- Luo, Yu. Music Source Separation with Band-split RNN. IEEE/ACM TASLP 2023 (arXiv:2209.15174)

## 代码位置

- `01_models/_third_party/bsrnn`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

## 本机权重（真实落点）

本目录下 `weights/` 是指向**同一份物理文件**的硬链接 / 目录联接，不占额外空间。

| 权重文件 | 本机真实路径 |
|---|---|
| `bsrnn-opt` | `01_models/_weights/BSRNN/bsrnn-opt` |

> 物理落点唯一：`01_models/_weights/`。改路径只改 `tools/paths.py`。

## 本机实测画像

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `bsrnn` | 1 | 40.777 M | 955.494 G | 1.0814 s | 1675.0 MB | 44100 Hz / 2 ch |
| `bsrnn_all` | 4 | 164.119 M | 3891.294 G | 5.7328 s | 2637.4 MB | 44100 Hz / 2 ch |

口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合注意力不计入），跨架构比较必须配合时延看。

数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

## 本模型在项目里的位置

- 产物目录：`03_outputs/BSRNN-opt/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
- 主结果表：`04_reports/separation/docs/`
- 板块标签：`board-1`
