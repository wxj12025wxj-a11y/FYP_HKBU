# MDX-Net

> 板块：**板 1 分离** ｜ 目录：`01_models/MDX-Net/`

mdx_extra 由 4 个 submission 权重组成；全场参数最多（334.5 M）

## 论文 / 出处

- Kim, Choi, Chung, Lee. KUIELab-MDX-Net: A Two-Stream Neural Network for Music Demixing. MDX Workshop 2021; 评测框架 Mitsufuji et al., Frontiers in Signal Processing 2022

## 代码位置

- `01_models/_third_party/mdx-net`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

## 本机权重（真实落点）

本目录下 `weights/` 是指向**同一份物理文件**的硬链接 / 目录联接，不占额外空间。

| 权重文件 | 本机真实路径 |
|---|---|
| `mdx_extra` | `01_models/_weights/MDX-Net/mdx_extra` |
| `mixer.ckpt` | `01_models/_weights/MDX-Net/mixer.ckpt` |

> 物理落点唯一：`01_models/_weights/`。改路径只改 `tools/paths.py`。

## 本机实测画像

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `mdx` | 4 | 334.544 M | 777.218 G | 0.7373 s | 1912.8 MB | 44100 Hz / 2 ch |

口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合注意力不计入），跨架构比较必须配合时延看。

数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

## 本模型在项目里的位置

- 产物目录：`03_outputs/MDX-Net/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
- 主结果表：`04_reports/separation/docs/`
- 板块标签：`board-1`
