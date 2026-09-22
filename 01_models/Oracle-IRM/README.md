# Oracle-IRM

> 板块：**板 1 分离** ｜ 目录：`01_models/Oracle-IRM/`

理论上界（IRM/IBM），零参数，用来给所有模型划天花板

## 论文 / 出处

- Rafii, Liutkus, Stoter, Mimilakis, Bittner. MUSDB18 - a corpus for music separation. 2017 (sigsep-mus-oracle); IRM 定义见 Wang & Wang, IEEE/ACM TASLP 2014

## 代码位置

- `01_models/_third_party/sigsep-mus-oracle`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

## 本机权重（真实落点）

**本模型无权重文件**——它是解析算法（零参数），公式即模型。

实现见 `01_models/_selfimpl/README.md` 指向的 `tools/compare_separation_methods.py`。

## 本机实测画像

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `oracle` | 4 | 0.0 M | 0.0 G | 0.0 s |  MB | 44100 Hz / 2 ch |

口径：44.1 kHz 立体声、10 s 输入、预热 2 次后取 5 次中位；FLOPs 由 `torch.utils.flop_counter.FlopCounterMode` 实测，是**下界**（STFT 与融合注意力不计入），跨架构比较必须配合时延看。

数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

## 本模型在项目里的位置

- 产物目录：`03_outputs/Oracle-IRM/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
- 主结果表：`04_reports/separation/docs/`
- 板块标签：`board-1`
