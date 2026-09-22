# ① 模型模块

每个模型一个独立文件夹。**权重不做拷贝**：`<模型>/weights/` 是指向 `01_models/_weights/` 同一份物理文件的硬链接 / 目录联接，零额外占用。

## 板 1 · 分离模型

| 文件夹 | 条目 key | 输出轨数 | 参数 | FLOPs | 时延(10 s) | 权重 |
|---|---|---|---|---|---|---|
| [`BS-RoFormer-L12/`](BS-RoFormer-L12/) | `bsroformer_l12` | 1 | 159.758 M | 43480.878 G | 4.5208 s | 有（2 项） |
| [`BS-RoFormer-L6/`](BS-RoFormer-L6/) | `bsroformer_l6` | 1 | 98.195 M | 14975.185 G | 2.1702 s | 有（2 项） |
| [`BSRNN-opt/`](BSRNN-opt/) | `bsrnn`, `bsrnn_all` | 1/4 | 164.119/40.777 M | 3891.294/955.494 G | 1.0814/5.7328 s | 有（1 项） |
| [`BSRNN-large/`](BSRNN-large/) | `bsrnn_large`, `bsrnn_large_all` | 1/4 | 146.656/36.417 M | 1126.935/276.819 G | 0.6294/2.9494 s | 有（1 项） |
| [`BSRNN-SIMO/`](BSRNN-SIMO/) | `bsrnn_simo` | 4 | 108.728 M | 1540.109 G | 1.2105 s | 有（1 项） |
| [`Demucs/`](Demucs/) | `demucs` | 4 | 41.984 M | 508.127 G | 0.2749 s | 有（1 项） |
| [`MDX-Net/`](MDX-Net/) | `mdx` | 4 | 334.544 M | 777.218 G | 0.7373 s | 有（2 项） |
| [`Open-Unmix/`](Open-Unmix/) | `umx` | 4 | 35.577 M | 14.292 G | 0.2 s | 有（4 项） |
| [`Conv-TasNet/`](Conv-TasNet/) | `convtasnet` | 4 | 13.395 M | 1171.744 G | 1.5221 s | 有（1 项） |
| [`MMDenseLSTM/`](MMDenseLSTM/) | `mmdenselstm` | 4 | 5.487 M | 294.469 G | 0.3331 s | 有（1 项） |
| [`DPRNN/`](DPRNN/) | `dprnn` | 4 | 3.686 M | 30.573 G | 0.103 s | 有（1 项） |
| [`RPCA/`](RPCA/) | `rpca` | 1 | 0.0 M | 0.0 G | 82.5301 s | 无（解析） |
| [`Oracle-IRM/`](Oracle-IRM/) | `oracle` | - | 0.0 M | 0.0 G | 0.0 s | 无（解析） |

## 板 2 · 降噪 / 去混响模型

| 文件夹 | 条目 key | 参数 | FLOPs | 时延(10 s) | 原生率 |
|---|---|---|---|---|---|
| [`denoiser-dns48/`](denoiser-dns48/) | `denoiser_dns48` | 18.868 M | 39.135 G | 0.0845 s | 16000 Hz |
| [`denoiser-dns64/`](denoiser-dns64/) | `denoiser_dns64` | 33.534 M | 69.032 G | 0.1263 s | 16000 Hz |
| [`denoiser-master64/`](denoiser-master64/) | `denoiser_master64` | 33.534 M | 69.032 G | 0.1269 s | 16000 Hz |
| [`Mel-RoFormer-Denoise/`](Mel-RoFormer-Denoise/) | `mel_roformer_denoise`, `mel_roformer_denoise_aggr` | 228.203 M | 4030.469 G | 0.9732/0.9802 s | 44100 Hz |
| [`Mel-RoFormer-Dereverb/`](Mel-RoFormer-Dereverb/) | `mel_roformer_dereverb` | 228.203 M | 4030.469 G | 0.9809 s | 44100 Hz |
| [`Mel-RoFormer-Dereverb-Echo/`](Mel-RoFormer-Dereverb-Echo/) | `mel_roformer_dereverb_echo` | 208.88 M | 3447.121 G | 1.0182 s | 44100 Hz |

## 共享层

| 目录 | 内容 |
|---|---|
| `_weights/` | **权重的唯一物理落点**（15 GB）。不能删，各模型文件夹的 `weights/` 都指向它 |
| `_third_party/` | 13 个官方/社区仓库（743 MB）。同一仓库被多个模型复用 |
| `_selfimpl/` | 自研算法索引（RPCA / NMF / ICA / HPSS / Oracle 掩码） |
| `MODEL_REGISTRY.md` | 全部模型在位状态 / 可运行性 / 精度总表 |
