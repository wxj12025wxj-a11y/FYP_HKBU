# 板 2 · 降噪 / 去混响 / 去回声（denoising）

> 6 个模型 ｜ 目录：`01_models/denoising/<模型>/`
>
> **本板块的模型目录只保留 `code/`**（指向 `01_models/_third_party/<repo>` 的目录联接，零额外占用）。
> 模型卡 `README.md` 与 `weights/` 联接层已于 2026-09-23 精简移除：
> 权重的**唯一物理落点**是 `01_models/_weights/`，全部脚本（含 `tools/paths.py`）都从那里直接加载。

## 索引

| 模型 | 条目 key | 参数量 | FLOPs | 时延(10 s) | 一句话定位 |
|---|---|---|---|---|---|
| [`denoiser-dns48/`](denoiser-dns48/) | `denoiser_dns48` | 18.868 M | 39.135 G | 0.0845 s | 板 2；16 kHz 因果波形域；参数量最小（18.87 M），门禁 SI-SDR 改善最高（+2.15 dB） |
| [`denoiser-dns64/`](denoiser-dns64/) | `denoiser_dns64` | 33.534 M | 69.032 G | 0.1263 s | 板 2；16 kHz；与 master64 同参数量（33.53 M），训练集不含 Valentini |
| [`denoiser-master64/`](denoiser-master64/) | `denoiser_master64` | 33.534 M | 69.032 G | 0.1269 s | 板 2；**训练集含 Valentini → 测 Valentini 属 in-domain，必须报 dns48/dns64 对照** |
| [`Mel-RoFormer-Denoise/`](Mel-RoFormer-Denoise/) | `mel_roformer_denoise`, `mel_roformer_denoise_aggr` | 228.203 M | 4030.469 G | 0.9802 s | 板 2；44.1 kHz 立体声非因果；掩码估计头独占 88.3% 参数 |
| [`Mel-RoFormer-Dereverb/`](Mel-RoFormer-Dereverb/) | `mel_roformer_dereverb` | 228.203 M | 4030.469 G | 0.9809 s | 板 2 去混响；社区权重，非论文官方 |
| [`Mel-RoFormer-Dereverb-Echo/`](Mel-RoFormer-Dereverb-Echo/) | `mel_roformer_dereverb_echo` | 208.88 M | 3447.121 G | 1.0182 s | 板 2 去混响+去回声；社区权重（Sucial），SDR 10.0169 |

## 逐模型明细（原模型卡内容）

### denoiser-dns48

> 板 2；16 kHz 因果波形域；参数量最小（18.87 M），门禁 SI-SDR 改善最高（+2.15 dB）

**论文 / 出处**

- Defossez, Synnaeve, Adi. Real Time Speech Enhancement in the Waveform Domain. Interspeech 2020 (facebookresearch/denoiser)

**代码位置**

- `01_models/_third_party/denoiser`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `dns48-11decc9d8e3f0998.th` | `01_models/_weights/denoiser/dns48-11decc9d8e3f0998.th` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `denoiser_dns48` | 1 | 18.868 M | 39.135 G | 0.0845 s | 233.3 MB | 16000 Hz / 1 ch |

- 目录：`01_models/denoising/denoiser-dns48/`（`code/` 为目录联接）
- 产物：`03_outputs/denoiser-dns48/`
- 主结果表：`04_reports/denoiser/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### denoiser-dns64

> 板 2；16 kHz；与 master64 同参数量（33.53 M），训练集不含 Valentini

**论文 / 出处**

- Defossez, Synnaeve, Adi. Real Time Speech Enhancement in the Waveform Domain. Interspeech 2020 (facebookresearch/denoiser)

**代码位置**

- `01_models/_third_party/denoiser`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `dns64-a7761ff99a7d5bb6.th` | `01_models/_weights/denoiser/dns64-a7761ff99a7d5bb6.th` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `denoiser_dns64` | 1 | 33.534 M | 69.032 G | 0.1263 s | 343.2 MB | 16000 Hz / 1 ch |

- 目录：`01_models/denoising/denoiser-dns64/`（`code/` 为目录联接）
- 产物：`03_outputs/denoiser-dns64/`
- 主结果表：`04_reports/denoiser/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### denoiser-master64

> 板 2；**训练集含 Valentini → 测 Valentini 属 in-domain，必须报 dns48/dns64 对照**

**论文 / 出处**

- Defossez, Synnaeve, Adi. Real Time Speech Enhancement in the Waveform Domain. Interspeech 2020 (facebookresearch/denoiser)

**代码位置**

- `01_models/_third_party/denoiser`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `master64-8a5dfb4bb92753dd.th` | `01_models/_weights/denoiser/master64-8a5dfb4bb92753dd.th` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `denoiser_master64` | 1 | 33.534 M | 69.032 G | 0.1269 s | 343.2 MB | 16000 Hz / 1 ch |

- 目录：`01_models/denoising/denoiser-master64/`（`code/` 为目录联接）
- 产物：`03_outputs/denoiser-master64/`
- 主结果表：`04_reports/denoiser/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### Mel-RoFormer-Denoise

> 板 2；44.1 kHz 立体声非因果；掩码估计头独占 88.3% 参数

**论文 / 出处**

- 架构源自 Lu et al. Mel-Band RoFormer for Music Source Separation (arXiv:2310.01809)
- ⚠ 权重由 **aufr33** 训练并经 ZFTurbo 框架分发，属**社区模型**，非论文官方权重，论文引用时须注明。

**代码位置**

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `mel_roformer_denoise` | `01_models/_weights/mel_roformer_denoise` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `mel_roformer_denoise` | 1 | 228.203 M | 4030.469 G | 0.9802 s | 2011.2 MB | 44100 Hz / 2 ch |
| `mel_roformer_denoise_aggr` | 1 | 228.203 M | 4030.469 G | 0.9732 s | 2011.2 MB | 44100 Hz / 2 ch |

- 目录：`01_models/denoising/Mel-RoFormer-Denoise/`（`code/` 为目录联接）
- 产物：`03_outputs/Mel-RoFormer-Denoise/`
- 主结果表：`04_reports/denoiser/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### Mel-RoFormer-Dereverb

> 板 2 去混响；社区权重，非论文官方

**论文 / 出处**

- 架构源自 Lu et al. Mel-Band RoFormer for Music Source Separation (arXiv:2310.01809)
- ⚠ 权重由 **anvuew** 训练并经 ZFTurbo 框架分发，属**社区模型**，非论文官方权重，论文引用时须注明。

**代码位置**

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `mel_roformer_dereverb` | `01_models/_weights/mel_roformer_dereverb` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `mel_roformer_dereverb` | 1 | 228.203 M | 4030.469 G | 0.9809 s | 2011.2 MB | 44100 Hz / 2 ch |

- 目录：`01_models/denoising/Mel-RoFormer-Dereverb/`（`code/` 为目录联接）
- 产物：`03_outputs/Mel-RoFormer-Dereverb/`
- 主结果表：`04_reports/denoiser/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`

### Mel-RoFormer-Dereverb-Echo

> 板 2 去混响+去回声；社区权重（Sucial），SDR 10.0169

**论文 / 出处**

- 架构源自 Lu et al. Mel-Band RoFormer for Music Source Separation (arXiv:2310.01809)
- ⚠ 权重由 **Sucial** 训练并经 ZFTurbo 框架分发，属**社区模型**，非论文官方权重，论文引用时须注明。

**代码位置**

- `01_models/_third_party/Music-Source-Separation-Training`
- 统一加载器：`tools/benchmark_model_universal.py::REGISTRY`（板 2 见 `tools/_denoise_registry.py`）

**权重（真实落点 `01_models/_weights/`）**

| 文件 | 本机真实路径 |
|---|---|
| `mel_roformer_dereverb_echo` | `01_models/_weights/mel_roformer_dereverb_echo` |

**本机实测画像**

| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |
|---|---|---|---|---|---|---|
| `mel_roformer_dereverb_echo` | 2 | 208.88 M | 3447.121 G | 1.0182 s | 1869.2 MB | 44100 Hz / 2 ch |

- 目录：`01_models/denoising/Mel-RoFormer-Dereverb-Echo/`（`code/` 为目录联接）
- 产物：`03_outputs/Mel-RoFormer-Dereverb-Echo/`
- 主结果表：`04_reports/denoiser/docs/`
- 架构原理图：`04_reports/_shared/figures/model_arch/`
