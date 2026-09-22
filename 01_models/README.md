# ① 模型模块

顶层只分**两块板**，模型按板归类；每一块板下每个模型一个目录：

```
01_models/
  separation/     板 1 · 分离        （13 个模型）
  denoising/      板 2 · 降噪/去混响  （6 个模型）
  _third_party/   代码唯一物理落点（14 个仓库）
  _weights/       权重唯一物理落点（51 个文件 / 11 GB）
  _selfimpl/      自研算法索引
```

## 模型目录里有什么

每个 `<板块>/<模型>/` 下**只有 `code/`** —— 指向 `01_models/_third_party/<repo>` 的
**目录联接（junction）**，零额外占用。

```
separation/Demucs/
  code/
    demucs -> 01_models/_third_party/demucs
```

## 为什么没有「模型卡 + weights/ 联接」了（2026-09-23 精简）

原先每个模型目录是「`README.md` 模型卡 + `code/` + `weights/` 联接」三件套。后两项被证明冗余：

| 被删除的层 | 为什么冗余 | 现在从哪里取 |
|---|---|---|
| `<模型>/weights/` | 只是 `01_models/_weights/**` 的目录联接/硬链接，且**所有脚本都直接读 `_weights/`**（`tools/paths.py::WEIGHTS`），从不走这个别名 | `01_models/_weights/` |
| `<模型>/README.md` | 内容（论文出处 / 代码落点 / 权重真实路径 / 画像）可从 `model_profile.csv` + 板块索引完全还原 | `<板块>/README.md` |

> 删除前已逐个文件核验 `weights/` 里每个 inode 都存在于 `_weights/`（无一独有），
> 并把 19 张模型卡原文归档到 `tools/_scratch/_archive_model_cards_2026-09-23/`。

## 两块板

### 板 1 · 分离（separation）

| 模型 | 条目 key |
|---|---|
| [`BS-RoFormer-L12/`](separation/BS-RoFormer-L12/) | `bsroformer_l12` |
| [`BS-RoFormer-L6/`](separation/BS-RoFormer-L6/) | `bsroformer_l6` |
| [`BSRNN-large/`](separation/BSRNN-large/) | `bsrnn_large`, `bsrnn_large_all` |
| [`BSRNN-opt/`](separation/BSRNN-opt/) | `bsrnn`, `bsrnn_all` |
| [`BSRNN-SIMO/`](separation/BSRNN-SIMO/) | `bsrnn_simo` |
| [`Conv-TasNet/`](separation/Conv-TasNet/) | `convtasnet` |
| [`Demucs/`](separation/Demucs/) | `demucs` |
| [`DPRNN/`](separation/DPRNN/) | `dprnn` |
| [`MDX-Net/`](separation/MDX-Net/) | `mdx` |
| [`MMDenseLSTM/`](separation/MMDenseLSTM/) | `mmdenselstm` |
| [`Open-Unmix/`](separation/Open-Unmix/) | `umx` |
| [`Oracle-IRM/`](separation/Oracle-IRM/) | `oracle` |
| [`RPCA/`](separation/RPCA/) | `rpca` |

索引与逐模型明细：`01_models/separation/README.md`

### 板 2 · 降噪 / 去混响 / 去回声（denoising）

| 模型 | 条目 key |
|---|---|
| [`denoiser-dns48/`](denoising/denoiser-dns48/) | `denoiser_dns48` |
| [`denoiser-dns64/`](denoising/denoiser-dns64/) | `denoiser_dns64` |
| [`denoiser-master64/`](denoising/denoiser-master64/) | `denoiser_master64` |
| [`Mel-RoFormer-Denoise/`](denoising/Mel-RoFormer-Denoise/) | `mel_roformer_denoise`, `mel_roformer_denoise_aggr` |
| [`Mel-RoFormer-Dereverb/`](denoising/Mel-RoFormer-Dereverb/) | `mel_roformer_dereverb` |
| [`Mel-RoFormer-Dereverb-Echo/`](denoising/Mel-RoFormer-Dereverb-Echo/) | `mel_roformer_dereverb_echo` |

索引与逐模型明细：`01_models/denoising/README.md`

## 自检

```bash
PY="C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe"

# 1) 重建 / 补齐每个模型目录的 code/ 联接（幂等）
$PY tools/_model_deploy_layout.py

# 2) 核验：模型目录 + 代码联接 + 权重真源
$PY tools/_verify_model_deployment.py --json 04_reports/_shared/data/model_deployment_audit.json
```

**环境依赖**：项目根 `requirements-frozen-2026-09-23.txt`（pip freeze 快照）。
⚠️ 该快照**不含** PESQ（conda-forge 手工解包）与 CUDA 版 torch/torchaudio（须走
`--index-url https://download.pytorch.org/whl/cu126`），文件头已注明。

> 版 3（2026-09-23）：按板块分类 `separation/` + `denoising/`，
> 移除模型卡与 `weights/` 联接层；详见 `04_reports/_shared/docs/MODEL_DEPLOYMENT_2026-09-23.md`。
