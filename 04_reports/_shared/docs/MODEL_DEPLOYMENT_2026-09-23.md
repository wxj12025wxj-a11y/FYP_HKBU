# 模型本地部署报告（19 模型 · 完整代码 + 权重）

> 项目：HKBU FYP — 实时音乐源分离 / 降噪
> 机器：Windows 10 (26200) + RTX 4070 Laptop 8 GiB + Python 3.13.12
> 日期：**2026-09-23** ｜ 关联：`01_models/README.md`、`01_models/MODEL_REGISTRY.md`
> 数据源：`04_reports/_shared/data/model_deployment_audit.json`、`model_deploy_layout.json`

---

## 0. 一句话结论

**能。本机已把 19 个模型的「完整代码 + 完整权重」全部落到本地，本轮补齐了三处缺口，核验 19/19 PASS。**

不能下载的只有 **2 个**（RPCA+DRNN、Pac-HuBERT-SEP）——原因是**上游从未公开代码与权重**，
与本机磁盘 / 网络无关（见 §5）。

---

## 1. 本机可行性判定（三项硬约束全部满足）

| 约束 | 实测值 | 判定 |
|---|---|---|
| **磁盘** | E: 总 150.00 GB，可用 **64.44 GB**（43.0%） | ✅ 本次新增仅 **≈1.37 GB**，余量充足 |
| **网络** | `github.com` / `huggingface.co` / `zenodo.org` / `pypi.org` 全部 HTTP 200，直连无需代理 | ✅ |
| **工具链** | `git 2.55.0.windows.3`，`core.longpaths=true` 已开启 | ✅ |

> ⚠️ **`core.longpaths=true` 是必需的**，不是可选项：本项目曾因 Windows 260 字符路径上限
> 导致 `DNN-based_source_separation` 克隆后文件全丢，必须 `git config core.longpaths true` 后
> 重新 checkout 才还原。

**去重后的真实占用（硬链接只算一次）**

| 目录 | 唯一占用 | 内容 |
|---|---|---|
| `01_models/_third_party/` | **2.11 GB** | 14 个仓库（工作区 0.49 GB + 完整 git 历史 1.62 GB） |
| `01_models/_weights/` | **11.38 GB** | 51 个权重 / 配置文件 |
| `01_models/` 合计 | **13.49 GB** | 含 19 个模型文件夹（卡片 + 联接，0 字节） |

---

## 2. 部署结构：模型卡 + `code/` + `weights/` 三件套

每个模型文件夹固定三样，**代码与权重都不做拷贝**：

```
01_models/
├── _third_party/                     ← 代码的唯一物理落点（14 个仓库，2.11 GB）
│   ├── demucs/  bsrnn/  denoiser/  mdx-net*/  Music-Source-Separation-Training/
│   ├── Conv-TasNet/  DNN-based_source_separation/  Dual-Path-RNN-Pytorch/
│   ├── open-unmix-pytorch/  BS-RoFormer/  sigsep-mus-oracle/
├── _weights/                         ← 权重的唯一物理落点（11.38 GB）
│   ├── BS-RoFormer/  BSRNN/  Demucs/  denoiser/  DNN-based_source_separation/
│   ├── dprnn_musdb/  MDX-Net/  mel_roformer_{denoise,dereverb,dereverb_echo}/  open-unmix/
├── BSRNN-SIMO/                       ← 每个模型一个文件夹
│   ├── README.md                     ←   ①模型卡（出处 / 代码 / 权重 / 画像 / 复现命令）
│   ├── code/bsrnn         ──联接──→  _third_party/bsrnn
│   └── weights/simo-bsrnn-opt ─联接→ _weights/BSRNN/simo-bsrnn-opt
```

| 联接方式 | 用在 | 特性 |
|---|---|---|
| **目录联接**（junction，`mklink /J`） | `code/<repo>`；部分多目录权重的 `weights/<pkg>` | 目录级、**不需要管理员权限**（`mklink /D` 符号链接需要，故不用） |
| **硬链接**（`os.link`） | 单文件权重 | 文件级、同卷、删任一路径不影响另一路径的数据 |

**两种都是零额外占用**——19 个模型目录里没有一份真正的代码或权重副本。

---

## 3. 本轮补齐的三处缺口

部署前，19 个模型文件夹只有「模型卡 + `weights/`」，且 `_third_party` 里的代码**不完整**：

| # | 缺口 | 证据 | 处理 |
|---|---|---|---|
| **A** | **12 个仓库是 depth=1 浅克隆** | `git rev-parse --is-shallow-repository` 全部返回 `true`，`rev-list --count HEAD` = **1** | `git fetch --unshallow`（12/12 成功） |
| **B** | **`demucs` 根本不是 git 仓库** | `_third_party/demucs/` 有 `.gitignore` 却**没有 `.git`**，无法说出是哪个版本 | 克隆官方仓库比对后替换（见下） |
| **C** | **Open-Unmix 的源码不在项目里** | `_third_party/` 里没有 open-unmix；代码只存在于 venv 的 `site-packages/openunmix/` | 克隆 `sigsep/open-unmix-pytorch` 到 `_third_party/` |

### 3.1 A：补齐完整 git 历史（12 个仓库）

全部 `rc=0`、`shallow=false`。`_third_party` 由 **743 MB → 2.11 GB**（+1.37 GB）。

| 仓库 | 补前 commits | 补后 commits | `.git` | remote |
|---|---|---|---|---|
| `DNN-based_source_separation` | 1 | **2937** | 307 MB | `tky823/DNN-based_source_separation` |
| `bsrnn` | 1 | **74** | 890 MB | `magronp/bsrnn` |
| `Music-Source-Separation-Training` | 1 | **626** | 2 MB | `ZFTurbo/Music-Source-Separation-Training` |
| `mdx-net` | 1 | **101** | 14 MB | `kuielab/mdx-net` |
| `mdx-net-submission` | 1 | **50** | 125 MB | `kuielab/mdx-net-submission` |
| `mdx-net-submission-leaderboard_A` | 1 | **62** | 7 MB | 同上（分支 `leaderboard_A`） |
| `mdx-net-submission-leaderboard_B` | 1 | **57** | 118 MB | 同上（分支 `leaderboard_B`） |
| `BS-RoFormer` | 1 | **86** | 1 MB | `lucidrains/BS-RoFormer` |
| `denoiser` | 1 | **73** | 2 MB | `facebookresearch/denoiser` |
| `Dual-Path-RNN-Pytorch` | 1 | **46** | 1 MB | `JusperLee/Dual-Path-RNN-Pytorch` |
| `Conv-TasNet` | 1 | **32** | 2 MB | `kaituoxu/Conv-TasNet` |
| `sigsep-mus-oracle` | 1 | **18** | 1 MB | `sigsep/sigsep-mus-oracle` |

> ⚠️ `bsrnn` 的完整历史达 **890 MB**（该仓库历史里含大体积二进制），远超它 2.9 MB 的工作区 —— 这是本轮最大的单项开销。

### 3.2 B：`demucs` 补上版本控制（并已证明替换无损）

原目录是从发布包解出的**非 git** 副本，版本无法追溯。处理流程：

1. `git clone https://github.com/facebookresearch/demucs.git` → HEAD `e976d93`（2023-11-16）
2. **逐文件比对**原目录 vs 克隆：112 个文件（含 11 个 `__pycache__`）对 101 个；
   **排除行尾差异后 101/101 内容完全一致**（`diff --strip-trailing-cr`），证明是同一份代码
3. 旧目录移到 `tools/_scratch/_clone/demucs_old_nonGit/` 备份 → 克隆版就位
4. **版本对齐**：克隆版 `demucs/__version__ = "4.1.0a2"`，与原目录**完全一致**（该仓库已归档，HEAD 即未发布的 4.1.0a2）

**替换后端到端验证（只读探针，未写任何结果文件）**

| 检查 | 结果 |
|---|---|
| `demucs.__file__` | `E:\FYP_HKBU\01_models\_third_party\demucs\demucs\__init__.py` ✅ 从项目内加载 |
| `htdemucs` 参数量 | **41.984 M** ✅ 与 `model_profile.csv` 完全一致 |
| 真实音频 5 s 前向（CUDA） | 输出 `(4, 2, 220500)`，全有限值 ✅ |
| MDX-Net（复用 demucs 运行时） | `mdx_extra` 参数量 **334.544 M** ✅ 与画像表一致 |

### 3.3 C：补上 Open-Unmix 源码

- 目标：`sigsep/open-unmix-pytorch`，**tag `v1.3.0`**（与 venv 内 `openunmix==1.3.0` 精确对齐）
- 校验：仓库 `openunmix/` 模块清单与 `site-packages/openunmix/` **逐项一致**
  （`__init__ / cli / data / evaluate / filtering / model / predict / transforms / utils`）
- 落点：`01_models/_third_party/open-unmix-pytorch/`

### 3.4 附带：`code/` 联接层 + 权重落点规范化

- **新建 26 个 `code/` 目录联接**，覆盖全部 19 个模型，并同步写进各自的模型卡（新增「本地代码（零拷贝）」小节）
- **修正权重落点**：`Demucs`（1 个）与 `Open-Unmix`（4 个）的权重此前**只存在于 `tools/_scratch/.cache/torch/hub/checkpoints/`**
  （torch.hub 缓存），并非 `_weights/`。现已在 `_weights/Demucs/`、`_weights/open-unmix/` 建立硬链接镜像
  （`nlink=3`：缓存 + `_weights/` + 模型目录，**同一份物理数据，零占用**），使「`_weights/` 为唯一物理落点」这条约定真正成立
- `.gitignore` 补 `01_models/*/code/`（沿用既有 `01_models/*/weights/` 的写法）——
  **必须忽略**，否则 git 会把目录联接当普通目录递归进去，把 2 GB 第三方仓库再抄一遍

---

## 4. 逐模型部署清单（19/19 PASS）

| 模型 | 模型卡 | `code/` 联接 | `weights/` 文件 | 权重归属 |
|---|---|---|---|---|
| BS-RoFormer-L12 | ✅ | 2（MSST, BS-RoFormer） | 2 | OK |
| BS-RoFormer-L6 | ✅ | 2（MSST, BS-RoFormer） | 2 | OK |
| BSRNN-opt | ✅ | 1（bsrnn） | 4 | OK |
| BSRNN-large | ✅ | 1（bsrnn） | 4 | OK |
| BSRNN-SIMO | ✅ | 1（bsrnn） | 1 | OK |
| Demucs | ✅ | 1（demucs） | 1 | OK |
| MDX-Net | ✅ | 4（mdx-net + 3 submission 分支） | 6 | OK |
| Open-Unmix | ✅ | 1（open-unmix-pytorch） | 4 | OK |
| Conv-TasNet | ✅ | 2（Conv-TasNet, DNN-based） | 6 | OK |
| MMDenseLSTM | ✅ | 1（DNN-based） | 8 | OK |
| DPRNN | ✅ | 1（Dual-Path-RNN-Pytorch） | 2 | OK |
| RPCA | ✅ | 1（`_selfimpl` 索引） | 0（解析法） | OK |
| Oracle-IRM | ✅ | 2（sigsep-mus-oracle, `_selfimpl`） | 0（解析法） | OK |
| denoiser-dns48 | ✅ | 1（denoiser） | 1 | OK |
| denoiser-dns64 | ✅ | 1（denoiser） | 1 | OK |
| denoiser-master64 | ✅ | 1（denoiser） | 1 | OK |
| Mel-RoFormer-Denoise | ✅ | 1（MSST） | 3 | OK |
| Mel-RoFormer-Dereverb | ✅ | 1（MSST） | 2 | OK |
| Mel-RoFormer-Dereverb-Echo | ✅ | 1（MSST） | 2 | OK |

**合计数**：26 个代码联接、50 个模型侧权重文件（`_weights/` 共 51 个，多出的 1 个是
`MDX-Net/onnx_A.zip` 备查包，未联接到任何模型目录）。

「权重归属 = OK」的判据：每个 `weights/` 下的大文件都能在 `_weights/` 找到**同一 inode**
（`os.path.samefile` 为真），即不存在独立副本。

---

## 5. 不能下载的 2 个（上游未公开，非本机限制）

| 模型 | 类别 | 阻塞根因 | 处理 |
|---|---|---|---|
| **RPCA+DRNN** | 混合（古典 + DL） | Lai & Wang, *EURASIP JASMP 2022:4*（DOI `10.1186/s13636-022-00236-9`）**未公开任何代码或权重**，仅放出分离音频附件。且该论文口径为**单声道 2 源**（人声/伴奏），与 MUSDB18 4-stem **不可横比** | 已定稿：降级为「相关工作」。其两条思路本项目已各自独立实现 —— 古典分解走 `--model rpca`，时域 DL 走 `--model dprnn`（本机自训） |
| **Pac-HuBERT-SEP** | 自监督 + DL | MERL 项目页（Ke Chen et al., arXiv:2304.02160）**未提供代码与权重下载** | 已定稿：论文中仅作相关工作引用，不再投入工时 |

> 这两个既不在 `01_models/` 的 19 个文件夹里，也不在这 22 条画像条目里 ——
> 项目从一开始就按「无公开实现」处理，未纳入部署范围。

---

## 6. 自检与复现

```bash
PY="C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe"

# 重建 / 补齐所有 code/ 与 weights/ 联接（幂等，可反复跑）
$PY tools/_model_deploy_layout.py

# 同步更新 19 个模型卡的「本地代码」小节（幂等）
$PY tools/_model_deploy_layout.py --cards

# 全量核验（模型卡 + 代码联接 + 权重归属 + git 完整性 + 零拷贝），输出 PASS/FAIL
$PY tools/_verify_model_deployment.py --json 04_reports/_shared/data/model_deployment_audit.json
```

核验脚本的判据（全部机器可验，不靠"看起来对"）：

1. 模型卡存在且非空；
2. `code/<repo>` **必须能 `os.readlink`** 解析出指向 `_third_party/<repo>` 的联接 ——
   解析不出来的就是**真的拷了一份**，直接 FAIL；
3. `weights/**` 每个文件必须能在 `_weights/` 找到同一 inode，否则判为「有独立副本」FAIL；
4. 每个第三方仓库 `is-shallow == false` 且有 remote；
5. `_weights` 按 inode 去重后的占用，必须等于各模型 `weights/` 路径求和的量级
   （差值仅来自未联接的备查包）→ 证明零拷贝。

---

## 7. 遗留与建议

| # | 事项 | 说明 | 建议 |
|---|---|---|---|
| 1 | **根仓库 `.git` 有 496 个松散对象 / 542 MB** | `.gitignore` 里记录了 2026-09-23 的事故：`git add .` 后从未 commit，把 47 GB 数据集 + 21 GB 产物塞进对象库，E 盘可用空间归零；`git prune --expire=now` 回收了 57 GB。但当前仍有 542 MB 松散对象，而 `HEAD` 的树只有 **270 文件 / 54.5 MB** | 可 `git gc --prune=now` 收口。**注意根仓库内嵌了 14 个第三方 git 仓库，操作前先确认 `.gitignore` 覆盖完整** |
| 2 | **`weight/onnx_A.zip`（105 MB）与 `mdx-net-submission/onnx_A/` 重复** | `MODEL_REGISTRY` 记为「备查」 | 保留；若要回收空间，删 zip 即可（已解压内容在位） |
| 3 | **`FYP_HKBU/FYP_HKBU/` 是空壳克隆** | 你的 GitHub 仓库 `wxj12025wxj-a11y/FYP_HKBU` 远端**只有一个 `.gitattributes`**（initial commit `fdedbed`）；这个嵌套目录是那次克隆的遗留，仅 75 KB、无工作区文件 | 确认不用后可删；若打算把项目推上去，注意**权重与第三方仓库已在 `.gitignore` 里排除**，最终仓库约 55 MB 量级 |
| 4 | **环境依赖此前无任何锁定文件** | 项目里没有 requirements / pyproject / lock | ✅ 已补 `requirements-frozen-2026-09-23.txt`（115 个包）。⚠️ 其中**不含** PESQ（conda-forge 手工解包）与 CUDA 版 torch/torchaudio（须走 `--index-url .../whl/cu126`），文件头已注明 |

---

## 8. 本次改动文件清单

| 类型 | 路径 |
|---|---|
| 新增脚本 | `tools/_model_deploy_layout.py`（建联接 + 刷模型卡，幂等）<br>`tools/_verify_model_deployment.py`（全量核验） |
| 新增数据 | `04_reports/_shared/data/model_deploy_layout.json`<br>`04_reports/_shared/data/model_deployment_audit.json` |
| 新增文档 | 本文件；`requirements-frozen-2026-09-23.txt` |
| 新增联接 | 26 个 `01_models/<模型>/code/<repo>`（目录联接，0 字节） |
| 新增硬链接 | `_weights/Demucs/`（1 个）+ `_weights/open-unmix/`（4 个） |
| 代码补齐 | `_third_party/demucs`（非 git → git `e976d93`）<br>`_third_party/open-unmix-pytorch`（新，`v1.3.0`）<br>12 个仓库 `--unshallow` |
| 修改 | `01_models/README.md`（部署约定 + 共享层 + 自检）<br>19 个 `01_models/<模型>/README.md`（+「本地代码」小节）<br>`.gitignore`（+`01_models/*/code/`） |
| 备份（可删） | `tools/_scratch/_clone/demucs_old_nonGit/` |

---

## 9. 版本 3：模型按板块分类 + 移除两个冗余层（2026-09-23 稍晚）

### 9.1 为什么改

第 6 节的核验已证明：`<模型>/weights/` 里的每个文件都能在 `_weights/` 找到**同一 inode**，
而**所有脚本（含 `paths.py::WEIGHTS`）都直接从 `_weights/` 加载权重**，从不读这个别名。
模型卡 `README.md` 的内容（论文出处 / 代码落点 / 权重真实路径 / 画像）也完全可从
`model_profile.csv` + 板块索引还原。→ 这两层是**纯冗余**，删除不影响任何功能。

### 9.2 新结构

```
01_models/
  README.md              模块总览（重写）
  MODEL_REGISTRY.md      全模型登记表（保留）
  separation/            板 1 · 分离（13 个）  ← README.md 为板块索引
    <模型>/code/<repo>    目录联接 -> _third_party/<repo>（零额外占用）
  denoising/             板 2 · 降噪/去混响/去回声（6 个）  ← README.md 为板块索引
    <模型>/code/<repo>
  _third_party/          代码唯一物理落点（14 个仓库，2.11 GB）
  _weights/              权重唯一物理落点（51 个文件，11.38 GB）
  _selfimpl/
```

### 9.3 删除前的安全核验（先证伪「有独有数据」再删）

| 动作 | 结果 |
|---|---|
| 逐个文件核验 `<模型>/weights/**` 的 inode 是否都在 `_weights/` | **50/50 全部同源，无一独有** → 安全 |
| 19 张模型卡原文归档 | `tools/_scratch/_archive_model_cards_2026-09-23/`（19 个 .md） |
| 卡片信息结构化并入板块索引 | `separation/README.md`（417 行）+ `denoising/README.md`（204 行） |
| 可逆性 | `04_reports/_shared/data/_reclassify_manifest.json`（79 条操作） |

删除内容：**19 个 `README.md` + 17 个 `weights/` 目录**（RPCA / Oracle-IRM 无权重，本来就没有）。
净释放空间 ≈ 0（都是联接/小文本），但**目录层数从 3 层降到 2 层，语义清晰**。

### 9.4 同步改动的路径中枢

| 文件 | 改动 |
|---|---|
| `tools/paths.py` | 新增 `MODEL_BOARD` / `SEP_MODELS` / `DEN_MODELS` / `board_of_model()` / `model_board_dir()` / `model_code_dir()`；`model_dir()` 改为**板块感知**（带旧扁平路径回退）；**移除 `model_weights()`** |
| `tools/_model_deploy_layout.py` | 改为板块感知；**移除 `--cards`**（模型卡已不存在） |
| `tools/_verify_model_deployment.py` | 判据改为「板块归属 + 目录纯净（不得残留 README.md/weights/）+ code 联接目标须落在 `_third_party`/`_selfimpl` + `_weights` 真源完整性」 |
| `tools/selfcheck_project.py` | `01_models` 子项改为 `separation` / `denoising` / `<板块>/<模型>/code (抽样)` |
| `.gitignore` | `01_models/*/weights/` → `01_models/*/*/code/`（+ `01_models/*/*/` 兜底）；保留追踪 `01_models/*/README.md` |

> ⚠️ **Windows 细节（本次踩到）**：junction 的 `os.readlink()` 返回值带**扩展长度前缀** `\\?\`
> （如 `\\?\E:\FYP_HKBU\01_models\_third_party\bsrnn`）。直接拿它做 `Path.relative_to()`
> 会**判不中**（`is_dir()` 却为 True，极易造成"脚本能跑但判定反向"）。必须剥前缀再比较 ——
> 已在 `_verify_model_deployment.py::_norm_target()` 里处理。

### 9.5 版本 3 核验结果

```
模型 19 个（separation 13 / denoising 6）：PASS=19 FAIL=0
第三方仓库 14 个（完整历史 14 个）
权重真源：51 个文件 / 11.38 GB（唯一占用），内部重复 0.00 MB
```

复现入口：`tools/_reclassify_models_2026-09-23.py`（默认干跑，`--execute` 执行）。
