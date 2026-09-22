# FYP_HKBU — 实时音乐源分离 / 深度降噪

香港浸会大学 FYP ｜ 数据集：MUSDB18-HQ（音乐分离）+ DNS-Challenge（语音降噪）+ Valentini-Botinhao（降噪评测）

交付日期 **2026-10-27** ｜ 当前版本目录结构：**2026-09-23 第二次重组**

---

## ① 五大模块（顶层只有这 5 项）

```
E:\FYP_HKBU\
│
├── 01_models\          ① 模型 —— 每个模型一个独立文件夹
│   ├── BS-RoFormer-L12\     ← 模型卡 README.md + weights\（硬链接 / 目录联接）
│   ├── BS-RoFormer-L6\
│   ├── BSRNN-opt\  BSRNN-large\  BSRNN-SIMO\
│   ├── Demucs\  MDX-Net\  Open-Unmix\
│   ├── Conv-TasNet\  MMDenseLSTM\  DPRNN\
│   ├── RPCA\  Oracle-IRM\
│   ├── denoiser-dns48\  denoiser-dns64\  denoiser-master64\        ← 板 2
│   ├── Mel-RoFormer-Denoise\  Mel-RoFormer-Dereverb\
│   ├── Mel-RoFormer-Dereverb-Echo\
│   ├── _third_party\       13 个官方 / 社区仓库（共享层，743 MB）
│   ├── _weights\           ★ 权重的唯一物理落点（15 GB）
│   ├── _selfimpl\          自研算法索引（RPCA / NMF / ICA / HPSS / Oracle 掩码）
│   └── MODEL_REGISTRY.md   12 个指定模型在位状态 / 可运行性 / 精度总表
│
├── 02_databases\       ② 数据（只读）
│   ├── MUSDB18-HQ\         train(100) / test(50) / _smoketest
│   ├── DNS-Challenge\      clean / noise / dev_testset / impulse_responses(60248 条 RIR)
│   └── Valentini\          板 2 降噪评测集（824 noisy + 824 clean 配对，48 kHz 单声道）
│
├── 03_outputs\         ③ 产物 —— 每个模型一个独立文件夹
│   └── <模型名>\
│       ├── test\            MUSDB18 test 50 首整曲分离结果（每首一个子目录）
│       └── stage0_gate\     阶段 0 门禁单文件输出（仅板 2 模型有）
│
├── 04_reports\         ④ 报告 —— **按板块分文件夹**
│   ├── separation\         板 1 · 分离（阶段 1-4）
│   │   ├── docs\  figures\  data\  html\
│   ├── denoiser\           板 2 · 降噪 + 去混响 / 去回声（阶段 5-7）
│   ├── cascade\            级联实验（分离 → 降噪 端到端）
│   ├── _shared\            跨板块共用（总纲、模型画像、14 张架构图、阶段 0 门禁数据）
│   └── slides\             答辩材料
│
└── tools\              ⑤ 工具
    ├── *.py                69 个脚本（含唯一入口 paths.py）
    ├── _bat\               Task Scheduler 启动器（长任务防会话硬杀）
    ├── _logs\              运行日志（120 项）
    └── _scratch\           临时与缓存（.cache = TORCH_HOME / HF_HOME）
```

### 模块职责速查

| 模块 | 放什么 | 不放什么 |
|---|---|---|
| `01_models` | 模型代码、权重、模型卡、注册表 | 运行产物、报告 |
| `02_databases` | 原始数据集（只读） | 分离结果 |
| `03_outputs` | **每个模型自己的 stem wav** | 跨模型汇总、图表 |
| `04_reports` | 报告文档、**全部图表**、**全部数据 json/csv** | 大体积 wav |
| `tools` | 脚本、日志、缓存、批处理启动器 | 任何交付物 |

---

## ② 模型文件夹里的 `weights/` 是什么

**不是拷贝，是链接。** 为了让「一个模型一个文件夹」成立，同时不把 15 GB 权重复制一遍：

| 权重形态 | 链接方式 | 特性 |
|---|---|---|
| 目录（如 `BSRNN/bsrnn-large/`） | NTFS **目录联接**（`mklink /J`） | 秒级、零拷贝、无需管理员权限 |
| 单个文件（如 `model_bs_roformer_ep_317_*.ckpt`） | NTFS **硬链接**（`mklink /H`） | 同一 inode，`-rw-r--r-- 2`；删任一侧都不丢数据 |

因此：**物理落点始终唯一（`01_models/_weights/`），脚本不需要改任何权重路径。**
改动目录时只改 `tools/paths.py`。

---

## ③ 报告按板块归档

| 文件夹 | 归属 | 覆盖的计划阶段 |
|---|---|---|
| `04_reports/separation/` | 板 1 · 分离 | 阶段 1-4：模型画像、14 张架构原理图、频谱面板、混响鲁棒性 |
| `04_reports/denoiser/` | 板 2 · 降噪 | 阶段 5-7：降噪双口径评测（PESQ-WB/STOI/SI-SDR）、去混响、去回声 |
| `04_reports/cascade/` | 级联 | 阶段 6-7：分离输出 → 降噪输入 |
| `04_reports/_shared/` | 跨板块 | 总纲（计划 / 路线图 / 复盘）、模型画像表、架构图数据、阶段 0 门禁 |

每个板块下固定四类子目录：`docs/`（md 报告）、`figures/`（png）、`data/`（json/csv）、`html/`（自包含看板）。

---

## ④ 路径中枢（唯一真理源）

**所有脚本不再自己拼路径**，统一从 `tools/paths.py` 取：

```python
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()                       # 自动设 TORCH_HOME / HF_HOME / MPLCONFIGDIR

_paths.THIRD_PARTY, _paths.WEIGHTS, _paths.DEMUCS_SRC
_paths.MUSDB18_ROOT, _paths.DNS_ROOT, _paths.VALENTINI_ROOT
_paths.OUTPUTS
_paths.REPORTS, _paths.SEPARATION, _paths.DENOISER, _paths.CASCADE, _paths.SHARED
_paths.board_dir("denoiser")             # 取板块目录
_paths.board_sub("separation", "figures")
_paths.model_dir("BS-RoFormer-L12")      # 01_models/BS-RoFormer-L12
_paths.model_weights("BS-RoFormer-L12")  # 该模型文件夹内的 weights/
_paths.out_dir("BS-RoFormer-L12")        # 03_outputs/BS-RoFormer-L12
_paths.out_test("BS-RoFormer-L12")       # 03_outputs/BS-RoFormer-L12/test
_paths.LOGS, _paths.SCRATCH, _paths.BAT_DIR
```

**旧名兼容**：`DOCS / HTML / FIGURES / DATA` 四个常量现在一律指向 **板 1（separation）**
—— 既有脚本的产物本来就全属板 1，所以 62 个老脚本无需改动即可继续运行。
板 2 用 `DEN_DOCS / DEN_FIGURES / DEN_DATA / DEN_HTML`，共用资产用 `SHR_*`。

以后目录再调整，**只改 `tools/paths.py` 一个文件**。

---

## ⑤ 目录变更史

### 2026-09-23（第二次重组）

| 变更 | 从 | 到 |
|---|---|---|
| 杂项模块撤销 | `05_misc/logs/` | `tools/_logs/` |
| | `05_misc/scratch/` | `tools/_scratch/` |
| | `05_misc/*.bat` | `tools/_bat/` |
| 报告改为按板块 | `04_reports/{docs,figures,data,html}/` | `04_reports/{separation,denoiser,cascade,_shared}/{docs,figures,data,html}/` |
| 产物按模型独立 | `03_outputs/_test_run/<模型>/` | `03_outputs/<模型>/test/` |
| | `03_outputs/_stage0_gate/<模型>/` | `03_outputs/<模型>/stage0_gate/` |
| 模型按模型独立 | `01_models/_weights/…`（按仓库分组） | 新增 `01_models/<模型>/`（模型卡 + 权重链接），`_weights/` 保留为物理落点 |
| **移出项目** | `05_misc/lyrics-game/`（501 MB） | `E:\FYP_HKBU_evicted\lyrics-game\` |
| | `05_misc/_archive/`（74 MB） | `E:\FYP_HKBU_evicted\_archive_2026-09-23\` |
| | `05_misc/scratch/.tmp_dl/`（279 MB） | `E:\FYP_HKBU_evicted\scratch_tmp_dl_2026-09-23\` |
| | `05_misc/scratch` 下 63 个探针脚本 | `E:\FYP_HKBU_evicted\scratch_probes_2026-09-23\` |

> 移出 ≠ 删除。全部为同盘 `mv`，逆向清单见
> `04_reports/_shared/docs/RESTRUCTURE_2026-09-23.md`
> 与机器可读的 `04_reports/_shared/docs/_restructure_manifest.json`。

### 2026-09-16（第一次重组）

| 旧位置 | 新位置 |
|---|---|
| `outputs/comparison/*.json\|csv` | `04_reports/separation/data/comparison/` |
| `outputs/comparison/*.png`、`figs_time/` | `04_reports/separation/figures/` |
| `Output-demucs-main/figures/` | `04_reports/separation/figures/demucs/` |
| `reports/*.md` | `04_reports/separation/docs/` |
| `reports/*.html` | `04_reports/separation/html/` |
| `datasets/` | `02_databases/` |

### 2026-09-23 事故记录（磁盘）

根目录 `.git` 曾积累 **76,427 个不可达松散对象 = 57 GB**，成因是「`git add .` 之后从未 commit」，
把 47 GB 数据集与 21 GB 推理产物全塞进对象库，导致 E 盘可用空间归零、所有脚本写入失败（ENOSPC）。
`git prune --expire=now` 已回收全部 57 GB，`.gitignore` 已补防复发规则。

---

## ⑥ 常用命令

```bash
PY="C:/Users/jerry/.workbuddy/binaries/python/envs/fyp_audio/Scripts/python.exe"

# 路径自检（先跑这个，确认结构没坏）
$PY tools/paths.py

# 项目全量自检（模块 / 模型 / 数据集 / 产物）
$PY tools/selfcheck_project.py

# 模型本体画像 → 04_reports/_shared/data/model_analysis/model_profile.csv
$PY tools/_measure_model_profile.py

# 抓真实层级树（forward hook）→ graph_*.json，再渲染 14 张架构图
$PY tools/_extract_model_graph.py --models all
$PY tools/_plot_model_arch.py

# 整曲分离（单模型单曲） / 全模型扫描
$PY tools/bench_mark_model_universal.py --help
$PY tools/bench_musdb_test_songmajor.py --models all --songs all

# 人耳聆听页（双击也行）
start_listen.bat          # → http://127.0.0.1:8123/04_reports/separation/html/listen_compare.html
```

> ⚠️ 解释器**只能用 venv**（`fyp_audio`）。WorkBuddy 默认的 `python` 指向系统 3.13.12，没有 numpy，
> 子进程会秒退 `ModuleNotFoundError`，外层看起来像「推理失败」，实则根本没跑起来。

> ⚠️ 长任务（数小时~数天）**必须走 `schtasks`**，不能靠后台任务：Agent 会话结束会 `TerminateProcess`
> 硬杀子进程，`finally` 不执行、锁残留。启动器模板见 `tools/_bat/_sweep_launch.bat`。
