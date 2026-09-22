# 目录重构报告（2026-09-23）

> 范围：`E:\FYP_HKBU` ｜ 执行方式：**只移动 / 新建，未删除任何文件** ｜ 可逆
> 机器可读清单：`04_reports/_shared/docs/_restructure_manifest.json`（289 条操作）

---

## 1. 为什么要做

第一次重组（2026-09-16）解决了「文件乱放」，但留了三个问题：

| 问题 | 具体表现 |
|---|---|
| `05_misc` 是个筐 | 日志、缓存、批处理、**无关子项目（歌词猜歌游戏 501 MB）** 全塞在一起，无法一眼看出项目边界 |
| 报告按**文件类型**分 | `docs/` `figures/` `data/` `html/` 下混着板 1 和板 2 的东西，看某个板块要四头找 |
| 产物按**运行批次**分 | `03_outputs/_test_run/<模型>/`，模型自己的输出不是独立文件夹 |

本次目标（用户明确要求）：**顶层只留模型 / 数据 / output / report / 工具五项**；
模型下每个模型一个独立文件夹；output 下每个模型一个独立文件夹；report 下按板块分。

---

## 2. 重构后的顶层

```
E:\FYP_HKBU\
├── 01_models\      ① 模型（19 个模型文件夹 + 3 个共享层）
├── 02_databases\   ② 数据（MUSDB18-HQ / DNS-Challenge / Valentini）
├── 03_outputs\     ③ 产物（19 个模型文件夹）
├── 04_reports\     ④ 报告（separation / denoiser / cascade / _shared / slides）
├── tools\          ⑤ 工具（69 脚本 + _logs + _scratch + _bat）
├── README.md
├── .gitignore
└── .git
```

**顶层从 6 项（`01..05_misc` + `tools`）收敛为 5 项。**

---

## 3. 逐项变更对照

### 3.1 杂项模块撤销：`05_misc/` → 并入 `tools/`

| 原路径 | 新路径 | 说明 |
|---|---|---|
| `05_misc/logs/` | `tools/_logs/` | 120 项运行日志 |
| `05_misc/scratch/.cache/` | `tools/_scratch/.cache/` | **TORCH_HOME / HF_HOME**，含 Demucs、UMX 权重，不能丢 |
| `05_misc/scratch/.tmp_pesq/` | `tools/_scratch/.tmp_pesq/` | SAC 拦截期手工解包 pesq 的产物 |
| `05_misc/_sweep_launch.bat` 等 3 个 | `tools/_bat/` | Task Scheduler 启动器（长任务防会话硬杀） |
| `05_misc/README.md` | `tools/_logs/_FROM_05_misc_README_ORIGINAL.md` | 原文留档 |

### 3.2 报告改为按板块

| 原路径 | 新路径 |
|---|---|
| `04_reports/docs/DEMUCS_REPORT.md` 等 9 份 | `04_reports/separation/docs/` |
| `04_reports/docs/FYP_PLAN / ROADMAP / RETROSPECTIVE / MODEL_PROFILE / Audio_Setup_Guide` | `04_reports/_shared/docs/` |
| `04_reports/figures/{comparison,demucs,test_run,figs_time,audio_analysis,spectrograms,dprnn,reverb}/` | `04_reports/separation/figures/` |
| `04_reports/figures/{model_arch,_smoketest}/` | `04_reports/_shared/figures/` |
| `04_reports/figures/{denoise}/` | `04_reports/denoiser/figures/` |
| `04_reports/figures/cascade/` | `04_reports/cascade/figures/` |
| `04_reports/data/{comparison,reverb}/`、`dprnn_training*.json` | `04_reports/separation/data/` |
| `04_reports/data/model_analysis/` | `04_reports/_shared/data/model_analysis/` |
| `04_reports/data/denoise/` | `04_reports/denoiser/data/` |
| `04_reports/data/cascade/` | `04_reports/cascade/data/` |
| `04_reports/html/` 4 份现行页面 | `04_reports/separation/html/` |
| `04_reports/html/METHOD_VISUAL_COMPARISON_1.0/1.2.html` | `04_reports/_shared/html/_superseded/`（被 1.3 取代） |

### 3.3 产物改为按模型独立

| 原路径 | 新路径 |
|---|---|
| `03_outputs/_test_run/<模型>/<歌曲>/` | `03_outputs/<模型>/test/<歌曲>/` |
| `03_outputs/_stage0_gate/<模型 key>/` | `03_outputs/<模型>/stage0_gate/<模型 key>/` |

### 3.4 模型改为按模型独立（新增，不动原权重）

`01_models/` 下新增 **19 个模型文件夹**，每个含 `README.md` 模型卡 + `weights/`。
`weights/` 是**链接不是拷贝**：

| 权重形态 | 链接方式 | 证据 |
|---|---|---|
| 目录 | NTFS 目录联接 `mklink /J` | 无需管理员权限，读通即有效 |
| 单文件 | NTFS 硬链接 `mklink /H` | `ls -l` 显示链接计数 `2` |

`01_models/_weights/`（15 GB）仍是**唯一物理落点**，`_third_party/`（743 MB）保持共享
——同一仓库（如 `Music-Source-Separation-Training`）服务 BS-RoFormer-L12 / L6 与 3 个
Mel-RoFormer，拆开复制会造成 3 份冗余且破坏加载链路。

### 3.5 移出项目（未删除）

| 原路径 | 新位置 | 体积 | 文件数 |
|---|---|---|---|
| `05_misc/lyrics-game/` | `E:\FYP_HKBU_evicted\lyrics-game\` | 501 MB | 116 |
| `05_misc/_archive/` | `E:\FYP_HKBU_evicted\_archive_2026-09-23\` | 74 MB | 107 |
| `05_misc/scratch/.tmp_dl/` | `E:\FYP_HKBU_evicted\scratch_tmp_dl_2026-09-23\` | 279 MB | 4 |
| `05_misc/scratch/` 下 63 个探针脚本 / 调试图 / `.bak` | `E:\FYP_HKBU_evicted\scratch_probes_2026-09-23\` | 751 KB | 62 |
| **合计** | | **855 MB** | **289** |

`lyrics-game` 是**完全独立的另一个项目**（50 首粤语 mp3 + 繁体 lrc + 歌词猜歌网页），
与本项目的音乐源分离主线**零代码耦合**，此前只是借住在 `05_misc/` 下。

---

## 4. 校验结果

| 校验项 | 结果 |
|---|---|
| 项目内文件数 | 重组前 **85,481** → 重组后 **85,245**，移出 **289** → 85,245 + 289 = **85,534** |
| 文件数差额 +53 | 全部是**本次新建**：19 张模型卡 + 3 份 README + 13 个硬链接 + `__pycache__` 重生成，**无文件丢失** |
| 体积 | 项目内 **83 GB** + 移出区 **855 MB** ≈ 重组前 **84 GB**，**零体积损失** |
| 整曲分离音频 | `03_outputs/*/test/` 下 **1,665 个 flac**，`test_sweep_ledger.json` 记 **563 PASS / 572 次尝试**，与重组前一致 |
| 权重链接 | 23 条全部创建成功，抽样 4 条读通字节数正确 |
| 脚本可编译 | `tools/*.py` **69 个全部编译通过** |
| `paths.py` 自检 | 5 大模块 + 4 板块 + `_logs/_scratch/_bat` + 缓存全部命中 |
| `selfcheck_project.py` | 5 大模块状态 ✅，数据集 wav 数不变（MUSDB 750 / DNS 77,210 / Valentini 1,648） |
| 生成链路回归 | `_write_model_profile_md.py`（22 行表）、`_plot_model_arch.py`（14 张图）、`_check_model_arch.py`（0 项待处理）全部在新路径下重跑成功 |

### 扫描账本口径（`test_sweep_ledger.json`，2026-09-22 10:03 更新）

| 模型 | PASS | FAIL | TIMEOUT | 已停用 |
|---|---:|---:|---:|---|
| oracle / demucs / umx / mdx / convtasnet / mmdenselstm | 各 50 | 0 | 0 | — |
| bsroformer_l12 / bsroformer_l6 | 各 50 | 0 | 0 | — |
| bsrnn / bsrnn_large | 各 50 | 0 | 0 | — |
| dprnn | 49 | 1 | 0 | — |
| bsrnn_simo | 14 | 0 | 3 | ✔ 停用 |
| bsrnn_all / bsrnn_large_all | 0 | 0 | 2 / 1 | ✔ 停用 |
| rpca | 0 | 0 | 2 | ✔ 停用 |
| **合计** | **563** | **1** | **8** | 4 个 |

---

## 5. 代码侧改动

| 文件 | 改动 |
|---|---|
| `tools/paths.py` | **重写**。撤销 `MISC`；新增 `SEPARATION/DENOISER/CASCADE/SHARED` 与 `board_dir()/board_sub()`、`model_dir()/model_weights()/out_dir()/out_test()/out_gate()`。旧名 `DOCS/HTML/FIGURES/DATA` **保留但指向板 1**，使 62 个既有脚本零改动继续可跑 |
| `tools/selfcheck_project.py` | 模块清单改为新结构 |
| 24 个脚本 | 硬编码的 `04_reports/...` 路径重定向（共 78 处） |
| 3 个 `.bat` + `start_listen.bat` | 路径更新 |
| `README.md` / `.gitignore` | 重写 |
| `01_models/MODEL_REGISTRY.md` | 产物路径对齐 |

**新增脚本**：`tools/_restructure_2026-09-23.py`（本次重构的执行器，带 `--step` 与 `--dry-run`，
每一步写 manifest，可完整重放或逆向）。

---

## 6. 回滚方法

```bash
# 逐条逆向（manifest 里 kind=move 的条目，把 dst 移回 src）
PY="C:/Users/jerry/.workbuddy/binaries/python/envs/fyp_audio/Scripts/python.exe"
$PY - <<'EOF'
import json, os, pathlib
m = json.load(open("04_reports/_shared/docs/_restructure_manifest.json", encoding="utf-8"))
for op in reversed(m["ops"]):
    if op["kind"] == "move" and pathlib.Path(op["dst"]).exists() and not pathlib.Path(op["src"]).exists():
        pathlib.Path(op["src"]).parent.mkdir(parents=True, exist_ok=True)
        os.rename(op["dst"], op["src"])
        print("restored", op["src"])
EOF
```

- **硬链接**：直接删掉 `01_models/<模型>/weights/` 即可，`_weights/` 里的原文件不受影响。
- **目录联接**：`rmdir` 掉 `weights/` 下的联接目录即可，不会删到目标内容。
- **移出区**：`E:\FYP_HKBU_evicted\` 里四个目录整体 `mv` 回原位即可。

---

## 7. 遗留项（需你确认后处理）

| # | 事项 | 体积 | 建议 |
|---|---|---:|---|
| 1 | `01_models/_weights/BSRNN/` 下 3 个 `.zip` 与同名解压目录并存 | **4.6 GB** | 解压目录已在位且加载正常，zip 属冗余。**需你确认后**再删（属删除操作，本次未动） |
| 2 | `03_outputs/RPCA/` 有 2 个歌曲目录但 **0 个 flac** | — | 与账本一致（RPCA 单曲超时被停用），属正常截断，非本次重构造成 |
| 3 | `03_outputs/BSRNN-SIMO/` **14/50** | — | 已决策不补跑，报告中须标注截断 |
| 4 | `tools/build_html_v13.py` 读取 `METHOD_VISUAL_COMPARISON_1.2.html` | — | 该文件已在 `_shared/html/_superseded/`，此脚本是一次性生成器，若要重跑需改一行 SRC |
| 5 | `04_reports/**/*.md` 内部引用的旧路径 | — | 历史报告保留原样（记录当时口径）；新写的文档一律用新路径 |
