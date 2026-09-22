# FYP_HKBU 项目全面审查报告

> 审查时间：**2026-09-19 10:56**（GMT+8）
> 审查范围：`E:\FYP_HKBU\` 全部五大模块 + `tools\`
> 审查方式：目录实测 + 关键文件内容核对 + 数据一致性交叉验证
> **执行更新：2026-09-19 11:25 —— P0 两项 + P1 两项已在本轮完成，见 §6；剩余待办 8 项**
>
> 📌 **阅读顺序建议**：§0 结论 → §6 本轮执行结果 → §3 任务清单 → 其余为审查时的基线快照

---

## 0. 一句话结论

项目结构与文档体系**健康、自洽、可交付**；审查时发现的 **1 个状态脱节 + 1 个数据 bug** 已在同轮修复。

> ✅ **DPRNN 收尾链已于 2026-09-19 11:17 执行完毕**（6 步全通过）：
> 四轨均值 SDR **−4.54 → −1.72 dB**（+2.82，提升 62%），所有报告/图表/HTML 已同步。
> ✅ MDX-Net 鼓轨异常已复测**结案**。✅ 训练历史元数据矛盾已修复。

**当前待办 8 项**（P0 已完成）：P1 × 1（#5 跑 MUSDB18 test 集）、P2 × 5（#6–#10）、P3 × 2（#11–#12）。
唯一影响**论文可发表性**的是 **#5**；唯一遗留的**可信度缺口**是 `training_curve.png` 未重绘（#8）。详见 §3。

> ⚠️ 本次工作暴露一个**流程教训**：收尾链写完 ≠ 收尾链跑过。`_post_dprnn.txt` 里躺着 4 次 dry-run
> 却从未真跑，导致所有报告带旧数据 30+ 小时。**今后凡「等训练结束自动触发」的链路，
> 必须在其后安排一次人工核对**——本环境 `automation_update` 不可用（`connector-proxy: fetch failed`），
> 定时任务这条路走不通，只能靠主动复查。

---

## 6. 本轮执行结果（2026-09-19 11:17–11:25）

### 6.1 执行了「第一件事」：DPRNN 收尾链

```bash
$PY tools/post_dprnn_chain.py --verify-mdx
```

**先修补了收尾链的判据缺陷**（否则永远等不到）：
训练是被**外部中断**的，日志里**没有 `训练结束:` 行** → 原「只认日志标记」的判据永不满足。
已改为**双路判据**：正常路径（日志标记 + ckpt 稳定 + 进程已退）**或**
兜底路径（无标记，但进程已退且 ckpt 静止 ≥270 s）。
→ 实测 12 秒内即正确判定「疑似外部中断」，自动进入收尾。

**6 步全部通过（耗时 3.5 min）**

| 步 | 结果 |
|---|---|
| ① 重跑 DPRNN 三片段 | ✅ 四轨均值 **−4.54 → −1.72 dB** |
| ② 聚合中位数 + 刷新 `MEDIAN_TABLE.md` | ✅ |
| ③ 重绘 3 张对比图 | ✅ |
| ④ 重建 `METHOD_VISUAL_COMPARISON_1.3.html` | ✅ |
| ⑤ 复测 MDX-Net 鼓轨 | ✅ 得 **6.97**（旧 GPU 值 6.99）→ 结案 |
| ⑥ 再聚合 | ✅ |

**DPRNN 新旧对照（3 均衡片段中位数 museval SDR）**

| | vocals | drums | bass | other | **均值** |
|---|---|---|---|---|---|
| 旧（step 3045） | −3.83 | −3.57 | −4.35 | −6.40 | **−4.54** |
| **新（step 31900）** | −2.18 | −1.86 | −0.82 | −2.02 | **−1.72** |
| 提升 | +1.65 | +1.71 | **+3.53** | **+4.38** | **+2.82** |

> ⚠️ 四轨**仍全为负** → 排名结论不变（DPRNN 仍不参与精度排名，仅作负面证据）。
> 但 SI-SDR 侧 vocals 3.34 / drums 4.78 / bass 5.05 已明显转正，说明确实学到了东西。

### 6.2 结案：MDX-Net 鼓轨异常

GPU 复跑得 **6.97**，与旧 GPU 值 **6.99** 一致 → **是真实的设备差异，不是随机抖动**。
但三片段中位数口径下鼓轨为 **8.65**，该异常**不影响任何对外结论**。
已在报告 §1.6 写明引用口径。

### 6.3 修复：训练历史元数据矛盾

`--resume` 原只合并 `steps`/`vals`，导致 `batch` / `finished` / `final_step` / `best_val_mean`
停在**上一轮**。已做两件事：
1. **修脚本** `train_dprnn_musdb.py`：续跑时刷新全部超参与收尾字段（并加 `resumed_from_step`）；
2. **修历史文件**：备份后按数组真值回填 —— `batch` 4→**8**、`final_step` 3404→**31900**、
   `best_val_mean` 1.3661→**2.7449**、`finished` 11:07→**15:19:50**。
   备份：`dprnn_training.bak-before-metafix.json`

### 6.4 同步的文档

| 文件 | 改动 |
|---|---|
| `MODEL_REGISTRY.md` | §1 DPRNN 行改 −1.72 / RTF 0.03；§3 决策行改「已完成」；§6.3 重写（含新旧对照表）；§6.5 判据说明改为双路 + 执行记录 |
| `MODEL_PROVISIONING_REPORT.md` | §0 加复训进展；§1.6 主表 DPRNN 行 + MDX 结案说明 + DPRNN 注释重写；§3.4c 新增完整训练表；§7 未完成项 5 行状态更新 |
| `MEDIAN_TABLE.md` / 3 张对比图 / HTML | 由收尾链自动刷新 |

### 6.5 备份清单

| 备份 | 大小 |
|---|---|
| `model_runs_pre-dprnn-20260919-111730.json` | 89,503 B |
| `dprnn_training.bak-before-metafix.json` | 74,255 B |
| `model_runs_median.bak-before-offsetguard.json` | 8,316 B |

> 均位于 `04_reports\data\comparison\` 或 `05_misc\logs\`（按原文件所在位置）,可用于回滚。


---

## 1. 文件结构（实测）

```
E:\FYP_HKBU\                                     总计约 685 MB（不含数据集与权重）
│
├── README.md                         6.4 KB   ★ 项目总入口（五大模块 + 路径映射表 + 常用命令）
├── tools\                            48 个脚本（含 2 个 .ps1），不属于任何模块
│
├── 01_models\                        11.8 GB / 1682 文件
│   ├── MODEL_REGISTRY.md            20.5 KB  ★ 12 模型状态 / 可运行性 / 精度总表
│   ├── README.md                     2.3 KB
│   ├── _third_party\                 12 个官方仓库（原样克隆）
│   ├── _weights\                     11.2 GB
│   └── _selfimpl\                    （仅 README，自研实现实际在 tools\compare_separation_methods.py）
│
├── 02_databases\                     47 GB / 只读
│   ├── MUSDB18-HQ\   train 19G / test 11G / _smoketest 5.1M
│   ├── DNS-Challenge\ clean 3.4G / noise 2.1G / dev_testset 1.7G /
│   │                  impulse_responses 5.9G / _download 3.9G
│   └── README.md
│
├── 03_outputs\                       19 个模型目录 / 约 500 MB / 236 个 wav
│   └── 每模型一目录 = <模型>\<歌曲>\{vocals,drums,bass,other}.wav
│
├── 04_reports\                       14.2 MB / 69 文件  ★ 所有面向人的产出
│   ├── docs\     6 份 .md
│   ├── html\     4 份自包含 HTML
│   ├── figures\  6 组图表 / 42 张 png
│   ├── data\     comparison\ 16 个 json/csv + dprnn_training.json
│   └── slides\   （空）
│
└── 05_misc\                          851.9 MB / 314 文件
    ├── logs\        75 个日志
    ├── _archive\    3 份历史归档
    ├── scratch\     缓存 + 探针脚本
    └── lyrics-game\ 独立子项目（50 首 mp3 + 歌词）
```

---

## 2. 各模块内容明细

### 01_models — 模型代码与权重

**`MODEL_REGISTRY.md`（20.5 KB，★ 核心）**
- §0 一句话结论：12 模型中 10 可运行 / 2 受阻
- §1 总表：12 个模型的类别 / 代码在位 / 权重在位 / 可运行 / RTF(CPU) / RTF(GPU) / 四轨均值 SDR
- §2 可运行清单（10 个）+ 受阻清单（2 个）
- §3 决策记录（5 项决策全部执行）
- §4 多输出模型 state_dict 按位置对齐陷阱
- §5 CUDA 带来的实时性改写（CPU→GPU RTF 对照表）
- §6 DPRNN 自训完整说明（含 3 个致命 bug + 睡眠事故 + 4 个坑 + 限制 + **§6.5 收尾链**）

**`_third_party\`（12 个仓库）**
`BS-RoFormer`、`Conv-TasNet`、`DNN-based_source_separation`、`Dual-Path-RNN-Pytorch`、
`Music-Source-Separation-Training`、`bsrnn`、`demucs`、`mdx-net`、
`mdx-net-submission`、`mdx-net-submission-leaderboard_A`、`mdx-net-submission-leaderboard_B`、`sigsep-mus-oracle`

**`_weights\`（11.2 GB，实测）**

| 目录 | 大小 | 内容 |
|---|---|---|
| `BS-RoFormer\` | 985 MB | `ep_317`（609.7 MB，L12，vocals 单目标）、`ep_937`（374.9 MB，L6）+ 2 个 yaml |
| `BSRNN\` | 9.0 GB | `bsrnn-opt.zip` 1.7G / `bsrnn-large.zip` 1.5G / `simo-bsrnn-opt.zip` 1.1G + 已解包 3 套 ckpt（opt 4 个 stem ckpt + large 4 个 + SIMO separator） |
| `DNN-based_source_separation\` | 302 MB | ConvTasNet ×6（4sec_L20 / 8sec_L20 / 8sec_L64 各 best+last）、MMDenseLSTM ×8（4 stem 各 best+last） |
| `MDX-Net\` | 744 MB | `mdx_extra\` 4 个 .th（各 159.6 MB）+ `onnx_A.zip` 105 MB + `mixer.ckpt` |
| `dprnn_musdb\` | 56 MB | **本机自训**：`best.pt` 14.1 MB + `last.pt` 42.3 MB |

**`_selfimpl\README.md`（1.3 KB）** — ⚠️ 仅说明文档，无实际代码；RPCA/NMF/ICA/HPSS/oracle 的自研实现位于
`tools\compare_separation_methods.py`（25.8 KB）。属结构不一致，见 §4 问题清单。

### 02_databases — 数据集（只读，47 GB）

| 数据集 | 子目录 | 内容 |
|---|---|---|
| **MUSDB18-HQ** | `train\` 19 GB | 100 首 × 4 stem（44.1 kHz 立体声 HQ）|
| | `test\` 11 GB | 50 首 × 4 stem |
| | `_smoketest\` 5.1 MB | 3 首合成数据，用于快速冒烟测试 |
| **DNS-Challenge** | `clean\` 3.4 GB | VocalSet 独唱 + 情感语音 |
| | `noise\` 2.1 GB | Freesound 环境噪声 |
| | `dev_testset\` 1.7 GB | 官方带噪测试集（含 reference，可算指标）|
| | `impulse_responses\` 5.9 GB | 房间冲激响应（加混响）|
| | `_download\` 3.9 GB | 原始 .tar.bz2 压缩包与下载日志 |

`README.md`（6.2 KB）记录来源 URL、分片清单与扩量方法。

### 03_outputs — 每个模型的 stem wav（实测 wav 数）

| 目录 | wav | 大小 | 说明 |
|---|---|---|---|
| `Open-Unmix\` | 36 | 77.4 MB | 含 baseline 对比 |
| `Demucs\` | 20 | 47.1 MB | htdemucs 4-stem |
| `Baseline\` | 20 | 50.5 MB | 含 `metrics.csv` |
| `HPSS\` / `NMF\` | 各 20 | 各 50.5 MB | 各含 `metrics.csv` |
| `BSRNN-opt\` / `BSRNN-large\` / `BSRNN-SIMO\` | 各 16 | 各 26.9 MB | BSRNN 三变体 |
| `Conv-TasNet\` / `MMDenseLSTM\` / `MDX-Net\` / `Oracle-IRM\` | 各 16 | 各 26.9 MB | |
| `RPCA\` | 14 | 32.0 MB | 含 `metrics.csv` |
| `ICA\` | 10 | 25.2 MB | 含 `metrics.csv` |
| `DPRNN\` | 12 | 20.2 MB | ⚠️ **旧 ckpt 产物**，待重跑覆盖 |
| `BS-RoFormer-L12\` / `BS-RoFormer-L6\` | 各 4 | 各 6.7 MB | 单目标，文件名即语义（`vocals.wav` / `vocals+other.wav`）|
| `RPCA-DRNN\` / `Pac-HuBERT-SEP\` | 0 | — | 受阻占位，各含 `README.md` 说明原因 |

命名规范：`03_outputs\<模型名>\<歌曲名>\{vocals,drums,bass,other}.wav`

### 04_reports — 报告（★ 所有面向人的产出）

**`docs\`（6 份）**

| 文件 | 大小 | 内容 |
|---|---|---|
| `MODEL_PROVISIONING_REPORT.md` | 47.5 KB | ★ **主报告**：§0 结论 / §1 核查总表 / §1.5 评测口径 / §1.6 主结果表 / §2 SAC 问题 / §3 逐模型详解 / §4 工程改造 / §5 对「实时」目标的影响 / §6 产物清单 / §7 未完成项 |
| `METHOD_COMPARISON_REPORT.md` | 27.3 KB | 方法百科 + 技术演进与市场分层 + 文献数据 + 同数据实测 + 实时性 + 实测vs文献交叉验证 + 对 FYP 启示 |
| `CODE_REVIEW_REPORT.md` | 17.6 KB | 代码审阅：结论速览 / 审阅方法 / 逐份代码简介 / 问题（按严重度）/ 运行验证 / 每份代码的意义 / 建议行动 |
| `FYP_Audio_Setup_Guide.md` | 11.8 KB | 环境配置执行手册（两条路线 / 目录 / 数据集格式 / 运行 / 常见报错）|
| `DEMUCS_REPORT.md` | 7.8 KB | Demucs 纵向深入：分离质量 / 复杂度 Params-MACs-FLOPs / 实时性 / PESQ / 产出清单 |
| `MEDIAN_TABLE.md` | 1.9 KB | 多片段中位数表（SDR + SI-SDR，供直接引用）|

**`html\`（4 份自包含）**
`METHOD_VISUAL_COMPARISON_1.3.html`（1.28 MB，**最新**）、`1.2`、`1.0`、`RUNTIME_BENCH_PPT.html`（610 KB）

**`figures\`（42 张 png）**
- `comparison\`（4 张）：`fig_median_sdr_bars`、`fig_median_per_stem`、`fig_median_pareto`、`fig_compare_dsdr`
- `figs_time\`（7 张）：时长基准 / 累计 / RTF / 分布 / 时长vs精度 / 对比面板 / 速度-精度 pareto
- `demucs\`（9 张）、`audio_analysis\`（9 张）、`_smoketest\`（9 张）：波形 / 频谱 / MFCC / 裁剪对比
- `dprnn\training_curve.png`（1 张）

**`data\`**
- `comparison\`（16 个）：`model_runs.json` 87.4 KB（**全量原始记录**）、`model_runs_median.json` 8.3 KB（**聚合结果**）、`model_runs_cpu_2026-09-16.json`（CPU 基线）、`model_runs_median.bak-before-offsetguard.json`（守卫前备份）、`weight_loadability.json`、`dependency_probe.json`、`model_provisioning.json`、`comparison_results.json`、`demucs_metrics.json`、`bench_10songs.json` 等
- `dprnn_training.json` 72.3 KB — 训练历史（steps 217 条 + vals 297 条 + 超参）

**`slides\`** — ⚠️ **空目录**（`make_bench_ppt.py` 已存在，但 PPT 未产出）

### 05_misc — 杂项

| 子目录 | 内容 |
|---|---|
| `logs\`（75 个） | 各模型每轮完整运行日志（09-16 CPU 轮 / 09-17 GPU 轮）、`train_dprnn_musdb.txt` 65.3 KB（训练全程）、`_post_dprnn.txt`（收尾链）、`_multisong.txt` 86.9 KB、`_dprnn_train2_driver.txt` 50.4 KB |
| `_archive\` | `tools_backup_20260916\`（重组前全量备份）、`_smoke_backup\`、`comparison_2026-09-14_prev_run\`、`env_snapshot_pre_cuda_20260916.txt` |
| `scratch\` | `.cache\`（TORCH_HOME/HF_HOME）、`.tmp_dl\`、`.tmp_pesq\`、探针脚本（`_dprnn_timing_probe.py`、`_probe_proc_*.py`、`_survey_tree.py` 等）|
| `lyrics-game\` | **独立子项目**：50 首 mp3 + 繁体 lrc + `songs.json` + `script.js` + `start_game.bat`；含 `DESIGN_v3.2.md` / `GAP_ANALYSIS_v3.2.md` |

### tools\ — 48 个脚本（按职责分类）

| 类别 | 脚本 |
|---|---|
| **路径中枢** | `paths.py`（唯一真理源）、`_migrate_layout.py`、`_rewire_paths.py` |
| **基准与评测** | `benchmark_model_universal.py`（39.5 KB，统一骨架）、`bench_multisong.py`、`median_over_clips.py`、`benchmark_realtime.py`、`benchmark_models_10songs.py`、`summarize_model_runs.py`、`_scan_balanced_clip.py` |
| **模型加载器** | `_bsrnn_loader.py`、`_dprnn_loader.py`、`_librosa_shim.py` |
| **DPRNN 自训** | `train_dprnn_musdb.py`（22.3 KB）、`post_dprnn_chain.py`（15.9 KB）|
| **自研算法** | `compare_separation_methods.py`（25.8 KB，RPCA/NMF/ICA/HPSS/oracle）|
| **数据集** | `dns_fetch.py`、`dns_synthesize.py`、`par_download.py`、`make_synthetic_musdb.py`、`_verify_testset.py` |
| **报告生成** | `build_html_v13.py`（31.1 KB）、`plot_median_summary.py`、`plot_bench_10songs.py`、`make_bench_ppt.py`、`_embed_images.py` |
| **分析** | `run_audio_analysis.py`、`run_demucs_deepdive.py`、`prepare_listening_testset.py` |
| **核查/探针** | `model_provisioning_inventory.py`、`probe_model_deps.py`、`verify_weights_loadable.py`、`selfcheck_project.py`、`_probe_*.py`、`_diag_*.py` |
| **下载** | `download_all_weights.ps1`、`fetch_bsrnn.ps1`、`dl_dnnbased_weights.py`、`extract_bsrnn.py` |

---

## 3. 任务清单（按优先级）

> **状态图例**：✅ 已完成 ｜ ⏳ 待办 ｜ 🔴 高优先
> 更新于 2026-09-19 11:25（§6 执行后）

### P0 — ✅ 已全部完成（影响论文数据正确性）

| # | 任务 | 状态 | 结果 |
|---|---|---|---|
| **1** | **跑 DPRNN 收尾链** | ✅ **已完成** | 2026-09-19 11:17 执行，6 步全通过，耗时 3.5 min（先修补了判据缺陷，见 §6.1）|
| **2** | **同步 DPRNN 新数字进文档** | ✅ **已完成** | 报告 / MEDIAN_TABLE / HTML / 对比图 / `model_runs_median.json` 全部刷新为 **−1.72 dB** |

~~**为什么是 P0**：当前 `04_reports\` 里所有 DPRNN 数字都是旧 ckpt 的。若不修，论文会引用作废数据。~~
→ **已完成，不再有作废数据**。

---

### P1 — 应当处理（数据/文档可信度）

| # | 任务 | 状态 | 说明 |
|---|---|---|---|
| **3** | 🔴 **修 `dprnn_training.json` 元数据记录 bug** | ✅ **已完成** | 脚本 + 历史文件双修：`batch` 4→**8**、`final_step` 3404→**31900**、`best_val_mean` 1.3661→**2.7449**、`finished` 11:07→**15:19:50**；自洽性校验通过 |
| **4** | **MDX-Net 鼓轨数值异常复测** | ✅ **已结案** | 复测得 **6.97**（旧 GPU 6.99）→ 真实设备差异；中位数口径鼓轨 **8.65**，不影响对外结论 |
| **5** | **跑 MUSDB18 官方 test 集（50 首整曲）** | ⏳ **待办** | 当前只用 train 的 3 个 10 s 片段；要出**可发表成绩**必须跑 test 集 |

---

### P2 — 完善性工作

| # | 任务 | 状态 | 现状 |
|---|---|---|---|
| **6** | **产出 slides** | ⏳ 待办 | `04_reports\slides\` **空目录**；`make_bench_ppt.py`（20.2 KB）已就绪但未运行 |
| **7** | **修 `_selfimpl\` 结构不一致** | ⏳ 待办 | 目录仅 `README.md`，自研实现实际在 `tools\compare_separation_methods.py` |
| **8** | **重绘 `training_curve.png`** | ⏳ 待办 | 图时间戳仍为 **Sep 17 11:07**，未随收尾链更新 → 只反映 step 3045 之前的训练，**不覆盖完整 5 小时 / 31900 步** |
| **9** | **RPCA 性能优化**（报告 §7 标注「建议」）| ⏳ 待办 | `_rpca_ialm` 每轮全秩 SVD ×120 → 改 `randomized_svd` 低秩截断，RTF 有望压到 1 以下 |
| **10** | **清理 docs 冗余** | ⏳ 待办 | `04_reports\html\` 有 v1.0 / v1.2 / v1.3 三版并存；`model_runs_median.bak-*` 备份可归档 |

---

### P3 — 独立子项目（与主线无耦合）

| # | 任务 | 状态 | 现状 |
|---|---|---|---|
| **11** | 🔴 `lyrics-game\script.js::TRADITIONAL_MAP` 是**恒等映射** | ⏳ 待办 | 繁简归一功能失效 |
| **12** | 🔴 `fix_audio_map.py` 依赖未安装的 `opencc`，且**原地覆写** `songs.json` 无备份 | ⏳ 待办 | 运行有数据丢失风险 |

---

### 小结：剩余待办 8 项

| 优先级 | 编号 | 一句话 |
|---|---|---|
| **P1（1 项）** | #5 | 跑 MUSDB18 官方 test 集 —— **唯一影响论文可发表性**的待办 |
| **P2（5 项）** | #6 #7 #8 #9 #10 | slides 产出 / `_selfimpl` 结构 / 训练曲线图重绘 / RPCA 提速 / docs 清冗余 |
| **P3（2 项）** | #11 #12 | lyrics-game 两个 P1 缺陷（与主线无耦合）|

---

## 4. 审查中发现的其他问题

| 问题 | 详情 | 状态 |
|---|---|---|
| **训练日志缺「训练结束」行** | 末次训练（11:08 起，预算 300 min）日志到 **step 31900 / 已用 250.7 min** 就停了，**没有写 `训练结束:` 行** → 收尾链的第一重判据（日志结束标记）永远不满足，这是它「等不到」的直接原因之一。而 `last.pt`(15:19) 与 `best.pt`(15:18) 都正常落盘，说明进程是被**外部中断**（非正常退出）| ✅ 已修：收尾链改**双路判据**，加「进程已退 + ckpt 静止 ≥270 s」兜底路径（见 §6.1）|
| **`dprnn_training.json` 与 ckpt 不一致** | 文件 mtime 是 09-17 15:19（训练期间持续写入），但内容 `finished` 停在 11:07:56 —— 印证了「训练未走到收尾代码」| ✅ 已修（见 §6.3）|
| **`03_outputs\DPRNN\` 是旧产物** | 12 个 wav 来自 step 3045 的 ckpt，重跑后会被覆盖（这是预期行为）| ✅ 已随收尾链覆盖为 step 31900 产物 |

---

## 5. 状态可信度评估

> 下表的「可信度」为**审查时点（09-19 10:56）**的评估；带 ✅ 的行表示已在 §6 执行后恢复。

| 模块 | 可信度 | 说明 |
|---|---|---|
| `01_models` 权重 | ✅ 高 | MD5 全过（BSRNN），逐键 missing=0 |
| `02_databases` | ✅ 高 | 结构与 README 一致 |
| `03_outputs` | ✅ 高 | 除 DPRNN 外均为最新 GPU 轮产物；DPRNN 已随收尾链刷新 |
| `04_reports` 非 DPRNN 部分 | ✅ 高 | 09-17 GPU 重跑后已刷新 |
| **`04_reports` DPRNN 部分** | ~~🔴 失效~~ → ✅ **高** | 已重跑并同步为 **−1.72 dB**（旧值 −4.54）|
| `dprnn_training.json` | ~~🟠 部分失效~~ → ✅ **高** | 元数据已回填，自洽性校验通过 |
| `05_misc` | ✅ 高 | 日志完整可追溯 |

**唯一遗留的可信度缺口**：`04_reports\figures\dprnn\training_curve.png`
（时间戳 Sep 17 11:07，未随收尾链更新 → 只反映 step 3045 之前的训练，不覆盖完整 31900 步；对应待办 #8）
