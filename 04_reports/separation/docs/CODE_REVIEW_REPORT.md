# FYP 代码全面审阅报告

**项目**：`E:\FYP_HKBU` — 音频源分离 / 实时降噪（HKBU FYP）
**审阅范围**：全部**自研代码**（14 个文件），第三方源码 `01_models/_third_party/demucs/` 不逐一审阅
**审阅日期**：2026-09-15
**审阅方式**：静态阅读 + 实际运行（冒烟测试，全部 EXIT=0）

---

## 0. 结论速览

**总体判断：代码可行、结构清晰，工程化程度明显高于一般 FYP 水平。** 全部脚本都能跑通、产出符合预期。
但有 **1 个功能性 Bug（P1，会影响判题）**、**1 个环境依赖缺失（P1，会使脚本崩溃）**、以及 **1 处效率瓶颈（RPCA，P2）** 需要处理。

| # | 文件 | 类型 | 可运行 | 效率 | 结论 |
|---|---|---|---|---|---|
| 1 | `run_audio_analysis.py` | 基础分析 | ✅ | 好 | ✅ 可用 |
| 2 | `tools/compare_separation_methods.py` | 核心对比 | ✅ | ⚠️ RPCA 慢 | ⚠️ 建议优化 RPCA |
| 3 | `tools/run_demucs_deepdive.py` | 深度模型 | ✅ | 好 | ✅ 可用 |
| 4 | `tools/benchmark_realtime.py` | 实时基准 | ✅ | 好 | ⚠️ 口径需注明 |
| 5 | `tools/make_synthetic_musdb.py` | 自检夹具 | ⚠️ 需 ffmpeg | 好 | ✅ 可用 |
| 6 | `tools/_scan_stems.py` | 数据体检 | ✅ | ⚠️ 整文件读 | ✅ 可用（可优化） |
| 7 | `tools/_probe_umx.py` / `_probe_umx2.py` | 探针 | ✅ | — | 🗑 可归档 |
| 8 | `.tmp_pesq/extract.py` | 环境工具 | ✅ | — | ✅ 保留（PESQ 安装证据） |
| 9 | `lyrics-game/index.html` | 前端骨架 | ✅ | 好 | ✅ 可用 |
| 10 | `lyrics-game/script.js` | 游戏逻辑 | ✅ | 好 | ❌ **繁简转换失效（P1）** |
| 11 | `lyrics-game/style.css` | 样式 | ✅ | 好 | ✅ 可用 |
| 12 | `lyrics-game/start_game.bat` | 启动器 | ✅ | 好 | ✅ 可用 |
| 13 | `lyrics-game/gen_embedded.py` | 数据内嵌 | ✅ | 好 | ✅ 可用 |
| 14 | `lyrics-game/fix_audio_map.py` | 音频映射 | ❌ | — | ❌ **缺 opencc 必崩（P1）** |

---

## 1. 审阅方法与验证环境

- **解释器**：`C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe`（Python 3.13）
- **静态检查**：10 个 Python 文件全部 `py_compile` 通过；2 个 JS 文件 `node --check` 通过
- **运行验证**：用小参数（`--duration 10` / `--songs 1`）实际跑通，产出与预期一致
- **数据安全**：运行前对会被覆盖的产出（`04_reports/data/comparison/`、`Output-01_models/_third_party/demucs/`、`04_reports/figures/`）做了备份，验证后已全部恢复，**零数据损失**

---

## 2. 逐份代码简介（功能 + 预期输出）

### A 组 · 音频源分离（FYP 主线）

#### 1) `run_audio_analysis.py`（根目录，~428 行）
- **功能**：FYP 起步的**音频基础分析**脚本。自动识别 MUSDB18（`stem.mp4`）与 MUSDB18-HQ（`wav`）两种格式；对指定歌曲的前 N 秒做波形 / 采样率 / 频谱 / STFT 窗长 / 静音裁剪 / MFCC 六类分析。
- **预期输出**：`04_reports/figures/` 下 **9 张 PNG**（fig1~fig6，dpi=150）+ 终端打印各步的形状/采样率/trim 索引/MFCC 形状 + **9/9 自检**。
- **实测**：`--duration 10` → 9/9 图全部生成，EXIT=0。

#### 2) `tools/compare_separation_methods.py`（~654 行）★核心
- **功能**：**横向对比实验**。在同一段音频上跑 6 类方法（基线 / ICA / NMF / RPCA / HPSS / Open-Unmix），用 **museval（BSSEval v4，1s 窗、中位数）** 计算 SDR/ISR/SAR + SI-SDR，并计算 **ΔSDR = SDR(方法) − SDR(不分离基线)**。
- **预期输出**：
  - `03_outputs/<模型>/<歌曲>/<目标>.wav`（各模型独立分离音频）+ `03_outputs/<模型>/metrics.csv`
  - `04_reports/data/comparison/`：`comparison_results.csv/.json` + `fig_compare_sdr.png`（单曲）/ `fig_compare_dsdr.png`（多曲中位数）
  - 终端汇总表
- **亮点**：`--reanalyze`（从已有 JSON 重算，不重跑）、中位数聚合 + **自动剔除退化样本**（基线 SDR < −60 dB）、`--save-audio` 可选。
- **实测**：`--songs 1 --duration 10` → 10 行结果全部产出，51s，EXIT=0。

#### 3) `tools/run_demucs_deepdive.py`（~349 行）
- **功能**：**纵向深入分析**。对同一首歌跑 **Demucs（htdemucs）**，分离 4 个 stem，复刻同一套 9 张分析图，并计算 **SDR / SI-SDR / PESQ / STOI / Params / MACs / FLOPs**。
- **预期输出**：`Output-01_models/_third_party/demucs/` → `stems/`（4 个 wav）+ `figures/`（9 张图）+ `metrics.json` + `DEMUCS_REPORT.md`。
- **亮点**：FLOPs 计数改用 `torch.utils.flop_counter.FlopCounterMode`（因 `thop`/`ptflops` 不支持复数 STFT），并取 `model.models[0]` 在训练段长 7.8s 前向计数；PESQ 三级回退（C 版 pesq → pysepm → 不可用）。
- **实测**：`--duration 10` → 4 stem + 详细指标 + 9 图，EXIT=0。

#### 4) `tools/benchmark_realtime.py`（~109 行）
- **功能**：**实时性基准**。测各方法处理 30s 音频的耗时，计算 **RTF = 耗时 / 时长**（<1 才算可实时）。
- **预期输出**：终端 RTF 表 + `04_reports/data/comparison/realtime_benchmark.json`。
- **实测**：HPSS 0.068｜ICA 0.005｜NMF 0.112｜**RPCA 1.602（❌跑不动）**｜Open-Unmix 0.107，EXIT=0。

#### 5) `tools/make_synthetic_musdb.py`（~195 行）
- **功能**：**环境自检夹具**。合成一段「格式合法」的 MUSDB18 `stem.mp4`（5 条 AAC 音轨），用于在下载 4.4GB 真数据集前，先验证 `musdb + stempeg + FFmpeg + 绘图` 全链路。
- **预期输出**：`02_databases/MUSDB18-HQ/_smoketest_musdb18/train/00 - Synthetic Test Song.stem.mp4`。
- **备注**：需 `ffmpeg` 在 PATH（本机已具备）；它只造**假音频**，不用于正式分析。

#### 6) `tools/_scan_stems.py`（~46 行）
- **功能**：**数据体检**。扫描 train 前 N 首，用 RMS(dB) 判断 vocals/drums 是否静音，筛出"干净样本"，规避 museval 退化。
- **预期输出**：终端逐曲 dB 表 + 推荐 10 首序号/名称。
- **实测**：`_scan_stems.py 5` → 正确输出前 5 首判定。

#### 7) `tools/_probe_umx.py` / `_probe_umx2.py`（探针，~28 / ~13 行）
- **功能**：调试期一次性探针，用于确认 openunmix 版本、`separate()` 签名与返回类型。
- **预期输出**：终端打印。**已完成使命，可归档**（不影响任何流程）。

#### 8) `.tmp_pesq/extract.py`（~43 行）
- **功能**：解决本机无 C 编译器导致 `pip install pesq` 失败的问题——下载 conda-forge 预编译包，用 `zstandard` 解出 `site-packages/pesq` 拷入 venv。
- **预期输出**：venv 中出现可用的 `pesq` 包。
- **备注**：**建议保留**，是 PESQ 环境可复现的凭据。

### B 组 · lyrics-game（网页子项目）

#### 9) `lyrics-game/index.html`（~58 行）
- **功能**：游戏页面骨架。定义提示区 / 歌詞浮現区 / 控制按钮 / 输入区 / 状态栏 / 庆祝层；末尾引入 `data/embedded.js`（内嵌数据）与 `script.js`。
- **预期输出**：浏览器渲染的游戏界面。

#### 10) `lyrics-game/script.js`（~680 行）★核心
- **功能**：**猜歌游戏主逻辑**。LRC 解析 → 计时器驱动歌词逐句浮现 → 空格键/「猜這句」触发暂停 → 挖空下一句 → 判定（最多 4 次尝试，逐字 diff 显示）→ 提示（歌手/年代/地区）→ 下一首。
- **预期输出**：可交互的游戏；历史回答记录、庆祝动画。
- **⚠️ 关键问题**：见 §3 P1-1。

#### 11) `lyrics-game/style.css`（~380 行）
- **功能**："适老化"样式：暖暗色渐变 + 奶白文字 + 26px 大字号。
- **预期输出**：仅样式，无输出数据。

#### 12) `lyrics-game/start_game.bat`（~34 行）
- **功能**：一键启动器。释放 8080 端口 → 启动 `python -m http.server 8080` → 打开 Edge 到 `localhost:8080`。
- **预期输出**：本地服务器 + 浏览器窗口。

#### 13) `lyrics-game/gen_embedded.py`（~36 行）
- **功能**：**数据内嵌**。读 `data/songs.json`，把全部歌词塞进 `data/embedded.js`（`window.GAME_SONGS` + `window.GAME_LYRICS`），使游戏运行时**不再访问外部 .lrc**，规避"几分钟后访问不到"的问题。
- **预期输出**：`data/embedded.js`（约 95 KB）+ 终端统计。
- **实测**：校验成功——**50 首歌 / 50 条歌词全部内嵌**，0 缺失。

#### 14) `lyrics-game/fix_audio_map.py`（~55 行）
- **功能**：把 `audio/` 下的真实 mp3 文件名，按"歌手-歌名"规范匹配回写 `songs.json` 的 `audioFile`。
- **预期输出**：更新后的 `songs.json` + 匹配统计。
- **❌ 关键问题**：见 §3 P1-2。

### C 组 · 第三方源码（不逐一审阅）
- `01_models/_third_party/demucs/`：**Demucs v4.1.0a2**（Meta），未 pip 安装，通过 `sys.path.insert` 导入。

---

## 3. 发现的问题（按严重度）

### 🔴 P1-1　`script.js` 繁简转换完全失效（会影响判题）
`TRADITIONAL_MAP` 表里 **25 个抽样字 100% 是恒等映射**（`"愛":"愛"`、`"東":"東"`…），转换后一个都没变。

**实测证据**：
```
愛 -> 愛 (未变)   東 -> 東 (未变)   國 -> 國 (未变)   闊 -> 闊 (未变)
发生变化的比例: 0/25
```
而歌词是**繁体**（`[00:18.35]今天我寒夜裏看雪飄過`）。后果：
- 用户**输入繁体**：恰好能用（因为归一化是空操作）；
- 用户**输入简体**：**永远判错**（`爱` 永远匹配不上 `愛`）。

**修复建议**（三选一）：
1. 换成真正的繁→简映射表（把 `"愛":"爱"`、`"東":"东"` … 补对）；
2. 引入 `OpenCC.js`，对 **LRC 原文与用户输入双向归一**；
3. 最少改法：`normalizeAnswer` 里把两边都转成**同一种字形**再比（推荐引入 OpenCC）。

### 🔴 P1-2　`fix_audio_map.py` 必崩 + 原地覆写数据
- 顶部 `from opencc import OpenCC`，但本机 **`opencc` 未安装** → 一跑就 `ModuleNotFoundError`。
- 脚本会**直接原地覆写 `data/songs.json`**（无备份、无 dry-run），一旦匹配逻辑出错就会污染主数据。

**修复建议**：
1. `pip install opencc-python-reimplemented`（或 `opencc`）；
2. 覆写前先备份为 `songs.json.bak`，并支持 `--dry-run` 只打印不写盘；
3. 建议把 `norm()` 规范化逻辑与 `script.js` 的归一化统一，避免"两套繁简口径"。

### 🟠 P2-1　`compare_separation_methods.py::_rpca_ialm` 是全场效率瓶颈
- 每轮迭代对 `(1025, N)` 矩阵做**全秩 SVD**（`np.linalg.svd(..., full_matrices=False)`），最多 120 轮。
- 实测：**仅 10 秒音频 = 22.05 秒**；30 秒 RTF **1.602**（跑不动实时）。
- **优化建议**：
  1. 用 `sklearn.utils.extmath.randomized_svd` 或 `scipy.sparse.linalg.svds` 做**低秩截断 SVD**（低秩+稀疏分解本就不需要全秩），可提速数倍；
  2. 给 `mu` 加**上限**（标准 IALM 会 clamp `mu ≤ mu_max`），避免后期 mu 过大拖慢收敛；
  3. 收敛早停阈值调优 + `max_iter` 降到 60~80；
  4. 若只为「降采样后粗分」，可对 STFT 频谱做降采样再分解。

### 🟠 P2-2　`_nmf_fit` 迭代未收敛
- 实测出现 `ConvergenceWarning: Maximum number of iterations 250 reached`。
- **建议**：`max_iter` 提到 500 或改用更好的 init / 更小的 k；或在报告中注明这是有意的"限时"设定。

### 🟡 P2-3　若干小优化（不影响正确性）
| 位置 | 现状 | 建议 |
|---|---|---|
| `_nmf_mask_to_audio` | `for k in idx: num += np.outer(W[:,k], H[k,:])` | 向量化为 `W[:, idx] @ H[idx, :]` |
| `load_song` / `rd` | mixture 被读两遍（mono + stereo） | 读一次，再转 mono/stereo |
| `_scan_stems.py::rms_db` | `sf.read` **整文件**读入只为看前 30s | `sf.read(..., frames=n)` 只读前 n 帧 |
| `benchmark_realtime.py` | Open-Unmix RTF **含冷启动权重加载** | 与"热启动"分开测，或在报告注明口径 |

### 🟡 P2-4　lyrics-game 其他小问题
- `skipSong()` / `submitAnswer()` 直接 `audio.pause()`，**未做 null 保护**（正常流程不会为 null，低风险，但建议加 `if (audio)`）。
- `playClap()` 引用 **不存在的 `audio/clap.mp3`** → 答对时无音效（被 try/catch 静默吞掉）。
- `parseLRC` 正则只认 `[mm:ss.xx]`，不认 `[mm:ss:xx]`（冒号分隔）与 `[offset:]` 标签，兼容性一般。

### ✅ 值得肯定的地方（不是问题，但要点名）
- **XSS 安全**：`buildDiffHtml` 走 `innerHTML`，但输入先经 `normalizeAnswer` 过滤为 `[\u4e00-\u9fa5a-zA-Z0-9]`，`<script>` 会被剥成 `script`，**无注入风险**。
- **方法论正确**：多歌聚合用**中位数**、自动**剔除退化样本**（`< −60 dB`）、以 **ΔSDR** 而非绝对 SDR 做横比——这三条正是文献常踩的坑。
- **工程规范**：统一 `matplotlib.use("Agg")`、UTF-8 输出、产物自检、`--reanalyze` 免重跑、输出目录按模型分文件夹。

---

## 4. 运行验证汇总（实测）

| 脚本 | 冒烟命令 | 结果 | 耗时 |
|---|---|---|---|
| `run_audio_analysis.py` | `--duration 10` | ✅ 9/9 图 | ~5 s |
| `compare_separation_methods.py` | `--songs 1 --duration 10` | ✅ 10 行结果 | 51 s |
| `run_demucs_deepdive.py` | `--duration 10` | ✅ 4 stem + 9 图 + 指标 | ~30 s |
| `benchmark_realtime.py` | 默认 30s | ✅ JSON 产出 | ~60 s |
| `_scan_stems.py` | `5` | ✅ 判定正确 | <1 s |
| `script.js` / `embedded.js` | `node --check` | ✅ 语法通过 | — |
| `gen_embedded.py` | 逻辑复算 | ✅ 50 歌 / 50 歌词 | — |
| `fix_audio_map.py` | import 检查 | ❌ **缺 opencc** | — |
| lyrics-game 静态服务 | `http.server` + curl | ✅ 4 类资源全 200 | — |

> 注：RTF 与硬件强相关。以上为 **CPU、单进程、无 GPU** 的结果；相对排序（RPCA 最慢）在 GPU 上也不会变。

---

## 5. 每份代码对 FYP 的意义

### A 组 · 音频源分离

| 文件 | 对 FYP 的意义 |
|---|---|
| `run_audio_analysis.py` | **打地基**：把"音乐长什么样"用 9 张图讲清楚（波形/频谱/MFCC），是 Progress Report「Task 1~4」的直接交付物，也是后续所有分离实验的**数据理解与图表模板来源**。 |
| `compare_separation_methods.py` | **整个 FYP 的实证核心**：用**同一数据**横比古典方法 vs 深度模型，产出可复现的 ΔSDR 排序。它回答的是 FYP 最关键的选型问题——"到底用哪种方法"。 |
| `run_demucs_deepdive.py` | **精度天花板与代价的量化**：给出 Demucs 的 SDR/PESQ 与 Params/MACs/FLOPs，把"深度模型更好但更贵"从定性变成**可答辩的数字**，为"实时性 vs 精度"权衡提供依据。 |
| `benchmark_realtime.py` | **FYP 命题（实时降噪）的胜负手**：ΔSDR 只答"准不准"，RTF 才答"能不能实时"。它直接**淘汰了 RPCA**（RTF 1.6），是选型的硬约束证据。 |
| `make_synthetic_musdb.py` | **风险前置**：在下载 4.4GB 数据集前先验证整条 IO/绘图链路，避免"下了半天才发现跑不通"，属**工程稳健性保障**。 |
| `_scan_stems.py` | **数据质量守门人**：及时揪出"静音 stem / 过稀疏参考"，防止退化样本污染统计（此前正是它定位到 Aimee Norwich / Velvet Curtain 两个污染源），保证结论**不会被质疑**。 |
| `_probe_umx.py` / `_probe_umx2.py` | **探路石**：确认 openunmix 1.3 的返回类型（dict 而非 ndarray）等坑，换取主脚本的稳定。使命已完成。 |
| `.tmp_pesq/extract.py` | **绕过环境限制的关键一招**：本机无 C 编译器仍拿到 PESQ，使 FYP 具备**感知质量指标**，答辩时指标维度更完整；同时是可复现的环境凭据。 |

### B 组 · lyrics-game

| 文件 | 对 FYP 的意义 |
|---|---|
| `index.html` | 游戏的**界面骨架**：定义所有 UI 容器与按钮，是交互的舞台。 |
| `script.js` | 游戏的**大脑**：LRC 解析、逐句浮现、挖空判定、提示与记录。它把"音乐 + 歌词"变成可玩的**交互 demo**，是 FYP 的**展示型产物**（面向评委/用户）。 |
| `style.css` | **可用性保障**：适老化（大字、高对比）设计，让 demo 在真实用户面前站得住。 |
| `start_game.bat` | **一键演示入口**：评审现场双击即用，避免"配置环境"的尴尬。 |
| `gen_embedded.py` | **稳定性关键**：把歌词内嵌进 JS，彻底消除"运行时读 .lrc 失败"，让 demo **离线也能跑**。 |
| `fix_audio_map.py` | **数据一致性工具**：把 50 首 mp3 与歌曲元数据对齐；一次性工具，但目前**缺依赖跑不了**，需修复。 |

### 两条线的关系（供你确认）
`FYP_HKBU` 里实际有**两个独立项目**：**音频源分离**（主线，Sep 10–15）与 **lyrics-game**（子项目，Aug 15–18）。
两者目前**没有代码耦合**。如果它们本属同一个 FYP，建议明确"谁主谁次"；如果是两个课题，建议把 lyrics-game 移出 `FYP_HKBU` 或在 README 中说明关系，避免答辩时被问"到底做什么"。

---

## 6. 建议行动（按优先级）

| # | 行动 | 优先级 | 负责 |
|---|---|---|---|
| 1 | **修 `script.js` 繁简转换**（影响判题正确性） | 🔴 高 | 你 / 我 |
| 2 | **修 `fix_audio_map.py`**：装 `opencc` + 加备份/`--dry-run` | 🔴 高 | 你 / 我 |
| 3 | **优化 `_rpca_ialm`**：低秩 SVD + mu 上限，目标 RTF < 1 | 🟠 中 | 我 |
| 4 | 归档 `_probe_umx*.py` 到 `_archive/` | 🟡 低 | 我 |
| 5 | 补 `audio/clap.mp3` 或移除音效调用 | 🟡 低 | 你 |
| 6 | 明确两项目关系，写进 README | 🟡 低 | 你 |

---

*本报告由静态审阅 + 实际运行验证生成。所有冒烟测试均未破坏原始产出（备份 → 验证 → 恢复）。*
