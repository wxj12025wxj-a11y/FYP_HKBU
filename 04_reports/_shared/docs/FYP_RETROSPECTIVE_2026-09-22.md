# FYP 工作回顾（2026-09-10 → 2026-09-22）

> 整理日：2026-09-22 ｜ 覆盖 9 个工作段
> 用途：① 让你一眼看清"已经攒下什么资产"（避免重复劳动）② 答辩时"我做过什么"的事实来源
> 数据来源：本项目 9 份工作日志（`E:\Intern\2026-09-10-11-25-38\.workbuddy\memory\`）+ 项目内实物核对

---

## 0. 一页纸时间线

| 日期 | 主题 | 里程碑性产出 |
|---|---|---|
| 09-10 | 音频分析链路搭建（Progress Report Task 1–4） | 首个分析脚本 + 9 张图 + 环境手册 |
| 09-11 | 适配 MUSDB18-HQ，跑出真实结果 | 双格式数据集自动识别 |
| 09-14 | 源分离方法对比研究（学术调研 + 实测） | 406 行主报告（8 章）+ 首份实测 SDR/RTF |
| 09-15 | 目录重构 + 稳健基线 + Demucs 深入 + **PESQ 打通** + 代码审阅 + 可视化图谱 + 试听包 | 结论数据集 + 自包含证据图谱 HTML |
| 09-16 | **五模块目录重组** + paths.py + DNS 导入 + 12 模型核查 + BSRNN 三变体 | 项目规范化 + 11 模型可运行 |
| 09-17 | **CUDA 启用** + 全模型 GPU 重跑 + DPRNN 自训 | 12 模型 GPU 指标 + 首个自训模型 |
| 09-19 | 准入规则审计 + test 集扫描启动 | 规则判定 + song-major 调度器 |
| 09-21 | 排查"为什么这么慢" + 长任务可靠性攻坚 | 耗时真相 + 3 个 bug 修复 + schtasks 方案 |
| 09-22 | test 集 50 首数据分析 + 人耳聆听页 + 路线图 | 10 张图 + 看板 + 正式立项板 2 |

**一句话总结这 9 天**：从"能跑一个模型"走到了"12 个模型在 50 首整曲上跑完、指标口径已统一、耗时结构已摸清、板 2 已立项"。

---

## 1. 逐次回顾

### ① 09-10（周三）· 音频分析链路搭建

**你提出**：为 FYP（HKBU，目标实时降噪）搭建音频基础分析链路，完成 Progress Report Task 1–4。

**做了什么**
- 建 `E:\FYP_HKBU\run_audio_analysis.py`：读 MUSDB18，自动选曲、截 30 s、出 9 张 PNG，打印 shape/sr/trim/MFCC
- 建 `tools/make_synthetic_musdb.py`：用 FFmpeg 合成合法 `stem.mp4`，供无数据集时冒烟测试
- 建 `FYP_Audio_Setup_Guide.md`：环境 + 下载 + 运行 + 排错

**关键事实（后来一直复用）**
- FFmpeg **已装**（`8.1.2-full_build`，在 PATH）——你提示词里写"未安装"是过时的
- 无 conda；venv = `C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio`
- matplotlib 用 `Agg` 后端；脚本内 `sys.stdout.reconfigure(encoding='utf-8')` 修中文乱码

**踩的坑**：刚 pip 装的编译型 DLL（numpy `_mt19937`、scipy）首次导入被 **Windows 应用控制策略拦截**（`应用程序控制策略已阻止此文件`），重试即恢复。→ 这是后来 **SAC** 问题的第一次露面。

---

### ② 09-11（周四）· 适配 MUSDB18-HQ

**背景变更**：你实际只找到 **MUSDB18-HQ（wav 版）**，不是原始 `stem.mp4`。

**做了什么**
- 给脚本加 `_looks_like_dataset()` / `detect_dataset()` 自动识别两种布局，新增 `--format auto|hq|stem`
- 核对结构：`datasets\train\<歌名>\{mixture,vocals,drums,bass,other}.wav`；train 100 / test 50

**验证**：冒烟测试 9/9 通过、真实数据第一首 9/9 通过（`mixture shape: (1323000,) sr: 44100`）

**踩的坑**：bash 重定向 `> log 2>&1` + `MPLCONFIGDIR` 组合会导致脚本**静默失败**（EXIT=1、日志为空）→ 改用管道 `| tail -n`。

---

### ③ 09-14（周日）· 源分离方法对比研究

**你提出**：对比研究各源分离方法。

**做了什么**：交付 406 行 / 8 章主报告 `METHOD_COMPARISON_REPORT.md`（方法百科 + 市场分层 + 公开精度表 + 同数据实测 + 实时性基准 + 交叉验证 + FYP 启示）。修复了报告里重复的章节块与交叉引用错误。

**关键实测结论（歌曲 NightOwl，30 s，CPU）**
- 单曲人声 SDR：Open-Unmix **+6.55** > RPCA −1.21 > NMF −8.56 > 基线 −9.16 > ICA −26.44
- RTF：ICA 0.005 / HPSS 0.069 / Open-Unmix 0.117 / NMF 0.121 / **RPCA 1.645（不能实时）**
- **两条定调结论**：① **ΔSDR 是唯一可横比指标** ② "古典方法快、深度方法慢"**在源分离上是错的**（RPCA 比 Open-Unmix 慢 14 倍）

---

### ④ 09-15（周一）· 这是产出最密集的一天（7 件事）

**你先后提出 6 个新需求**，逐条落地：

| 你的需求 | 做了什么 |
|---|---|
| 目录要分清 | 把 `reports/comparison/` 拆为 `outputs/<模型>/` + `outputs/comparison/`，旧目录**零删除**归档 |
| 基线的稳健性 | 10 首歌跑完发现**退化样本污染均值**（人声极稀疏、鼓声静音）→ 改**中位数聚合 + 剔除退化**，sd 从 ±30 dB 降到 ±1~6 dB |
| Demucs 深入 | 新脚本输出 4 stems + 9 图 + metrics；测得人声 SDR 8.74 / Params 41.98 M / FLOPs 224.47 G |
| **PESQ 能不能用** | 本机无 C 编译器、`pip install pesq` 构建失败 → **下载 conda-forge 预编译包，用 zstandard 解包后拷进 venv** → 打通。同时发现**鼓声 PESQ 倒挂**（1.08 < 基线 2.79）→ 定论：**PESQ 只对人声有意义** |
| 全量代码审阅 | 14 个自研文件逐个"功能 + 预期输出 + 对 FYP 意义"；全部冒烟 EXIT=0（**测前备份、测后恢复，零数据损失**）。**发现 P1**：`lyrics-game` 的 `TRADITIONAL_MAP` 是 100% 恒等映射 → 繁简归一失效 |
| 可视化图谱 | 自包含 HTML（Chart.js），后整合为 **1.18 MB 七章证据图谱**，19 张产出图 base64 内联 |
| 运行时长基准 + 汇报 PPT | 10 首歌同组对比 UMX vs Demucs：**41.98 s vs 113.27 s → Demucs ≈ 2.70×**；出 7 张图 + 13 页自包含 HTML PPT |
| 试听包 | `DATA.zip` + `Open-Unmix_umxhq.zip` + `Demucs_htdemucs.zip`（10 首全曲，断点续跑） |

**这一天抓到的重要口径问题**
- 🔴 `openunmix.predict.separate` 返回 **(1,2,N) 3D**，旧管线只 mean 一次 → 形状错，人声 SDR 6.55 应修正为 **6.98**
- 🔴 **Demucs 非确定性**：同输入两次 8.83 / 8.73（±0.1~0.3 dB）
- 🔴 **长曲 RTF 不可由 30 s 外推**：UMX 全曲 RTF 0.428（30 s 基准 0.142）
- 🔴 **事实修正**：Open-Unmix 单目标 **8.90 M**（不是我先前写的 4.6 M），与 Demucs 41.98 M 同量级
- ⚠️ **主动承认边界**：UMX 的 FLOPs 3.57 G 被严重低估（抓不到 LSTM fused 算子）→ **不可与 Demucs 224 G 横比**

---

### ⑤ 09-16（周二）· 项目规范化 + 11 模型可运行

**你提出**：目录要"清晰明了地看到几个模块"。

**做了什么**
1. **五模块重组**：`01_models / 02_databases / 03_outputs / 04_reports / 05_misc` + `tools/`
   - **只移动、未删除**；同盘 mv 秒级完成；脚本 `_migrate_layout.py`（幂等）+ `_rewire_paths.py`（改写 128 处硬编码路径）
2. **`tools/paths.py` —— 路径唯一真理源**：以后目录再变只改这一个文件
3. **DNS-Challenge 导入**：全量 892 GB 不可行 → 拍板 5 个官方分片 **共 4.17 GB**
4. **12 模型可运行性核查** → 11 个指定模型：**8 可运行 / 3 受阻**
5. **BSRNN 三变体跑通**：4.67 GB 权重 MD5 双重校验，逐键 `missing=0 / unexpected=0`
6. Demucs 并入统一基准，`model_runs.json` 从 11 → **12 模型**

**这一天最重要的两个技术发现**
- 🔴🔴 **SIMO 的 state_dict 按位置对齐**：ckpt 记 `targets=['vocals','bass','drums','other']`；顺序错时 `load_state_dict(strict=False)` **不报错、不丢键、能出声、wav 正常落盘，但 4 个 masker 静默互换 → 指标串味**。修法：从 ckpt 的 `hyper_parameters` 读原始 targets 顺序。
- 🔴 **评测片段的"能量陷阱"**：NightOwl 前 10 s bass 占 **85%**、鼓仅 **1.4%** → **所有模型**的 bass SDR 一起虚高到 20 dB+。修法：加 `--offset` + 写 `_scan_balanced_clip.py` 筛均衡片段。
- ⚠️ **SAC 拦截面不是静态的**：上午记录"拦 numpy.random 系列 → museval 全挂"，**下午复测已恢复**。→ 方法论：**历史"被拦"记录不要采信，先重跑探测**。

**操作事故（值得记一辈子）**：`mv -f` 把新写的 2.9 KB 文件覆盖掉 78 KB 的 12 模型历史记录 → **目标已有同名文件时绝不直接 `mv -f`**。

---

### ⑥ 09-17（周三）· CUDA 启用 + DPRNN 自训

**做了什么**
1. **torch 切 CUDA**：`2.14.0+cpu` → `+cu126`（PyPI 没有 CUDA wheel，必须走 `--index-url`）
2. **全模型 GPU 重跑**：12 模型 × 3 片段，13/13 PASS；CPU 基线归档不覆盖
3. **DPRNN 自训**：复用 `Dual-Path-RNN-Pytorch` 模型定义 + 自写 MUSDB18-HQ 4-stem 管线（3.69 M 参数、11.025 kHz）
4. 建**训练收尾链** `post_dprnn_chain.py` + **中位数聚合加 offset 守卫**

**三个大坑**
- 🔴 **CUDA sysmem fallback**：8 GiB 卡超显存**不报 OOM**，张量悄悄进系统内存 → 步耗时 0.37 s → 3.4 s。**这是本机最危险的静默故障源**
- 🔴 **首轮 110 分钟预算被一次 8 小时系统睡眠吃掉**（Kernel-Power EventID 42）→ 真实算力只有 ~23 分钟。修法：`SetThreadExecutionState` 做**进程级禁睡**（比改 `powercfg` 可靠）
- 🔴 **假 batch**：`x[None].expand(batch,-1)` 只是复制同一段音频 → 梯度等价 batch=1

**DPRNN 最终结果**：step 31900，best mean SI-SDR **+2.745 dB**（旧 ckpt +0.781）；四轨均值 **−4.54 → −1.72**（+2.82 dB），**四轨仍全为负**
→ 定调：**不参与精度排名，只作"自训 5 小时无法达 SOTA"的负面证据**

---

### ⑦ 09-19（周五）· 准入规则审计 + test 集扫描启动

**你新增了一条规则**：只有「**模型本身能输出分离音轨**」才进工作流。

**做了什么**
- 审计全部模型 → 宽松读 **9 个符合** / 严格读 4 符合 + 2 半符合；**不符合 3 个**：`dprnn`（自训权重 + 原生 11.025 kHz）、`rpca_drnn`（论文未公开代码）、`pac_hubert_sep`（MERL 未发布）
- 启动 MUSDB18-HQ test 集 50 首整曲扫描；建 **song-major 扫描调度器**

**长任务启动的坑（比推理本身更耗时）**
1. 🔴 WorkBuddy 的 `python` 解析到**系统 Python（无 numpy）** → 子进程一秒就死，外层只显示「ERR 0.5s」，**看着像推理失败实则没跑起来**
2. 🔴 bash 包装层让子进程 stdout 退回 **GBK** → print emoji 就崩；**最恶的是崩在写锁之后**，留下陈旧 `.lock` 挡住后续启动

---

### ⑧ 09-21（周日）· 排查"为什么这么慢" + 可靠性攻坚

**你提出**：「暂停这个后台任务，然后检查为什么需要这么久」

**诊断结论（推翻了直觉）**
- oracle 实测 **24.7 s/首**，其中**推理只有 2.0 s（8.3%）**
- 拆解：解释器 1.5 s + import torch/CUDA **2.9 s** + **museval 评测 4 轨 12.9 s（★ 最大单项）** + 其它 IO ~1 s
- **根因**：`evaluate()` 对每一轨单跑一次 museval → 4 次/首；热点是 `compute_GsfC`（512-tap 滤波器矩阵求逆）
- 🔴 **RTF 外推法系统性低估**：原 39.11 h 预估只算了推理，真实 42 h

**随后你下指令**：确认可运行模型 → 按 song-major 顺序跑 50 首 → 单曲超 2 分钟暂停该模型 → 超时 3 次即标记不可用。

**执行中的发现与决策**
- 🔴 **把 15 个模型塞进同一进程会 OOM**（权重常驻显存）→ 反过来证明了「每组合独立子进程」**不是浪费，是显存隔离**
- 🔴 **磁盘约束**：12 模型 WAV 要 **98 GB**，但只剩 66 GB → **FLAC 无损压缩到 24.4 GB**（实测**逐比特一致**，误差 0.00e+00，对指标零影响）
- 🔴 **120 s 超时会淘汰精度前 3 名**（含第一名 BSRNN-SIMO）→ 你拍板放宽到 **180 s**，存活 9 个

**"为什么暂停了"的根因（本轮最有价值的发现）**
- 逐项排除：机器睡眠 ❌ / 进程崩溃 ❌ / OOM 被杀 ❌ / **会话结束被硬杀 ✅**
- **判据**：`finally` 里的解锁代码没执行 → 说明是 `TerminateProcess` 而非异常退出
- **修法**：改走 **Windows 计划任务 `schtasks`**（父进程链落到 `services.exe`，不随会话死）

**修掉的 3 个 bug**
1. **假 PASS**：脚本内部捕获异常写 `status: FAIL` 却**仍以 0 退出** → 调度器按返回码判定，把失败记成 PASS
2. **分块推理 stereo 广播崩溃**：DPRNN（mono）的累加器按原始维度分配 → 修复后 `FAIL` 变 `PASS`，RTF 0.0133
3. **僵尸进程误判**：`OpenProcess` 成功 ≠ 进程存活（被杀但句柄未释放）→ 改用 `GetExitCodeProcess == 259`

---

### ⑨ 09-22（周二）· 数据分析 + 人耳聆听 + 立项板 2

**你先后提出 3 个需求**：

**1) 梳理 12 个模型的实机运行数据并可视化**
- 🔴 先核出**三处口径问题**：① "12 个跑完"实际是 **11 满 + 1 半**（BSRNN-SIMO 目录里 17 首但只有 14 首有效，另 3 首是超时半成品）② DPRNN 账本里那笔 FAIL 是**修复前的过期记录**（实际已重跑 PASS）③ `n_tracks` 经 JSON 往返**键变字符串**，`4 in {"4":50}` 恒为 False → 所有四轨模型被误判成单轨
- **真实耗时结构**：12 模型累计墙钟 **10.17 h，推理只占 4.25 h（41.8%）**
- 🔴 **最深的一条新结论**：**开销与"输出几轨"强绑定**。白盒实测 museval 评 4 轨 **19.5 s** vs 评 1 轨 **6.5 s**；同曲配对 50 首后，4 轨模型比 1 轨模型**每首多花中位 31.0 s**，且随歌长放大（<150 s +13.8 s → 430 s 那首 +56.0 s）
- **实机排名与 RTF 排名完全不同**：最快的 Oracle 推理只要 3.2 s/首，端到端却要 52 s/首（**93% 是开销**）
- 产出：10 张图 + 自包含看板 + 论文用数字摘要 + 2 张 CSV

**2) 做三首歌的人耳聆听对比页**
- 三首：`Georgia Wonder - Siren`（430.4 s，最长）、`PR - Oh No`（76.2 s，最短）、**`M.E.R.C. Music - Knockout`**（270.0 s，"歌与歌差距"中位，第 26/50 位）
- 产出：**111 个可播音频**（源混音 + GT 四轨 + 11 模型全部产出），支持单轨 A/B、叠听重建整曲、逐轨音量、盲听打乱、键盘快捷键
- **三重播放验证**：Python 全量解码 111 文件（478.9 min，全部完整无静音）+ **真实 Chrome 探针**（http 与 `file://` 各 5/5，可解码/播放/拖动定位）+ jsdom 33 项 DOM 断言
- 又抓到 3 个坑：① **半成品音频**（`BSRNN-SIMO/.../vocals.flac` 只有 5.7 MB，FLAC `total_samples=0`）→ 收录前必须比对帧数与源 mixture ② 键盘事件在非元素 target 上抛异常 ③ "全停"后行状态没清
- ⚠️ 新教训：**同一消息里并行发多个 Edit 会互相覆盖**（丢了 3 处修改）→ 改文件一律一条消息一个 Edit

**3) 和教授沟通后 → 立项板 2，出路线图**
- 产出 `FYP_ROADMAP_2026-09-22.md`：完整梳理 + 7 阶段 + 13 条风险
- 调研中 6 个改变计划的发现（详见路线图）：
  1. 你给的两个数据集来源**是同一个东西**（Edinburgh DataShare 那条链接就是 Valentini-Botinhao 本身）
  2. **Mel-RoFormer-Denoise 不需要装新框架**——它走的 ZFTurbo 框架你**本机已克隆**
  3. 参考图的 Reverberation / Target Echo **本领域有专门模型**（去混响 / 去回声 Mel-RoFormer），不用硬凑
  4. 频谱面板不要照抄水下声学形式 → 本领域的理论等价物是 **BSS-Eval 四分量分解**，而 museval 本来就产出这些
  5. **PESQ 只支持 8/16 kHz，在 48 kHz 上跑是错的** → 定双口径
  6. **`master64` 训练集含 Valentini** → 测 Valentini 属 in-domain，必须并列 `dns48`/`dns64` 对照

**你拍板的三项决策**
- 降噪范围 = **评测 + 级联实验**（不做微调）
- **纳入**去混响 / 去回声模型（+2 个）
- **BSRNN-SIMO 暂不补跑** → 保留 14/50 并显式标注截断

---

## 2. 累积资产盘点（现在手里有什么）

### 2.1 数据与产物

| 资产 | 体量 | 状态 |
|---|---|---|
| MUSDB18-HQ | train 100 + test 50，44.1 kHz 立体声四轨 | ✅ 完整 |
| **test 集整曲分离产物** | `03_outputs/_test_run/<模型>/<歌曲>/*.flac`，**21 GB / 564 组合** | ✅ 与源混音样本级对齐 |
| DNS-Challenge 子集 | 5 个官方分片，**4.17 GB，77210 个 wav** | ✅ 含 **dev_testset 921 条带参考** + **RIR 60248 条** |
| 模型权重 | 12 个模型全套（含 BSRNN 三变体、DPRNN 自训） | ✅ MD5 校验通过 |
| Valentini | — | ❌ 待下载（阶段 0） |
| 降噪 / 去混响模型 | — | ❌ 待下载（阶段 0） |

### 2.2 指标与报告

| 资产 | 落点 |
|---|---|
| 主结果表（3 均衡片段中位数） | `docs/MEDIAN_TABLE.md` |
| 整曲表 | `docs/MEDIAN_TABLE_TEST.md` |
| 逐曲逐模型指标（564 行） | `data/comparison/test_run_per_song.csv` |
| 耗时结构分析 | `data/comparison/test_run_analysis.json` |
| 供给与可运行性报告（485 行） | `docs/MODEL_PROVISIONING_REPORT.md` |
| 方法对比报告（406 行） | `docs/METHOD_COMPARISON_REPORT.md` |
| 准入规则审计 | `docs/MODEL_RULE_AUDIT_2026-09-19.md` |
| test 集扫描摘要 | `docs/MUSDB18_TEST_SWEEP_2026-09-22.md` |

### 2.3 可视化与交互

| 资产 | 说明 |
|---|---|
| **人耳聆听对比页** | `html/listen_compare.html`，111 个可播音频，双击可用 |
| test 集分析看板 | `html/test_run_dashboard.html`，11 节 + 10 图 |
| 证据图谱 | `html/METHOD_VISUAL_COMPARISON*.html`，1.25 MB |
| 图表 | `figures/test_run/`（10）、`figures/comparison/`（3）、`figures/audio_analysis/`（9） |

### 2.4 工具链沉淀

| 类别 | 关键脚本 |
|---|---|
| 基准骨架 | `benchmark_model_universal.py`（15 模型统一入口） |
| 批量调度 | `bench_musdb_test_songmajor.py`（song-major + 超时熔断 + 断点续跑） |
| 推理 | `_chunked_infer.py`（分块，显存降 11×、快 15×） |
| 分析出图 | `_analyze_test_run.py` / `_plot_test_run.py` / `_build_test_run_html.py` / `_probe_overhead_split.py` |
| 聆听站 | `_listen_server.py`（Range 206）/ `_build_listen_page.py` / `_verify_listen_audio.py` |
| 基建 | `paths.py`（路径唯一真理源）/ `dns_fetch.py` / `par_download.py` |
| 已沉淀为技能 | `windows-long-inference-batch` / `sac-blocked-python` |

---

## 3. 跨阶段教训（写进方法论，不是抱怨）

| # | 教训 | 出处 | 现在的应对 |
|---|---|---|---|
| 1 | **"跑通但指标静默错"是本项目最高频的陷阱**（已踩实 6 次：state_dict 位置对齐、能量陷阱、假 batch、sysmem fallback ×2、有目录≠成功） | 09-16 / 09-17 | 判成功**只读结果 JSON 的 status**；多输出模型**从 ckpt 读 targets 顺序**；收录音频**必对长度** |
| 2 | **"有目录"≠"成功"、"返回码 0"≠"成功"** | 09-19 / 09-21 / 09-22 | 半成品同样会留下输出目录（BSRNN-SIMO 的 3 首）；脚本内部捕获异常仍以 0 退出 |
| 3 | **RTF 不能用来外推墙钟** | 09-21 / 09-22 | 实机排名与 RTF 排名完全不同；Oracle 93% 是开销 |
| 4 | **口径不写死，数字就会悄悄漂移** | 09-15 / 09-21 | `evaluate()` 未传 window/hop，实为 2 s/1.5 s 而非文档写的 1 s 窗；跨组比较有明令禁令 |
| 5 | **链条"写完"≠"跑过"** | 09-17 / 09-19 | 收尾链 4 次 dry-run 却从未真跑，报告带旧数据 30+ 小时 → 凡自动触发链路，其后必须安排人工核对 |
| 6 | **坏消息越早说成本越低** | 09-15 / 09-22 | 主动承认 UMX FLOPs 被低估、Demucs 非确定性、DPRNN 四轨为负 |
| 7 | **环境事实有时效性，必须重测** | 09-16 | SAC 拦截面当天内就会变；"环境阻塞"结论必须带时间戳 |
| 8 | **长任务的最大敌人不是算法是宿主** | 09-17 / 09-19 / 09-21 | 系统睡眠、会话结束、GBK 编码、僵尸 PID、锁残留——五类宿主问题合起来比推理本身耗时更多 |

---

## 4. 下一步

正式计划见 **`FYP_PLAN_2026-09-22.md`**：5 周（至 2026-10-27）、8 个阶段、5 个里程碑、17 条风险登记。

**建议立即做的 3 件事**
1. 今晚启动阶段 0 的下载（Valentini + denoiser + 4 个新模型权重）
2. 本周内向导师确认中间检查点与提交形式（目前"待确认"）
3. 明天开工写 `_measure_model_profile.py`（教授最直接要的参数/FLOPs 表，几乎不占 GPU）
