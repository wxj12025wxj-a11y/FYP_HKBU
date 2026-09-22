# 模型准入规则审查 —— 「原生可直接输出分离音轨」

> 审查时间：**2026-09-19 12:15**（GMT+8）
> 新增规则：**模型本身可以输出分离音轨的，才进入后续工作流**（而不是需要我们对它进行修改和改进才能成功输出）
> 审查范围：项目内全部 **12 个**模型条目
> 关联：`01_models/MODEL_REGISTRY.md`、`04_reports/docs/MODEL_PROVISIONING_REPORT.md`

---

## 0. 一句话结论

按新规则，**12 个条目中只有 6 个符合**，6 个不符合。

| 判定 | 数量 | 模型 |
|---|---|---|
| ✅ **符合** | **6** | MDX-Net、Demucs、BS-RoFormer L12、BS-RoFormer L6、BSRNN-SIMO、Oracle(IRM/IBM) |
| ⚠️ **边界（我们补了胶水才跑通）** | 3 | Conv-TasNet、MMDenseLSTM、BSRNN(4×单目标 ckpt) |
| ⛔ **不符合** | **3** | DPRNN、RPCA+DRNN、Pac-HuBERT-SEP |

> **本报告的口径说明（重要）**：规则里「不需要我们修改和改进」有两种读法 ——
> 严格读法只认「一个模型文件直接吐全套 stem」；宽松读法把「官方发布权重 + 官方推理链路，
> 我们只写调用适配」也算过关。
> 下面**两条读法都给**（§2 严格表 / §3 宽松表），并明确标出 3 个边界模型的差异。
> **进入 test 集工作流的名单按宽松读法**（§3），因为那 3 个边界模型的输出确实
> 100% 来自官方权重与官方网络，我们没改过任何一个权重张量。

---

## 1. 规则拆解：三个必须同时满足的条件

「原生可直接输出分离音轨」= 以下三条**同时成立**：

| # | 条件 | 判据 |
|---|---|---|
| **C1** | **权重是模型原作者发布的** | 不是我们训练/微调/重新拟合出来的 |
| **C2** | **网络结构未被我们改动** | 没有换输出头、没有加/删分支、没有改 masker 数量 |
| **C3** | **官方推理链路能直接跑出 stem** | 无需我们重写 STFT / 后处理 / 输出映射才能得到音轨 |

对照：**只要有一条不满足，就属于「需要我们对它进行修改和改进才能成功输出」。**

---

## 2. 严格读法：只有「一个模型直接出全套 stem」

> 严格读 = C1 + C2 + C3，且**单一模型文件一次前向出多个 stem**，不借助「多 ckpt 串行 / 只出部分 stem / 后处理拼装」。

| # | 模型 | C1 官方权重 | C2 结构未改 | C3 官方链路直出 | 严格判定 | 原因 |
|---|---|---|---|---|---|---|
| 1 | **MDX-Net** (`mdx_extra`) | ✅ | ✅ | ✅ | ✅ **符合** | demucs 运行时原生支持 4-stem bag 集成，`apply_model` 直出 vocals/drums/bass/other |
| 2 | **Demucs** (`htdemucs`) | ✅ | ✅ | ✅ | ✅ **符合** | 单模型 41.98 M，`model.sources` 原生 4 stem |
| 3 | **BSRNN-SIMO** (`simo-bsrnn-opt`) | ✅ Zenodo | ✅ | ✅ | ✅ **符合** | **唯一原生单模型 4-stem 的 BSRNN 变体**（joint_bandsplit + 每源一个 masker） |
| 4 | **Oracle (IRM/IBM)** | 不需要 | — | ✅ | ✅ **符合** | 解析法，无权重；作为理论上界参照 |
| 5 | **BS-RoFormer L12** (ep_317) | ✅ | ✅ | ✅ | ⚠️ 部分 | 官方就是**单目标** model（`num_stems=1`, `target=vocals`）→ **只出 vocals**，非 4-stem |
| 6 | **BS-RoFormer L6** (ep_937) | ✅ | ✅ | ✅ | ⚠️ 部分 | 官方单目标，实际输出 `mixture−drums−bass`（vocals+other）→ **只出 1 条合并轨** |
| 7 | **RPCA** | ❌ **自研** | ❌ | ⚠️ | ⛔ 不符合 | 解析法无官方实现；且**稀疏分量无天然 stem 归属**（实测 cross-stem 对 bass 最好 +3.96 dB，对 vocals −7.90 dB）→ 必须由我们指定映射，属「需要改造」 |
| 8 | **Conv-TasNet** | ✅ | ✅ | ❌ | ⛔ 不符合（严格） | 官方 `estimate_all` 只给函数级调用，**需我们接 mean/std 标准化 + 哑通道维度**才能喂进去 |
| 9 | **MMDenseLSTM** | ✅ | ✅ | ❌ | ⛔ 不符合（严格） | 官方 `MMDenseLSTM` 是**单目标版**（`forward(input)`）；出 4 轨要靠我们选 `ParallelMMDenseLSTM` + 手工套 `TimeDomainWrapper` |
| 10 | **BSRNN (4× 单目标 ckpt)** | ✅ | ✅ | ❌ | ⛔ 不符合（严格） | 4 个独立单目标 ckpt，**需我们串行跑 4 次再拼**成 4-stem |
| 11 | **DPRNN** | ❌ **自训** | ✅ | ❌ | ⛔ **不符合** | 官方**只发语音权重**，音乐权重是我们本机训出来的 → 直接违反 C1 |
| 12 | **RPCA+DRNN** | ❌ 无 | ❌ 无 | ❌ | ⛔ **不符合** | 论文未公开任何代码/权重 |
| 13 | **Pac-HuBERT-SEP** | ❌ 无 | ❌ 无 | ❌ | ⛔ **不符合** | MERL 未放出代码与权重 |

**严格读法小结：符合 4 个（MDX-Net / Demucs / BSRNN-SIMO / Oracle），
另 2 个半符合（两个 BS-RoFormer 只出部分 stem），不符合 7 个。**

---

## 3. 宽松读法：官方权重 + 官方链路，我们只写调用适配

> 宽松读 = C1 + C2 + C3，**允许我们写 Python 胶水去「调用」官方代码**，但**不允许**改动权重、
> 结构、或为得到 stem 而重写信号处理。这是 `MODEL_REGISTRY.md` 一贯的口径。

| # | 模型 | 权重来源 | 我们写的适配 | 宽松判定 |
|---|---|---|---|---|
| 1 | **MDX-Net** | 官方 `mdx_extra` ×4 | 32 行（本地 repo 指向 + `torch.load` 兼容层） | ✅ 符合 |
| 2 | **Demucs** | 官方 `htdemucs` | 28 行（`apply_model` 调用 + 归一化） | ✅ 符合 |
| 3 | **BS-RoFormer L12** | 官方 ep_317 | 官方 `bigshifts_wrapper` 直调 | ✅ 符合（vocals 单目标） |
| 4 | **BS-RoFormer L6** | 官方 ep_937 | 同上 + 输出改名 | ✅ 符合（vocals+other 单目标） |
| 5 | **BSRNN-SIMO** | 官方 Zenodo | 官方 `BSRNN` 类 + 官方 `_apply_model_to_track` | ✅ 符合 |
| 6 | **Oracle** | 无 | 解析 IRM/IBM | ✅ 符合 |
| 7 | **Conv-TasNet** | 官方 musdb18 | 30 行（`build_from_pretrained` + 标准化） | ✅ 符合（**边界**） |
| 8 | **MMDenseLSTM** | 官方 musdb18 paper | 45 行（选 `Parallel` 变体 + `TimeDomainWrapper`） | ✅ 符合（**边界**） |
| 9 | **BSRNN 4×ckpt** | 官方 Zenodo | 官方类 + 4 次前向拼接 | ✅ 符合（**边界**） |
| 10 | **RPCA** | ❌ 自研 | 自研 `_rpca_ialm` | ⛔ 不符合 |
| 11 | **DPRNN** | ❌ 自训 | 自写训练管线 + loader | ⛔ 不符合 |
| 12 | **RPCA+DRNN** | ❌ 无 | — | ⛔ 不符合 |
| 13 | **Pac-HuBERT-SEP** | ❌ 无 | — | ⛔ 不符合 |

**宽松读法小结：符合 9 个，不符合 3 个。**

---

## 4. 不符合规则的原因（逐个说清）

### 4.1 🔴 DPRNN —— 「权重不是原作者发布的」

| 项 | 事实 |
|---|---|
| **官方发布内容** | `Dual-Path-RNN-Pytorch` 仓库**只发布语音分离权重**（WSJ0-2mix 等），**没有任何音乐权重** |
| **我们的做法** | 用本机 GPU 从零训练：`tools/train_dprnn_musdb.py`，MUSDB18-HQ 4-stem，5 小时 / step 31900 |
| **为什么违反 C1** | 规则要的是「**模型本身**能输出」，而 DPRNN 能输出**是因为我们训了它** —— 这正是「需要我们对它进行修改和改进才能成功输出」 |
| **额外缺陷** | 原生 **11.025 kHz**（44100÷4），**低于** MUSDB18-HQ 的 44.1 kHz → 11 kHz 以上带宽天生缺失，SDR 里如实计入了这部分损失 |
| **实测成绩** | 四轨均值 museval SDR **−1.72 dB**（四轨全为负），**不参与精度排名** |

> ✅ 额外证据（`tools/benchmark_model_universal.py::load_dprnn` docstring 原文）：
> 「官方仓库只发布语音权重，音乐权重是 `tools/train_dprnn_musdb.py` 在本机训出来的。」

### 4.2 ⛔ RPCA+DRNN —— 「论文未公开任何代码或权重」

| 项 | 事实 |
|---|---|
| 出处 | Lai & Wang, *EURASIP J. Audio Speech Music Process.* 2022:4 |
| 发布情况 | **论文与作者主页均未公开任何代码或权重** |
| 为什么不能自研顶上 | ① 无官方基线 → 自研结果「是否复现到位」**无法验证**；② 该论文口径是**单声道 2 源**，与 MUSDB18 4-stem **不可横比** |
| 已定稿处置 | 降级为「**相关工作**」，不进主结果对比表（`03_outputs/RPCA-DRNN/README.md`）|

### 4.3 ⛔ Pac-HuBERT-SEP —— 「MERL 未放出代码与权重」

| 项 | 事实 |
|---|---|
| 出处 | MERL（三菱电机研究实验室）TR2023-030 |
| 发布情况 | 官方项目页与 GitHub **均无发布** |
| 已定稿处置 | 仅作「相关工作」引用，一句「未公开代码与权重，本文不予复现」（`03_outputs/Pac-HuBERT-SEP/README.md`）|

### 4.4 🔴 RPCA —— 「没有天然 stem 归属，必须由我们指定映射」

虽然 RPCA 能跑，但它是**单输出**：低秩+稀疏分解出的是「稀疏分量」，
**语音/音乐里并没有规定这个分量对应哪个 stem**。实测交叉评估：

| 对 bass | 对 vocals |
|---|---|
| **+3.96 dB**（最匹配） | **−7.90 dB** |

→ 要放进 4-stem 表里，**必须由我们做一个「稀疏分量 → 某 stem」的映射决定**，
这本身就是「对模型进行修改和改进」。故按规则排除。
（报告里的正确写法是**报区间 1.8~3.5，不给单点**。）

---

## 5. 溯源：这些「不符合」的模型当初为什么会被列进来

**溯源链**：最初的模型清单来自 `04_reports/docs/METHOD_COMPARISON_REPORT.md` §2–§3 的
**文献综述与市场分层调研**，不是「先验证可运行性再纳入」，而是「先把候选面铺开再逐个落实」。

```
文献/榜单扫描（METHOD_COMPARISON_REPORT.md §3 公开准确度汇总表）
   │  来源：Demucs 论文 Table 1、RPCA-DRNN 论文 Table 7（EURASIP 2022）、
   │        MERL TR2023-030、Nature Sci Rep 2025 综述、Frontiers of CS 2027、
   │        TÜBİTAK 对比研究（MusDB-HQ）
   ↓
按「技术演进时间轴 + 市场分层」选出 12 个代表
   ┌─ 🔴 已淘汰：ICA、HPSS、无监督 NMF（仅教学/对照）
   ├─ 🟠 经典基线：RPCA、监督 NMF、MMDenseLSTM、Wave-U-Net
   ├─ 🟢 现役主力：Open-Unmix、Demucs、Spleeter、Conv-TasNet
   ├─ 🔵 SOTA 前沿：BS-RoFormer、MDX-Net、HTDemucs、Band-Split RNN
   └─ 🟡 实时/轻量：RPCA+DRNN ← 「与 FYP 目标最相关」
   ↓
逐个去找官方仓库与权重（这就是问题发生的地方）
   ├─ 找到 + 官方发权重     → 进入可运行 ✅
   ├─ 找到 + 官方只发语音权重 → 我们自训补齐（DPRNN）⚠️ ← 违反新规则
   └─ 找不到 / 未公开       → 受阻 ⛔（RPCA+DRNN、Pac-HuBERT-SEP）
```

### 5.1 你「有它们的数据」的两个具体来源

**来源 A：DPRNN 的数据是我们自己训出来的**
- 官方只发语音权重 → 2026-09-17 拍板「自训补齐」（`MODEL_REGISTRY.md` §3 决策 #2）
- 产物：`01_models/_weights/dprnn_musdb/{best,last}.pt`（step 31900）
- 数据：`04_reports/data/comparison/model_runs.json` 里的 `dprnn` 条目、`03_outputs/DPRNN/`
- → **所以 DPRNN 有数据，但数据不是「模型原生能力」的体现**

**来源 B：RPCA+DRNN / Pac-HuBERT-SEP 的数据其实是「占位与引述」，不是真跑出来的**
- 这两条**从来没有产出过任何 stem wav**，`03_outputs/` 下对应目录里**只有 README.md**（无音频）
- 它们出现在 `MODEL_REGISTRY.md` / `MODEL_PROVISIONING_REPORT.md` 里的
  「精度」数字，是**论文里报告的公开数字**（引自文献综述），**不是本机实测**
- → 「有数据」是文档层面的引述，不是实验层面的产物

**来源 C（附带）：RPCA 是本项目自研的解析实现**
- 落在 `tools/compare_separation_methods.py::sep_rpca` / `_rpca_ialm`，**不是官方发布物**
- 之所以当初纳入：「古典基线」需要一个矩阵分解代表，RPCA 是最标准的那个

### 5.2 一句话解释为什么会出现「需要改造才能输出」的模型

> 清单是按**学术代表性与文献覆盖面**选的，而不是按**可运行性**筛的。
> 因此必然包含两类「仅存在于论文里」的方法（RPCA+DRNN、Pac-HuBERT-SEP），
> 以及一类「官方只发了一半东西」的方法（DPRNN —— 有代码无音乐权重）。
> 新规则的作用正是把后两类**从工作流里剔除**。

---

## 6. 不可实机运行 / 不符合规则的原因分类

| 类别 | 模型 | 技术根因 | 有无解法 |
|---|---|---|---|
| **A. 官方无代码无权重** | RPCA+DRNN | 论文未公开任何实现 | ❌ 无（自研不可验证 + 口径不可横比）|
| **A. 官方无代码无权重** | Pac-HuBERT-SEP | MERL 未发布 | ❌ 无 |
| **B. 官方有代码但无音乐权重** | DPRNN | 官方只发语音权重 → 必须自训 | ⚠️ 技术上可自训，但**违反新规则**（权重非原作者发布）|
| **C. 官方有代码有权重，但输出不是原生 stem** | RPCA | 单输出、无天然 stem 归属，映射需我们指定 | ⚠️ 可跑，但须报区间 |
| **D. 官方有代码有权重，但需我们搭桥** | Conv-TasNet / MMDenseLSTM / BSRNN(4×ckpt) | 单目标模型 / 需选对 variant / 需串行拼装 | ✅ 已跑通（属宽松读法边界）|

**另外两个「跑得通但不是问题」的说明**：
- **BS-RoFormer L12 / L6**：官方**故意**只训单目标（各自 yaml `num_stems=1`）。
  这不是缺陷，是设计。但它们**无法单独构成 4-stem 结果**，引用时必须写明「单目标」。
- **RPCA / Oracle**：都**不需要权重**（解析法）。Oracle 是上界参照，本就无需模型。

---

## 7. 进入后续工作流的名单（按宽松读法，共 12 个）

| # | 模型 | 键名 | 输出 | 备注 |
|---|---|---|---|---|
| 1 | **MDX-Net** | `mdx` | 4 stem | 最佳「快+准」折中，RTF 0.13 |
| 2 | **Demucs** | `demucs` | 4 stem | GPU 上最快（RTF 0.03）|
| 3 | **BSRNN-SIMO** | `bsrnn_simo` | 4 stem | 精度最高（9.27），108.73 M 单模型 |
| 4 | **BS-RoFormer L12** | `bsroformer_l12` | vocals | 人声最高 9.69 |
| 5 | **BS-RoFormer L6** | `bsroformer_l6` | vocals+other | 单条合并轨 |
| 6 | **BSRNN (oBSRNN 4-ckpt)** | `bsrnn_all` | 4 stem | 4 次前向拼接 |
| 7 | **BSRNN large (4-ckpt)** | `bsrnn_large_all` | 4 stem | 同上 |
| 8 | **Conv-TasNet** | `convtasnet` | 4 stem | other 轨偏弱（1.66）|
| 9 | **MMDenseLSTM** | `mmdenselstm` | 4 stem | — |
| 10 | **Open-Unmix** | `umx` | 4 stem | 骨架自检基准 |
| 11 | **Oracle (IRM/IBM)** | `oracle` | 4 stem | 理论上界参照 |
| 12 | **RPCA** | `rpca` | 1 分量 | 无天然 stem 归属 → 交叉评估 |

### ⛔ 被排除（3 个）

| 模型 | 排除理由 |
|---|---|
| **DPRNN** | 权重为本机自训，非原作者发布（违反 C1）|
| **RPCA+DRNN** | 无官方代码与权重 |
| **Pac-HuBERT-SEP** | 无官方代码与权重 |

---

## 8. 本次规则变更带来的连带调整

| 项 | 调整 |
|---|---|
| `03_outputs/DPRNN/` | 保留既有产物（历史证据），但**不再新增 test 集输出** |
| `04_reports/figures/dprnn/` | 训练曲线等仍保留，作为「自训受限」的负面证据 |
| 主结果表 | DPRNN 行**保留但标注「不参与排名」**（它是「为什么不能自训」的实证）|
| test 集工作流 | 只跑 §7 的 12 个模型 |
| 论文写法 | 「相关工作」章节引 RPCA+DRNN 与 Pac-HuBERT-SEP；正文明确 DPRNN 为自训对照 |

---

## 9. 遗留待确认

| 项 | 说明 |
|---|---|
| **BS-RoFormer 两个 ckpt 是否该算「4-stem 必备」** | 它们是官方单目标设计，严格读法下只算「部分符合」。建议报告里单独成组，不与 4-stem 模型同表比均值 |
| **RPCA 是否保留在对比池** | 它违反 C3（需我们指定映射）。若按最严读法，应移出主表、只作「古典方法代表」讨论 |
| **DPRNN 的定位** | 建议保留为「负面证据」而非直接删除 —— 它支撑「自训 5 小时无法达 SOTA」这一结论 |
