# 模型本体画像：参数量 · 计算量 · 显存 · 时延

> 生成日：2026-09-22 ｜ 交付物：**D1 模型画像表** ｜ 数据源：`04_reports/_shared/data/model_analysis/model_profile.csv`

> 本表回答的是「模型要花多少代价」，与「分离得准不准」（见 `MEDIAN_TABLE*.md`）互为补充。


## 1. 测量口径（引用本表时必须一并声明）

| 项 | 取值 |
|---|---|
| 输入 | **44.1 kHz / 立体声 / 10 s**（统一构造，不来自数据集） |
| 参数量 | `sum(p.numel())`，取自模型实例，非配置文件声明值 |
| FLOPs | `torch.utils.flop_counter.FlopCounterMode` 实测，**口径 = 2×MACs** |
| 时延 | 预热 2 次后取 **5 次中位数**；CUDA 前后各同步一次 |
| 峰值显存 | `torch.cuda.max_memory_allocated()`，失败路径不计入 |
| 设备 | RTX 4070 Laptop **8 GiB**（sm_89），torch 2.14.0+cu126 |
| 隔离 | **每个模型一个独立子进程**（15 个模型同进程会显存耗尽） |

### ⚠️ FLOPs 是**下界**，不是等价比较

`FlopCounterMode` 只能统计被 torch 调度的算子（conv / linear / matmul / norm）。
**`torch.stft` 与融合的 `scaled_dot_product_attention` 不在统计范围内**。
因此对成本主要落在「STFT + 注意力」的 RoFormer 系列，本列数值只能作为下界；
跨架构的 FLOPs 比较必须配合**实测时延**一起看，不可单独下结论。

> 例：BS-RoFormer-L12 的 FLOPs（43.5 T）是 HTDemucs（508 G）的 **86 倍**，
> 但实测时延只差 **16 倍**（4.52 s vs 0.279 s）——差额就来自未被计入的算子。


### FLOPs 交叉验证：与 thop 的独立第二口径

`thop` 走 **nn.Module 钩子** 计数（口径同为 2×MACs），与 `FlopCounterMode` 的
**算子调度计数**机制不同，两者只在「纯 conv/linear 堆叠」的模型上才应当吻合。
下表只列 thop 能给出数值的模型——它给不出的，如实写原因，不用它去覆盖主口径。

| 模型 | FlopCounterMode (G) | thop (G) | 偏差 | 说明 |
|---|---:|---:|---:|---|
| BSRNN (oBSRNN, vocals) | 955.5 | 3428.8 | +258.9% | BSRNN @ 1,C,T; thop=2xMACs  [FLAG: dev>5% -> count coverage differs] |
| BSRNN (oBSRNN, 4-stem) | 3891.3 | 4596.9 | — | BSRNN @ 1,C,T; thop=2xMACs; AGGREGATED loader (4 nets) -> thop saw 1 net only, deviation NOT comparable |
| BSRNN large (vocals) | 276.8 | 2972.8 | +973.9% | BSRNN @ 1,C,T; thop=2xMACs  [FLAG: dev>5% -> count coverage differs] |
| BSRNN large (4-stem) | 1126.9 | 3985.2 | — | BSRNN @ 1,C,T; thop=2xMACs; AGGREGATED loader (4 nets) -> thop saw 1 net only, deviation NOT comparable |
| BSRNN SIMO (4-stem) | 1540.1 | 4978.7 | +223.3% | BSRNN @ 1,C,T; thop=2xMACs  [FLAG: dev>5% -> count coverage differs] |

> thop 能给出数值的模型只有 **5/22**。
> 其中**两边数的是同一个模型**、偏差可比较的只有 **3 个**，偏差区间 **223% ~ 974%**——**远超 D1 设定的 5%**，
> 也就是说该验收条款**未达成**。但原因不是实现错误：
> `FlopCounterMode` 数的是 torch 真正 dispatch 的算子，`thop` 数的是 「nn.Module 权重 × 注册到的输入形状」，
> 凡是 `forward` 里走 `F.conv*` / `einsum` / 多频带循环的地方，两者必然分叉。
> 另有 **2 个聚合权重**（同一模型由多个 ckpt 组成）thop 只拿到了其中一个子网，与被比较的整体口径不同源，故**不给偏差值**。
> 余下 **17 个**模型 thop 根本跑不动，原因逐条记在 `thop_crosscheck.json` 与 CSV 的 `thop_note` 列。
> **处置**：本表以 `FlopCounterMode` 为**唯一正式口径**（它对全部模型同源可比），
> thop 仅作覆盖度披露，**不用它去校正任何数字**；band-split RNN 一类的 FLOPs 本就随口径浮动 3~10 倍，引用时必须带此说明。


## 2. 全表（22 个模型/权重）

| 分组 | 模型 | 参数量 (M) | FLOPs (G) | 单次时延 (s) | 峰值显存 (MB) | 输出轨 |
|---|---|---:|---:|---:|---:|---:|
| 四轨分离模型 | BSRNN (oBSRNN, 4-stem) | 164.119 | 3891.3 | 5.7328 | 2637.4 | 4 |
|  | BSRNN SIMO (4-stem) | 108.728 | 1540.1 | 1.2105 | 2483.0 | 4 |
|  | Conv-TasNet (musdb18) | 13.395 | 1171.7 | 1.5221 | 606.6 | 4 |
|  | BSRNN large (4-stem) | 146.656 | 1126.9 | 2.9494 | 2567.9 | 4 |
|  | MDX-Net (mdx_extra) | 334.544 | 777.2 | 0.7373 | 1912.8 | 4 |
|  | Demucs (htdemucs) | 41.984 | 508.1 | 0.2749 | 599.7 | 4 |
|  | MMDenseLSTM (musdb18) | 5.487 | 294.5 | 0.3331 | 643.0 | 4 |
|  | DPRNN (自训, 4-stem) | 3.686 | 30.6 | 0.1030 | 579.7 | 4 |
|  | Open-Unmix (umxhq) | 35.577 | 14.3 | 0.2000 | 1227.9 | 4 |
| 单目标模型（仅目标 stem） | BS-RoFormer L12 (ep_317, vocals) | 159.758 | 43480.9 | 4.5208 | 3394.3 | 1 |
|  | BSRNN (oBSRNN, vocals) | 40.777 | 955.5 | 1.0814 | 1675.0 | 1 |
|  | BSRNN large (vocals) | 36.417 | 276.8 | 0.6294 | 1657.4 | 1 |
| 双目标模型 | BS-RoFormer L6 (ep_937, vocals+other) | 98.195 | 14975.2 | 2.1702 | 2087.4 | 1（=vocals+other 残差） |
| 降噪（板 2） | Mel-RoFormer-Denoise (aufr33) | 228.203 | 4030.5 | 0.9802 | 2011.2 | 1 |
|  | Mel-RoFormer-Denoise (aggr) | 228.203 | 4030.5 | 0.9732 | 2011.2 | 1 |
|  | denoiser dns64 (16 kHz) | 33.534 | 69.0 | 0.1263 | 343.2 | 1 |
|  | denoiser master64 (16 kHz) | 33.534 | 69.0 | 0.1269 | 343.2 | 1 |
|  | denoiser dns48 (16 kHz) | 18.868 | 39.1 | 0.0845 | 233.3 | 1 |
| 去混响 / 去回声（板 2） | Dereverb-Mel-RoFormer | 228.203 | 4030.5 | 0.9809 | 2011.2 | 1 |
|  | Dereverb-Echo-Mel-RoFormer | 208.880 | 3447.1 | 1.0182 | 1869.2 | 2 |
| 解析 / 无参方法 | IRM/IBM Oracle | 0.000 | 0（无算子调度） | 0.0000 | - | 4 |
|  | RPCA (Inexact-ALM) | 0.000 | 0（无算子调度） | 82.5301 | - | —（分量无天然归属） |

> ⚠️ **「输出轨」列不可直接当「目标 stem」读。** BS-RoFormer 的 L6 权重其 yaml 声明为 
> `other`，但实际输出是 **mixture − drums − bass**（即 vocals + other 的合成残差），
> 与 `L12` 的单目标 `vocals` 不是同一口径。凡引用本列做跨模型比较，必须回查 
> `04_reports/separation/docs/MEDIAN_TABLE_TEST.md` 的目标定义，不可仅看数字。


### 配套图表

- `04_reports/_shared/figures/model_arch/profile_params_flops.png` —— 双面板条形图：参数量 vs 对数坐标 FLOPs，按分组着色。
- `04_reports/_shared/figures/model_arch/profile_pareto.png` —— 气泡图：x = 时延（对数），y = 参数量，气泡面积 = 峰值显存。


## 3. 分组观察

**① 最省的一档是 denoiser 三兄弟，最贵的单点是 MDX-Net。**
MDX-Net 参数量 **334.5 M** 是全场最大，约等于 DPRNN（3.7 M）的 91 倍；
而 `denoiser dns48` 只用 **18.868 M**（板 2 最小）。

**② 「参数量小 ≠ 计算量小」在本项目里被反复验证。**
`Conv-TasNet` 只有 13.4 M，FLOPs 却达 1172 G（靠极长的时序卷积堆算力）；
`MMDenseLSTM` 仅 5.5 M，FLOPs 294 G。反之 `BSRNN-opt` 参数 40.8 M 而 FLOPs 仅 955 G。

**③ 板 2 的性价比反差极大。**
Mel-RoFormer 系（228.2 M / 4030 G / 0.98 s / 2011 MB）在参数上是 denoiser dns48（18.9 M）的约 12 倍，
时延约 12 倍（0.98 s vs 0.085 s），但它换来的 SI-SDR 改善是 **+9.88 dB vs +2.15 dB**（见 `stage0_gate.json`）。

**④ 最省时延 = denoiser dns48 (16 kHz)（0.085 s）；最费 = BSRNN (oBSRNN, 4-stem)（5.7328 s）。**
峰值显存最高的是 **BS-RoFormer L12 (ep_317, vocals)**（3394 MB），在 8 GiB 卡上需注意与其它进程共存时的余量。

**⑤ 解析方法的代价结构完全不同。**
`IRM/IBM Oracle` 无参数、无算子调度；`RPCA (Inexact-ALM)` 同样零参数，
但单次 10 s 处理耗时 **82.53 s**，远高于全场所有深度模型——它是纯 CPU 的矩阵迭代求解，这也是它在 50 首整曲扫描中必然触发超时的根因。


## 4. 板 2 门禁结果（`stage0_gate.json`）

探针：Valentini `p232_001.wav` 的真实含噪段（**非白噪声**——降噪器对白噪声输出 ~0 是正确行为，会伪装成「未加载成功」），SI-SDR 相对同段干净参考计算。

| 权重 | 类型 | 状态 | SI-SDR 输入 → 输出 | 改善 |
|---|---|---|---:|---:|
| Mel-RoFormer denoise | 降噪 | PASS | +15.48 → +25.36 dB | **+9.88** |
| Mel-RoFormer denoise (aggr) | 降噪 | PASS | +15.48 → +25.15 dB | **+9.67** |
| Mel-RoFormer dereverb (anvuew) | 去混响 | PASS | +15.48 → +19.07 dB | **+3.59** |
| Mel-RoFormer dereverb-echo (Sucial) | 去混响/回声 | PASS | +15.48 → +19.08 dB | **+3.60** |
| denoiser dns48 | 降噪 | PASS | +15.47 → +17.62 dB | **+2.15** |
| denoiser dns64 | 降噪 | PASS | +15.47 → +17.16 dB | **+1.69** |
| denoiser master64 | 降噪 | PASS | +15.47 → +15.62 dB | **+0.15** |

> 门禁 **7/7 PASS**（判据只认 `status` 字段，不看返回码/目录）。最强改善 
> **+9.88 dB**，最弱 **+0.15 dB**。


> 🔴 **待核（R2 / in-domain 检查）**：`master64` 的训练集据称**包含 Valentini**，按理应在 Valentini 上表现最好，但此处单样本改善仅 **+0.15 dB**，反而低于 dns48/dns64。
> 该结论**基于单段样本，不足以定论**——Stage 5 需在 Valentini testset 上按 824 对样本重测中位数，
> 并明确标注 in-domain / out-of-domain。


## 5. 与参考论文表的差异说明（答辩预案）

| 项 | 参考表 | 本机实测 | 差异原因 |
|---|---|---|---|
| Demucs 参数量 | 128 M | **41.98 M** | 参考表是**原版 Demucs (v1)**；本项目用 **HTDemucs**（混合时域/频域架构），是不同的模型 |
| DPRNN-TasNet | 官方权重 6.01 M | **本机自训 5 h**（3.69 M） | 官方未发布 MUSDB 分离权重，自训欠拟合（四轨 SDR 为负），仅作负面证据 |
| 评价片段 | 未声明 | 3 个能量均衡片段取中位数 | 已排除「开头低频主导」的能量陷阱片段 |

> 建议在图注中写明：**本表为同一任务的本机复现口径**，与参考论文的实现版本、
> 评价片段选择不同，数值不可逐格对齐；差异来源如上表，而非实现错误。


## 6. 复现命令

```bash
# 全量（分离 15 + 降噪/去混响 7）
python tools/_measure_model_profile.py --models all --device cuda

# 仅板 2
python tools/_measure_model_profile.py \
  --models mel_roformer_denoise,mel_roformer_denoise_aggr,\
           mel_roformer_dereverb,mel_roformer_dereverb_echo,\
           denoiser_dns48,denoiser_dns64,denoiser_master64

# 出图
python tools/_plot_model_profile.py
```
