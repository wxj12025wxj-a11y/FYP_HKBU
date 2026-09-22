# 自研算法实现索引

本目录**当前为空**：本项目自己写的分离算法目前都在 `tools/compare_separation_methods.py` 里
（统一基准骨架 `tools/benchmark_model_universal.py` 也直接复用它）。

| 算法 | 函数 | 说明 |
|---|---|---|
| RPCA | `sep_rpca()` / `_rpca_ialm()` | Inexact ALM 迭代；输出「稀疏分量」，**无天然 stem 归属**，需交叉评估决定映射到哪一轨 |
| NMF | `sep_nmf()` / `_nmf_fit()` / `_nmf_mask_to_audio()` | 非负矩阵分解 + 软掩码 |
| ICA | `sep_ica()` | 独立成分分析（项目历史结论：快但**有害**，ΔSDR 为负）|
| HPSS | `sep_hpss()` | 谐波/打击分离（`--which percussive` 取打击） |
| Oracle | `sep_nmf_oracle()` + 通用 oracle 掩码 | IRM / IBM 理论上界，`ref_map` 机制支持线性组合重建参考 |

> 为什么没搬进本目录：这些函数与评估骨架（`evaluate` / `si_sdr` / `SR` / `UMX_ORDER`）**强耦合**，
> 拆分会产生循环导入。等 FYP 定稿、不再频繁改动后可整体搬迁。
>
> 对应的第三方 oracle 实现（IRM/IBM/MWF）在 `01_models/_third_party/sigsep-mus-oracle/`。
> 想补 `RPCA+DRNN`（论文未公开代码的那个）时，建议直接在本目录新建 `rpca_drnn.py`，再从 `tools/` 引用。
