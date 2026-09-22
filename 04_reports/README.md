# ④ 报告模块

按**板块**分文件夹，每个板块下再分 `docs / figures / data / html` 四类。

| 文件夹 | 归属 | 计划阶段 |
|---|---|---|
| `separation/` | **板 1 · 分离** | 阶段 1-4（模型画像、架构图、频谱面板、混响鲁棒性）|
| `denoiser/` | **板 2 · 降噪** | 阶段 5-7（降噪评测、去混响/去回声）|
| `cascade/` | **级联**（板 2 收尾）| 阶段 6-7（分离→降噪串联）|
| `_shared/` | 跨板块共用 | 总纲文档、模型画像、架构图、门禁数据 |
| `slides/` | 答辩材料 | 阶段 8 |

## 各板块内容

### `separation/`

板 1：模型本体画像、14 张架构原理图、频谱面板、混响/去混响实验

- `docs/` —— `CODE_REVIEW_REPORT.md`, `DEMUCS_REPORT.md`, `MEDIAN_TABLE.md`, `MEDIAN_TABLE_TEST.md`, `METHOD_COMPARISON_REPORT.md`, `MODEL_PROVISIONING_REPORT.md`, `MODEL_RULE_AUDIT_2026-09-19.md`, `MUSDB18_TEST_SWEEP_2026-09-22.md`, `PROJECT_AUDIT_2026-09-19.md`
- `figures/` —— `audio_analysis`, `comparison`, `demucs`, `dprnn`, `figs_time`, `reverb`, `spectrograms`, `test_run`
- `data/` —— `comparison`, `dprnn_training.bak-before-metafix.json`, `dprnn_training.json`, `reverb`
- `html/` —— `METHOD_VISUAL_COMPARISON_1.3.html`, `RUNTIME_BENCH_PPT.html`, `listen_compare.html`, `test_run_dashboard.html`

### `denoiser/`

板 2：降噪双口径评测表、去混响与去回声模型结果

- `docs/` —— （空，待阶段产出）
- `figures/` —— `denoise`
- `data/` —— `denoise`
- `html/` —— （空，待阶段产出）

### `cascade/`

级联实验：分离输出 → 降噪输入 的端到端结果

- `docs/` —— （空，待阶段产出）
- `figures/` —— `cascade`
- `data/` —— `cascade`
- `html/` —— （空，待阶段产出）

### `_shared/`

总纲（计划/路线图/复盘）、模型画像表、架构图数据、阶段 0 门禁

- `docs/` —— `FYP_Audio_Setup_Guide.md`, `FYP_PLAN_2026-09-22.md`, `FYP_RETROSPECTIVE_2026-09-22.md`, `FYP_ROADMAP_2026-09-22.md`, `MODEL_PROFILE_2026-09-22.md`
- `figures/` —— `README.md`, `_smoketest`, `model_arch`
- `data/` —— `model_analysis`
- `html/` —— `_qa_audio_probe.html`, `_superseded`
