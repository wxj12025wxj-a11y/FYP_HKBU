# 图表目录

| 子目录 | 内容 | 生成脚本 |
|---|---|---|
| `comparison/` | 跨模型对比图：ΔSDR + **12 模型总览 3 张**（四轨均值条形 / 精度-速度散点 / 分轨对照）| `tools/plot_bench_10songs.py`、`tools/compare_separation_methods.py`、**`tools/plot_median_summary.py`** |
| `figs_time/` | 运行时长基准 7 张图（UMX vs Demucs vs 各方法）| `tools/plot_bench_10songs.py` |
| `demucs/` | Demucs 纵向深入分析 9 张图（波形/频谱/MFCC/裁剪前后）| `tools/run_demucs_deepdive.py` |
| `audio_analysis/` | 单曲音频分析 9 张图 | `tools/run_audio_analysis.py` |
| `_smoketest/` | 冒烟测试图（可随时删）| — |

`comparison/` 下的总览图数据源是 `04_reports/data/comparison/model_runs_median.json`（3 个均衡片段中位数），
与报告 §1.6 主结果表同源 —— 换片段重跑后请一并重生成：`python tools/plot_median_summary.py`。

> 报告里的图都已 base64 内联进 `04_reports/html/*.html`，单独分享 HTML 不需要带本目录。
