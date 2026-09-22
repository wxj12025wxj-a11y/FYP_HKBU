#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FYP_HKBU 路径中枢（single source of truth）。

设计原则
--------
1. **所有脚本只从这里取路径**，不要在自己文件里拼 `"weights"` / `"outputs"` 之类的字面量。
   以后目录再调整，只改这一个文件。
2. **新路径优先 + 旧路径回退**（`resolve()`）：迁移期间新旧两种布局都能跑，
   迁移中途中断也不会把脚本弄崩。
3. 提供 `setup_env()` 统一设置 `TORCH_HOME` / `HF_HOME`。

五大模块（2026-09-23 第二次重组）
--------------------------------
    01_models/      ① 模型：**每个模型一个独立文件夹**（模型卡 + weights 链接）
                    + 共享层 _third_party / _weights / _selfimpl
    02_databases/   ② 数据：MUSDB18-HQ、DNS-Challenge、Valentini
    03_outputs/     ③ 产物：**每个模型一个独立文件夹**（test/ + stage0_gate/）
    04_reports/     ④ 报告：按板块 —— separation / denoiser / cascade / _shared
                    每个板块下再分 docs / figures / data / html
    tools/          ⑤ 工具：脚本 + _logs + _scratch(含 torch·hf 缓存) + _bat

本次重组（相对 2026-09-16 版）的三处变化
---------------------------------------
* `05_misc/` 撤销：`logs/` -> `tools/_logs/`，`scratch/` -> `tools/_scratch/`，
  批处理启动器 -> `tools/_bat/`；**无关子项目 lyrics-game 与历史归档已移出项目**，
  落在 `E:\\FYP_HKBU_evicted\\`（未删除，见 `04_reports/_shared/docs/RESTRUCTURE_2026-09-23.md`）。
* `04_reports/` 由「按文件类型」改为「按板块」：板 1 = `separation/`，板 2 = `denoiser/`，
  级联 = `cascade/`，跨板块 = `_shared/`。
  **为兼容既有脚本，下面 `DOCS / HTML / FIGURES / DATA` 四个旧常量一律指向板 1（separation）**
  —— 现有脚本的产物本来就全属板 1。板 2 请用 `DEN_*`，共用资产用 `SHR_*`。
* `03_outputs/_test_run/<模型>/` -> `03_outputs/<模型>/test/`，
  `03_outputs/_stage0_gate/<模型>/` -> `03_outputs/<模型>/stage0_gate/`。

用法
----
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import PROJECT_ROOT, WEIGHTS, MUSDB18_ROOT, OUTPUTS, REPORTS, setup_env

    _paths.model_dir("BS-RoFormer-L12")     # 01_models/BS-RoFormer-L12
    _paths.out_dir("BS-RoFormer-L12")       # 03_outputs/BS-RoFormer-L12
    _paths.board_dir("denoiser")            # 04_reports/denoiser
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- 根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_EXPERIMENTAL = False

# ---------------------------------------------------------------- ① 模型
MODELS = PROJECT_ROOT / "01_models"
THIRD_PARTY = MODELS / "_third_party"
WEIGHTS = MODELS / "_weights"          # 权重的**唯一物理落点**
SELF_IMPL = MODELS / "_selfimpl"

# ---------------------------------------------------------------- ② 数据集
DATABASES = PROJECT_ROOT / "02_databases"
MUSDB18_ROOT = DATABASES / "MUSDB18-HQ"          # train / test / _smoketest
DNS_ROOT = DATABASES / "DNS-Challenge"           # clean / noise / dev_testset / impulse_responses
VALENTINI_ROOT = DATABASES / "Valentini"         # 板 2 降噪评测集（824 noisy + 824 clean 配对）

# ---------------------------------------------------------------- ③ 产物
OUTPUTS = PROJECT_ROOT / "03_outputs"

# ---------------------------------------------------------------- ④ 报告（按板块）
REPORTS = PROJECT_ROOT / "04_reports"
SEPARATION = REPORTS / "separation"      # 板 1 分离
DENOISER = REPORTS / "denoiser"          # 板 2 降噪 / 去混响
CASCADE = REPORTS / "cascade"            # 级联实验
SHARED = REPORTS / "_shared"             # 跨板块共用（总纲、模型画像、架构图、门禁）
SLIDES = REPORTS / "slides"

BOARD_NAMES = ("separation", "denoiser", "cascade", "_shared")

# 旧名兼容：现有脚本的产物全部属板 1，故这四个常量指向 separation。
DOCS = SEPARATION / "docs"
HTML = SEPARATION / "html"
FIGURES = SEPARATION / "figures"
DATA = SEPARATION / "data"
DATA_COMPARISON = DATA / "comparison"                  # 跨模型对比 json/csv
FIGURES_COMPARISON = FIGURES / "comparison"            # 跨模型对比图

# 板 2 / 级联 / 共用，显式命名
DEN_DOCS = DENOISER / "docs"
DEN_FIGURES = DENOISER / "figures"
DEN_DATA = DENOISER / "data"
DEN_HTML = DENOISER / "html"

CAS_DOCS = CASCADE / "docs"
CAS_FIGURES = CASCADE / "figures"
CAS_DATA = CASCADE / "data"
CAS_HTML = CASCADE / "html"

SHR_DOCS = SHARED / "docs"
SHR_FIGURES = SHARED / "figures"
SHR_DATA = SHARED / "data"
SHR_HTML = SHARED / "html"

# 模型画像 / 架构图（跨板共用，2026-09-23 起落在这里）
MODEL_ANALYSIS = SHR_DATA / "model_analysis"
MODEL_ARCH_FIGURES = SHR_FIGURES / "model_arch"

# ---------------------------------------------------------------- ⑤ 工具与运行支撑
TOOLS = PROJECT_ROOT / "tools"
LOGS = TOOLS / "_logs"
SCRATCH = TOOLS / "_scratch"
BAT_DIR = TOOLS / "_bat"

# ---------------------------------------------------------------- 缓存
CACHE = SCRATCH / ".cache"
TORCH_CACHE = CACHE / "torch"
HF_CACHE = CACHE / "huggingface"

# ---------------------------------------------------------------- 已移出项目（只读参考）
# 2026-09-23：lyrics-game（无关子项目）、历史归档、旧 tools 备份全部移出 FYP_HKBU，
# 未删除。若非必要不要从这里读；需要回滚时按 RESTRUCTURE 报告的清单反向 mv。
EVICTED = PROJECT_ROOT.parent / "FYP_HKBU_evicted"
ARCHIVE = EVICTED                       # 兼容旧名
LYRICS_GAME = EVICTED / "lyrics-game"

# ---------------------------------------------------------------- 常用具体路径
DEMUCS_SRC = THIRD_PARTY / "demucs"
BSRNN_SRC = THIRD_PARTY / "bsrnn"
BSROFORMER_SRC = THIRD_PARTY / "Music-Source-Separation-Training"
DENOISER_SRC = THIRD_PARTY / "denoiser"
AUDIO_ANALYSIS = TOOLS / "run_audio_analysis.py"

# 分离模型的标准入口（唯一骨架 + 15 模型 REGISTRY）
BENCH_UNIVERSAL = TOOLS / "benchmark_model_universal.py"
DENOISE_REGISTRY = TOOLS / "_denoise_registry.py"


# ---------------------------------------------------------------- 便捷函数
def board_dir(board: str) -> Path:
    """取板块目录。board ∈ {separation, denoiser, cascade, _shared}。"""
    b = str(board).strip().lower().lstrip("_")
    m = {"separation": SEPARATION, "sep": SEPARATION,
         "denoiser": DENOISER, "denoise": DENOISER,
         "cascade": CASCADE, "shared": SHARED}
    if b not in m:
        raise KeyError("unknown board %r; expected one of %s" % (board, BOARD_NAMES))
    return m[b]


def board_sub(board: str, sub: str) -> Path:
    """取板块下的子类目录，sub ∈ {docs, figures, data, html}。"""
    if sub not in ("docs", "figures", "data", "html"):
        raise KeyError("unknown sub %r; expected docs/figures/data/html" % sub)
    return board_dir(board) / sub


def model_dir(name: str) -> Path:
    """模型文件夹：01_models/<name>（含 README.md 模型卡与 weights/ 链接）。"""
    return MODELS / str(name)


def model_weights(name: str) -> Path:
    """模型文件夹内的权重入口（硬链接 / 目录联接，物理落点仍是 _weights/）。"""
    return MODELS / str(name) / "weights"


def out_dir(name: str) -> Path:
    """产物文件夹：03_outputs/<name>。"""
    return OUTPUTS / str(name)


def out_test(name: str) -> Path:
    """整曲分离结果：03_outputs/<name>/test/<歌曲名>/。"""
    return OUTPUTS / str(name) / "test"


def out_gate(name: str) -> Path:
    """阶段 0 门禁单文件输出：03_outputs/<name>/stage0_gate/。"""
    return OUTPUTS / str(name) / "stage0_gate"


# ---------------------------------------------------------------- 兼容回退
def resolve(*candidates: Path | str) -> Path:
    """返回第一个存在的候选路径；都不存在时返回最后一个（作为「将要创建」的目标）。

    迁移期用法：resolve(DATA_COMPARISON, PROJECT_ROOT / "outputs" / "comparison")
    新旧布局都能命中，谁在就用谁。
    """
    cands = [Path(c) for c in candidates]
    for c in cands:
        if c.exists():
            return c
    return cands[-1]


def comparison_dir() -> Path:
    """跨模型对比数据目录（新：04_reports/separation/data/comparison；旧：outputs/comparison）。"""
    return resolve(DATA_COMPARISON, PROJECT_ROOT / "outputs" / "comparison")


def musdb_root() -> Path:
    """MUSDB18-HQ 根（新：02_databases/MUSDB18-HQ；旧：datasets）。"""
    return resolve(MUSDB18_ROOT, PROJECT_ROOT / "datasets")


def demucs_out() -> Path:
    """Demucs 产物根（新：03_outputs/Demucs；旧：Output-demucs-main）。"""
    return resolve(OUTPUTS / "Demucs", PROJECT_ROOT / "Output-demucs-main")


def test_run_dir() -> Path:
    """整曲 test 结果的根（2026-09-23 起分散到各模型目录下）。

    旧布局 03_outputs/_test_run/<模型>/ 若有残留，回退到旧路径，
    让迁移期脚本两种布局都能跑。
    """
    return resolve(OUTPUTS, OUTPUTS / "_test_run")


def ensure_dirs() -> None:
    for d in (MODELS, THIRD_PARTY, WEIGHTS, SELF_IMPL,
              DATABASES, MUSDB18_ROOT, DNS_ROOT, VALENTINI_ROOT,
              OUTPUTS, REPORTS, SLIDES,
              SEPARATION, DENOISER, CASCADE, SHARED,
              *[board_sub(b, s) for b in BOARD_NAMES for s in ("docs", "figures", "data", "html")],
              TOOLS, LOGS, SCRATCH, BAT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_env() -> None:
    """统一设置缓存环境变量（必须在 import torch / 下载权重之前调用）。"""
    os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "mpl"))
    # 显存碎片治理（8 GiB 卡上跑 DPRNN 自训时必需）：
    # 参见 tools/train_dprnn_musdb.py 的 batch 实测 —— batch 8 峰值 5.88 GiB，
    # 不做可扩展段管理时容易因碎片触发 OOM。用 setdefault，尊重用户显式设置。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for d in (TORCH_CACHE, HF_CACHE, CACHE / "mpl"):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("PROJECT_ROOT = %s" % PROJECT_ROOT)
    rows = [("01_models", MODELS), ("02_databases", DATABASES),
            ("03_outputs", OUTPUTS), ("04_reports", REPORTS), ("tools", TOOLS)]
    for label, p in rows:
        mark = "OK " if p.is_dir() else "缺 "
        print("  %s%-14s %s" % (mark, label, p))
    print("  -- 04_reports 板块 --")
    for b in BOARD_NAMES:
        p = board_dir(b)
        print("     %s%-12s %s" % ("OK " if p.is_dir() else "缺 ", b, p))
    print("  -- 运行支撑 --")
    for label, p in [("tools/_logs", LOGS), ("tools/_scratch", SCRATCH),
                     ("tools/_bat", BAT_DIR), ("cache(torch)", TORCH_CACHE)]:
        print("     %s%-14s %s" % ("OK " if p.is_dir() else "缺 ", label, p))
    print("  -- 模型文件夹抽样 --")
    for name in ("BS-RoFormer-L12", "BS-RoFormer-L6", "denoiser-master64"):
        d = model_dir(name)
        w = model_weights(name)
        n = len(list(w.iterdir())) if w.is_dir() else 0
        print("     %s%-22s weights=%d 项" % ("OK " if d.is_dir() else "缺 ", name, n))
    print("  -> comparison_dir() = %s" % comparison_dir())
    print("  -> musdb_root()      = %s" % musdb_root())
    print("  -> demucs_out()      = %s" % demucs_out())
    print("  -> evicted           = %s" % EVICTED)
