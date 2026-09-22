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

五大模块（2026-09-23 第三次重组）
--------------------------------
    01_models/      ① 模型：**按板块 separation / denoising 分组**；
                    每个模型目录下**只保留 `code/`**（指向 _third_party 的目录联接）
                    + 共享层 _third_party / _weights / _selfimpl
    02_databases/   ② 数据：MUSDB18-HQ、DNS-Challenge、Valentini
    03_outputs/     ③ 产物：**每个模型一个独立文件夹**（test/ + stage0_gate/）
    04_reports/     ④ 报告：按板块 —— separation / denoiser / cascade / _shared
                    每个板块下再分 docs / figures / data / html
    tools/          ⑤ 工具：脚本 + _logs + _scratch(含 torch·hf 缓存) + _bat

历次重组
--------
* 2026-09-16：`05_misc/` 撤销 → `tools/_logs` `tools/_scratch` `tools/_bat`；
  无关子项目 lyrics-game 与历史归档移出到 `E:\\FYP_HKBU_evicted\\`。
* 2026-09-23（第二次）：`04_reports/` 由「按文件类型」改为「按板块」
  （板 1 `separation/`、板 2 `denoiser/`、级联 `cascade/`、跨板 `_shared/`）；
  `03_outputs/_test_run/<模型>/` → `03_outputs/<模型>/test/`。
* 2026-09-23（第三次，本次）：`01_models/<模型>/` → `01_models/<板块>/<模型>/`
  （板块 = `separation` / `denoising`）。同时**移除两个冗余层**：
  - `<模型>/weights/`（只是 `_weights/` 的目录联接/硬链接，脚本从不读它）——权重统一在 `WEIGHTS`；
  - `<模型>/README.md` 模型卡——内容并入 `01_models/<板块>/README.md` 与
    `04_reports/_shared/data/model_analysis/model_profile.csv`，
    原文归档在 `tools/_scratch/_archive_model_cards_2026-09-23/`。

用法
----
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import PROJECT_ROOT, WEIGHTS, MUSDB18_ROOT, OUTPUTS, REPORTS, setup_env

    _paths.model_dir("BS-RoFormer-L12")     # 01_models/separation/BS-RoFormer-L12
    _paths.model_code_dir("Demucs")         # 01_models/separation/Demucs/code
    _paths.board_of_model("Demucs")         # "separation"
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

# 模型按板块分组（2026-09-23 第三次重组）。
# 目录名：separation = 板 1 分离；denoising = 板 2 降噪 / 去混响 / 去回声。
SEP_MODELS = MODELS / "separation"
DEN_MODELS = MODELS / "denoising"

MODEL_BOARD: dict[str, str] = {
    # ---- 板 1 · 分离（13） ----
    "BS-RoFormer-L12": "separation",
    "BS-RoFormer-L6": "separation",
    "BSRNN-opt": "separation",
    "BSRNN-large": "separation",
    "BSRNN-SIMO": "separation",
    "Demucs": "separation",
    "MDX-Net": "separation",
    "Open-Unmix": "separation",
    "Conv-TasNet": "separation",
    "MMDenseLSTM": "separation",
    "DPRNN": "separation",
    "RPCA": "separation",
    "Oracle-IRM": "separation",
    # ---- 板 2 · 降噪 / 去混响 / 去回声（6） ----
    "denoiser-dns48": "denoising",
    "denoiser-dns64": "denoising",
    "denoiser-master64": "denoising",
    "Mel-RoFormer-Denoise": "denoising",
    "Mel-RoFormer-Dereverb": "denoising",
    "Mel-RoFormer-Dereverb-Echo": "denoising",
}

# 板块名 -> 目录（模型侧；板名别名见 _MODEL_BOARD_ALIAS）
MODEL_BOARDS = {"separation": SEP_MODELS, "denoising": DEN_MODELS}
_MODEL_BOARD_ALIAS = {
    "separation": "separation", "sep": "separation", "board1": "separation", "board-1": "separation",
    "denoising": "denoising", "denoise": "denoising", "denoiser": "denoising",
    "board2": "denoising", "board-2": "denoising",
}

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

# 模型卡原文归档（2026-09-23 精简时从各模型目录移除）
MODEL_CARDS_ARCHIVE = PROJECT_ROOT / "tools" / "_scratch" / "_archive_model_cards_2026-09-23"

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
    """取报告板块目录。board ∈ {separation, denoiser, cascade, _shared}。"""
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


def board_of_model(name: str) -> str:
    """模型所属板块：`separation` 或 `denoising`。未登记的模型一律归 separation。"""
    return MODEL_BOARD.get(str(name), "separation")


def model_board_dir(board: str) -> Path:
    """模型侧板块目录：01_models/<separation|denoising>。"""
    b = _MODEL_BOARD_ALIAS.get(str(board).strip().lower())
    if b is None:
        raise KeyError("unknown model board %r; expected separation / denoising" % (board,))
    return MODEL_BOARDS[b]


def model_dir(name: str) -> Path:
    """模型文件夹：`01_models/<板块>/<name>`（目录内只含 `code/` 联接）。

    兼容回退：迁移期若旧扁平路径 `01_models/<name>/` 仍存在且新路径不存在，
    返回旧路径，保证半迁移状态下脚本仍能跑。
    """
    name = str(name)
    new = MODEL_BOARDS[board_of_model(name)] / name
    if new.is_dir():
        return new
    flat = MODELS / name
    if flat.is_dir():
        return flat
    return new                      # 都不存在 -> 返回「将要创建」的新路径


def model_code_dir(name: str) -> Path:
    """模型文件夹内的代码入口：`<模型目录>/code`（一组指向 _third_party 的目录联接）。

    2026-09-23 起，模型目录**不再**有 `weights/` 与 `README.md`：
    权重统一取 `WEIGHTS`，模型说明见 `01_models/<板块>/README.md`。
    """
    return model_dir(name) / "code"


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
    for d in (MODELS, THIRD_PARTY, WEIGHTS, SELF_IMPL, SEP_MODELS, DEN_MODELS,
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
    print("  -- 01_models 板块（2026-09-23 起模型按板分类） --")
    for b, root in MODEL_BOARDS.items():
        n = len([x for x in root.iterdir() if x.is_dir() and not x.name.startswith("_")]) \
            if root.is_dir() else 0
        print("     %s%-12s %-46s %d 个模型" % ("OK " if root.is_dir() else "缺 ", b, root, n))
    for label, p in [("_third_party", THIRD_PARTY), ("_weights", WEIGHTS),
                     ("_selfimpl", SELF_IMPL), ("MODEL_REGISTRY.md", MODELS / "MODEL_REGISTRY.md")]:
        print("     %s%-16s %s" % ("OK " if p.exists() else "缺 ", label, p))
    print("  -- 运行支撑 --")
    for label, p in [("tools/_logs", LOGS), ("tools/_scratch", SCRATCH),
                     ("tools/_bat", BAT_DIR), ("cache(torch)", TORCH_CACHE)]:
        print("     %s%-14s %s" % ("OK " if p.is_dir() else "缺 ", label, p))
    print("  -- 模型文件夹抽样（目录内应只有 code/） --")
    for name in ("BS-RoFormer-L12", "BS-RoFormer-L6", "denoiser-master64"):
        d = model_dir(name)
        c = model_code_dir(name)
        n = len(list(c.iterdir())) if c.is_dir() else 0
        inner = sorted(x.name for x in d.iterdir()) if d.is_dir() else []
        print("     %s%-24s %-46s 内含 %s  code=%d 联接" % (
            "OK " if d.is_dir() else "缺 ", name, d.relative_to(PROJECT_ROOT), inner, n))
    print("  -> comparison_dir() = %s" % comparison_dir())
    print("  -> musdb_root()      = %s" % musdb_root())
    print("  -> demucs_out()      = %s" % demucs_out())
    print("  -> evicted           = %s" % EVICTED)
