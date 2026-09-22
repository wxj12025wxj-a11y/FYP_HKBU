#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01_models 结构再设计（2026-09-23 第三次）—— 精简卡片/权重联接 + 按板块分类。

背景
----
第二次重组后每个模型目录是「三件套」：`README.md`（模型卡）+ `code/` + `weights/`。
其中 **模型卡** 与 **`weights/` 联接** 已被证明是冗余层：

* `weights/` 只是指向 `01_models/_weights/` 的目录联接/硬链接，**全部脚本都直接从
  `01_models/_weights/**` 加载权重**（`paths.py::WEIGHTS`），从不读 `<模型>/weights/`；
* 模型卡的内容（论文出处 / 代码落点 / 权重真实路径 / 画像）已可从
  `model_profile.csv` + 本脚本生成的板块索引完全还原。

目标结构
--------
    01_models/
      README.md                 模块总览（重写）
      MODEL_REGISTRY.md         全模型登记表（保留，补新路径说明）
      separation/               板 1 · 分离（13 个模型）
        README.md               板块索引（吸收 13 张模型卡的内容）
        <模型>/code/<repo>       目录联接 -> _third_party/<repo>
      denoising/                板 2 · 降噪 / 去混响（6 个模型）
        README.md               板块索引（吸收 6 张模型卡的内容）
        <模型>/code/<repo>
      _third_party/             代码唯一物理落点（不动）
      _weights/                 权重唯一物理落点（不动，11 GB / 51 文件）
      _selfimpl/

安全设计
--------
* 删除前先把**每张模型卡的原文**归档到 `tools/_scratch/_archive_model_cards_2026-09-23/`，
  且把卡片信息结构化并入板块 README —— 删掉的是「散落的重复」，不是信息。
* `weights/` 删除前逐个文件核验 inode **确实存在于 `_weights/`**；任一文件找不到出处即中止。
* 每一步写 `_reclassify_manifest.json`，可逆向还原。
* 全部 print 用 ASCII 标记（GBK 控制台下 emoji 会崩）。

用法
----
    python tools/_reclassify_models_2026-09-23.py            # 干跑（默认）
    python tools/_reclassify_models_2026-09-23.py --execute  # 真正执行
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths  # noqa: E402

ROOT = _paths.PROJECT_ROOT
MODELS = _paths.MODELS
WEIGHTS = _paths.WEIGHTS
THIRD_PARTY = _paths.THIRD_PARTY
ARCHIVE = ROOT / "tools" / "_scratch" / "_archive_model_cards_2026-09-23"
MANIFEST = ROOT / "04_reports" / "_shared" / "data" / "_reclassify_manifest.json"

BOARD_DIRNAME = {"separation": "separation", "denoising": "denoising"}
BOARD_LABEL = {"separation": "板 1 · 分离（separation）",
               "denoising": "板 2 · 降噪 / 去混响 / 去回声（denoising）"}

# 19 个模型 -> 板块（唯一权威映射；paths.py 也引用同一份语义）
BOARD_OF: dict[str, str] = {
    # ---- 板 1 分离 ----
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
    # ---- 板 2 降噪 ----
    "denoiser-dns48": "denoising",
    "denoiser-dns64": "denoising",
    "denoiser-master64": "denoising",
    "Mel-RoFormer-Denoise": "denoising",
    "Mel-RoFormer-Dereverb": "denoising",
    "Mel-RoFormer-Dereverb-Echo": "denoising",
}

_ops: list[dict] = []


def log(tag: str, msg: str) -> None:
    print("[%-6s] %s" % (tag, msg))


def record(kind: str, src, dst) -> None:
    _ops.append({"kind": kind, "src": str(src), "dst": str(dst)})


# --------------------------------------------------------------------------
# 模型卡解析（把卡片原文变成结构化信息，供板块索引复用）
# --------------------------------------------------------------------------
def parse_card(text: str) -> dict:
    """解析既有模型卡的固定格式。缺失字段返回空，不抛异常。"""
    out = {"title": "", "note": "", "cite": [], "code": [], "weights": [],
           "board_line": "", "profile": []}
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        out["title"] = lines[0][2:].strip()

    sec = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("> 板块："):
            out["board_line"] = s
            continue
        m = re.match(r"^##\s+(.+)$", s)
        if m:
            sec = m.group(1).strip()
            continue
        if not s:
            continue
        if sec is None and not s.startswith("#"):
            if not out["note"] and not s.startswith(">"):
                out["note"] = s
            continue
        if sec == "论文 / 出处":
            if s.startswith("- "):
                out["cite"].append(s[2:].strip())
        elif sec == "代码位置":
            if s.startswith("- "):
                out["code"].append(s[2:].strip())
        elif sec and sec.startswith("本机权重"):
            if s.startswith("|") and "---" not in s:
                cells = [c.strip() for c in s.strip("|").split("|")]
                if len(cells) >= 2 and cells[0] not in ("权重文件",) and cells[0].startswith("`"):
                    out["weights"].append((cells[0].strip("`"), cells[1].strip("`")))
        elif sec and sec.startswith("本机实测画像"):
            if s.startswith("|") and "---" not in s:
                cells = [c.strip() for c in s.strip("|").split("|")]
                if cells and cells[0] not in ("条目",) and cells[0].startswith("`"):
                    out["profile"].append(cells)
    return out


def weight_inodes() -> dict:
    """_weights/ 下所有文件的 (dev, ino) -> 相对路径。"""
    d = {}
    for p in WEIGHTS.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
            except OSError:
                continue
            d[(st.st_dev, st.st_ino)] = str(p.relative_to(ROOT)).replace("\\", "/")
    return d


def verify_links_safe(model_dir: Path, w_inodes: dict) -> tuple[bool, list[str]]:
    """核验 <模型>/weights/ 里每个文件都能在 _weights/ 里找到同 inode 的真身。"""
    wd = model_dir / "weights"
    if not wd.exists():
        return True, []
    bad = []
    for p in wd.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError as e:
            bad.append("%s (stat 失败 %s)" % (p, e))
            continue
        if (st.st_dev, st.st_ino) not in w_inodes:
            bad.append("%s (%.2f MB) 在 _weights/ 中无同源文件" % (
                p.relative_to(ROOT), st.st_size / 1e6))
    return (not bad), bad


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def discover_models() -> list[tuple[str, str]]:
    """返回 [(模型名, 板块)]，按 SPECS 的板块映射；只取实际存在的扁平目录。"""
    found = []
    for d in sorted(MODELS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if d.name in BOARD_DIRNAME.values():
            continue
        if d.name not in BOARD_OF:
            log("warn", "未知模型目录（未在 BOARD_OF 中）：%s -> 归入 separation" % d.name)
            found.append((d.name, "separation"))
        else:
            found.append((d.name, BOARD_OF[d.name]))
    return found


def build_board_index(board: str, entries: list[dict]) -> str:
    """生成板块 README.md，吸收原模型卡的信息。"""
    label = BOARD_LABEL[board]
    L = ["# %s" % label, ""]
    L.append("> %d 个模型 ｜ 目录：`01_models/%s/<模型>/`" % (len(entries), board))
    L.append(">")
    L.append("> **本板块的模型目录只保留 `code/`**（指向 `01_models/_third_party/<repo>` 的目录联接，零额外占用）。")
    L.append("> 模型卡 `README.md` 与 `weights/` 联接层已于 2026-09-23 精简移除：")
    L.append("> 权重的**唯一物理落点**是 `01_models/_weights/`，全部脚本（含 `tools/paths.py`）都从那里直接加载。")
    L.append("")
    L.append("## 索引")
    L.append("")
    L.append("| 模型 | 条目 key | 参数量 | FLOPs | 时延(10 s) | 一句话定位 |")
    L.append("|---|---|---|---|---|---|")

    def fmt(v: str) -> str:
        return v if v and v != "nan" else "—"

    for e in entries:
        pf = e["card"]["profile"]
        if pf:
            params = fmt(pf[0][2]) if len(pf[0]) > 2 else "—"
            flops = fmt(pf[0][3]) if len(pf[0]) > 3 else "—"
            lat = fmt(pf[0][4]) if len(pf[0]) > 4 else "—"
        else:
            params = flops = lat = "—"
        keys = e["keys"] or "—"
        note = e["card"]["note"].replace("|", "\\|")
        L.append("| [`%s/`](%s/) | %s | %s | %s | %s | %s |" % (
            e["name"], e["name"], keys, params, flops, lat, note))
    L.append("")
    L.append("## 逐模型明细（原模型卡内容）")
    L.append("")
    for e in entries:
        c = e["card"]
        L.append("### %s" % e["name"])
        L.append("")
        if c["note"]:
            L.append("> %s" % c["note"])
            L.append("")
        if c["cite"]:
            L.append("**论文 / 出处**")
            L.append("")
            for x in c["cite"]:
                L.append("- %s" % x)
            L.append("")
        if c["code"]:
            L.append("**代码位置**")
            L.append("")
            for x in c["code"]:
                L.append("- %s" % x)
            L.append("")
        if c["weights"]:
            L.append("**权重（真实落点 `01_models/_weights/`）**")
            L.append("")
            L.append("| 文件 | 本机真实路径 |")
            L.append("|---|---|")
            for fn, real in c["weights"]:
                L.append("| `%s` | `%s` |" % (fn, real))
            L.append("")
        else:
            L.append("**权重**：无（解析算法，零参数）。实现见 `01_models/_selfimpl/README.md`。")
            L.append("")
        if c["profile"]:
            L.append("**本机实测画像**")
            L.append("")
            L.append("| 条目 | 输出轨数 | 参数量 | FLOPs | 单次前向(10 s) | 峰值显存 | 原生率 |")
            L.append("|---|---|---|---|---|---|---|")
            for row in c["profile"]:
                L.append("| " + " | ".join(row) + " |")
            L.append("")
        L.append("- 目录：`01_models/%s/%s/`（`code/` 为目录联接）" % (board, e["name"]))
        L.append("- 产物：`03_outputs/%s/`" % e["name"])
        L.append("- 主结果表：`04_reports/%s/docs/`" % ("separation" if board == "separation" else "denoiser"))
        L.append("- 架构原理图：`04_reports/_shared/figures/model_arch/`")
        L.append("")
    return "\n".join(L)


def build_models_readme(by_board: dict) -> str:
    L = ["# ① 模型模块", "",
         "顶层只分**两块板**，模型按板归类；每一块板下每个模型一个目录：", "",
         "```",
         "01_models/",
         "  separation/     板 1 · 分离        （%d 个模型）" % len(by_board["separation"]),
         "  denoising/      板 2 · 降噪/去混响  （%d 个模型）" % len(by_board["denoising"]),
         "  _third_party/   代码唯一物理落点（14 个仓库）",
         "  _weights/       权重唯一物理落点（51 个文件 / 11 GB）",
         "  _selfimpl/      自研算法索引",
         "```", "",
         "## 模型目录里有什么", "",
         "每个 `<板块>/<模型>/` 下**只有 `code/`** —— 指向 `01_models/_third_party/<repo>` 的",
         "**目录联接（junction）**，零额外占用。", "",
         "```",
         "separation/Demucs/",
         "  code/",
         "    demucs -> 01_models/_third_party/demucs",
         "```", "",
         "## 为什么没有「模型卡 + weights/ 联接」了（2026-09-23 精简）", "",
         "原先每个模型目录是「`README.md` 模型卡 + `code/` + `weights/` 联接」三件套。后两项被证明冗余：", "",
         "| 被删除的层 | 为什么冗余 | 现在从哪里取 |",
         "|---|---|---|",
         "| `<模型>/weights/` | 只是 `01_models/_weights/**` 的目录联接/硬链接，且**所有脚本都直接读 `_weights/`**（`tools/paths.py::WEIGHTS`），从不走这个别名 | `01_models/_weights/` |",
         "| `<模型>/README.md` | 内容（论文出处 / 代码落点 / 权重真实路径 / 画像）可从 `model_profile.csv` + 板块索引完全还原 | `<板块>/README.md` |", "",
         "> 删除前已逐个文件核验 `weights/` 里每个 inode 都存在于 `_weights/`（无一独有），",
         "> 并把 19 张模型卡原文归档到 `tools/_scratch/_archive_model_cards_2026-09-23/`。", "",
         "## 两块板", ""]
    for b in ("separation", "denoising"):
        L.append("### %s" % BOARD_LABEL[b])
        L.append("")
        L.append("| 模型 | 条目 key |")
        L.append("|---|---|")
        for e in by_board[b]:
            L.append("| [`%s/`](%s/%s/) | %s |" % (e["name"], b, e["name"], e["keys"] or "—"))
        L.append("")
        L.append("索引与逐模型明细：`01_models/%s/README.md`" % b)
        L.append("")
    L += ["## 自检", "",
          "```bash",
          'PY="C:\\Users\\jerry\\.workbuddy\\binaries\\python\\envs\\fyp_audio\\Scripts\\python.exe"',
          "",
          "# 1) 重建 / 补齐每个模型目录的 code/ 联接（幂等）",
          "$PY tools/_model_deploy_layout.py",
          "",
          "# 2) 核验：模型目录 + 代码联接 + 权重真源",
          "$PY tools/_verify_model_deployment.py --json 04_reports/_shared/data/model_deployment_audit.json",
          "```", "",
          "**环境依赖**：项目根 `requirements-frozen-2026-09-23.txt`（pip freeze 快照）。",
          "⚠️ 该快照**不含** PESQ（conda-forge 手工解包）与 CUDA 版 torch/torchaudio（须走",
          "`--index-url https://download.pytorch.org/whl/cu126`），文件头已注明。", "",
          "> 版 3（2026-09-23）：按板块分类 `separation/` + `denoising/`，",
          "> 移除模型卡与 `weights/` 联接层；详见 `04_reports/_shared/docs/MODEL_DEPLOYMENT_2026-09-23.md`。", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="真正执行（默认只干跑）")
    args = ap.parse_args()
    dry = not args.execute

    print("=" * 84)
    print("01_models 再设计 :: %s" % ("DRY-RUN（不修改任何文件）" if dry else "EXECUTE"))
    print("=" * 84)

    models = discover_models()
    if not models:
        log("ERR", "没有发现任何待分类的模型目录（可能已分类完成）")
        return 2
    log("info", "发现 %d 个模型目录" % len(models))

    # ---- 阶段 0：解析模型卡 ----
    entries: dict[str, dict] = {}
    for name, board in models:
        mdir = MODELS / name
        card_p = mdir / "README.md"
        text = card_p.read_text(encoding="utf-8") if card_p.is_file() else ""
        card = parse_card(text)
        keys = []
        for x in card["code"]:
            if "benchmark_model_universal.py::REGISTRY" in x:
                pass
        # 条目 key 从画像表首列取（最可靠）
        keys = ", ".join("`%s`" % r[0].strip("`") for r in card["profile"]) or ""
        entries[name] = {"name": name, "board": board, "card": card,
                         "keys": keys, "has_card": card_p.is_file(),
                         "has_weights": (mdir / "weights").exists(),
                         "raw_card": text}

    # ---- 阶段 1：核验 weights/ 无独有数据 ----
    print()
    print("-- 阶段 1：核验 <模型>/weights/ 是否全为 _weights/ 的别名 --")
    w_in = weight_inodes()
    log("info", "_weights/ 唯一 inode 数 = %d" % len(w_in))
    unsafe: list[tuple[str, list[str]]] = []
    n_links = 0
    for name, e in entries.items():
        ok, bad = verify_links_safe(MODELS / name, w_in)
        wd = MODELS / name / "weights"
        nf = sum(1 for p in wd.rglob("*") if p.is_file()) if wd.exists() else 0
        n_links += nf
        if ok:
            log("ok", "%-30s weights/ %2d 文件，全部同源" % (name, nf))
        else:
            log("ERR", "%-30s *** 存在独有数据，禁止删除 ***" % name)
            for b in bad:
                log("", "      %s" % b)
            unsafe.append((name, bad))
    if unsafe:
        print()
        log("ABORT", "有模型存在独有数据，已中止（未修改任何文件）")
        return 3
    log("ok", "全部 %d 个 weights/ 目录共 %d 个文件均可在 _weights/ 找到同源 → 可安全删除" % (
        sum(1 for e in entries.values() if e["has_weights"]), n_links))

    # ---- 阶段 2：计划 ----
    print()
    print("-- 阶段 2：迁移计划 --")
    by_board: dict[str, list[dict]] = {"separation": [], "denoising": []}
    for name, e in entries.items():
        by_board[e["board"]].append(e)
    for b in ("separation", "denoising"):
        print("  [%s] %d 个模型 -> 01_models/%s/" % (b, len(by_board[b]), b))
        for e in by_board[b]:
            acts = ["mv"]
            if e["has_card"]:
                acts.append("del README.md")
            if e["has_weights"]:
                acts.append("del weights/")
            print("      %-30s %s" % (e["name"], " + ".join(acts)))
    n_card = sum(1 for e in entries.values() if e["has_card"])
    n_wt = sum(1 for e in entries.values() if e["has_weights"])
    print()
    log("plan", "目录移动 %d ｜ 删除模型卡 %d 张 ｜ 删除 weights/ 联接 %d 个" % (
        len(entries), n_card, n_wt))

    # ---- 阶段 3：执行 ----
    print()
    print("-- 阶段 3：执行 --")
    if dry:
        log("dry", "跳过实际修改。加 --execute 真正执行。")
        return 0

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (MODELS / "_weights").mkdir(exist_ok=True)

    # 3a 归档卡片原文
    for name, e in entries.items():
        if not e["has_card"]:
            continue
        (ARCHIVE / ("%s.md" % name)).write_text(e["raw_card"], encoding="utf-8")
        record("archive-card", MODELS / name / "README.md", ARCHIVE / ("%s.md" % name))
    log("ok", "已归档 %d 张模型卡原文 -> %s" % (n_card, ARCHIVE))

    # 3b 建板块目录
    for b in ("separation", "denoising"):
        (MODELS / BOARD_DIRNAME[b]).mkdir(exist_ok=True)
        record("mkdir", "", MODELS / BOARD_DIRNAME[b])

    # 3c 删卡片 + 删 weights/，再移动
    for name, e in entries.items():
        mdir = MODELS / name
        if e["has_card"]:
            (mdir / "README.md").unlink()
            record("delete-card", mdir / "README.md", "")
        if e["has_weights"]:
            shutil.rmtree(mdir / "weights")
            record("delete-weights-links", mdir / "weights", "")
        dst = MODELS / BOARD_DIRNAME[e["board"]] / name
        if dst.exists():
            log("warn", "目标已存在，跳过移动：%s" % dst)
            continue
        os.rename(str(mdir), str(dst))
        record("move", mdir, dst)
    log("ok", "卡片与 weights/ 已删除，%d 个模型目录已移入板块" % len(entries))

    # 3d 生成板块索引
    for b in ("separation", "denoising"):
        readme = MODELS / BOARD_DIRNAME[b] / "README.md"
        readme.write_text(build_board_index(b, by_board[b]), encoding="utf-8")
        record("write-index", "", readme)
        log("ok", "写入板块索引 %s（%d 个模型）" % (
            readme.relative_to(ROOT), len(by_board[b])))

    # 3e 重写模块总览
    readme = MODELS / "README.md"
    readme.write_text(build_models_readme(by_board), encoding="utf-8")
    record("write-readme", "", readme)
    log("ok", "重写 %s" % readme.relative_to(ROOT))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "when": "2026-09-23",
        "board_of": BOARD_OF,
        "archive": str(ARCHIVE),
        "ops": _ops,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    log("ok", "清单 -> %s（%d 条操作）" % (MANIFEST.relative_to(ROOT), len(_ops)))
    print()
    log("done", "执行完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
