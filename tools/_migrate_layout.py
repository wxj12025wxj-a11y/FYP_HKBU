#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一次性目录重组：旧布局 → 五大模块布局。

安全设计
--------
- **只做 move，不做 delete**。同盘 move 在 NTFS 上是元数据操作（毫秒级），不复制数据。
- **幂等**：源不存在就跳过，目标已存在就下沉合并，可反复运行。
- **先 dry-run 后执行**：加 `--apply` 才真正动。
- 全程写日志到 `05_misc/logs/_migrate_layout.txt`。

用法
----
    python tools/_migrate_layout.py            # dry-run（只打印计划）
    python tools/_migrate_layout.py --apply    # 真正执行
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "01_models"
DATABASES = ROOT / "02_databases"
OUTPUTS = ROOT / "03_outputs"
REPORTS = ROOT / "04_reports"
MISC = ROOT / "05_misc"

LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LINES.append(msg)


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def move(src: Path, dst: Path, apply: bool) -> None:
    if not src.exists():
        log(f"  ·  跳过（源不存在）    {rel(src)}")
        return
    if dst.is_dir() and src.is_dir():
        log(f"  ⇢  合并目录          {rel(src)}  →  {rel(dst)}/")
        if apply:
            for child in sorted(src.iterdir()):
                move(child, dst / child.name, apply)
            try:
                src.rmdir()
            except OSError:
                pass
        return
    if dst.exists():
        log(f"  ·  跳过（目标已存在）  {rel(dst)}")
        return
    log(f"  →  {rel(src):58s} →  {rel(dst)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def build_plan() -> list[tuple[Path, Path]]:
    plan: list[tuple[Path, Path]] = []

    # ---------- ① 模型：第三方仓库 ----------
    tp = ROOT / "third_party"
    if tp.is_dir():
        for repo in sorted(tp.glob("*")):
            plan.append((repo, MODELS / "_third_party" / repo.name))
    plan.append((ROOT / "demucs-main", MODELS / "_third_party" / "demucs"))

    # ---------- ① 模型：预训练权重 ----------
    w = ROOT / "weights"
    if w.is_dir():
        for sub in sorted(w.glob("*")):
            plan.append((sub, MODELS / "_weights" / sub.name))

    # ---------- ② 数据集：MUSDB18-HQ ----------
    for sub, tgt in [("train", "train"), ("test", "test"),
                     ("_smoketest_musdb18", "_smoketest")]:
        plan.append((ROOT / "datasets" / sub, DATABASES / "MUSDB18-HQ" / tgt))

    # ---------- ③ 产物：逐模型下沉（comparison 单独处理） ----------
    old_out = ROOT / "outputs"
    if old_out.is_dir():
        for sub in sorted(old_out.iterdir()):
            if sub.name == "comparison":
                continue
            plan.append((sub, OUTPUTS / sub.name))

    # ---------- ③④ 拆分 comparison：数据 → report/data，图 → report/figures ----------
    cmp_old = old_out / "comparison"
    if cmp_old.is_dir():
        for f in sorted(cmp_old.iterdir()):
            if f.is_dir():
                if f.name == "figs_time":
                    plan.append((f, REPORTS / "figures" / "figs_time"))
                else:
                    plan.append((f, REPORTS / "figures" / f.name))
            elif f.suffix.lower() in (".png", ".jpg", ".svg", ".pdf"):
                plan.append((f, REPORTS / "figures" / "comparison" / f.name))
            else:
                plan.append((f, REPORTS / "data" / "comparison" / f.name))

    # ---------- ③④ Demucs 产物 ----------
    dm = ROOT / "Output-demucs-main"
    plan.append((dm / "stems", OUTPUTS / "Demucs" / "stems"))
    plan.append((dm / "figures", REPORTS / "figures" / "demucs"))
    plan.append((dm / "DEMUCS_REPORT.md", REPORTS / "docs" / "DEMUCS_REPORT.md"))
    plan.append((dm / "metrics.json", REPORTS / "data" / "comparison" / "demucs_metrics.json"))
    plan.append((dm / "run.log", MISC / "logs" / "demucs_deepdive_run.log"))

    # ---------- ④ 报告 ----------
    rep = ROOT / "reports"
    if rep.is_dir():
        for f in sorted(rep.iterdir()):
            if f.is_dir():
                if f.name == "_smoketest_figures":
                    plan.append((f, REPORTS / "figures" / "_smoketest"))
                elif f.name == "figures":
                    plan.append((f, REPORTS / "figures"))
                else:
                    plan.append((f, REPORTS / "docs" / f.name))
            elif f.suffix.lower() == ".html":
                plan.append((f, REPORTS / "html" / f.name))
            else:
                plan.append((f, REPORTS / "docs" / f.name))

    # ---------- ⑤ 杂项 ----------
    for sub in ("logs", "_archive", "lyrics-game"):
        plan.append((ROOT / sub, MISC / sub))
    for sub in (".cache", ".tmp_dl", ".tmp_pesq", "__pycache__", "_par_download.txt"):
        plan.append((ROOT / sub, MISC / "scratch" / sub))

    # ---------- 根目录散落文件 ----------
    plan.append((ROOT / "FYP_Audio_Setup_Guide.md", REPORTS / "docs" / "FYP_Audio_Setup_Guide.md"))
    plan.append((ROOT / "run_audio_analysis.py", ROOT / "tools" / "run_audio_analysis.py"))

    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正执行（默认 dry-run）")
    args = ap.parse_args()

    log(f"===== 目录重组 {'APPLY' if args.apply else 'DRY-RUN'}  "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    plan = build_plan()
    for src, dst in plan:
        move(src, dst, args.apply)
    log("")
    log(f"计划条目数：{len(plan)}")

    if args.apply:
        (MISC / "logs").mkdir(parents=True, exist_ok=True)
        with open(MISC / "logs" / "_migrate_layout.txt", "a", encoding="utf-8") as fh:
            fh.write("\n".join(LINES) + "\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
