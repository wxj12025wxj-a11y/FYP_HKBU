#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 tools/*.py 里散落的硬编码路径，统一改接到 tools/paths.py 的常量。

背景：目录重组后旧路径（weights/ datasets/ outputs/ reports/ logs/ demucs-main/ ...）
已不存在，20+ 个脚本共 100+ 处硬编码引用需要一次性重定向。

做法：
  1. 在 `PROJECT_ROOT = Path(r"E:\\FYP_HKBU")` 这类根定义处，注入 bootstrap 并改成 `_paths.PROJECT_ROOT`；
     同时用 `_paths.setup_env()` 取代各脚本自己写的 TORCH_HOME/HF_HOME setdefault。
  2. 把 `PROJECT_ROOT / "weights"` 这类表达式替换为对应的 `_paths.XXX` 常量。

用法：
    python tools/_rewire_paths.py            # dry-run，只统计
    python tools/_rewire_paths.py --apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
SKIP = {"paths.py", "_rewire_paths.py", "_migrate_layout.py"}

BOOTSTRAP = '''# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
'''

ROOT_RE = re.compile(
    r'^(?P<indent>[ \t]*)(?P<var>PROJECT_ROOT|ROOT_DIR|ROOT)\s*=\s*'
    r'Path\(\s*r?["\']E:[\\/]+FYP_HKBU["\']\s*\)[ \t]*$',
    re.M)

# 顺序敏感：长模式（带子路径）必须排在短模式之前。
# 前缀 `(?:_paths\.)?` 让脚本可重复运行（幂等）：`_paths.PROJECT_ROOT / "weights"`
# 与裸 `PROJECT_ROOT / "weights"` 都会被整体替换，不会出现 `_paths._paths.` 叠字。
_ROOTV = r'(?:_paths\.)?(?:PROJECT_ROOT|ROOT_DIR|ROOT)'
REPL: list[tuple[str, str]] = [
    (rf'{_ROOTV}\s*/\s*"datasets"\s*/\s*"_smoketest_musdb18"',
     '_paths.MUSDB18_ROOT / "_smoketest"'),
    (rf'{_ROOTV}\s*/\s*"datasets"\s*/\s*"(?P<sub>train|test)"',
     r'_paths.MUSDB18_ROOT / "\g<sub>"'),
    (rf'{_ROOTV}\s*/\s*"Output-demucs-main"\s*/\s*"figures"',
     '_paths.FIGURES / "demucs"'),
    (rf'{_ROOTV}\s*/\s*"outputs"\s*/\s*"comparison"', '_paths.comparison_dir()'),
    (rf'{_ROOTV}\s*/\s*"Output-demucs-main"', '_paths.OUTPUTS / "Demucs"'),
    (rf'{_ROOTV}\s*/\s*"weights"', '_paths.WEIGHTS'),
    (rf'{_ROOTV}\s*/\s*"datasets"', '_paths.MUSDB18_ROOT'),
    (rf'{_ROOTV}\s*/\s*"outputs"', '_paths.OUTPUTS'),
    (rf'{_ROOTV}\s*/\s*"logs"', '_paths.LOGS'),
    (rf'{_ROOTV}\s*/\s*"demucs-main"', '_paths.DEMUCS_SRC'),
    (rf'{_ROOTV}\s*/\s*"third_party"', '_paths.THIRD_PARTY'),
    (rf'{_ROOTV}\s*/\s*"reports"', '_paths.REPORTS'),
    (rf'{_ROOTV}\s*/\s*"tools"', '_paths.TOOLS'),
]

# 绝对路径字面量首段 → 常量
TAIL_MAP = {
    "weights": "_paths.WEIGHTS",
    "datasets": "_paths.MUSDB18_ROOT",
    "outputs": "_paths.OUTPUTS",
    "reports": "_paths.REPORTS",
    "logs": "_paths.LOGS",
    "demucs-main": "_paths.DEMUCS_SRC",
    "third_party": "_paths.THIRD_PARTY",
    "Output-demucs-main": '_paths.OUTPUTS / "Demucs"',
    "tools": "_paths.TOOLS",
}

CACHE_LINE_RE = re.compile(
    r'^[ \t]*os\.environ\.setdefault\(\s*["\'](?:TORCH_HOME|HF_HOME|MPLCONFIGDIR)["\'].*$',
    re.M)

# 残留的绝对路径字面量
ABS_RE = re.compile(r'r?["\']E:[\\/]+FYP_HKBU(?:[\\/][^"\']*)?["\']')


def rewire(text: str) -> tuple[str, int]:
    n = 0

    # 1) 根定义 → bootstrap + _paths.PROJECT_ROOT
    def _root(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return f"{m.group('indent')}{BOOTSTRAP}{m.group('indent')}{m.group('var')} = _paths.PROJECT_ROOT"

    text, k = ROOT_RE.subn(_root, text)
    if k == 0:
        # 没有根定义：可能已经有 bootstrap，或路径写在别的形式里
        if "import paths as _paths" not in text and ABS_RE.search(text):
            text = BOOTSTRAP + text
            n += 1

    # 2) 删掉各脚本自设的缓存环境变量（由 setup_env 统一负责）
    text, k = CACHE_LINE_RE.subn("", text)
    n += k

    # 3) 路径表达式 → 常量
    for pat, rep in REPL:
        text, k = re.subn(pat, rep, text)
        n += k

    # 4) 收尾：残留的 E:\FYP_HKBU 绝对路径
    def _abs(m: re.Match[str]) -> str:
        nonlocal n
        raw = m.group(0).strip('r"\'')
        tail = raw.replace("/", "\\").split("FYP_HKBU", 1)[1].strip("\\")
        n += 1
        if not tail:
            return "_paths.PROJECT_ROOT"
        parts = [p for p in tail.split("\\") if p]
        if len(parts) >= 2 and parts[0] == "datasets" and parts[1] == "_smoketest_musdb18":
            parts = ["datasets", "_smoketest"]
        head = TAIL_MAP.get(parts[0])
        if head:
            return head + "".join(f' / "{p}"' for p in parts[1:])
        return "_paths.PROJECT_ROOT / " + " / ".join(f'"{p}"' for p in parts)

    text = ABS_RE.sub(_abs, text)

    # 5) 清理连续空行
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total = 0
    for f in sorted(TOOLS.glob("*.py")):
        if f.name in SKIP:
            continue
        src = f.read_text(encoding="utf-8")
        if "E:\\FYP_HKBU" not in src and "E:/FYP_HKBU" not in src \
                and "import paths as _paths" not in src:
            continue
        new, n = rewire(src)
        if n:
            total += n
            print(f"  {f.name:40s} 改动 {n:3d} 处")
            if args.apply:
                f.write_text(new, encoding="utf-8")
    print(f"\n{'已写入' if args.apply else 'DRY-RUN'}：{total} 处改动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
