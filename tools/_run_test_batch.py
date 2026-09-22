# -*- coding: utf-8 -*-
"""在**进程内**以 UTF-8 启动 test 集批次 —— 绕开 bash 包装层导致的编码/锁问题。

为什么不用 `nohup python tools/bench_musdb_test.py ... &`
--------------------------------------------------------
本机 bash -> python 的链路里，子进程 stdout 会退回 GBK（`python -X utf8` 也压不住），
一旦脚本 print 带 emoji 就 `UnicodeEncodeError` 崩掉；更糟的是崩在**写锁之后**，
留下陈旧 `_bench_musdb_test.lock`，后续启动会被「已有实例在跑」挡住。

本脚本在**已配置好 UTF-8 的当前解释器内**：
  1. 强制 `sys.stdout/stderr` 与 `PYTHONIOENCODING` 为 utf-8
  2. 清掉陈旧锁（PID 已死或本进程复用时）
  3. 通过 `subprocess.Popen(..., stdout=open(...,'w',encoding='utf-8'))`
     直接把子进程输出写进 UTF-8 文本文件，不经 shell

用法：
    python tools/_run_test_batch.py fast      # 9 个快模型（~7.0 h）
    python tools/_run_test_batch.py slow      # 3 个慢模型（~32 h，建议先优化）
    python tools/_run_test_batch.py all       # 全部 12 个（~39 h）
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths  # noqa: E402

ROOT = _paths.PROJECT_ROOT
LOGS = _paths.LOGS
LOCK = LOGS / "_bench_musdb_test.lock"
TOOL = _paths.TOOLS / "bench_musdb_test.py"

FAST = "oracle,umx,mdx,demucs,convtasnet,mmdenselstm,bsroformer_l12,bsroformer_l6,bsrnn_simo"
SLOW = "bsrnn_all,bsrnn_large_all,rpca"

GROUPS = {"fast": FAST, "slow": SLOW, "all": FAST + "," + SLOW}


def _pid_alive(pid: int) -> bool:
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:
        return False


def clear_stale_lock() -> None:
    if not LOCK.is_file():
        return
    try:
        pid = int(LOCK.read_text(encoding="utf-8").split()[0])
    except Exception:
        pid = 0
    if pid and pid != os.getpid() and _pid_alive(pid):
        raise SystemExit(f"[!] PID {pid} 仍在运行，拒绝启动第二个实例。")
    LOCK.unlink()
    print(f"[i] 清掉陈旧锁（PID {pid} 已死）")


def _pick_python() -> str:
    """选出**带 torch/numpy 的 venv 解释器**。

    ⚠️ 坑：直接用 `sys.executable` 是不可靠的。
    本机 WorkBuddy 调用链里，`python` 解析到的是 **系统 Python 3.13.12**（无 numpy），
    而项目依赖全在 venv `fyp_audio` 里。子进程一启动就
    `ModuleNotFoundError: No module named 'numpy'`，而外层脚本只会打印一条
    「ERR 0.5s」——看起来像推理失败，实则根本没跑起来。

    优先序：环境变量 FYP_PY -> 已知 venv 路径 -> 逐个探测能用 numpy 的解释器。
    """
    cands: list[str] = []
    if os.environ.get("FYP_PY"):
        cands.append(os.environ["FYP_PY"])
    cands.append(str(Path.home() / ".workbuddy/binaries/python/envs/fyp_audio/Scripts/python.exe"))
    cands.append(sys.executable)
    cands.append("python")

    for exe in cands:
        if not exe:
            continue
        if os.path.sep in exe and not Path(exe).is_file():
            continue
        try:
            r = subprocess.run([exe, "-c", "import numpy, torch; print('ok')"],
                               capture_output=True, text=True, timeout=90)
        except Exception:
            continue
        if r.returncode == 0 and "ok" in (r.stdout or ""):
            return exe
    raise SystemExit("[!] 找不到带 numpy/torch 的解释器，请设 FYP_PY")


def main() -> int:
    group = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if group not in GROUPS:
        raise SystemExit(f"未知分组 {group}，可选 {list(GROUPS)}")

    LOGS.mkdir(parents=True, exist_ok=True)
    clear_stale_lock()

    py = _pick_python()
    out_path = LOGS / f"_musdb_test_{group}.out"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [py, "-X", "utf8", str(TOOL),
           "--models", GROUPS[group], "--songs", "all", "--progress-every", "5"]
    print(f"[>] 解释器: {py}")
    print(f"[>] 启动 {group} 批次：{GROUPS[group]}")
    print(f"    输出 -> {out_path}")

    fh = open(out_path, "w", encoding="utf-8", buffering=1)
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                         env=env, text=True, encoding="utf-8")
    print(f"    PID {p.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
