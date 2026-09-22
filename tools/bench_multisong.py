# -*- coding: utf-8 -*-
"""多曲验证：把同一批模型在 N 首歌上各跑一遍，便于取中位数。

单片段数字极易误导（例如某首歌 bass 稀疏，所有模型的 bass SDR 都会虚高），
所以论文级结论必须用「多曲中位数」。本脚本只是循环调用
tools/benchmark_model_universal.py，结果合并进同一个 model_runs.json。

用法：
    python tools/bench_multisong.py                      # 全部模型 × 3 首均衡片段
    python tools/bench_multisong.py ANiMAL               # 只跑含 ANiMAL 的曲目
    python tools/bench_multisong.py --models dprnn       # 只重跑 dprnn（收尾链用）

⚠️ 用 `--models` 只重跑单个模型时**仍走同一把单实例锁**，这是故意的：
   自训 DPRNN 收尾时需要「训练结束 → 重跑基准」，最容易出现两个实例同时写
   model_runs.json（2026-09-17 真实事故）。让它必须经过本脚本，锁才有意义。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
ROOT = _paths.PROJECT_ROOT
PY = r"C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\python.exe"
BENCH = _paths.TOOLS / "benchmark_model_universal.py"

MODELS = (
    "oracle,rpca,umx,mdx,convtasnet,mmdenselstm,demucs,dprnn,"
    "bsroformer_l12,bsroformer_l6,bsrnn_all,bsrnn_large_all,bsrnn_simo"
)

LOCK = _paths.LOGS / "_bench_multisong.lock"


def _pid_alive(pid: int) -> bool:
    """本机进程是否还活着（跨平台最小的实现）。"""
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        k32.CloseHandle(h)
        return True
    except AttributeError:                     # 非 Windows
        import os
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

# 取自 tools/_scan_balanced_clip.py：按「4 个 stem 能量占比的最小值」挑选的均衡片段。
# 不用 offset=0 是因为 MUSDB18 很多歌的开头是低频引子（例如 NightOwl 前 10 s
# bass 占 85%、鼓只占 1.4%），会让所有模型的 bass SDR 一起虚高到 20 dB 以上。
SONGS = [
    ("ANiMAL - Clinic A", 140.0),            # 23.0 / 18.5 / 22.9 / 35.6  ← 最均衡
    ("Creepoid - OldTree", 60.0),            # 42.4 / 15.6 / 26.7 / 15.3
    ("Dark Ride - Burning Bridges", 180.0),  # 20.7 / 36.9 / 28.4 / 14.0
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="多曲基准：循环调用 benchmark_model_universal.py，结果累加进同一份 model_runs.json")
    ap.add_argument("songs", nargs="*",
                    help="按歌名子串过滤曲目（默认全部 3 首均衡片段）。例：ANiMAL")
    ap.add_argument("--models", default=MODELS,
                    help=f"逗号分隔的模型名，默认全部。只重跑单个模型时用，例：--models dprnn")
    args = ap.parse_args()

    only = args.songs
    models = args.models
    songs = [s for s in SONGS if (not only or any(o.lower() in s[0].lower() for o in only))]
    if not songs:
        print(f"[bench_multisong] ⛔ 过滤后没有曲目（输入={only}）。可用曲目："
              + "; ".join(s[0] for s in SONGS))
        return 2
    log = _paths.LOGS / "_multisong.txt"

    # ---- 单实例锁 ----------------------------------------------------------
    # 2026-09-17 真实事故：一条「以为已被网络中断杀掉」的后台流水线其实还活着，
    # 训练一结束它就自己启动了基准，与我手动启动的实例**同时写 model_runs.json**。
    # 靠人记得「有没有在跑」不可靠 → 用 PID 锁兜底。
    import os
    _paths.LOGS.mkdir(parents=True, exist_ok=True)
    if LOCK.is_file():
        try:
            old = int(LOCK.read_text(encoding="utf-8").split()[0])
        except Exception:                                              # noqa: BLE001
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            print(f"[bench_multisong] ⛔ 已有实例在跑 (PID {old})，拒绝重复启动。\n"
                  f"   确认是残留时删除 {LOCK} 再试。")
            return 2
        print(f"[bench_multisong] 清理失效锁 (PID {old})")
    LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                    encoding="utf-8")

    try:
        return _run(songs, log, models)
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass


def _run(songs, log, models) -> int:
    with open(log, "a", encoding="utf-8") as lg:
        lg.write(f"\n===== multisong start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        lg.write(f"models = {models}\n")
        for i, (song, off) in enumerate(songs, 1):
            lg.write(f"--- [{i}/{len(songs)}] {song} @ {off:g}s ---\n")
            lg.flush()
            t0 = time.time()
            p = subprocess.run(
                [PY, str(BENCH), "--model", models, "--song", song,
                 "--duration", "10", "--offset", f"{off:g}"],
                cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            lg.write(p.stdout or "")
            if p.stderr:
                lg.write("\n[stderr]\n" + p.stderr[-3000:])
            lg.write(f"\n[{song}@{off:g}s] exit={p.returncode} 用时 {time.time()-t0:.0f}s\n")
            lg.flush()
    print(f"done -> {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
