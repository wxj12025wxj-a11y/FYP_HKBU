#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DPRNN 自训「收尾链」：等训练结束 → 重跑基准 → 刷新聚合/图/HTML。

为什么需要这个脚本
------------------
`tools/train_dprnn_musdb.py --minutes 300` 要跑 5 小时，训完的那一刻才是真正需要
动作的时刻。中途靠人守着不现实，靠「一个后台 nohup 等着」又出过事故：

    2026-09-17 真实事故：一条「以为已被网络中断杀掉」的后台流水线其实还活着，
    训练一结束它就自己启动了基准，与手动启动的实例**同时写 model_runs.json**。
    → 本脚本因此强制走 `bench_multisong.py --models dprnn`（它自带单实例锁），
      并且自己再加一把锁。两层锁都是「宁可拒绝启动，也不并发写」。

三重「训练是否结束」判据（缺一不可，避免把「卡住」误判成「跑完」）
--------------------------------------------------------------
1. `train_dprnn_musdb.txt` 在开始轮询之后**新增**了 `训练结束:` 这一行
   （训练脚本退出前必写；这是唯一的权威信号）；
2. `01_models/_weights/dprnn_musdb/last.pt` 的 mtime 至少稳定 `--stable` 秒；
3. 没有 `train_dprnn_musdb` 进程还在跑（ctypes 枚举，不依赖 psutil）。

判据 2/3 是给「日志被别的进程写」和「日志写完但进程没退干净」兜底的。

用法
----
    python tools/post_dprnn_chain.py --dry-run     # 只校验，不执行
    python tools/post_dprnn_chain.py               # 等训练结束再跑（推荐）
    python tools/post_dprnn_chain.py --force       # 训练还在跑也直接开跑（会拿到欠训练权重！）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths                                          # noqa: E402
_paths.setup_env()

ROOT = _paths.PROJECT_ROOT
PY = sys.executable
TRAIN_LOG = _paths.LOGS / "train_dprnn_musdb.txt"
CHAIN_LOG = _paths.LOGS / "_post_dprnn.txt"
LOCK = _paths.LOGS / "_post_dprnn.lock"
CKPT_DIR = _paths.WEIGHTS / "dprnn_musdb"
RUNS_JSON = _paths.comparison_dir() / "model_runs.json"

MARKER = "训练结束"                    # train_dprnn_musdb.py 退出前必写

_log_fh = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def keep_awake(on: bool = True) -> None:
    """与训练脚本同一套 Windows 禁睡兜底（链子本身只有十几分钟，但别在关键时刻睡着）。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if on
            else ES_CONTINUOUS)
    except Exception:                                              # noqa: BLE001
        pass


# ---------------------------------------------------------------- 进程探测
def _pid_alive(pid: int) -> bool:
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        k32.CloseHandle(h)
        return True
    except AttributeError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _running_pids() -> list[int]:
    """所有「正在跑训练脚本」的 PID（不依赖 psutil）。

    🔴 2026-09-17 实测的假阳性陷阱
    ------------------------------
    最初只按 `CommandLine -like '*train_dprnn_musdb*'` 过滤，结果一次匹配到 **7 个**
    进程：`WorkBuddy.exe`、3 个 `bash.exe` 包装壳、以及**查询自身的 powershell**——
    它们的命令行里都恰好带着这个字符串（因为是我在命令行里输入的）。
    → 收尾链会误判「训练永远没结束」，死等到超时。

    修法：**加上进程名过滤** `Name -like 'python*'`，并排除本脚本自己。
    实测（同日 11:15）这样只剩真正的两个 python：
    `Scripts\\python.exe`（启动器壳，~5 MB，父）与它 spawn 的
    真实解释器（~2.1 GB，子）——两者命令行完全一致，退出时一起退。

    另外 `wmic` 在 Windows 11 26200 上已被移除，所以走 CIM；查询失败时返回 []，
    此时只剩「日志结束标记」这一条判据（调用方已说明）。
    """
    cmd = ("Get-CimInstance Win32_Process | "
           "Where-Object { $_.Name -like 'python*' -and "
           "$_.CommandLine -like '*train_dprnn_musdb*' -and "
           "$_.CommandLine -notlike '*post_dprnn_chain*' } | "
           "ForEach-Object { $_.ProcessId }")
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                           capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
    except Exception:                                              # noqa: BLE001
        return []
    out = []
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            out.append(int(line))
    return out


# ---------------------------------------------------------------- 锁
def acquire_lock() -> bool:
    _paths.LOGS.mkdir(parents=True, exist_ok=True)
    if LOCK.is_file():
        try:
            old = int(LOCK.read_text(encoding="utf-8").split()[0])
        except Exception:                                          # noqa: BLE001
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            log(f"⛔ 已有收尾链在跑 (PID {old})，拒绝重复启动。确认是残留就删 {LOCK}")
            return False
        log(f"清理失效锁 (PID {old})")
    LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                    encoding="utf-8")
    return True


def release_lock() -> None:
    try:
        LOCK.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------- 等待训练
def wait_for_training(poll: int, timeout: float, stable: int, force: bool) -> bool:
    """等到训练真的结束。返回 False 表示超时/异常。

    🔴 判据缺陷与修补（2026-09-19）
    ------------------------------
    原设计只认日志里新增的 `训练结束:` 行。但 09-17 那轮训练是**被外部中断**的
    （预算 300 min 只跑到 250.7 min，step 31900 后戛然而止），
    `train_dprnn_musdb.py` 的收尾代码根本没执行到 → 日志里**永远不会有这一行**，
    于是收尾链会一路空等到超时。

    真实判据应当是「**训练已经不可能再产出更好结果**」，而不是「训练自己说了结束」。
    因此这里改成：结束标记 **或** 「进程已退 + ckpt 长时间不再变动」二者满足其一即可；
    后者触发时会在日志里明确标注为 `[兜底判据]`，避免和正常结束混淆。
    """
    if force:
        log("⚠️ --force：跳过等待，直接用当前 ckpt")
        return True

    if not TRAIN_LOG.is_file():
        log(f"⛔ 找不到训练日志 {TRAIN_LOG}")
        return False

    # 只认「开始轮询之后新增」的结束标记，避免把上一轮的结束语当成本轮结束
    off0 = TRAIN_LOG.stat().st_size
    log(f"开始等待训练结束（日志起点 {off0} 字节，轮询 {poll}s，上限 {timeout/3600:.1f}h）")
    t0 = time.time()
    last_note = 0.0

    def ckpt_age() -> float:
        lp = CKPT_DIR / "last.pt"
        return (time.time() - lp.stat().st_mtime) if lp.is_file() else -1.0

    while time.time() - t0 < timeout:
        with open(TRAIN_LOG, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(off0)
            tail = fh.read()

        alive = [p for p in _running_pids() if p != os.getpid()]
        age = ckpt_age()
        marker_seen = MARKER in tail

        # ---- 正常路径：日志有结束标记，且 ckpt 稳定、进程退干净 ----
        if marker_seen and age >= stable and not alive:
            log(f"✅ 训练结束（日志标记，等待 {(time.time()-t0)/60:.1f} min）")
            for m in tail.splitlines():
                if MARKER in m or "best mean SI-SDR" in m:
                    log("   " + m.strip())
            return True

        # ---- 兜底路径：无结束标记，但进程已退且 ckpt 长时间静止 ----
        # 要求静止时间达到 stable 的 3 倍，避免把「验证间隙」误判成「训练结束」
        if not marker_seen and not alive and age >= stable * 3:
            log(f"✅ [兜底判据] 训练进程已退出，且 last.pt 已 {age/60:.1f} min 未变动"
                f"（日志无「{MARKER}」行 → 疑似被外部中断）")
            for m in tail.splitlines()[-3:]:
                log("   " + m.strip())
            return True

        if time.time() - last_note > 300:
            why = []
            if alive:
                why.append(f"训练进程在跑(PID {alive})")
            if not marker_seen:
                why.append(f"无「{MARKER}」标记")
            why.append(f"last.pt {age/60:.1f} min 前写过")
            log(f"…仍在等（{'；'.join(why)}；已等 {(time.time()-t0)/60:.1f} min）")
            last_note = time.time()
        time.sleep(poll)

    log(f"⛔ 等待超时（{timeout/3600:.1f}h）——训练可能仍没结束，放弃收尾")
    return False


# ---------------------------------------------------------------- 备份
def backup_runs() -> Path | None:
    """动手前备份 model_runs.json。

    🔴 本项目的 `mv -f` 覆盖事故（曾把 78 KB 的 model_runs.json 覆盖成 2.9 KB）说明
    「聚合脚本会原地重写全集」这件事必须有退路。
    """
    if not RUNS_JSON.is_file():
        log(f"（{RUNS_JSON.name} 不存在，跳过备份）")
        return None
    dst = RUNS_JSON.with_name(
        f"{RUNS_JSON.stem}_pre-dprnn-{time.strftime('%Y%m%d-%H%M%S')}.json")
    shutil.copy2(RUNS_JSON, dst)
    log(f"已备份 {RUNS_JSON.name} → {dst.name}（{dst.stat().st_size} 字节）")
    return dst


# ---------------------------------------------------------------- 读中位数
def dprnn_median() -> dict:
    p = _paths.comparison_dir() / "model_runs_median.json"
    if not p.is_file():
        return {}
    import json
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("dprnn") or {}
    except Exception:                                              # noqa: BLE001
        return {}


def fmt_row(m: dict) -> str:
    if not m:
        return "（无）"
    sdr = m.get("sdr") or {}
    stems = " ".join(f"{k}={sdr.get(k,{}).get('median')}"
                     for k in ("vocals", "drums", "bass", "other") if k in sdr)
    return (f"均值 {m.get('sdr_avg4')} ｜ {stems} ｜ RTF {m.get('rtf')}"
            f" ｜ n={m.get('n_clips')}")


# ---------------------------------------------------------------- 步骤表
def steps(models: str, verify_mdx: bool = False) -> list[tuple[str, list[str]]]:
    """收尾链的步骤表。

    全部复用**已测试过**的脚本，不在链条里内联新逻辑 —— 否则出问题时分不清是
    「模型/数据的问题」还是「链条临时写的代码的问题」。

    `--verify-mdx` 会额外重跑 MDX-Net 在 `ANiMAL - Clinic A` 的片段，用于结掉
    `MODEL_PROVISIONING_REPORT.md` §1.6 里挂着的 P1：
      「MDX 鼓轨 CPU 11.96 / GPU 6.99，其余轨 CPU/GPU 差 < 0.1 dB → 待复测」。
    默认关闭：它会**改写** MDX 的既有记录，属于会影响正文数字的动作，应当显式选择。
    """
    out = [
        ("① 重跑 DPRNN 三片段基准",
         [PY, str(_paths.TOOLS / "bench_multisong.py"), "--models", models]),
        ("② 聚合中位数（并刷新 MEDIAN_TABLE.md）",
         [PY, str(_paths.TOOLS / "median_over_clips.py"),
          "--write", str(_paths.DOCS / "MEDIAN_TABLE.md")]),
        ("③ 重绘对比图",
         [PY, str(_paths.TOOLS / "plot_median_summary.py")]),
        ("④ 重建 HTML",
         [PY, str(_paths.TOOLS / "build_html_v13.py")]),
    ]
    if verify_mdx:
        out.append(
            ("⑤ 复测 MDX-Net 鼓轨异常（ANiMAL - Clinic A）",
             [PY, str(_paths.TOOLS / "benchmark_model_universal.py"), "--model", "mdx",
              "--song", "ANiMAL - Clinic A", "--duration", "10", "--offset", "140"]))
        out.append(
            ("⑥ 再聚合（纳入 MDX 复测结果）",
             [PY, str(_paths.TOOLS / "median_over_clips.py"),
              "--write", str(_paths.DOCS / "MEDIAN_TABLE.md")]))
    return out


def run_step(title: str, cmd: list[str]) -> bool:
    log(f"--- {title} ---")
    log("    " + " ".join(Path(c).name if c.endswith(".py") else c for c in cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-40:]
    for line in tail:
        log("    | " + line)
    if p.stderr:
        for line in (p.stderr.strip().splitlines()[-15:]):
            log("    ! " + line)
    ok = p.returncode == 0
    log(f"    {'✅' if ok else '❌'} exit={p.returncode} 用时 {time.time()-t0:.0f}s")
    return ok


def main() -> int:
    global _log_fh
    ap = argparse.ArgumentParser(description="DPRNN 自训收尾链")
    ap.add_argument("--models", default="dprnn", help="要重跑的模型（默认 dprnn）")
    ap.add_argument("--poll", type=int, default=60, help="轮询间隔秒（默认 60）")
    ap.add_argument("--timeout", type=float, default=12 * 3600.0,
                    help="等待训练结束的上限秒数（默认 12h）")
    ap.add_argument("--stable", type=int, default=90,
                    help="last.pt 需要保持静的秒数（默认 90）")
    ap.add_argument("--force", action="store_true", help="不等训练结束，立刻开跑")
    ap.add_argument("--verify-mdx", action="store_true",
                    help="额外复测 MDX-Net 鼓轨异常（会改写 MDX 记录，见 steps() 说明）")
    ap.add_argument("--dry-run", action="store_true", help="只校验路径与命令，不执行")
    args = ap.parse_args()

    plan = steps(args.models, args.verify_mdx)

    _paths.LOGS.mkdir(parents=True, exist_ok=True)
    _log_fh = open(CHAIN_LOG, "a", encoding="utf-8")

    if not acquire_lock():
        return 2
    try:
        keep_awake(True)
        log("=" * 72)
        log(f"DPRNN 收尾链启动  models={args.models}  dry-run={args.dry_run}")

        # ---------- dry-run：只校验 ----------
        if args.dry_run:
            ok = True
            for label, p in [("训练日志", TRAIN_LOG), ("ckpt 目录", CKPT_DIR),
                             ("model_runs.json", RUNS_JSON)]:
                log(f"  {label:16s} {'OK ' if p.exists() else '缺 '} {p}")
                ok &= p.exists()
            for n in ("best.pt", "last.pt"):
                f = CKPT_DIR / n
                log(f"  {n:16s} {'OK ' if f.is_file() else '缺 '}"
                    + (f"  {f.stat().st_size} 字节  "
                       f"{time.strftime('%H:%M:%S', time.localtime(f.stat().st_mtime))}"
                       if f.is_file() else ""))
            for title, cmd in plan:
                log(f"  {title}: {' '.join(cmd)}")
                ok &= Path(cmd[1]).is_file()
            log(f"dry-run 结束：{'全部就绪' if ok else '有缺失项'}")
            return 0 if ok else 2

        # ---------- 等训练 ----------
        if not wait_for_training(args.poll, args.timeout, args.stable, args.force):
            return 3

        # ---------- 备份 ----------
        backup_runs()
        before = dprnn_median()
        log(f"收尾前 DPRNN：{fmt_row(before)}")

        # ---------- 逐步执行 ----------
        results = []
        for title, cmd in plan:
            ok = run_step(title, cmd)
            results.append((title, ok))
            if not ok:
                log(f"⛔ 步骤失败，链条中止（后续步骤依赖它）：{title}")
                break

        after = dprnn_median()
        log(f"收尾后 DPRNN：{fmt_row(after)}")
        log("=" * 72)
        for title, ok in results:
            log(f"  {'✅' if ok else '❌'} {title}")
        if all(ok for _, ok in results) and len(results) == len(plan):
            log("🎉 收尾链全部完成 —— 记得复核 04_reports 里的 DPRNN 数字并同步文档")
            return 0
        return 1
    finally:
        release_lock()
        keep_awake(False)
        if _log_fh:
            _log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
