# -*- coding: utf-8 -*-
"""MUSDB18-HQ test 集 —— **按歌曲推进**的全模型扫描 + 超时熔断。

用户规则（2026-09-21）
--------------------
1. 顺序：**先让所有模型跑同一首歌，全部跑完后再换下一首**（song-major）
2. 单个 (模型, 歌曲) **超时即中止**该次运行，直接换下一首歌（默认 180s，理由见 TIMEOUT_S 注释）
3. 某模型累计 **3 次超时** → 标记为不可用，从后续轮次中剔除
4. 产物集中写入**新文件夹** `03_outputs/_test_run/<模型>/<歌曲>/`
5. 50 首全部跑完后**再回头分析**为什么某些模型慢（本脚本只负责执行与记账）

为什么每个 (模型, 歌曲) 开一个子进程
----------------------------------
不是浪费，是**必须**：实测把 15 个模型塞进同一个进程，跑到第 10 个就开始
`CUDA error: out of memory` —— 因为模型权重常驻显存、前一个模型不释放。
子进程隔离既能保证显存干净，又让 `subprocess.run(timeout=...)` 能干净地
超时杀进程（不会留下半死的 CUDA context）。

用法
----
    python tools/bench_musdb_test_songmajor.py --models all --dry-run
    python tools/bench_musdb_test_songmajor.py --models all
    python tools/bench_musdb_test_songmajor.py --report      # 只出汇总，不跑
    python tools/bench_musdb_test_songmajor.py --songs 0:3   # 先试 3 首
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths  # noqa: E402

ROOT = _paths.PROJECT_ROOT
TOOLS = _paths.TOOLS
UNIVERSAL = TOOLS / "benchmark_model_universal.py"
TEST_ROOT = _paths.MUSDB18_ROOT / "test"
LOGS = _paths.LOGS
RUNS_JSON = _paths.DATA_COMPARISON / "model_runs_run.json"
LEDGER = _paths.DATA_COMPARISON / "test_sweep_ledger.json"
LOCK = LOGS / "_test_sweep.lock"

# ---- 用户规则参数 ----
# ⚠️ 阈值定为 180s（而非最初说的 120s）—— 2026-09-21 实测：
#    精度第一名 BSRNN-SIMO 在中位曲上要约 103.6s，50 首里有 **10 首** 超过 120s
#    → 会触发 3 次熔断被淘汰，把最好的模型筛掉。180s 下它最长也只要 165s，可保住；
#    同时仍能淘汰 bsrnn/bsrnn_large/l6（220–250s）与 bsrnn_all/bsrnn_large_all/rpca（700s+）。
TIMEOUT_S = 180.0      # 单个 (模型, 歌曲) 墙钟上限
MAX_TIMEOUTS = 3       # 累计超时次数上限，达到即剔除

OUT_TAG = "run"        # 走 03_outputs/_test_run/<模型>/<歌>/ + model_runs_run.json
AUDIO_FORMAT = "flac"  # 无损，约为 WAV 的 25%（磁盘只剩 66 GB，WAV 要 98 GB）


def _pick_python() -> str:
    """选出带 torch/numpy 的解释器。

    ⚠️ `sys.executable` 在本机指向**系统 Python（无 numpy）**，
    子进程一启动就 ModuleNotFoundError，而外层只看到「ERR 0.5s」。
    """
    cands = [os.environ.get("FYP_PY"),
             str(Path.home() / ".workbuddy/binaries/python/envs/fyp_audio/Scripts/python.exe"),
             sys.executable, "python"]
    for exe in cands:
        if not exe:
            continue
        if os.path.sep in exe and not Path(exe).is_file():
            continue
        try:
            r = subprocess.run([exe, "-c", "import numpy, torch; print('ok')"],
                               capture_output=True, text=True, timeout=120)
        except Exception:
            continue
        if r.returncode == 0 and "ok" in (r.stdout or ""):
            return exe
    raise SystemExit("[!] 找不到带 numpy/torch 的解释器，请设 FYP_PY")


def _registry_keys() -> list[str]:
    """从源码里正则抠出 REGISTRY 的键 —— 避免 import 时把 torch 拉进调度进程。"""
    src = UNIVERSAL.read_text(encoding="utf-8")
    m = re.search(r"^REGISTRY\s*=\s*\{(.*?)^\}", src, re.S | re.M)
    if not m:
        raise SystemExit("[!] 无法从 benchmark_model_universal.py 解析 REGISTRY")
    return re.findall(r'^\s{4}"([a-z_0-9]+)"\s*:', m.group(1), re.M)


def _list_songs() -> list[str]:
    return sorted(d.name for d in TEST_ROOT.iterdir() if d.is_dir())


def _pid_alive(pid: int) -> bool:
    """PID 是否**真的**在运行。

    🔴 为什么不能用 `OpenProcess` 成功与否当判据（2026-09-21 实测踩到）：
    进程被杀之后，只要有别人还持有它的句柄（这里是 WorkBuddy 的作业对象），
    进程对象就会**残留在内核里**：`OpenProcess` 照样成功、`GetProcessTimes`
    照样返回原始创建时间，但 `tasklist` 已经查不到它。
    于是上一次被杀的扫描进程（PID 24404）被误判成「仍在运行」，
    计划任务被「已有实例在跑」挡在门外，长任务根本起不来。
    必须用 `GetExitCodeProcess` 看它是否还是 `STILL_ACTIVE`。
    """
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        h = k32.OpenProcess(0x1000, False, pid)      # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = wintypes.DWORD(0)
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == 259                 # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    except Exception:
        return False


def _proc_create_unix(pid: int) -> float | None:
    """取 PID 的创建时间（Unix 秒）。进程不存在 / 无权限时返回 None。"""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        h = k32.OpenProcess(0x1000, False, pid)
        if not h:
            return None
        try:
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if not k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                                       ctypes.byref(k), ctypes.byref(u)):
                return None
            ft = (c.dwHighDateTime << 32) | c.dwLowDateTime
            return ft / 1e7 - 11644473600.0              # 1601-01-01 -> Unix 纪元
        finally:
            k32.CloseHandle(h)
    except Exception:
        return None


def _clear_stale_lock() -> None:
    """清掉陈旧锁。

    判活用 `_pid_alive`（`GetExitCodeProcess`）。**不要**退回「OpenProcess 成功即存活」——
    被杀的进程只要有句柄残留就会被误判成存活，把长任务永久挡死（见 `_pid_alive` 注释）。
    另加一道创建时间校验，防止 PID 被系统回收后撞车。
    """
    if not LOCK.is_file():
        return
    try:
        pid = int(LOCK.read_text(encoding="utf-8").split()[0])
    except Exception:
        pid = 0
    if pid and pid != os.getpid() and _pid_alive(pid):
        lock_mtime = LOCK.stat().st_mtime
        created = _proc_create_unix(pid)
        # 创建时间早于锁文件（+3s 容差）→ 大概率就是原持有者，拒绝双开
        if created is not None and created <= lock_mtime + 3:
            raise SystemExit(f"[!] PID {pid} 仍在运行，拒绝启动第二个实例。删 {LOCK} 再试。")
        print(f"[i] 锁里的 PID {pid} 已被系统回收（创建时间晚于锁文件），按陈旧锁处理")
    LOCK.unlink()
    print(f"[i] 清掉陈旧锁（PID {pid} 已失效）")


# ------------------------------------------------------------------ #
# 台账（ledger）：记录每次尝试的墙钟、超时次数、被剔除的模型
# ------------------------------------------------------------------ #
def load_ledger() -> dict:
    if LEDGER.is_file():
        try:
            return json.loads(LEDGER.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"attempts": {}, "timeouts": {}, "disabled": [], "updated": None}


def save_ledger(led: dict) -> None:
    led["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=2), encoding="utf-8")


def _done_set() -> set[tuple[str, str]]:
    """已 PASS 的 (model, song) —— 断点续跑用。"""
    if not RUNS_JSON.is_file():
        return set()
    try:
        runs = json.loads(RUNS_JSON.read_text(encoding="utf-8")).get("runs", {})
    except Exception:
        return set()
    return {(m, s) for m, recs in runs.items() if isinstance(recs, dict)
            for s, r in recs.items()
            if isinstance(r, dict) and r.get("status") == "PASS"}


def _out_dir(model: str, song: str) -> Path:
    """镜像 benchmark_model_universal._out_dir_for 的 run-tree 布局。"""
    src = UNIVERSAL.read_text(encoding="utf-8")
    mm = re.search(r"MODEL_DIR\s*=\s*\{(.*?)\}", src, re.S)
    mapping = dict(re.findall(r'"([a-z_0-9]+)"\s*:\s*"([^"]+)"', mm.group(1))) if mm else {}
    return _paths.OUTPUTS / "_test_run" / mapping.get(model, model) / song


def _json_status(model: str, song: str) -> str | None:
    """读 model_runs_run.json 里这条 (模型, 歌曲) 的真实状态。

    🔴 为什么不能只看子进程返回码（2026-09-21 实测踩到的假 PASS）：
    `benchmark_model_universal.py` 会把模型异常**内部捕获**、照常写 JSON
    （`status: FAIL`），然后**仍然以 0 退出**。于是调度器把
    `dprnn` 的 `ValueError: operands could not be broadcast ...` 误记成
    「PASS 6.8s」写进台账，而产物目录其实是空的。
    这条假记录会一路污染 `_done_set()` 的续跑判断与最终统计，必须在源头掐掉。
    """
    if not RUNS_JSON.is_file():
        return None
    try:
        runs = json.loads(RUNS_JSON.read_text(encoding="utf-8")).get("runs", {})
        rec = runs.get(model, {}).get(song)
        if isinstance(rec, dict):
            return rec.get("status")
    except Exception:
        pass
    return None


def run_combo(py: str, model: str, song: str, timeout: float) -> dict:
    """跑单个 (模型, 歌曲)，带超时。返回 {status, seconds, stdout, stderr}。"""
    cmd = [py, "-X", "utf8", str(UNIVERSAL),
           "--model", model, "--song", song, "--subset", "test",
           "--duration", "0",                       # 整曲
           "--out-tag", OUT_TAG,
           "--audio-format", AUDIO_FORMAT]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env)
        dt = time.time() - t0
        # 真实状态以结果 JSON 为准；返回码非 0 直接判 FAIL
        jst = _json_status(model, song)
        if p.returncode != 0:
            st = "FAIL"
        elif jst == "PASS":
            st = "PASS"
        else:
            st = "FAIL"
        return {"status": st, "seconds": round(dt, 1), "json_status": jst,
                "stdout": p.stdout or "", "stderr": (p.stderr or "")[-2000:]}
    except subprocess.TimeoutExpired as e:
        dt = time.time() - t0
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return {"status": "TIMEOUT", "seconds": round(dt, 1), "json_status": None,
                "stdout": out, "stderr": err[-2000:]}


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
def main() -> int:
    ap = argparse.ArgumentParser(description="test 集 song-major 扫描 + 超时熔断")
    ap.add_argument("--models", default="all")
    ap.add_argument("--songs", default="all", help="all / a:b 区间 / 名称子串")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--max-timeouts", type=int, default=MAX_TIMEOUTS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-skip-done", action="store_true")
    ap.add_argument("--report", action="store_true", help="只出汇总报告")
    args = ap.parse_args()

    if args.report:
        return report()

    all_models = _registry_keys()
    models = all_models if args.models == "all" else [m.strip() for m in args.models.split(",")]
    bad = [m for m in models if m not in all_models]
    if bad:
        raise SystemExit(f"未知模型 {bad}\n可用: {all_models}")

    songs_all = _list_songs()
    if args.songs == "all":
        songs = songs_all
    elif ":" in args.songs and all(x.strip().isdigit() for x in args.songs.split(":")):
        a, b = (int(x) for x in args.songs.split(":"))
        songs = songs_all[a:b]
    else:
        pats = [x.strip() for x in args.songs.split(",")]
        songs = [s for s in songs_all if any(p.lower() in s.lower() for p in pats)]
    if not songs:
        raise SystemExit("过滤后没有曲目")

    print("=" * 70)
    print("MUSDB18-HQ test 集 —— song-major 扫描 + 超时熔断")
    print("=" * 70)
    print(f"  模型 ({len(models)}): {', '.join(models)}")
    print(f"  曲目 ({len(songs)}): {songs[0]} ... {songs[-1]}")
    print(f"  超时阈值: {args.timeout:.0f}s / 组合     熔断: 累计 {args.max_timeouts} 次超时即剔除")
    print(f"  产物: {_paths.OUTPUTS}/_test_run/<模型>/<歌曲>/  ({AUDIO_FORMAT})")
    print(f"  台账: {LEDGER}")
    print(f"  上限组合数: {len(models) * len(songs)}")
    print()

    if args.dry_run:
        print("[dry-run] 不执行")
        return 0

    py = _pick_python()
    print(f"  解释器: {py}")
    LOGS.mkdir(parents=True, exist_ok=True)
    _clear_stale_lock()
    LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")

    led = load_ledger()
    led.setdefault("attempts", {})
    led.setdefault("timeouts", {})
    led.setdefault("disabled", [])
    for m in models:
        led["timeouts"].setdefault(m, 0)

    done = set() if args.no_skip_done else _done_set()
    active = [m for m in models if m not in led["disabled"]]
    if led["disabled"]:
        print(f"  [i] 台账中已剔除 {len(led['disabled'])} 个模型: {led['disabled']}")
    if done:
        print(f"  [i] 已 PASS {len(done)} 个组合，将跳过")

    log_path = LOGS / "_test_sweep.txt"
    t_start = time.time()
    n_pass = n_fail = n_to = n_skip = 0
    wall_by_model: dict[str, list[float]] = {}
    slowest: list[tuple[float, str, str, str]] = []   # (秒, 状态, 模型, 歌)

    try:
        with open(log_path, "a", encoding="utf-8", buffering=1) as lg:
            lg.write("\n" + "=" * 70 + "\n")
            lg.write(f"===== test sweep start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            lg.write(f"models={models}\nsongs={len(songs)}\ntimeout={args.timeout}\n\n")

            # ---- 外层：歌曲（用户要求的推进顺序）----
            for si, song in enumerate(songs, 1):
                if not active:
                    print(f"\n[!] 所有模型均已剔除，剩余 {len(songs) - si + 1} 首不再执行。")
                    break
                print(f"\n{'=' * 70}")
                print(f"第 {si}/{len(songs)} 首: {song}")
                print(f"  在场模型 {len(active)}/{len(models)}: {', '.join(active)}")
                lg.write(f"\n--- song {si}/{len(songs)}: {song} ---\n")

                # ---- 内层：模型 ----
                for model in list(active):
                    if (model, song) in done:
                        n_skip += 1
                        print(f"    [skip] {model:<16} 已完成")
                        continue

                    r = run_combo(py, model, song, args.timeout)
                    st, sec = r["status"], r["seconds"]
                    led["attempts"].setdefault(model, {})[song] = {
                        "status": st, "seconds": sec,
                        "json_status": r.get("json_status"),
                        "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                    wall_by_model.setdefault(model, []).append(sec)
                    slowest.append((sec, st, model, song))

                    if st == "PASS":
                        n_pass += 1
                        print(f"    [OK ] {model:<16} {sec:7.1f}s")
                        lg.write(f"  PASS   {model:<16} {sec:7.1f}s\n")
                    elif st == "TIMEOUT":
                        n_to += 1
                        led["timeouts"][model] = led["timeouts"].get(model, 0) + 1
                        k = led["timeouts"][model]
                        print(f"    [T/O] {model:<16} {sec:7.1f}s  "
                              f"超时 {k}/{args.max_timeouts}"
                              + ("  -> 剔除该模型" if k >= args.max_timeouts else ""))
                        lg.write(f"  TIMEOUT {model:<16} {sec:7.1f}s  "
                                 f"({k}/{args.max_timeouts})\n")
                        if k >= args.max_timeouts:
                            active.remove(model)
                            if model not in led["disabled"]:
                                led["disabled"].append(model)
                            lg.write(f"  >>> DISABLED {model} (累计 {k} 次超时)\n")
                    else:
                        n_fail += 1
                        print(f"    [ERR] {model:<16} {sec:7.1f}s  "
                              f"(JSON={r.get('json_status')})")
                        lg.write(f"  FAIL   {model:<16} {sec:7.1f}s  "
                                 f"(JSON={r.get('json_status')})\n")
                        lg.write((r.get("stderr") or "")[-1200:] + "\n")
                    save_ledger(led)

                el = (time.time() - t_start) / 3600
                print(f"  -- 本首结束  用时 {el:.2f}h  "
                      f"PASS {n_pass} / ERR {n_fail} / T/O {n_to} / skip {n_skip}  "
                      f"在场 {len(active)} 模型")
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass
        save_ledger(led)

    el = (time.time() - t_start) / 3600
    print(f"\n{'=' * 70}")
    print(f"扫描结束  用时 {el:.2f}h")
    print(f"  PASS {n_pass} / FAIL {n_fail} / TIMEOUT {n_to} / SKIP {n_skip}")
    print(f"  被剔除模型: {led['disabled'] or '（无）'}")
    print(f"  日志: {log_path}")
    print(f"  台账: {LEDGER}")
    print(f"\n  下一步（用户规则第 5 步）：跑完后回头分析为什么慢 ->")
    print(f"    python tools/bench_musdb_test_songmajor.py --report")
    return 0


# ------------------------------------------------------------------ #
# 汇总报告
# ------------------------------------------------------------------ #
def report() -> int:
    if not LEDGER.is_file():
        print(f"[!] 台账不存在: {LEDGER}")
        return 2
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    attempts = led.get("attempts", {})
    print("=" * 78)
    print("test 集扫描台账汇总")
    print("=" * 78)
    print(f"  更新时间: {led.get('updated')}")
    print(f"  被剔除: {led.get('disabled') or '（无）'}")
    print()
    print(f"  {'模型':<18}{'尝试':>5}{'PASS':>6}{'T/O':>5}{'ERR':>5}"
          f"{'中位耗时':>10}{'最慢':>9}{'超时次数':>9}")
    print("  " + "-" * 74)
    rows = []
    for m, recs in attempts.items():
        st = [r["status"] for r in recs.values()]
        secs = sorted(r["seconds"] for r in recs.values())
        med = secs[len(secs) // 2] if secs else 0
        rows.append((m, len(recs), st.count("PASS"), st.count("TIMEOUT"),
                     st.count("FAIL"), med, max(secs) if secs else 0,
                     led.get("timeouts", {}).get(m, 0)))
    for r in sorted(rows, key=lambda x: -(x[5] or 0)):
        print(f"  {r[0]:<18}{r[1]:>5}{r[2]:>6}{r[3]:>5}{r[4]:>5}"
              f"{r[5]:>9.1f}s{r[6]:>8.1f}s{r[7]:>9}")
    print()
    if RUNS_JSON.is_file():
        d = json.loads(RUNS_JSON.read_text(encoding="utf-8"))
        n = sum(len(v) for v in d.get("runs", {}).values())
        print(f"  结果 JSON: {RUNS_JSON}（{n} 条记录）")
    print()
    print("  ⚠️ 慢因分析请见 04_reports/separation/docs/ 下的专项报告")
    return 0


if __name__ == "__main__":
    sys.exit(main())
