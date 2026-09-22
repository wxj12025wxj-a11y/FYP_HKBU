# -*- coding: utf-8 -*-
"""MUSDB18-HQ **官方 test 集**整曲基准 —— 与 train-片段基准完全隔离。

为什么必须单独一套（而不是直接复用 bench_multisong.py）
-----------------------------------------------------
`benchmark_model_universal.py` 的结果是**按 model -> song 覆盖**写进
`04_reports/separation/data/comparison/model_runs.json`，而该文件正是
`median_over_clips.py` 生成论文主表 （`model_runs_median.json` / `MEDIAN_TABLE.md`）的数据源。
它靠「歌名」挑片段 —— 如果 test 集的 50 首歌混进同一个 json，
主表的中位数会被**静默污染**（train 的 3 个均衡片段 + test 的 50 首整曲混在一起聚合）。

因此本脚本：
  * 结果写**独立文件** `04_reports/separation/data/comparison/model_runs_test.json`
  * 音频写**独立目录树** `03_outputs/<模型>/_musdb18_test/<歌曲>/`
  * 绝不触碰 `model_runs.json` / `model_runs_median.json`

整曲 vs 片段
------------
`load_clip(duration=None)` 会读整首歌（`--duration 0` 表示整曲，见 `_duration_arg`）。
整曲推理对显存是压力测试：分块类模型（MDX/Demucs/BS-RoFormer/BSRNN）内部有自己的
segment + overlap-add，长输入只影响耗时；但 **DPRNN 是整段前向**，300 s 的音频会直接爆显存，
故 DPRNN 强制分块（`--chunk-s` 控制，默认 30 s，带交叠淡入淡出）。

用法
----
    python tools/bench_musdb_test.py --list-songs
    python tools/bench_musdb_test.py --models umx,mdx --songs 0:2      # 前 2 首试跑
    python tools/bench_musdb_test.py --models all --workers 1          # 全量 50 首
    python tools/bench_musdb_test.py --aggregate                       # 仅聚合出表
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
_sys = sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()

ROOT = _paths.PROJECT_ROOT
PY = sys.executable
TEST_ROOT = _paths.MUSDB18_ROOT / "test"
OUT_ROOT = _paths.OUTPUTS
DATA_COMPARISON = _paths.DATA_COMPARISON
RUNS_JSON = DATA_COMPARISON / "model_runs_test.json"
MANIFEST = DATA_COMPARISON / "musdb_test_manifest.json"
LOGS = _paths.LOGS
LOCK = LOGS / "_bench_musdb_test.lock"

# ---- 符合新规则的模型（原生直接输出分离音轨，无需我们改造权重/网络）----
# 详见 04_reports/separation/docs/MODEL_RULE_AUDIT_2026-09-19.md
ELIGIBLE = [
    "oracle",          # IRM/IBM 解析上界（非模型，作为天花板参照）
    "umx",             # Open-Unmix umxhq
    "mdx",             # MDX-Net mdx_extra
    "demucs",          # htdemucs
    "convtasnet",      # Conv-TasNet musdb18
    "mmdenselstm",     # MMDenseLSTM musdb18 paper
    "bsroformer_l12",  # BS-RoFormer L12 ep_317（vocals 单目标）
    "bsroformer_l6",   # BS-RoFormer L6 ep_937（vocals+other 单目标）
    "bsrnn_all",       # oBSRNN 4-stem（4 ckpt 串行，非原生 4-stem）
    "bsrnn_large_all", # BSRNN large 4-stem（同上）
    "bsrnn_simo",      # oBSRNN SIMO 4-stem（★ 原生单模型 4 stem）
    "rpca",            # RPCA（自研解析实现，无天然 stem 归属 → 交叉评估）
]
# ⛔ 被排除：
#   dprnn    —— 权重是本机自训，非模型原生发布；且原生 11.025 kHz < 44.1 kHz
#   rpca_drnn / pac_hubert  —— 无官方代码与权重
EXCLUDED = ["dprnn", "rpca_drnn", "pac_hubert_sep"]

# 模型名 -> 03_outputs 下的目录名（与 benchmark_model_universal.py::MODEL_DIR 保持一致）
MODEL_DIR = {
    "oracle": "Oracle-IRM", "rpca": "RPCA", "umx": "Open-Unmix", "mdx": "MDX-Net",
    "mmdenselstm": "MMDenseLSTM", "bsroformer_l12": "BS-RoFormer-L12",
    "bsroformer_l6": "BS-RoFormer-L6", "bsrnn_all": "BSRNN-opt",
    "bsrnn_large_all": "BSRNN-large", "bsrnn_simo": "BSRNN-SIMO",
    "convtasnet": "Conv-TasNet", "demucs": "Demucs",
}

# DPRNN 之外都需要整曲；DPRNN 不分块会爆显存
NEEDS_CHUNK = {"dprnn"}
DEFAULT_CHUNK_S = 30.0
CHUNK_OVERLAP_S = 2.0


def _pid_alive(pid: int) -> bool:
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)
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


def list_songs() -> list[str]:
    if MANIFEST.is_file():
        d = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return [s["song"] for s in d["songs"]]
    return sorted(p.name for p in TEST_ROOT.iterdir() if p.is_dir())


# ------------------------------------------------------------------ #
# 整曲分块推理（仅 DPRNN 这类整段前向的模型需要）
# ------------------------------------------------------------------ #
def run_chunked(model_key: str, song: str, chunk_s: float, overlap_s: float) -> dict:
    """对整曲做带交叠的分块推理，再重叠相加拼回，避免显存爆掉。"""
    raise NotImplementedError(
        "DPRNN 已被排除在新规则之外（权重非原生发布），本脚本不跑 DPRNN。"
    )


def _run_one(model_key: str, song: str, duration: int) -> dict:
    """调用 benchmark_model_universal.py 跑单模型单曲，返回其写的记录。"""
    cmd = [PY, str(_paths.TOOLS / "benchmark_model_universal.py"),
           "--model", model_key, "--song", song, "--subset", "test",
           "--duration", str(duration)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    return {"model": model_key, "song": song, "exit": p.returncode,
            "seconds": round(dt, 1), "stdout": p.stdout, "stderr": p.stderr[-4000:]}


def _done_set() -> set[tuple[str, str]]:
    """已 PASS 的 (model, song) 集合 —— 供 --skip-done 断点续跑。

    ⚠️ 全量 50 首 × 12 模型预估 ~39 h，中途必被系统睡眠/断电/误杀打断。
    没有续跑能力就只能从头再来，因此默认开启跳过。
    """
    if not RUNS_JSON.is_file():
        return set()
    try:
        runs = json.loads(RUNS_JSON.read_text(encoding="utf-8")).get("runs", {})
    except Exception:
        return set()
    out = set()
    for m, recs in runs.items():
        if not isinstance(recs, dict):
            continue
        for s, rec in recs.items():
            if isinstance(rec, dict) and rec.get("status") == "PASS":
                out.add((m, s))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="MUSDB18-HQ test 集整曲基准（与 train 片段基准隔离）")
    ap.add_argument("--models", default="all",
                    help="逗号分隔的模型名，或 all（默认）。可选：" + ",".join(ELIGIBLE))
    ap.add_argument("--songs", default="all",
                    help="歌名子串 / 逗号分隔 / 索引区间 a:b / all")
    ap.add_argument("--duration", type=int, default=0, help="0 = 整曲（默认）")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--list-songs", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="只做聚合出表，不跑推理")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-skip-done", action="store_true",
                    help="不跳过已 PASS 的组合（默认跳过，用于断点续跑）")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="每 N 个组合打一条带 ETA 的进度行（默认 10）")
    args = ap.parse_args()

    songs_all = list_songs()
    if args.list_songs:
        print(f"MUSDB18-HQ test 集：{len(songs_all)} 首")
        for i, s in enumerate(songs_all, 1):
            print(f"  {i:3d}  {s}")
        return 0

    if args.aggregate:
        return aggregate()

    models = ELIGIBLE if args.models == "all" else [m.strip() for m in args.models.split(",")]
    bad = [m for m in models if m not in ELIGIBLE]
    if bad:
        raise SystemExit(
            f"未知/不符合新规则的模型: {bad}\n"
            f"符合规则的模型：{ELIGIBLE}\n"
            f"已排除（原因见 MODEL_RULE_AUDIT_2026-09-19.md）：{EXCLUDED}")

    # ---- 选歌 ----
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

    print(f"MUSDB18-HQ test 整曲基准")
    print(f"  模型 ({len(models)}): {', '.join(models)}")
    print(f"  曲目 ({len(songs)}): {songs[0]} … {songs[-1]}")
    print(f"  时长: {'整曲' if args.duration == 0 else f'{args.duration}s'}")
    print(f"  结果: {RUNS_JSON}")
    print(f"  音频: {OUT_ROOT}/<模型>/_musdb18_test/<歌曲>/")
    print()

    if args.dry_run:
        print("[dry-run] 不执行")
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    if LOCK.is_file():
        try:
            old = int(LOCK.read_text(encoding="utf-8").split()[0])
        except Exception:
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            print(f"[!] 已有实例在跑 (PID {old})，拒绝重复启动。删 {LOCK} 再试。")
            return 2
    LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")

    log = LOGS / "_musdb_test.txt"
    n_ok = n_fail = n_skip = 0
    done_before = set() if args.no_skip_done else _done_set()
    # 展开成待跑组合，先过滤已完成的
    todo = [(m, s) for m in models for s in songs]
    if done_before:
        before = len(todo)
        todo = [x for x in todo if x not in done_before]
        n_skip = before - len(todo)
        print(f"  [skip] 跳过已完成 {n_skip} 个组合（--no-skip-done 可关闭）")
    t_start = time.time()
    per_item: list[float] = []
    try:
        with open(log, "a", encoding="utf-8") as lg:
            lg.write(f"\n===== musdb-test start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            lg.write(f"models={models}\nsongs={len(songs)}\nduration={args.duration}\n"
                     f"skip_done={not args.no_skip_done}\n")
            total = len(todo)
            for k, (model_key, song) in enumerate(todo, 1):
                head = f"[{k}/{total}] {model_key:<16} {song}"
                print(head + " …", flush=True)
                lg.write(f"--- {head} ---\n")
                lg.flush()
                try:
                    r = _run_one(model_key, song, args.duration)
                except Exception:
                    r = {"model": model_key, "song": song, "exit": -1,
                         "seconds": 0, "stdout": "", "stderr": traceback.format_exc()[-2000:]}
                lg.write(r["stdout"] or "")
                if r.get("stderr"):
                    lg.write("\n[stderr]\n" + r["stderr"] + "\n")
                lg.write(f"exit={r['exit']} {r['seconds']}s\n")
                lg.flush()
                ok = r["exit"] == 0
                n_ok += ok
                n_fail += (not ok)
                per_item.append(r["seconds"])
                eta = ""
                if per_item and k < total:
                    avg = sum(per_item) / len(per_item)
                    rem = avg * (total - k)
                    eta = f"  ETA {rem / 3600:.2f}h"
                print(f"     {'OK ' if ok else 'ERR'} {r['seconds']}s{eta}", flush=True)
                if args.progress_every and k % args.progress_every == 0 and k < total:
                    el = (time.time() - t_start) / 3600
                    print(f"  -- 进度 {k}/{total}  已用 {el:.2f}h  "
                          f"成功 {n_ok} 失败 {n_fail} --", flush=True)
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass

    el = (time.time() - t_start) / 3600
    print(f"\n完成：成功 {n_ok} / 失败 {n_fail} / 跳过 {n_skip}   用时 {el:.2f}h")
    print(f"日志: {log}")
    print(f"结果: {RUNS_JSON}")
    return 0 if n_fail == 0 else 1


def aggregate() -> int:
    """把 model_runs_test.json 聚合成每模型四轨均值 + 中位数，写 MEDIAN_TABLE_TEST.md。"""
    if not RUNS_JSON.is_file():
        print(f"⛔ {RUNS_JSON} 不存在，先跑推理")
        return 2
    d = json.loads(RUNS_JSON.read_text(encoding="utf-8"))
    runs = d.get("runs", {})
    STEMS = ("vocals", "drums", "bass", "other")
    rows = []
    for model, recs in runs.items():
        per_stem = {s: [] for s in STEMS}
        rtf = []
        for song, rec in recs.items():
            if rec.get("status") != "PASS":
                continue
            sdr = rec.get("sdr") or {}
            for s in STEMS:
                v = sdr.get(s)
                if isinstance(v, (int, float)):
                    per_stem[s].append(v)
            if isinstance(rec.get("rtf"), (int, float)):
                rtf.append(rec["rtf"])
        if not any(per_stem.values()):
            continue
        med = {s: (sorted(v)[len(v) // 2] if v else None) for s, v in per_stem.items()}
        vals = [v for v in med.values() if v is not None]
        rows.append({
            "model": model,
            "n_songs": len(per_stem["vocals"]) or len(rtf),
            **{f"med_{s}": med[s] for s in STEMS},
            "med_mean": round(sum(vals) / len(vals), 3) if vals else None,
            "rtf_median": round(sorted(rtf)[len(rtf) // 2], 4) if rtf else None,
        })
    rows.sort(key=lambda r: -(r["med_mean"] or -99))

    out_json = DATA_COMPARISON / "model_runs_test_median.json"
    out_json.write_text(json.dumps({"rows": rows, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                                   ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# MUSDB18-HQ 官方 test 集 —— 四轨中位数 SDR", "",
             f"> 来源 `{RUNS_JSON.name}` ｜ 生成 {time.strftime('%Y-%m-%d %H:%M:%S')}",
             "> 口径：整曲（duration=0），museval BSSEval v4、1 s 窗、**按曲取中位数**。", "",
             "| 模型 | 曲数 | vocals | drums | bass | other | **均值** | RTF(中位) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def f(k, _r=r):
            v = _r[k]
            return "—" if v is None else f"{v:.2f}"
        rt = "—" if r["rtf_median"] is None else f"{r['rtf_median']:.3f}"
        lines.append(f"| {r['model']} | {r['n_songs']} | {f('med_vocals')} | {f('med_drums')} | "
                     f"{f('med_bass')} | {f('med_other')} | **{f('med_mean')}** | {rt} |")
    out_md = _paths.DOCS / "MEDIAN_TABLE_TEST.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"聚合完成：{out_json}\n           {out_md}")
    for r in rows:
        print(f"  {r['model']:<18} n={r['n_songs']:<3} mean={r['med_mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
