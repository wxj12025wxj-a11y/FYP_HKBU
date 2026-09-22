"""Measure the intrinsic profile of every model: params / FLOPs / VRAM / latency.

This produces the table the reference figure asks for (Average SDR + Size + FLOPs)
-- i.e. "what does the model cost", independent of how well it separates.

Design decisions (deliberate, each one bought with a past bug)
-------------------------------------------------------------
* **One subprocess per model.**  Loading 15 models in a single process OOMs on
  the 8 GiB card because every weight set stays resident.  Isolation is cheap
  (a few seconds of import) compared with a crashed sweep.
* **The loader chain is reused from `benchmark_model_universal.REGISTRY`, never
  re-implemented.**  A second load path would silently drift from the first and
  the profile table would describe models nobody ran.
* **Models are recovered from the loader's closure.**  Loaders return `run_fn`
  (a closure over the model) plus an `info` dict; rather than change 15 loader
  signatures we walk the closure cells and collect every `nn.Module`.  Verified
  to reproduce `info['params']` exactly for BSRNN-SIMO / BSRNN-all / Demucs.
* **Fixed measurement protocol so numbers are comparable:** 44.1 kHz stereo,
  10 s input, FLOPs = 2 x MACs, warm-up before timing, median of 5 runs.

Honest scope note
-----------------
`FlopCounterMode` counts what torch dispatches: conv / linear / matmul / norm.
It does **not** see fused `scaled_dot_product_attention`, nor `torch.stft`.
Models whose cost lives in STFT + attention therefore report a *lower bound*.
The CSV carries `flops_coverage` so nobody compares a lower bound with a
fully-counted number without noticing.

Usage
-----
    python tools/_measure_model_profile.py --models all
    python tools/_measure_model_profile.py --models demucs,bsrnn_simo,umx
    python tools/_measure_model_profile.py --worker demucs      # internal
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as _paths  # noqa: E402

_paths.setup_env()

ROOT = str(_paths.PROJECT_ROOT)
TOOLS = os.path.join(ROOT, "tools")
OUT_CSV = os.path.join(ROOT, "04_reports", "_shared", "data", "model_analysis", "model_profile.csv")
OUT_JSON = os.path.join(ROOT, "04_reports", "_shared", "data", "model_analysis", "model_profile.json")

# fixed measurement protocol
SR = 44100
CH = 2
SECONDS = 10.0
WARMUP = 2
REPEAT = 5

# group labels for the figure (kept explicit: group discipline is a project law,
# single-target models must never share an axis with 4-stem models unlabelled)
GROUP = {
    "oracle": "analytic",
    "rpca": "analytic",
    "umx": "4-stem", "mdx": "4-stem", "demucs": "4-stem",
    "convtasnet": "4-stem", "mmdenselstm": "4-stem",
    "bsrnn_all": "4-stem", "bsrnn_large_all": "4-stem",
    "bsrnn_simo": "4-stem", "dprnn": "4-stem",
    "bsroformer_l12": "1-target", "bsrnn": "1-target",
    "bsrnn_large": "1-target", "bsroformer_l6": "2-target",
}

try:                     # board-2 models share the exact same loader contract
    from _denoise_registry import REGISTRY as _DEN_REG, GROUP as _DEN_GROUP
except Exception:                                        # noqa: BLE001
    _DEN_REG, _DEN_GROUP = {}, {}
GROUP.update(_DEN_GROUP)


def _all_reg():
    """Separation + denoise registries behind a single lookup."""
    import benchmark_model_universal as B
    return {**B.REGISTRY, **_DEN_REG}


FIELDS = ["model_key", "label", "group", "kind", "params_M", "params_raw",
          "weights_mb", "n_modules", "n_targets", "targets", "native_sr", "native_ch",
          "macs_G", "flops_G", "flops_coverage", "peak_vram_mb", "peak_vram_infer_mb",
          "latency_s", "rtf", "load_s", "device", "notes"]


# ------------------------------------------------------------------ discovery
def extract_modules(run_fn, max_depth=5):
    """Collect every nn.Module reachable from a closure, without changing loaders."""
    import torch

    def walk(o, depth, seen):
        if id(o) in seen or depth > max_depth:
            return []
        seen.add(id(o))
        out = []
        if isinstance(o, torch.nn.Module):
            out.append(o)
        elif isinstance(o, (list, tuple, set)):
            for x in o:
                out += walk(x, depth + 1, seen)
        elif isinstance(o, dict):
            for x in o.values():
                out += walk(x, depth + 1, seen)
        return out

    mods, seen = [], set()
    for cell in (getattr(run_fn, "__closure__", None) or ()):
        try:
            mods += walk(cell.cell_contents, 0, seen)
        except ValueError:
            pass
    # de-duplicate while keeping order
    uniq, ids = [], set()
    for m in mods:
        if id(m) not in ids:
            ids.add(id(m))
            uniq.append(m)
    return uniq


def weights_mb(info):
    """On-disk weight size, from the info dict if the loader recorded it."""
    for k in ("ckpt", "weights", "path"):
        p = info.get(k)
        if isinstance(p, str) and os.path.isfile(p):
            return round(os.path.getsize(p) / 1e6, 2)
    if isinstance(info.get("ckpt_path"), str) and os.path.isfile(info["ckpt_path"]):
        return round(os.path.getsize(info["ckpt_path"]) / 1e6, 2)
    return ""


# ---------------------------------------------------------------------- worker
def worker(key, device):
    import numpy as np
    import torch

    sys.path.insert(0, TOOLS)
    import benchmark_model_universal as B

    spec = _all_reg()[key]
    rec = {"model_key": key, "label": spec["label"], "group": GROUP.get(key, "other"),
           "device": device}

    t0 = time.time()
    run_fn, load_s, info = spec["loader"]()
    rec["load_s"] = round(load_s, 2)
    rec["kind"] = info.get("kind", "")
    rec["native_sr"] = info.get("sample_rate") or info.get("sr") or SR
    rec["native_ch"] = info.get("channels") or info.get("num_channels") or CH
    rec["targets"] = ",".join(info.get("targets") or spec.get("targets") or [])
    rec["n_targets"] = len(info.get("targets") or []) or ""
    rec["weights_mb"] = weights_mb(info)
    rec["note_info"] = ""

    mods = extract_modules(run_fn)
    rec["n_modules"] = len(mods)
    if mods:
        n = sum(p.numel() for m in mods for p in m.parameters())
        rec["params_raw"] = n
        rec["params_M"] = round(n / 1e6, 3)
    else:
        rec["params_raw"] = 0
        rec["params_M"] = 0.0
        rec["note_info"] = "analytic/classical: no nn.Module"

    x = (np.random.default_rng(0).standard_normal((int(SR * SECONDS), CH))
         .astype(np.float32) * 0.1)

    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda and mods:
        for m in mods:
            m.to("cuda")

    def call():
        return run_fn(x)

    # ---- warm-up (also brings CUDA kernels / caches up) -------------------
    # Counting the keys of the real output is the only reliable way to know how
    # many stems a model emits: several loaders omit `targets` from their info
    # dict, and a missing field silently becomes an empty table cell.
    out_keys = None
    for _ in range(WARMUP):
        try:
            o = call()
            if isinstance(o, dict) and o:
                out_keys = sorted(o.keys())
        except Exception:                                     # noqa: BLE001
            break
    if out_keys:
        rec["targets"] = ",".join(out_keys)
        rec["n_targets"] = len(out_keys)

    # ---- FLOPs -----------------------------------------------------------
    macs = 0
    cov = "torch-dispatch (conv/linear/matmul/norm only)"
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fcm = FlopCounterMode(display=False)
        with fcm:
            call()
        macs = int(fcm.get_total_flops())
    except Exception as e:                                    # noqa: BLE001
        cov = "unavailable: %s" % type(e).__name__
    rec["macs_G"] = round(macs / 1e9, 3)
    # torch's flop counter already reports 2*MACs for conv/matmul
    rec["flops_G"] = round(macs / 1e9, 3)
    rec["flops_coverage"] = cov
    if not mods:
        rec["flops_coverage"] = "no torch ops dispatched (analytic/classical)"

    # ---- peak VRAM -------------------------------------------------------
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        call()
    except Exception:                                         # noqa: BLE001
        pass
    if use_cuda:
        rec["peak_vram_infer_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    else:
        rec["peak_vram_infer_mb"] = ""

    # ---- latency ---------------------------------------------------------
    ts = []
    for _ in range(REPEAT):
        if use_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            call()
        except Exception as e:                                # noqa: BLE001
            rec["latency_s"] = ""
            rec["note_info"] = (rec["note_info"] + " | latency failed: %s"
                                % type(e).__name__).strip(" |")
            break
        if use_cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    if ts:
        med = st.median(ts)
        rec["latency_s"] = round(med, 4)
        rec["rtf"] = round(med / SECONDS, 4)
        rec["latency_min_s"] = round(min(ts), 4)
        rec["latency_spread_pct"] = round(100 * (max(ts) - min(ts)) / med, 1) if med else ""
    else:
        rec["rtf"] = ""

    if use_cuda and mods:
        torch.cuda.reset_peak_memory_stats()
        for m in mods:
            m.to("cuda")
        try:
            call()
        except Exception:                                     # noqa: BLE001
            pass
        rec["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    else:
        rec["peak_vram_mb"] = ""

    rec["wall_worker_s"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------------ main
def _run_worker_subprocess(key, device):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", key, "--device", device]
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           timeout=1800, cwd=TOOLS)
    except subprocess.TimeoutExpired:
        return {"model_key": key, "status": "TIMEOUT", "error": "1800 s"}
    out = p.stdout or ""
    marker = "@@JSON@@"
    if marker in out:
        blob = out.split(marker, 1)[1].strip().splitlines()[0]
        try:
            rec = json.loads(blob)
            rec["status"] = "PASS"
            return rec
        except Exception as e:                                # noqa: BLE001
            return {"model_key": key, "status": "BADJSON", "error": "%s" % e,
                    "stdout_tail": out[-600:]}
    return {"model_key": key, "status": "FAIL",
            "error": "no JSON marker", "stdout_tail": out[-600:],
            "stderr_tail": (p.stderr or "")[-600:], "elapsed": round(time.time() - t, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--worker", default=None)
    a = ap.parse_args()

    if a.worker:
        try:
            rec = worker(a.worker, a.device)
        except Exception as e:                                # noqa: BLE001
            print("WORKER FAIL", type(e).__name__, e)
            print(traceback.format_exc()[-1200:])
            return 1
        print("@@JSON@@")
        print(json.dumps(rec, ensure_ascii=False))
        return 0

    sys.path.insert(0, TOOLS)
    import benchmark_model_universal as B

    reg = _all_reg()
    keys = list(reg.keys()) if a.models == "all" else \
        [k.strip() for k in a.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in reg]
    if unknown:
        print("unknown model keys: %s" % unknown)
        return 2

    print("=" * 78)
    print("profile protocol: %d Hz / %d ch / %.0f s / FLOPs = 2xMACs / median of %d"
          % (SR, CH, SECONDS, REPEAT))
    print("models: %d   device: %s" % (len(keys), a.device))
    print("=" * 78)

    recs = []
    for i, k in enumerate(keys, 1):
        print("[%2d/%2d] %s" % (i, len(keys), k), flush=True)
        rec = _run_worker_subprocess(k, a.device)
        recs.append(rec)
        if rec.get("status") == "PASS":
            print("        params=%-9s flops=%-8s latency=%-8s vram=%-8s"
                  % ("%sM" % rec.get("params_M"), "%sG" % rec.get("macs_G"),
                     ("%ss" % rec.get("latency_s")) if rec.get("latency_s") != "" else "-",
                     ("%sMB" % rec.get("peak_vram_mb")) if rec.get("peak_vram_mb") != "" else "-"))
        else:
            print("        %s  %s" % (rec.get("status"), rec.get("error", "")[:70]))

    # ---- write CSV -------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    import csv
    # Merge with previously measured models: the two boards are measured in two
    # passes, and a plain overwrite would silently drop board 1.
    merged = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("model_key"):
                    merged[row["model_key"]] = row
    for r in recs:
        merged[r["model_key"]] = {k: r.get(k, "") for k in FIELDS + ["status"]}
    order = [k for k in reg if k in merged]
    order += [k for k in merged if k not in order]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["status"], extrasaction="ignore")
        w.writeheader()
        for k in order:
            w.writerow(merged[k])

    prev_json = {}
    if os.path.exists(OUT_JSON):
        try:
            for r in json.load(open(OUT_JSON, encoding="utf-8")).get("records", []):
                prev_json[r.get("model_key")] = r
        except Exception:                                    # noqa: BLE001
            prev_json = {}
    for r in recs:
        prev_json[r["model_key"]] = r
    json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "protocol": {"sr": SR, "channels": CH, "seconds": SECONDS,
                            "warmup": WARMUP, "repeat": REPEAT, "flops": "2xMACs"},
               "records": [prev_json[k] for k in order if k in prev_json]},
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    ok = [r for r in recs if r.get("status") == "PASS"]
    print("=" * 78)
    print("PASS %d / %d   csv -> %s" % (len(ok), len(recs), OUT_CSV))
    for r in recs:
        if r.get("status") != "PASS":
            print("   %-16s %s %s" % (r.get("model_key"), r.get("status"),
                                      r.get("error", "")[:60]))
    return 0 if len(ok) == len(recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
