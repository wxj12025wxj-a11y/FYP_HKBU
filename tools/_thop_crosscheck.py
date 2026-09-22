"""Cross-check the FlopCounterMode FLOPs with thop  (T1.3 / D1 acceptance).

D1's acceptance rule is "FLOPs 与 thop 交叉验证偏差 <5%，超标的逐条标注", so the
table must carry a second, independent FLOP count.  The two counters work
differently and therefore disagree by design:

  * `torch.utils.flop_counter.FlopCounterMode` hooks the **dispatcher**.  It sees
    every conv / linear / matmul / norm torch actually dispatches, including
    fused kernels it knows about, but it is blind to `torch.stft` and to fused
    attention.
  * `thop` hooks **nn.Module** instances and multiplies each module's registered
    input shape by its weight shape.  It sees only the module types in its own
    table, and it cannot see through custom `forward` implementations.

Consequence: they agree only for plain conv/linear stacks.  For RoFormer-style
models (STFT + attention) thop typically under-counts or refuses outright.  This
script therefore **records the number when it is obtainable and names the reason
when it is not** -- it never invents a value and never averages the two.

Units: thop returns MACs.  The main table's `flops_G` is 2 x MACs, so the
comparison is `2 * thop_macs` against `flops_G`.

This script only ADDS three columns (`thop_G`, `thop_dev_pct`, `thop_note`) to
the existing CSV; latency / VRAM columns measured on the GPU are untouched, so
the table does not have to be re-measured.

Usage
-----
    python tools/_thop_crosscheck.py
    python tools/_thop_crosscheck.py --models demucs,umx,mdx
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as _paths  # noqa: E402

_paths.setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _measure_model_profile import extract_modules, _all_reg  # noqa: E402

ROOT = str(_paths.PROJECT_ROOT)
CSV_PATH = os.path.join(ROOT, "04_reports", "_shared", "data", "model_analysis", "model_profile.csv")
JSON_OUT = os.path.join(ROOT, "04_reports", "_shared", "data", "model_analysis", "thop_crosscheck.json")

NEW_COLS = ["thop_G", "thop_dev_pct", "thop_note"]
TOL_PCT = 5.0
SR, CH, SECONDS = 44100, 2, 10.0
LAYOUTS = [("1,C,T", lambda t: t),
           ("1,T,C", lambda t: t.transpose(1, 2)),
           ("C,T", lambda t: t[0]),
           ("T,C", lambda t: t[0].t())]


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def probe(key, flops_g):
    """Return (thop_G or '', dev_pct or '', note)."""
    reg = _all_reg()
    if key not in reg:
        return "", "", "not-in-registry"
    try:
        run_fn, _load_s, _info = reg[key]["loader"]()
    except Exception as e:                                       # noqa: BLE001
        return "", "", "load-failed:%s" % type(e).__name__

    mods = list({id(m): m for m in extract_modules(run_fn)}.values())
    if not mods:
        return "", "", "no nn.Module (analytic/classical) - thop N/A by construction"
    aggregate = len(mods) > 1

    mods.sort(key=_nparams, reverse=True)
    x = torch.randn(1, CH, int(SR * SECONDS))
    try:
        from thop import profile as thop_profile
    except Exception as e:                                       # noqa: BLE001
        return "", "", "thop-import-failed:%s" % type(e).__name__

    last = "no module accepted the input"
    for m in mods[:3]:
        for lname, f in LAYOUTS:
            try:
                with torch.no_grad():
                    macs, params = thop_profile(m, inputs=(f(x),), verbose=False)
            except Exception as e:                               # noqa: BLE001
                last = "%s:%s" % (type(e).__name__, str(e)[:60])
                continue
            if macs and macs > 0:
                thop_g = 2 * macs / 1e9                          # MACs -> FLOPs
                dev = ((thop_g - flops_g) / flops_g * 100.0) if flops_g else None
                note = "%s @ %s; thop=2xMACs" % (type(m).__name__, lname)
                if aggregate:
                    # bsrnn_all / bsrnn_large_all load N independent single-target
                    # nets; thop was handed only the largest one, while flops_G is
                    # the aggregate.  The two numbers are not the same model, so
                    # the deviation is meaningless and must not be quoted.
                    dev = ""
                    note += ("; AGGREGATED loader (%d nets) -> thop saw 1 net only, "
                             "deviation NOT comparable" % len(mods))
                elif dev is not None and abs(dev) > TOL_PCT:
                    note += "  [FLAG: dev>%.0f%% -> count coverage differs]" % TOL_PCT
                del mods
                gc.collect()
                return round(thop_g, 3), (round(dev, 2) if dev not in ("", None) else ""), note
    del mods
    gc.collect()
    return "", "", "thop cannot profile this model (%s); FlopCounterMode lower bound stands" % last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="'all' or comma list")
    ap.add_argument("--json-only", action="store_true",
                    help="rebuild thop_crosscheck.json from the CSV, no probing")
    args = ap.parse_args()

    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = [c for c in rd.fieldnames if c not in NEW_COLS]
        rows = list(rd)

    if args.json_only:
        cases = {}
        for r in rows:
            dev = r.get("thop_dev_pct")
            cases[r["model_key"]] = {
                "thop_G": (float(r["thop_G"]) if r.get("thop_G") else ""),
                "dev_pct": (float(dev) if dev not in ("", None) else ""),
                "note": r.get("thop_note") or "",
            }
        with open(JSON_OUT, "w", encoding="utf-8") as f:
            json.dump({"tol_pct": TOL_PCT, "units": "thop_G = 2*MACs",
                       "n_models": len(cases), "cases": cases}, f,
                      ensure_ascii=False, indent=1)
        ncov = sum(1 for v in cases.values() if v["thop_G"] != "")
        print("rebuilt %s from CSV: %d models, %d with a thop number"
              % (JSON_OUT, len(cases), ncov))
        return

    want = None if args.models == "all" else {k.strip() for k in args.models.split(",")}
    summary = {}
    for r in rows:
        if want and r["model_key"] not in want:
            continue
        flops_g = float(r.get("flops_G") or 0)
        t0 = time.time()
        g, dev, note = probe(r["model_key"], flops_g)
        r["thop_G"], r["thop_dev_pct"], r["thop_note"] = g, dev, note
        summary[r["model_key"]] = {"thop_G": g, "dev_pct": dev, "note": note}
        mark = "OK  " if g != "" else "n/a "
        if g == "":
            devtxt = "        -        "
        elif dev in ("", None):
            # aggregate: number obtained, but not comparable -> no deviation
            devtxt = "%9.3fG  dev=  n/a  " % g
        else:
            devtxt = "%9.3fG  dev=%+.1f%%" % (g, dev)
        print("%s %-26s flops=%9.3fG thop=%s  %s  (%.1fs)"
              % (mark, r["model_key"], flops_g, devtxt, note[:58], time.time() - t0))

    # merge: keep original column order, append the three new ones
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + NEW_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # The JSON is a *derived* view: rebuild it from the CSV (which accumulates
    # across partial runs) rather than from this run's `summary`, otherwise a
    # subset re-run silently shrinks the file to just the models it touched.
    cases = {}
    for r in rows:
        dev = r.get("thop_dev_pct")
        cases[r["model_key"]] = {
            "thop_G": (float(r["thop_G"]) if r.get("thop_G") else ""),
            "dev_pct": (float(dev) if dev not in ("", None) else ""),
            "note": r.get("thop_note") or "",
        }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump({"tol_pct": TOL_PCT, "units": "thop_G = 2*MACs",
                   "n_models": len(cases), "cases": cases}, f,
                  ensure_ascii=False, indent=1)

    got = [v for v in summary.values() if v["thop_G"] != ""]
    flag = [k for k, v in summary.items()
            if v["dev_pct"] != "" and abs(v["dev_pct"]) > TOL_PCT]
    print("\n[summary] thop obtainable for %d/%d models; %d exceed %.0f%% -> %s"
          % (len(got), len(summary), len(flag), TOL_PCT, flag or "none"))
    print("wrote %s" % CSV_PATH)


if __name__ == "__main__":
    main()
