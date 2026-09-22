"""Capture the *real* layer tree of every model via forward hooks (T2.1).

Why hooks instead of reading the source
---------------------------------------
Architecture diagrams drawn from a paper's figure are frequently wrong about the
implementation actually shipped in the checkpoint.  This script runs one real
forward pass and records, for every `nn.Module` that fired:

    module path  |  class name  |  #params  |  input shapes  |  output shapes

so every shape printed on a figure can be traced back to a JSON in
`04_reports/_shared/data/model_analysis/graph_<key>.json`.  That traceability is the
acceptance criterion for D2 ("shape 可溯源；参数量占比之和 = 100%").

Wrappers are a known trap: several loaders return a `run_fn` closure over the
model (and Demucs' `BagOfModels` even raises `NotImplementedError` if you call
`forward` directly).  We therefore never call the top module ourselves -- we
hook every module reachable from the closure and then invoke `run_fn(x)`, which
is the one entry point known to work for all 22 models.

Two bugs this rewrite fixes (both produced plausible-looking, wrong data)
------------------------------------------------------------------------
1. `max(mods, key=_nparams)` looked harmless but silently broke every
   **aggregated** entry.  `bsrnn_all` is four independently loaded
   single-target nets, so hooking only the largest one traced 45.5 M of a
   164.1 M model -- and a parameter-share bar drawn from that trace can never
   add up to 100 %.  We now hook *all* ancestor-free closure roots and prefix
   them `net0/`, `net1/`, ...
2. Ancestry was inferred from string prefixes, so `<root>` (whose children are
   `encoder.*`, not `<root>.encoder`) was classified as a **leaf** and its full
   parameter count was added on top of its own children's -- every
   single-network model came out at exactly 2.00x its true size.  Each node now
   carries an explicit `parent` field and downstream code builds a children map
   from that, never from string surgery.

Usage
-----
    python tools/_extract_model_graph.py --models all
    python tools/_extract_model_graph.py --models demucs,bsrnn_simo,mdx
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as _paths  # noqa: E402

_paths.setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import benchmark_model_universal as B  # noqa: E402
from _measure_model_profile import extract_modules, _all_reg  # noqa: E402

ROOT_DIR = str(_paths.PROJECT_ROOT)
OUT_DIR = os.path.join(ROOT_DIR, "04_reports", "_shared", "data", "model_analysis")
PROFILE_CSV = os.path.join(OUT_DIR, "model_profile.csv")
SR = 44100
CH = 2
SECONDS = 10.0
# Generous cap: parameter shares must be computed over the WHOLE tree, so
# truncating the node list would silently under-count the denominator.
MAX_MODULES_PER_JSON = 20000


def _profile_row(key):
    """Authoritative params/targets for `key`, from the profile table.

    The profile CSV is the single source of truth for a model's size, because
    for aggregated entries the traced tree may legitimately differ from the
    headline number.
    """
    try:
        with open(PROFILE_CSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("model_key") == key:
                    return r
    except OSError:
        pass
    return {}


# --------------------------------------------------------------------- shapes
def _shapes(o, depth=0, cap=6):
    """Compact, JSON-safe description of the tensors inside an arbitrary output."""
    if depth > 3:
        return "..."
    if isinstance(o, torch.Tensor):
        return list(o.shape)
    if isinstance(o, (list, tuple)):
        out = [_shapes(x, depth + 1, cap) for x in o[:cap]]
        if len(o) > cap:
            out.append("...(+%d)" % (len(o) - cap))
        return out
    if isinstance(o, dict):
        return {str(k): _shapes(v, depth + 1, cap) for k, v in list(o.items())[:cap]}
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return type(o).__name__


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def _closure_roots(run_fn):
    """Top-level network(s) in the loader closure.

    Loaders close over helper objects as well as the network, and the
    aggregated ones close over N sibling networks.  Only modules with **no
    ancestor inside the closure** are roots; anything else would be traced twice
    (once as itself, once as part of its parent) and inflate every parameter sum.
    """
    mods = list({id(m): m for m in extract_modules(run_fn)}.values())
    if not mods:
        return []
    ids = {id(m) for m in mods}
    roots = [m for m in mods
             if not any(a is not m and id(a) in ids for a in m.modules())]
    roots.sort(key=_nparams, reverse=True)
    return roots


# ------------------------------------------------------------------- capture
def capture(key, spec, device="cpu"):
    t0 = time.time()
    run_fn, load_s, info = spec["loader"]()
    roots = _closure_roots(run_fn)
    if not roots:
        return {"model_key": key, "status": "NO_MODULE",
                "note": "analytic / classical: no nn.Module in the loader closure",
                "load_s": round(load_s, 3)}

    x = (np.random.default_rng(0).standard_normal((int(SR * SECONDS), CH))
         .astype(np.float32) * 0.1)

    recs = {}
    order = []
    handles = []
    multi = len(roots) > 1

    def mk_hook(path, mod):
        def hook(module, inp, out):
            if path not in recs:
                recs[path] = {
                    "path": path,
                    "parent": "",
                    "cls": type(module).__name__,
                    # module identity: the SAME object is often reachable under two
                    # names (`BN` and `separator.0.BN`), and its parameters must be
                    # counted once, not once per alias
                    "mid": id(module),
                    "params": _nparams(module),
                    "in_shapes": _shapes(inp),
                    "out_shapes": _shapes(out),
                    "calls": 0,
                }
                order.append(path)
            recs[path]["calls"] += 1
        return hook

    mod_by_path = {}
    for i, r in enumerate(roots):
        # `net{i}/` prefix only when there really are siblings, so single-network
        # models keep the clean, paper-like paths (`encoder.3.2`).
        pre = ("net%d/" % i) if multi else ""
        for sub, mod in r.named_modules():
            p = (pre + sub) if sub else (pre + "<root>")
            mod_by_path[p] = mod
            handles.append(mod.register_forward_hook(mk_hook(p, mod)))

    err = ""
    try:
        with torch.no_grad():
            run_fn(x)
    except Exception as e:                                       # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, str(e)[:200])
    finally:
        for h in handles:
            h.remove()

    # parent links must be written AFTER the forward pass: `recs` does not exist
    # until the hooks fire, so doing this before it silently left every node with
    # an empty parent and the leaf-only sum below then counted parents + children.
    for i, r in enumerate(roots):
        pre = ("net%d/" % i) if multi else ""
        for sub, _mod in r.named_modules():
            if not sub:
                continue
            p = pre + sub
            if p in recs:
                recs[p]["parent"] = pre + (sub.rsplit(".", 1)[0]
                                           if "." in sub else "<root>")
        rp = pre + "<root>"
        if rp in recs:
            recs[rp]["parent"] = ""

    nodes = [recs[p] for p in order]

    # ------------------------------------------------------------------
    # Parameter attribution.
    #
    # Three traps inflate a naive "sum the leaves" total; all three were hit:
    #   (a) parent and child both fired -> the parent's parameters are counted
    #       once for itself and again for each child;
    #   (b) the same module OBJECT is reachable under two names (BSRNN's `BN` is
    #       also `separator.0.BN`) -> counted twice even after (a);
    #   (c) two DIFFERENT modules share the same parameter tensors (BSRNN's
    #       `Masker` and `separator` are weight-tied) -> invisible to (a) and
    #       (b) alike.  This is what left `bsrnn_simo` at 192.1 M vs a true
    #       108.7 M.
    # The only definition that survives all three: walk the root's own
    # `named_parameters()` and credit each DISTINCT tensor to the deepest fired
    # module that contains it.  Every parameter is then counted exactly once, so
    # the sum reconciles with the checkpoint by construction.
    # ------------------------------------------------------------------
    fired_set = set(recs)
    # Only modules that ACTUALLY FIRED may own parameters.  Using every
    # registered module here loses parameters silently: a module reachable as
    # both `BN` and `separator.0.BN` has `BN` as its canonical (shortest) path
    # even when only `separator.0.BN` fired, so its weights were credited to a
    # path that never appears in `exec_order` -- the budget then failed to
    # reconcile (demucs was short 2.63 M of 41.98 M) while every individual
    # number still looked plausible.
    live = {id(m): p for p, m in mod_by_path.items() if p in fired_set}
    canon = {}                               # module object -> canonical fired path
    for mid, p in live.items():
        cur = canon.get(mid)
        if cur is None or (len(p), p) < (len(cur), cur):
            canon[mid] = p
    canon_paths = set(canon.values())

    attr = {p: 0 for p in canon_paths}
    for i, r in enumerate(roots):
        pre = ("net%d/" % i) if multi else ""
        fallback = (pre + "<root>") if (pre + "<root>") in fired_set else ""
        for name, par in r.named_parameters():
            full = pre + name
            hit = ""
            parts = full.split(".")
            for k in range(len(parts) - 1, 0, -1):        # longest prefix first
                cand = ".".join(parts[:k])
                if cand in canon_paths:
                    hit = cand
                    break
            hit = hit or fallback
            if hit:
                attr[hit] = attr.get(hit, 0) + par.numel()

    root_p = sum(_nparams(r) for r in roots)
    traced_p = float(sum(attr.values()))
    # hard invariant: the attribution must reproduce the checkpoint exactly.
    # If it ever does not, the figure's "100 %" bar would be a lie.
    reconcile = abs(traced_p - root_p) <= max(1.0, 1e-6 * root_p)
    for n in nodes:
        if n["path"] != canon[n["mid"]]:
            n["role"] = "alias"              # same module, non-canonical name
            n["params_attr"] = 0
            continue
        n["params_attr"] = attr.get(n["path"], 0)
        n["role"] = "leaf" if n["params_attr"] > 0 else "container"
    for n in nodes:
        n.pop("mid", None)                   # process-local id: not a stable datum
    n_alias = sum(1 for n in nodes if n["role"] == "alias")

    prow = _profile_row(key)
    p_csv = prow.get("params_M") or ""
    note = ""
    if multi:
        note = ("aggregate entry: all %d sibling networks in the loader closure are "
                "traced (paths prefixed net0/ .. net%d/)" % (len(roots), len(roots) - 1))
    elif len(extract_modules(run_fn)) > 1:
        note = ("loader closure holds extra helper objects beside the network; "
                "only the %d ancestor-free root(s) are traced" % len(roots))
    if err:
        note = (err + ("  |  " + note if note else ""))
    if not reconcile:
        note = (note + "  |  " if note else "") + (
            "[FAIL] attribution does not reconcile with the checkpoint: "
            "traced %.4f M vs roots %.4f M" % (traced_p / 1e6, root_p / 1e6))
    if n_alias:
        note = (note + "  |  " if note else "") + (
            "%d path(s) are non-canonical aliases of a module that is also reachable "
            "under another name; they are marked role='alias' and carry no parameters"
            % n_alias)

    rec = {
        "model_key": key,
        "label": prow.get("label") or spec.get("label") or info.get("name") or key,
        "group": prow.get("group", ""),
        "kind": prow.get("kind", ""),
        "status": "OK" if nodes else "EMPTY",
        # authoritative size (single source of truth = model_profile.csv)
        "params_M": (round(float(p_csv), 4) if p_csv else round(root_p / 1e6, 4)),
        "params_M_source": ("model_profile.csv" if p_csv else "traced-roots"),
        # what this particular trace actually covered
        "n_roots": len(roots),
        "roots": [{"cls": type(r).__name__, "params_M": round(_nparams(r) / 1e6, 4)}
                  for r in roots],
        "traced_params_M": round(traced_p / 1e6, 4),
        "traced_vs_csv_ratio": (round(traced_p / 1e6 / float(p_csv), 4) if p_csv else ""),
        # True when the parameter attribution reproduces the checkpoint exactly.
        # A false here means a figure's 100 % bar cannot be trusted.
        "attribution_reconciles": bool(reconcile),
        "n_aliased_paths": n_alias,
        "hooked_root_cls": type(roots[0]).__name__,
        "hooked_root_params_M": round(_nparams(roots[0]) / 1e6, 4),
        "n_targets": prow.get("n_targets") or "",
        "native_sr": prow.get("native_sr") or info.get("sr") or info.get("sample_rate") or "",
        "n_modules_total": sum(len(list(r.named_modules())) for r in roots),
        "n_modules_fired": len(nodes),
        "load_s": round(load_s, 3),
        "capture_s": round(time.time() - t0, 2),
        "note": note,
        "exec_order": nodes[:MAX_MODULES_PER_JSON],
    }
    del roots, run_fn, recs, nodes
    gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="'all' or comma-separated model keys")
    args = ap.parse_args()

    reg = _all_reg()
    keys = sorted(reg) if args.models == "all" else [k.strip() for k in args.models.split(",") if k.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)

    ok = bad = 0
    for k in keys:
        if k not in reg:
            print("ERR  unknown model %r" % k)
            bad += 1
            continue
        try:
            rec = capture(k, reg[k])
        except Exception as e:                                    # noqa: BLE001
            rec = {"model_key": k, "status": "FAIL",
                   "note": "%s: %s" % (type(e).__name__, str(e)[:200]),
                   "trace": traceback.format_exc()[-800:]}
        fp = os.path.join(OUT_DIR, "graph_%s.json" % k)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        if rec.get("status") == "OK":
            ok += 1
            print("OK   %-26s roots=%d %-14s fired=%5d/%5d  csv=%.2fM traced=%.2fM (x%s)%s  %.1fs"
                  % (k, rec["n_roots"], rec["hooked_root_cls"], rec["n_modules_fired"],
                     rec["n_modules_total"], rec["params_M"], rec["traced_params_M"],
                     rec["traced_vs_csv_ratio"],
                     "" if rec["attribution_reconciles"] else "  [FAIL-NOT-RECONCILED]",
                     rec["capture_s"]))
        else:
            bad += 1
            print("ERR  %-26s %s  %s" % (k, rec.get("status"), (rec.get("note") or "")[:90]))

    print("\n[summary] %d OK / %d not-OK  ->  %s" % (ok, bad, OUT_DIR))


if __name__ == "__main__":
    main()
