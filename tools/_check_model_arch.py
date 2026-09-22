# -*- coding: utf-8 -*-
"""Self-check for D2: dump what every architecture figure actually claims.

Run after _plot_model_arch.py.  Prints, per family: the parameter-composition
buckets (must sum to 100 %), the top skeleton rows with their real shapes, and
any concept-flow stage whose shape annotation could not be sourced from the
trace.  This is the machine half of the T2.3 review -- the human half is the
supervisor/user looking at the PNGs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
import matplotlib
matplotlib.use("Agg")
import _plot_model_arch as M

bad = 0
for fam in M.FAMILIES:
    g = M.load_graph(fam["primary"])
    state = M.graph_state(g)
    items, tot = M.composition(g, fam["groups"])
    rows = M.skeleton_rows(g, max_rows=4)
    print("=" * 100)
    print("FAMILY %s" % fam["title"])
    print("| state=%s  traced_total=%.4f M  keys=%s" % (state, tot, ",".join(fam["keys"])))
    if tot:
        ssum = sum(v for _, v, _ in items)
        print("| buckets=%d  sum=%.4f M  (delta=%+.4f M)" % (len(items), ssum, ssum - tot))
        for lab, v, _ in items:
            print("|    %-44s %8.3f M  %6.2f%%" % (lab, v, 100 * v / tot))
        if abs(ssum - tot) > 1e-3:
            print("|    [FAIL] 占比之和 != 总量  (delta=%+.4f M)" % (ssum - tot))
            bad += 1
    for r in rows:
        print("| row %-30s %-14s %6.1f%% calls=%-3d %s"
              % (r["path"][:30], r["cls"][:14], r["pct"], r["calls"], r["io"][:52]))
    for sg in fam["stages"]:
        if sg.get("find") and not M.resolve_shape(g, sg["find"]):
            print("| [UNRESOLVED] stage %s find=%s" % (sg["name"], sg["find"]))
            bad += 1

print("\n[summary] 需要处理的项：%d" % bad)
