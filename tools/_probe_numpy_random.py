# -*- coding: utf-8 -*-
"""逐个测试 numpy.random 子模块能否加载，评估 Smart App Control 的受损面。"""
import importlib
import traceback

MODS = [
    "numpy.random._bounded_integers",
    "numpy.random._common",
    "numpy.random._generator",
    "numpy.random._mt19937",
    "numpy.random._pcg64",
    "numpy.random._philox",
    "numpy.random._sfc64",
    "numpy.random.bit_generator",
    "numpy.random.mtrand",
    "numpy.random",
]

for m in MODS:
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m}")
    except Exception as e:
        print(f"  BLOCK {m:<34} {type(e).__name__}: {str(e)[:90]}")

print()
try:
    import numpy as np
    print("  RandomState (mtrand 路径) ->", np.random.RandomState(0).rand(2))
except Exception as e:
    print("  RandomState FAIL:", str(e)[:120])
