# -*- coding: utf-8 -*-
"""探测哪些关键依赖在本机被应用程序控制策略拦截。只读诊断。"""
import importlib
import sys
import traceback

TESTS = [
    ("numpy", "import numpy"),
    ("numpy.random", "import numpy as np; np.random.randn(2)"),
    ("numpy.random.default_rng", "import numpy as np; np.random.default_rng(0).random(2)"),
    ("scipy", "import scipy"),
    ("scipy.signal", "from scipy.signal import stft, istft"),
    ("scipy.linalg", "from scipy.linalg import svd"),
    ("soundfile", "import soundfile as sf; sf.info"),
    ("sklearn", "from sklearn.decomposition import FastICA"),
    ("librosa.import", "import librosa"),
    ("librosa.stft", "import numpy as np, librosa; "
                     "librosa.stft(np.zeros(2048, np.float32), n_fft=2048, hop_length=512)"),
    ("numba", "import numba"),
    ("llvmlite.binding", "import llvmlite.binding as b; b.llvm_version_info"),
    ("museval", "import museval"),
    ("pystoi", "import pystoi"),
    ("mir_eval", "import mir_eval"),
    ("pysepm", "import pysepm"),
    ("torch", "import torch"),
    ("torch.matmul", "import torch; torch.matmul(torch.randn(4,4), torch.randn(4,4))"),
    ("torchaudio", "import torchaudio"),
    ("einops", "import einops"),
    ("pandas", "import pandas"),
    ("matplotlib", "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt"),
]

ok, bad = [], []
for name, code in TESTS:
    try:
        g = {}
        exec(code, g)
        ok.append(name)
        print(f"  OK    {name}")
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:150]}"
        bad.append((name, msg))
        print(f"  BLOCK {name:<26} {msg}")

print("\n---- summary ----")
print("OK   :", ", ".join(ok))
print("BLOCK:", ", ".join(n for n, _ in bad))
try:
    sys.stdout.flush()
except Exception:
    pass
