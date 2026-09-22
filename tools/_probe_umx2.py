# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
import os, traceback
import numpy as np, torch

from openunmix import predict as P

y = (np.random.randn(2, 44100 * 3) * 0.01).astype(np.float32)
try:
    out = P.separate(audio=torch.as_tensor(y), rate=44100,
                     targets=["vocals", "drums"],
                     model_str_or_path="umxhq", device="cpu", filterbank="torch")
    print("OK", type(out), np.shape(out))
except Exception:
    traceback.print_exc()
