# -*- coding: utf-8 -*-
"""BS-RoFormer 推理探针：绕开 librosa->numba->llvmlite 的 SAC 阻塞。

根因: Music-Source-Separation-Training/models/bs_roformer/__init__.py 会一并导入
      MelBandRoformer，而它在模块级 `from librosa import filters`，
      于是 librosa.filters -> numba -> llvmlite.dll 被 Smart App Control 拦截。

对策: 在导入该包之前，向 sys.modules 注入一个只有 `.filters` 属性的 librosa 占位模块。
      我们只用 BSRoformer（纯 torch 实现，内部不调用 librosa），占位模块不会被真正调用。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT

sys.path.insert(0, str(_paths.TOOLS))
sys.path.insert(0, str(_paths.DEMUCS_SRC))

import numpy as np  # noqa: E402


def install_librosa_stub():
    """注入最小 librosa 占位模块，避免 librosa.filters 触发 numba/llvmlite。"""
    if "librosa" in sys.modules:
        return False
    lib = types.ModuleType("librosa")
    filt = types.ModuleType("librosa.filters")
    lib.filters = filt
    lib.__path__ = []
    sys.modules["librosa"] = lib
    sys.modules["librosa.filters"] = filt
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="L12", choices=["L12", "L6"])
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--song", default="A Classic Education - NightOwl")
    ap.add_argument("--subset", default="train")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    stub = install_librosa_stub()
    print(f"[probe] librosa stub installed = {stub}")

    mss = _paths.THIRD_PARTY / "Music-Source-Separation-Training"
    sys.path.insert(0, str(mss))

    import torch
    import soundfile as sf
    from utils.model_utils import bigshifts_wrapper, prefer_target_instrument
    from utils.settings import get_model_from_config

    stem = {
        "L12": "model_bs_roformer_ep_317_sdr_12.9755",
        "L6": "model_bs_roformer_ep_937_sdr_10.5309",
    }[args.which]
    cfg_path = _paths.WEIGHTS / "BS-RoFormer" / "configs" / f"{stem}.yaml"
    ckpt = _paths.WEIGHTS / "BS-RoFormer" / f"{stem}.ckpt"

    t0 = time.time()
    model, config = get_model_from_config("bs_roformer", str(cfg_path))
    print(f"[probe] model built in {time.time()-t0:.2f}s  ({type(model).__name__})")
    print(f"[probe] target instruments = {prefer_target_instrument(config)}")
    print(f"[probe] num_stems = {config.model.get('num_stems')}  dim={config.model.get('dim')} "
          f"depth={config.model.get('depth')}")

    t1 = time.time()
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state", "state_dict", "model_state_dict"):
            if k in sd:
                print(f"[probe] ckpt wrapper key = '{k}'")
                sd = sd[k]
                break
    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"[probe] ckpt loaded in {time.time()-t1:.2f}s  tensors={len(sd)} params={n_params/1e6:.2f}M")

    model.load_state_dict(sd)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"[probe] device = {dev}  use_amp = {getattr(config.training, 'use_amp', True)}")

    # 片段
    d = _paths.MUSDB18_ROOT / args.subset / args.song
    y, sr = sf.read(str(d / "mixture.wav"), dtype="float32", always_2d=True)
    n = int(sr * args.duration)
    mix = np.ascontiguousarray(y[:n].T, dtype=np.float32)   # (2, T)
    print(f"[probe] mix {mix.shape} sr={sr}")

    t2 = time.time()
    with torch.no_grad():
        est = bigshifts_wrapper(config, model, torch.as_tensor(mix), torch.device(dev),
                                model_type="bs_roformer", pbar=False, bigshifts=1)
    infer = time.time() - t2
    print(f"[probe] infer {infer:.2f}s  RTF {infer/args.duration:.3f}")
    if isinstance(est, dict):
        for k, v in est.items():
            print(f"[probe]   {k}: shape={np.shape(v)} dtype={np.asarray(v).dtype} "
                  f"rms={float(np.sqrt(np.mean(np.asarray(v)**2))):.5f}")
    else:
        print(f"[probe] ndarray shape={np.shape(est)}")

    if args.save:
        out = _paths.OUTPUTS / f"BS-RoFormer-{args.which}" / args.song
        out.mkdir(parents=True, exist_ok=True)
        items = est.items() if isinstance(est, dict) else ((t, est[i]) for i, t in enumerate(["vocals", "drums", "bass", "other"]))
        for k, v in items:
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 2 and v.shape[0] == 2:
                v = v.T
            sf.write(str(out / f"{k}.wav"), np.clip(v, -1, 1), sr, subtype="PCM_16")
            print(f"[probe] saved {out / (k + '.wav')}")

    print("[probe] OK")


if __name__ == "__main__":
    main()
