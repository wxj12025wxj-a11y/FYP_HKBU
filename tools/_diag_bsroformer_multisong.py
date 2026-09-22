# -*- coding: utf-8 -*-
"""BS-RoFormer 多曲/多偏移复核：判断低 SI-SDR 是普遍现象还是本片段特例。"""
from __future__ import annotations

import argparse
import os
import sys
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

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def install_librosa_stub():
    try:
        import librosa.filters  # noqa: F401
        return False
    except Exception:
        pass
    lib = types.ModuleType("librosa"); lib.__path__ = []
    filt = types.ModuleType("librosa.filters")
    lib.filters = filt
    sys.modules["librosa"] = lib
    sys.modules["librosa.filters"] = filt
    return True


def si_sdr(ref, est):
    ref = np.asarray(ref, np.float64).ravel(); est = np.asarray(est, np.float64).ravel()
    n = min(len(ref), len(est)); ref, est = ref[:n], est[:n]
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    tgt = a * ref; nz = est - tgt
    return 10 * np.log10((tgt ** 2).sum() / ((nz ** 2).sum() + 1e-12) + 1e-12)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="L6")
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--songs", nargs="*", default=[
        "A Classic Education - NightOwl",
        "ANiMAL - Clinic A",
        "Actions - One Minute Smile",
        "Alexander Ross - Goodbye Bolero",
    ])
    ap.add_argument("--offsets", nargs="*", type=float, default=[0.0])
    args = ap.parse_args()

    install_librosa_stub()
    sys.path.insert(0, str(_paths.THIRD_PARTY / "Music-Source-Separation-Training"))
    import torch
    from utils.model_utils import bigshifts_wrapper, prefer_target_instrument
    from utils.settings import get_model_from_config

    stem = {"L12": "model_bs_roformer_ep_317_sdr_12.9755",
            "L6": "model_bs_roformer_ep_937_sdr_10.5309"}[args.which]
    model, config = get_model_from_config(
        "bs_roformer", str(_paths.WEIGHTS / "BS-RoFormer" / "configs" / f"{stem}.yaml"))
    sd = torch.load(str(_paths.WEIGHTS / "BS-RoFormer" / f"{stem}.ckpt"),
                    map_location="cpu", weights_only=False)
    for k in ("state", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]; break
    model.load_state_dict(sd); model.eval()
    tgt = prefer_target_instrument(config)[0]
    print(f"which={args.which} target={tgt}")

    print(f"\n{'song':38}{'off':>6}{'target SDR*':>12}{'corr':>8}{'rms_est':>9}{'rms_gt':>9}")
    for song in args.songs:
        d = _paths.MUSDB18_ROOT / "train" / song
        for off in args.offsets:
            y, sr = sf.read(str(d / "mixture.wav"), dtype="float32", always_2d=True)
            i0, n = int(sr * off), int(sr * args.duration)
            mix = np.ascontiguousarray(y[i0:i0 + n].T, np.float32)
            if mix.shape[1] < n:
                continue
            g = sf.read(str(d / f"{tgt}.wav"), dtype="float32", always_2d=True)[0][i0:i0 + n].mean(1)
            with torch.no_grad():
                est = bigshifts_wrapper(config, model, torch.as_tensor(mix), torch.device("cpu"),
                                        model_type="bs_roformer", pbar=False, bigshifts=1)
            v = np.asarray(est[tgt] if isinstance(est, dict) else est, np.float32)
            if v.ndim == 2 and v.shape[0] == 2:
                v = v.T
            em = v.mean(1) if v.ndim == 2 else v
            print(f"{song[:37]:38}{off:6.0f}{si_sdr(g, em):12.3f}{corr(g, em):8.3f}"
                  f"{np.sqrt((em**2).mean()):9.4f}{np.sqrt((g**2).mean()):9.4f}", flush=True)


if __name__ == "__main__":
    main()
