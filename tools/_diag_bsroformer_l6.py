# -*- coding: utf-8 -*-
"""诊断 BS-RoFormer L6 (ep_937) 输出是否真的对应 target=other。"""
from __future__ import annotations

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
sys.path.insert(0, str(_paths.DEMUCS_SRC))

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
    ref = np.asarray(ref, np.float64).ravel()
    est = np.asarray(est, np.float64).ravel()
    n = min(len(ref), len(est)); ref, est = ref[:n], est[:n]
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    tgt = a * ref
    noise = est - tgt
    return 10 * np.log10((tgt ** 2).sum() / ((noise ** 2).sum() + 1e-12) + 1e-12)


def main():
    stub = install_librosa_stub()
    mss = _paths.THIRD_PARTY / "Music-Source-Separation-Training"
    sys.path.insert(0, str(mss))
    import torch
    from utils.model_utils import bigshifts_wrapper, prefer_target_instrument
    from utils.settings import get_model_from_config

    stem = "model_bs_roformer_ep_937_sdr_10.5309"
    cfg_path = _paths.WEIGHTS / "BS-RoFormer" / "configs" / f"{stem}.yaml"
    ckpt = _paths.WEIGHTS / "BS-RoFormer" / f"{stem}.ckpt"

    model, config = get_model_from_config("bs_roformer", str(cfg_path))
    print(f"targets={prefer_target_instrument(config)} num_stems={config.model.get('num_stems')}")
    print(f"audio.chunk_size={config.audio.chunk_size} n_fft={config.audio.n_fft} "
          f"sample_rate={config.audio.sample_rate}")
    print(f"model.stft_n_fft={config.model.get('stft_n_fft')} "
          f"stft_hop_length={config.model.get('stft_hop_length')} "
          f"dim_freqs_in={config.model.get('dim_freqs_in')}")
    print(f"inference={dict(config.inference)}")
    print(f"training.instruments={config.training.instruments} "
          f"target_instrument={config.training.get('target_instrument')}")

    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    for k in ("state", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and k in sd:
            print(f"wrapper key='{k}'"); sd = sd[k]; break
    model.load_state_dict(sd)
    model.eval()

    song = "A Classic Education - NightOwl"
    d = _paths.MUSDB18_ROOT / "train" / song
    y, sr = sf.read(str(d / "mixture.wav"), dtype="float32", always_2d=True)
    n = sr * 10
    mix = np.ascontiguousarray(y[:n].T, np.float32)
    gt = {t: sf.read(str(d / f"{t}.wav"), dtype="float32", always_2d=True)[0][:n].mean(1)
          for t in ("vocals", "drums", "bass", "other")}

    with torch.no_grad():
        est = bigshifts_wrapper(config, model, torch.as_tensor(mix), torch.device("cpu"),
                                model_type="bs_roformer", pbar=False, bigshifts=1)
    key = list(est)[0] if isinstance(est, dict) else "arr"
    v = np.asarray(est[key] if isinstance(est, dict) else est, np.float32)
    if v.ndim == 2 and v.shape[0] == 2:
        v = v.T
    em = v.mean(1) if v.ndim == 2 else v
    print(f"\n输出 key='{key}' shape={v.shape} rms={np.sqrt((em**2).mean()):.5f}")

    mixm = mix.mean(0)
    print(f"mixture rms={np.sqrt((mixm**2).mean()):.5f}")
    print("\n-- 与各 GT 的 SI-SDR / 相关系数 --")
    for t, g in gt.items():
        print(f"  vs {t:7} SI-SDR {si_sdr(g, em):7.3f}   corr {np.corrcoef(g, em)[0,1]:6.3f}")
    # 残差假设: mixture - (vocals+drums+bass) 应约等于 other
    resid = mixm - (gt["vocals"] + gt["drums"] + gt["bass"])
    print(f"  vs resid(other=mixture-v-d-b) SI-SDR {si_sdr(resid, em):7.3f}")
    print(f"  est vs mixture SI-SDR {si_sdr(mixm, em):7.3f}")


if __name__ == "__main__":
    main()
