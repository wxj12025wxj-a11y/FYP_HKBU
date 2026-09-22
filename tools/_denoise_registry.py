"""Loader registry for the denoise / dereverb board (board 2).

Why a separate registry instead of appending to `benchmark_model_universal.REGISTRY`:
the separation registry describes models that take a 44.1 kHz stereo mixture and
return stems.  Board-2 models are a different task with a different native I/O
contract (denoiser is 16 kHz mono, causal).  Keeping the two registries apart
keeps the boundary honest, while the *loader contract is identical* so every
downstream tool (profiling, evaluation, sweep) can be reused unchanged:

    loader() -> (run_fn, load_seconds, info)
    run_fn(mix_stereo: (N, 2) float32 @ 44.1 kHz) -> {target: (N, 2) float32}

The 44.1 kHz stereo wrapper is deliberate: it makes board-1 and board-2 models
addressable by exactly the same code, and the cascade experiment (stage 6) needs
precisely this -- feed one noisy mixture to any model, in any order.  Native
sample rate / channel count are recorded in `info` so nothing is hidden.

Sources (all local, nothing is downloaded at import time)
--------------------------------------------------------
  mel_roformer_denoise / _aggr  Mel-RoFormer-Denoise-Aufr33   (44.1 kHz stereo)
  mel_roformer_dereverb         dereverb_mel_band_roformer_anvuew
  mel_roformer_dereverb_echo    Dereverb-Echo_Mel_Band_Roformer
  denoiser_dns48/dns64/master64 facebookresearch/denoiser      (16 kHz mono)

⚠️ Community models, not peer-reviewed papers.  denoiser is CC-BY-NC 4.0.
Any report using these must carry the provenance note.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as _paths  # noqa: E402

_paths.setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = str(_paths.PROJECT_ROOT)
W = os.path.join(ROOT, "01_models", "_weights")
ZF = os.path.join(ROOT, "01_models", "_third_party", "Music-Source-Separation-Training")
DEN = os.path.join(ROOT, "01_models", "_third_party", "denoiser")

SR = 44100


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_t(x):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)


# --------------------------------------------------------------- Mel-RoFormer
def _make_mel_loader(folder, ckpt, yaml, label):
    def load():
        if ZF not in sys.path:
            sys.path.insert(0, ZF)
        from utils.settings import get_model_from_config

        t0 = time.time()
        ck_path = os.path.join(W, folder, ckpt)
        cfg_path = os.path.join(W, folder, yaml)
        model, config = get_model_from_config("mel_band_roformer", cfg_path)
        sd = torch.load(ck_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        info_load = model.load_state_dict(sd, strict=True)
        model.eval()
        dev = _device()
        model.to(dev)
        dt = time.time() - t0

        m = config.model if hasattr(config, "model") else config["model"]
        n_stems = int(getattr(m, "num_stems", 1) if not isinstance(m, dict)
                      else m.get("num_stems", 1))
        native_sr = int(getattr(config.audio, "sample_rate", 44100) if hasattr(config, "audio")
                        else config["audio"]["sample_rate"])
        targets = ["dry", "other"] if n_stems >= 2 else ["enhanced"]

        def run(mix):
            x = _as_t(mix.T)[None].to(dev)                    # (1, C, T)
            with torch.no_grad():
                y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y = y.detach().float().cpu().numpy()
            out = y[0]                                        # (n_stem, C, T) or (C, T)
            if out.ndim == 2:                                 # single stem
                return {targets[0]: out.T}
            return {t: out[i].T for i, t in enumerate(targets)}

        return run, dt, {
            "kind": "deep", "repo": "ZFTurbo/Music-Source-Separation-Training",
            "arch": "MelBandRoformer", "label": label,
            "params": sum(p.numel() for p in model.parameters()),
            "targets": targets, "n_stems": n_stems,
            "sample_rate": native_sr, "channels": 2,
            "ckpt": ck_path, "yaml": cfg_path, "device": dev,
            "n_missing_keys": len(info_load.missing_keys),
            "n_unexpected_keys": len(info_load.unexpected_keys),
            "license": "community checkpoint (non-peer-reviewed)",
        }

    return load


# --------------------------------------------------------------- denoiser
def _make_denoiser_loader(fname, hidden):
    def load():
        if DEN not in sys.path:
            sys.path.insert(0, DEN)
        from denoiser import pretrained as P
        from denoiser.dsp import convert_audio

        t0 = time.time()
        path = os.path.join(W, "denoiser", fname)
        # pretrained=False -> official architecture, then load our local file
        model = P._demucs(False, P.DNS_48_URL, hidden=hidden)  # noqa: SLF001
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        info_load = model.load_state_dict(sd, strict=True)
        model.eval()
        dev = _device()
        model.to(dev)
        dt = time.time() - t0

        def run(mix):
            wav = _as_t(mix.T)                                # (C, T) @ 44.1k
            wav = convert_audio(wav, SR, 16000, 1)            # (1, T16)
            with torch.no_grad():
                y = model(wav[None].to(dev))
            y = y.detach().float().cpu()[0, 0].numpy()        # (T16,)
            import torchaudio.functional as AF
            back = AF.resample(torch.as_tensor(y), 16000, SR).numpy()
            return {"enhanced": np.stack([back, back], axis=-1).astype(np.float32)}

        return run, dt, {
            "kind": "deep", "repo": "facebookresearch/denoiser",
            "arch": "Demucs (causal, waveform)",
            "params": sum(p.numel() for p in model.parameters()),
            "targets": ["enhanced"], "n_stems": 1,
            "sample_rate": 16000, "channels": 1, "hidden": hidden,
            "ckpt": path, "device": dev,
            "n_missing_keys": len(info_load.missing_keys),
            "n_unexpected_keys": len(info_load.unexpected_keys),
            "license": "CC-BY-NC 4.0 (non-commercial)",
        }

    return load


# ------------------------------------------------------------------ registry
REGISTRY = {
    "mel_roformer_denoise": dict(
        loader=_make_mel_loader(
            "mel_roformer_denoise",
            "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            "model_mel_band_roformer_denoise.yaml",
            "Mel-RoFormer-Denoise-Aufr33"),
        needs_gt=False,
        label="Mel-RoFormer-Denoise (aufr33)"),

    "mel_roformer_denoise_aggr": dict(
        loader=_make_mel_loader(
            "mel_roformer_denoise",
            "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
            "model_mel_band_roformer_denoise.yaml",
            "Mel-RoFormer-Denoise-Aufr33 (aggressive)"),
        needs_gt=False,
        label="Mel-RoFormer-Denoise (aggr)"),

    "mel_roformer_dereverb": dict(
        loader=_make_mel_loader(
            "mel_roformer_dereverb",
            "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            "dereverb_mel_band_roformer_anvuew.yaml",
            "Dereverb-Mel-RoFormer (anvuew)"),
        needs_gt=False,
        label="Dereverb-Mel-RoFormer"),

    "mel_roformer_dereverb_echo": dict(
        loader=_make_mel_loader(
            "mel_roformer_dereverb_echo",
            "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
            "config_dereverb-echo_mel_band_roformer.yaml",
            "Dereverb-Echo-Mel-RoFormer (Sucial)"),
        needs_gt=False,
        label="Dereverb-Echo-Mel-RoFormer"),

    "denoiser_dns48": dict(
        loader=_make_denoiser_loader("dns48-11decc9d8e3f0998.th", 48),
        needs_gt=False, label="denoiser dns48 (16 kHz)"),
    "denoiser_dns64": dict(
        loader=_make_denoiser_loader("dns64-a7761ff99a7d5bb6.th", 64),
        needs_gt=False, label="denoiser dns64 (16 kHz)"),
    "denoiser_master64": dict(
        loader=_make_denoiser_loader("master64-8a5dfb4bb92753dd.th", 64),
        needs_gt=False, label="denoiser master64 (16 kHz)"),
}

# group label used by the profile figure
GROUP = {
    "mel_roformer_denoise": "denoise",
    "mel_roformer_denoise_aggr": "denoise",
    "denoiser_dns48": "denoise",
    "denoiser_dns64": "denoise",
    "denoiser_master64": "denoise",
    "mel_roformer_dereverb": "dereverb",
    "mel_roformer_dereverb_echo": "dereverb",
}
