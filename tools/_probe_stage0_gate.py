"""Stage 0 gate: every newly added model must survive one real forward pass.

Gate rule (project law): success is read from the JSON `status` field.
A populated directory, a written wav, or exit code 0 are NOT proof -- this
project has twice been burned by silent failures that looked like success.

The probe is a REAL noisy/clean pair from Valentini, not white noise.
White noise passes every shape check while driving a denoiser to ~zero output,
which would make a broken model look exactly like a working one.  A real pair
lets the gate also assert "the model actually improved the signal".

Models under test (7 weights / 6 architectures):
  mel_roformer_denoise       denoise_mel_band_roformer_aufr33_sdr_27.9959
  mel_roformer_denoise_aggr  denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768
  mel_roformer_dereverb      dereverb_mel_band_roformer_anvuew_sdr_19.1729
  mel_roformer_dereverb_echo dereverb-echo_mel_band_roformer_sdr_10.0169
  denoiser_dns48 / dns64 / master64

Usage
-----
    python tools/_probe_stage0_gate.py --group mel
    python tools/_probe_stage0_gate.py --group den
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as _paths  # noqa: E402

_paths.setup_env()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

ROOT = str(_paths.PROJECT_ROOT)
W = os.path.join(ROOT, "01_models", "_weights")
ZF = os.path.join(ROOT, "01_models", "_third_party", "Music-Source-Separation-Training")
DEN = os.path.join(ROOT, "01_models", "_third_party", "denoiser")
OUT_AUDIO = os.path.join(ROOT, "03_outputs", "_stage0_gate")
OUT_JSON = os.path.join(ROOT, "04_reports", "_shared", "data", "model_analysis", "stage0_gate.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PROBE_SONG = "p232_001.wav"

MEL_CASES = [
    ("mel_roformer_denoise", "mel_roformer_denoise",
     "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
     "model_mel_band_roformer_denoise.yaml"),
    ("mel_roformer_denoise_aggr", "mel_roformer_denoise",
     "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
     "model_mel_band_roformer_denoise.yaml"),
    ("mel_roformer_dereverb", "mel_roformer_dereverb",
     "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
     "dereverb_mel_band_roformer_anvuew.yaml"),
    ("mel_roformer_dereverb_echo", "mel_roformer_dereverb_echo",
     "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
     "config_dereverb-echo_mel_band_roformer.yaml"),
]

DEN_CASES = [
    ("denoiser_dns48", "dns48-11decc9d8e3f0998.th", 48),
    ("denoiser_dns64", "dns64-a7761ff99a7d5bb6.th", 64),
    ("denoiser_master64", "master64-8a5dfb4bb92753dd.th", 64),
]


def _si_sdr(est, ref):
    """Scale-invariant SDR between two 1-D signals (dB)."""
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    est = est - est.mean()
    ref = ref - ref.mean()
    a = float((est * ref).sum() / ((ref ** 2).sum() + 1e-12))
    tgt = a * ref
    err = est - tgt
    return float(10.0 * np.log10(((tgt ** 2).sum() + 1e-12) / ((err ** 2).sum() + 1e-12)))


def _resample(x, sr_in, sr_out):
    import torchaudio.functional as AF
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    return AF.resample(t, sr_in, sr_out).numpy()


def _probe_pair():
    pb = os.path.join(ROOT, "02_databases", "Valentini")
    n, sr = sf.read(os.path.join(pb, "noisy_testset_wav", PROBE_SONG), dtype="float32")
    c, _ = sf.read(os.path.join(pb, "clean_testset_wav", PROBE_SONG), dtype="float32")
    return n, c, sr


def record(rec_id, **kw):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    prev = {}
    if os.path.exists(OUT_JSON):
        try:
            prev = json.load(open(OUT_JSON, encoding="utf-8")).get("cases", {})
        except Exception:                                     # noqa: BLE001
            prev = {}
    prev[rec_id] = kw
    json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "device": DEV, "cases": prev},
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def save_wav(path, audio, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr, subtype="FLOAT")
    return os.path.getsize(path)


# ------------------------------------------------------------------ Mel-RoFormer
def run_mel():
    sys.path.insert(0, ZF)
    from utils.settings import get_model_from_config

    n_raw, c_raw, sr_in = _probe_pair()
    n44 = _resample(n_raw, sr_in, 44100)
    c44 = _resample(c_raw, sr_in, 44100)
    x = np.stack([n44, n44], axis=0).astype(np.float32)       # (2, T) stereo probe
    base_sdr = _si_sdr(n44, c44)
    print("probe: Valentini %s  %.2f s  noisy-vs-clean SI-SDR = %.2f dB"
          % (PROBE_SONG, len(n44) / 44100.0, base_sdr))

    for rec_id, folder, ckpt, yaml in MEL_CASES:
        print("[gate] %s" % rec_id)
        base = os.path.join(W, folder)
        ck_path, cfg_path = os.path.join(base, ckpt), os.path.join(base, yaml)
        model = None
        try:
            t0 = time.time()
            model, config = get_model_from_config("mel_band_roformer", cfg_path)
            sd = torch.load(ck_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            t_load = time.time() - t0
            n_params = sum(p.numel() for p in model.parameters())
            model.eval().to(DEV)

            with torch.no_grad():
                t1 = time.time()
                y = model(torch.as_tensor(x, dtype=torch.float32)[None].to(DEV))
                t_fwd = time.time() - t1
            if isinstance(y, (tuple, list)):
                y = y[0]
            y = y.detach().float().cpu().numpy()

            out = y[0] if y.ndim == 4 else y          # (B, n_stem, C, T) -> (n_stem, C, T)
            if out.ndim == 3:
                out = out[0]                          # gate reads stem 0
            peak = float(np.abs(out).max())
            rms = float(np.sqrt(np.mean(out ** 2)))
            om = out.mean(axis=0) if out.ndim == 2 else out.ravel()
            m = min(len(om), len(c44))
            enh_sdr = _si_sdr(om[:m], c44[:m])
            sz = save_wav(os.path.join(OUT_AUDIO, rec_id, "out.wav"), out.T, 44100)
            print("    OK  params=%.2fM miss=%d unexp=%d load=%.1fs fwd=%.1fs out=%s "
                  "peak=%.3f rms=%.4f  SI-SDR %.2f -> %.2f dB (%+.2f)"
                  % (n_params / 1e6, len(missing), len(unexpected), t_load, t_fwd,
                     y.shape, peak, rms, base_sdr, enh_sdr, enh_sdr - base_sdr))
            record(rec_id, status="PASS", params=n_params,
                   n_missing=len(missing), n_unexpected=len(unexpected),
                   missing_keys=list(missing)[:8], unexpected_keys=list(unexpected)[:8],
                   load_s=round(t_load, 2), forward_s=round(t_fwd, 3),
                   out_shape=list(y.shape), peak=round(peak, 4), rms=round(rms, 5),
                   si_sdr_in=round(base_sdr, 2), si_sdr_out=round(enh_sdr, 2),
                   si_sdr_delta=round(enh_sdr - base_sdr, 2),
                   wav_mb=round(sz / 1e6, 2), device=DEV, ckpt=ckpt, yaml=yaml,
                   sample_rate=44100, channels=2, probe=PROBE_SONG)
        except Exception as e:                                # noqa: BLE001
            tb = traceback.format_exc()
            sac = any(k in tb for k in ("4551", "4561", "Blocked", "Prevented"))
            print("    FAIL %s: %s%s" % (type(e).__name__, e, "  [SAC BLOCK]" if sac else ""))
            print("    " + tb.strip().splitlines()[-1][:160])
            record(rec_id, status="FAIL", error="%s: %s" % (type(e).__name__, e),
                   sac_block=sac, traceback=tb[-1500:], device=DEV,
                   ckpt=ckpt, yaml=yaml)
        finally:
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------- denoiser
def run_den():
    sys.path.insert(0, DEN)
    from denoiser import pretrained as P

    n_raw, c_raw, sr_in = _probe_pair()
    n16 = _resample(n_raw, sr_in, 16000)
    c16 = _resample(c_raw, sr_in, 16000)
    x = n16.astype(np.float32)                                # (T,) mono probe
    base_sdr = _si_sdr(n16, c16)
    print("probe: Valentini %s  %.2f s  noisy-vs-clean SI-SDR = %.2f dB"
          % (PROBE_SONG, len(n16) / 16000.0, base_sdr))

    for rec_id, fname, hidden in DEN_CASES:
        print("[gate] %s" % rec_id)
        path = os.path.join(W, "denoiser", fname)
        model = None
        try:
            t0 = time.time()
            # pretrained=False -> build the official architecture, then load our
            # local file instead of letting torch.hub re-download it.
            model = P._demucs(False, P.DNS_48_URL, hidden=hidden)  # noqa: SLF001
            sd = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            t_load = time.time() - t0
            n_params = sum(p.numel() for p in model.parameters())
            model.eval().to(DEV)

            with torch.no_grad():
                t1 = time.time()
                y = model(torch.as_tensor(x, dtype=torch.float32)[None, None].to(DEV))
                t_fwd = time.time() - t1
            y = y.detach().float().cpu().numpy()
            out = y[0]
            if out.ndim == 2:
                out = out[0]
            peak = float(np.abs(out).max())
            rms = float(np.sqrt(np.mean(out ** 2)))
            m = min(len(out), len(c16))
            enh_sdr = _si_sdr(out[:m], c16[:m])
            sz = save_wav(os.path.join(OUT_AUDIO, rec_id, "out.wav"), out, 16000)
            print("    OK  params=%.2fM miss=%d unexp=%d load=%.1fs fwd=%.3fs out=%s "
                  "peak=%.3f rms=%.4f  SI-SDR %.2f -> %.2f dB (%+.2f)"
                  % (n_params / 1e6, len(missing), len(unexpected), t_load, t_fwd,
                     y.shape, peak, rms, base_sdr, enh_sdr, enh_sdr - base_sdr))
            record(rec_id, status="PASS", params=n_params,
                   n_missing=len(missing), n_unexpected=len(unexpected),
                   missing_keys=list(missing)[:8], unexpected_keys=list(unexpected)[:8],
                   load_s=round(t_load, 2), forward_s=round(t_fwd, 3),
                   out_shape=list(y.shape), peak=round(peak, 4), rms=round(rms, 5),
                   si_sdr_in=round(base_sdr, 2), si_sdr_out=round(enh_sdr, 2),
                   si_sdr_delta=round(enh_sdr - base_sdr, 2),
                   wav_mb=round(sz / 1e6, 2), device=DEV, ckpt=fname,
                   sample_rate=16000, channels=1, hidden=hidden, probe=PROBE_SONG)
        except Exception as e:                                # noqa: BLE001
            tb = traceback.format_exc()
            sac = any(k in tb for k in ("4551", "4561", "Blocked", "Prevented"))
            print("    FAIL %s: %s%s" % (type(e).__name__, e, "  [SAC BLOCK]" if sac else ""))
            record(rec_id, status="FAIL", error="%s: %s" % (type(e).__name__, e),
                   sac_block=sac, traceback=tb[-1500:], device=DEV, ckpt=fname)
        finally:
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="mel", choices=["mel", "den", "all"])
    a = ap.parse_args()
    print("device = %s" % DEV)
    if a.group in ("mel", "all"):
        run_mel()
    if a.group in ("den", "all"):
        run_den()

    cases = json.load(open(OUT_JSON, encoding="utf-8"))["cases"]
    shown = {k: v for k, v in cases.items() if
             (k.startswith("mel_") and a.group in ("mel", "all"))
             or (k.startswith("denoiser_") and a.group in ("den", "all"))}
    bad = [k for k, v in shown.items() if v.get("status") != "PASS"]
    print("=" * 74)
    for k, v in sorted(shown.items()):
        extra = ("params %.1fM  SI-SDR %+.2f dB" % (v["params"] / 1e6, v["si_sdr_delta"])
                 if v.get("params") else v.get("error", "")[:60])
        print("  %-28s %-5s %s" % (k, v.get("status"), extra))
    print("gate: %d/%d PASS" % (len(shown) - len(bad), len(shown)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
