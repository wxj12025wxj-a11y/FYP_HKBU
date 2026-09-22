# -*- coding: utf-8 -*-
"""Micro-probe: decompose the per-song FIXED overhead, and explain why models that emit
1 track cost ~30 s less per song than models that emit 4 tracks.

Measures, on one real MUSDB18 test song:
  A. import cost (measured by the caller / reported separately)
  B. GT + mixture decode
  C. museval bss_eval for 1 track vs for 4 tracks  (dominant term)
  D. FLAC write for 1 track vs 4 tracks
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import numpy as np
import soundfile as sf

SONG = "Al James - Schoolboy Facination"
D = os.path.join(_ROOT, "02_databases", "MUSDB18-HQ", "test", SONG)
TRACKS = ("vocals", "drums", "bass", "other")
TMP = os.path.join(_ROOT, "tools", "_scratch", "_probe_out")
os.makedirs(TMP, exist_ok=True)


def tick():
    return time.perf_counter()


import museval  # noqa: E402
from museval.metrics import bss_eval  # noqa: E402

T = {}
t0 = tick()

# ---- B. decode -------------------------------------------------------
t = tick()
mix, sr = sf.read(os.path.join(D, "mixture.wav"), dtype="float32", always_2d=True)
T["mix_decode"] = tick() - t
n = mix.shape[0]
t = tick()
gt = {}
for k in TRACKS:
    y, _ = sf.read(os.path.join(D, f"{k}.wav"), dtype="float32", always_2d=True)
    gt[k] = y[:n]
T["gt_decode_4"] = tick() - t
print(f"song={SONG}  sr={sr}  dur={n / sr:.1f}s  channels={mix.shape[1]}")

# ---- C. museval ------------------------------------------------------
def ev(ref2d, est2d):
    r = np.asarray(ref2d, np.float32).reshape(1, -1, 1)
    e = np.asarray(est2d, np.float32).reshape(1, -1, 1)
    sdr, isr, sir, sar, _ = bss_eval(r, e, compute_permutation=False)
    return float(np.nanmedian(np.atleast_1d(sdr).astype(float)))


mono = {k: gt[k].mean(axis=1).astype(np.float32) for k in TRACKS}
ref_mix = mix.mean(axis=1).astype(np.float32)

per_track = {}
for k in TRACKS:
    t = tick()
    s = ev(mono[k], ref_mix)     # 用 mixture 当估计，只测耗时
    per_track[k] = tick() - t
    print(f"  bss_eval[{k:6s}] = {per_track[k]:6.2f} s   (SDR {s:6.2f})")
T["museval_4_tracks"] = sum(per_track.values())
T["museval_1_track"] = per_track["vocals"]
print(f"  -> museval 4 tracks = {T['museval_4_tracks']:.2f}s   1 track = {T['museval_1_track']:.2f}s"
      f"   delta = {T['museval_4_tracks'] - T['museval_1_track']:.2f}s")

# ---- D. FLAC write ---------------------------------------------------
t = tick()
for k in TRACKS:
    sf.write(os.path.join(TMP, f"_p4_{k}.flac"), gt[k], sr, format="FLAC", subtype="PCM_16")
T["flac_write_4"] = tick() - t
t = tick()
sf.write(os.path.join(TMP, "_p1_vocals.flac"), gt["vocals"], sr, format="FLAC", subtype="PCM_16")
T["flac_write_1"] = tick() - t
print(f"  -> flac write 4 tracks = {T['flac_write_4']:.2f}s   1 track = {T['flac_write_1']:.2f}s")

for f in os.listdir(TMP):
    os.remove(os.path.join(TMP, f))

print("\n--- summary (seconds, excl. interpreter/torch import) ---")
for k, v in T.items():
    print(f"  {k:18s} {v:7.2f}")
print(f"  {'TOTAL 4-track':18s} "
      f"{T['mix_decode'] + T['gt_decode_4'] + T['museval_4_tracks'] + T['flac_write_4']:7.2f}")
print(f"  {'TOTAL 1-track':18s} "
      f"{T['mix_decode'] + T['gt_decode_4'] + T['museval_1_track'] + T['flac_write_1']:7.2f}")
print(f"\n全程墙钟(含 import) = {tick() - t0:.2f}s")
