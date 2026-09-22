"""Verify every audio file referenced by the listening page.

Two things a browser would silently punish:
  1. a file that cannot be fully decoded (truncated FLAC / broken WAV)
  2. a file that decodes to silence (a "plays fine, means nothing" stem)

Full decode + RMS/peak here, so the page only ever links audio that is
provably complete and non-silent.
"""
from __future__ import annotations

import json
import os
import sys
from urllib.parse import unquote

import numpy as np
import soundfile as sf

ROOT = r"E:\FYP_HKBU"
HTML = os.path.join(ROOT, "04_reports", "separation", "html", "listen_compare.html")
BASE = os.path.join(ROOT, "04_reports", "separation", "html")


def main():
    man = json.load(open(os.path.join(ROOT, "04_reports", "separation", "data", "comparison",
                                     "listen_manifest.json"), encoding="utf-8"))
    bad = []
    silent = []
    clipped = []
    qc = {}
    n = 0
    tot_s = 0.0
    rows = []

    for s in man["songs"]:
        ref = [(t["src"], t["label"], s["dur_s"]) for t in s["ref"]]
        mods = [(t["src"], "%s / %s" % (m["label"], t["label"]), s["dur_s"])
                for m in s["models"] for t in m["tracks"]]
        for src, label, dur in ref + mods:
            path = os.path.normpath(os.path.join(BASE, unquote(src).replace("/", os.sep)))
            if not os.path.isfile(path):
                bad.append(("MISSING", label, path))
                qc[src] = {"ok": False, "why": "missing"}
                continue
            try:
                data, sr = sf.read(path, always_2d=True)
            except Exception as e:                       # noqa: BLE001
                bad.append(("DECODE", label, "%s: %s" % (type(e).__name__, e)))
                qc[src] = {"ok": False, "why": str(e)}
                continue
            n += 1
            tot_s += len(data) / sr
            rms = float(np.sqrt(np.mean(data ** 2)))
            peak = float(np.max(np.abs(data)))
            nclip = int(np.sum(np.abs(data) >= 0.9995))
            qc[src] = {"ok": True, "sr": sr, "ch": data.shape[1],
                       "dur": round(len(data) / sr, 3), "rms": round(rms, 6),
                       "peak": round(peak, 6), "clipped": nclip}
            if rms < 1e-5:
                silent.append((label, rms))
            if nclip > 0:
                clipped.append((label, nclip, peak))
            rows.append((label, sr, data.shape[1], len(data) / sr, rms, peak, nclip))

    print("%-56s %6s %5s %8s %8s %8s %9s" % ("file", "sr", "ch", "dur_s", "rms", "peak", "clip smp"))
    for label, sr, ch, dur, rms, peak, nclip in rows:
        print("%-56s %6d %5d %8.2f %8.4f %8.4f %9d" % (label[:56], sr, ch, dur, rms, peak, nclip))

    out = os.path.join(ROOT, "04_reports", "separation", "data", "comparison", "listen_audio_qc.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(qc, fh, ensure_ascii=False, indent=1)

    print("\nchecked %d files, %.1f min of audio" % (n, tot_s / 60))
    print("qc -> %s" % out)
    if clipped:
        print("\n== 削波（峰值触顶，人耳会听到失真/爆音） ==")
        for label, nclip, peak in clipped:
            print("   %-52s peak=%.4f  %d samples" % (label[:52], peak, nclip))
    if bad:
        print("\n!! FAILED (%d)" % len(bad))
        for kind, label, extra in bad:
            print("   [%s] %s  %s" % (kind, label, extra))
    if silent:
        print("\n!! SILENT (%d)" % len(silent))
        for label, rms in silent:
            print("   %s  rms=%.2e" % (label, rms))
    if not bad and not silent:
        print("\nOK: every referenced clip decodes completely and carries signal")
    return 1 if (bad or silent) else 0


if __name__ == "__main__":
    sys.exit(main())
