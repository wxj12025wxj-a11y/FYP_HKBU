"""Stage 0 asset downloader (idempotent, resumable, ASCII-only log).

Covers the three network prerequisites defined in FYP_PLAN_2026-09-22.md:

  testset   Valentini-Botinhao testset  (clean/noisy 824 pairs + SNR txt)   ~325 MB
  trainset  Valentini-Botinhao 28spk trainset                             ~5.3 GB   [optional]
  models    Mel-RoFormer-Denoise-Aufr33 x2 + dereverb + dereverb-echo      ~3.6 GB
  denoiser  facebookresearch/denoiser pretrained dns48/dns64/master64      ~0.4 GB

Design notes
------------
* Resumable: uses HTTP Range when a partial file exists; if the server
  ignores Range (returns 200 instead of 206) the file is restarted cleanly.
* Idempotent: a file whose size already matches the expected size is skipped.
  Agents get killed at session end on this machine, so re-running must be free.
* Log is pure ASCII (this shell's wrapper layer drops stdout to GBK; emoji crash).
* Writes a manifest JSON so the stage-0 gate can judge success from data,
  not from a directory listing.

Usage
-----
    python tools/_stage0_download.py --group testset       # small, do first
    python tools/_stage0_download.py --group models
    python tools/_stage0_download.py --group denoiser
    python tools/_stage0_download.py --group trainset      # optional, 5.3 GB
    python tools/_stage0_download.py --group all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

# bootstrap must come AFTER `from __future__ import annotations` (SyntaxError otherwise)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as _paths  # noqa: E402

_paths.setup_env()

ROOT = _paths.PROJECT_ROOT
DEST_DB = os.path.join(ROOT, "02_databases", "Valentini")
DEST_W = os.path.join(ROOT, "01_models", "_weights")
MANIFEST = os.path.join(ROOT, "tools", "_logs", "_stage0_manifest.json")

# ---------------------------------------------------------------- sources
VAL_ITEM = "6ed35425-bf14-4d2b-93a1-0a4984952757"
VAL_BUNDLE = "bdb21ae9-97ad-46dc-89b6-c1884783df6a"
VAL_API = "https://datashare.ed.ac.uk/server/api"

TEST_FILES = ["clean_testset_wav.zip", "noisy_testset_wav.zip", "testset_txt.zip"]
TRAIN_FILES = [
    "clean_trainset_28spk_wav.zip",
    "noisy_trainset_28spk_wav.zip",
    "trainset_28spk_txt.zip",
    "logfiles.zip",
]

HF_REPOS = {
    # denoise models (community redistribution of Aufr33's released checkpoints)
    "mel_roformer_denoise": (
        "oulianov/melband-roformer-denoise",
        [
            "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
            "model_mel_band_roformer_denoise.yaml",
        ],
    ),
    "mel_roformer_dereverb": (
        "anvuew/dereverb_mel_band_roformer",
        [
            "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            "dereverb_mel_band_roformer_anvuew.yaml",
        ],
    ),
    "mel_roformer_dereverb_echo": (
        "Sucial/Dereverb-Echo_Mel_Band_Roformer",
        [
            "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
            "config_dereverb-echo_mel_band_roformer.yaml",
        ],
    ),
}

DENOISER_ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
DENOISER_FILES = ["dns48-11decc9d8e3f0998.th",
                  "dns64-a7761ff99a7d5bb6.th",
                  "master64-8a5dfb4bb92753dd.th"]

CHUNK = 1 << 20          # 1 MiB
TOL = 0.995              # accept >=99.5% of expected size as "complete"
S = requests.Session()
S.headers["User-Agent"] = "fyp-hkbu/stage0"


def head_size(url: str) -> int:
    """Content-Length via HEAD, following redirects. 0 if unknown."""
    try:
        r = S.head(url, allow_redirects=True, timeout=30)
        return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def val_files(names):
    """Resolve (name, url, size) for the requested bitstreams of the Valentini item."""
    u = "%s/core/bundles/%s/bitstreams?size=100" % (VAL_API, VAL_BUNDLE)
    r = S.get(u, timeout=30)
    r.raise_for_status()
    got = {}
    for b in r.json()["_embedded"]["bitstreams"]:
        got[b["name"]] = (
            "%s/core/bitstreams/%s/content" % (VAL_API, b["uuid"]),
            int(b.get("sizeBytes") or 0),
        )
    out = []
    for n in names:
        if n not in got:
            print("[warn] bitstream not listed: %s" % n)
            continue
        url, size = got[n]
        out.append((n, url, size))
    return out


def hf_files(repo, names):
    out = []
    for n in names:
        url = "https://huggingface.co/%s/resolve/main/%s" % (repo, n)
        out.append((n, url, head_size(url)))
    return out


def fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f GB" % n


def fetch(url: str, dst: str, expect: int, label: str) -> dict:
    """Download url -> dst with resume. Returns a result record."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    have = os.path.getsize(dst) if os.path.exists(dst) else 0

    if expect and have >= expect * TOL:
        print("[skip] %-46s complete  %s" % (label, fmt(have)))
        return {"label": label, "dst": dst, "url": url, "status": "SKIP",
                "bytes": have, "expect": expect}

    t0 = time.time()
    # If the expected size is unknown we cannot distinguish a partial file from a
    # complete one.  HF does not always return Content-Length on HEAD, so a small
    # but complete file (yaml) would send "Range: bytes=<full>-" and get HTTP 416.
    # Re-downloading a few KB is cheaper than the ambiguity.
    if have and not expect:
        have = 0
    headers = {"Range": "bytes=%d-" % have} if have else {}
    try:
        r = S.get(url, headers=headers, stream=True, timeout=(30, 300))
    except Exception as e:                                    # noqa: BLE001
        print("[ERR ] %-46s %s: %s" % (label, type(e).__name__, e))
        return {"label": label, "dst": dst, "url": url, "status": "FAIL",
                "error": "%s: %s" % (type(e).__name__, e)}

    if r.status_code == 200 and have:
        have = 0                       # server ignored Range -> restart
    elif r.status_code == 416:
        # range not satisfiable == we already hold at least the full file
        r.close()
        print("[skip] %-46s complete (416)  %s" % (label, fmt(have)))
        return {"label": label, "dst": dst, "url": url, "status": "SKIP",
                "bytes": have, "expect": expect}
    elif r.status_code not in (200, 206):
        print("[ERR ] %-46s HTTP %s" % (label, r.status_code))
        r.close()
        return {"label": label, "dst": dst, "url": url, "status": "FAIL",
                "error": "HTTP %s" % r.status_code}

    total = have + int(r.headers.get("Content-Length") or 0)
    done = have
    last = 0.0
    mode = "ab" if have else "wb"
    try:
        with open(dst, mode) as f:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last >= 15:                     # progress every 15 s
                    pct = (100.0 * done / total) if total else 0.0
                    rate = (done - have) / max(1e-6, now - t0)
                    eta = (total - done) / rate if rate > 0 and total else 0
                    print("[dl  ] %-40s %5.1f%%  %s  %s/s  eta %dm"
                          % (label, pct, fmt(done), fmt(rate), eta / 60))
                    last = now
    except Exception as e:                                    # noqa: BLE001
        print("[ERR ] %-46s write: %s" % (label, e))
        return {"label": label, "dst": dst, "url": url, "status": "FAIL",
                "error": str(e), "bytes": done, "expect": expect}
    finally:
        r.close()

    ok = (not expect) or done >= expect * TOL
    print("[%s] %-46s %s in %ds" % ("done" if ok else "PART",
                                    label, fmt(done), time.time() - t0))
    return {"label": label, "dst": dst, "url": url,
            "status": "OK" if ok else "PARTIAL", "bytes": done, "expect": expect}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=None,
                    choices=["testset", "trainset", "models", "denoiser", "all"],
                    help="repeatable: --group models --group denoiser")
    args = ap.parse_args()
    if not args.group:
        args.group = ["testset"]

    jobs = {}          # group -> list[(name, url, size, dst)]
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

    want = {"testset", "trainset", "models", "denoiser"} if "all" in args.group \
        else set(args.group)

    if "testset" in want or "trainset" in want:
        names = []
        if "testset" in want:
            names += TEST_FILES
        if "trainset" in want:
            names += TRAIN_FILES
        try:
            jobs["valentini"] = [(n, u, s, os.path.join(DEST_DB, n))
                                 for n, u, s in val_files(names)]
        except Exception as e:                                # noqa: BLE001
            print("[ERR ] cannot list Valentini bitstreams: %s" % e)

    if "models" in want:
        for grp, (repo, files) in HF_REPOS.items():
            jobs[grp] = [(n, u, s, os.path.join(DEST_W, grp, n))
                         for n, u, s in hf_files(repo, files)]

    if "denoiser" in want:
        jobs["denoiser"] = [(n, DENOISER_ROOT + n, head_size(DENOISER_ROOT + n),
                             os.path.join(DEST_W, "denoiser", n))
                            for n in DENOISER_FILES]

    grand = sum(s for v in jobs.values() for _, _, s, _ in v)
    print("=" * 74)
    print("group(s)=%s   jobs=%d   total=%s"
          % (",".join(args.group), sum(len(v) for v in jobs.values()), fmt(grand)))
    print("=" * 74)

    results = []
    for grp, items in jobs.items():
        print("---- %s ----" % grp)
        for name, url, size, dst in items:
            results.append(fetch(url, dst, size, "%s/%s" % (grp, name)))

    prev = {}
    if os.path.exists(MANIFEST):
        try:
            prev = json.load(open(MANIFEST, encoding="utf-8")).get("results", {})
        except Exception:                                     # noqa: BLE001
            prev = {}
    for r in results:
        prev[r["label"]] = r

    ok = sum(1 for r in prev.values() if r["status"] in ("OK", "SKIP"))
    fail = [k for k, r in prev.items() if r["status"] not in ("OK", "SKIP")]
    json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "results": prev, "n_ok": ok, "n_bad": len(fail)},
              open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=" * 74)
    print("summary: %d ok / %d bad   manifest -> %s" % (ok, len(fail), MANIFEST))
    for k in fail:
        print("   FAIL %s  %s" % (k, prev[k].get("error", "")))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
