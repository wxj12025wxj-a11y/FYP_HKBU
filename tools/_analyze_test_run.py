# -*- coding: utf-8 -*-
"""Aggregate the 50-song MUSDB18 test sweep into multi-dimensional per-model stats.

Outputs:
  04_reports/separation/data/comparison/test_run_analysis.json
  04_reports/separation/data/comparison/test_run_per_song.csv
  04_reports/separation/data/comparison/test_run_model_summary.csv
"""
from __future__ import annotations

import json
import os
import statistics as st
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMP = os.path.join(_ROOT, "04_reports", "separation", "data", "comparison")
OUT_ROOT = os.path.join(_ROOT, "03_outputs", "_test_run")

LABEL = {
    "oracle": "Oracle-IRM",
    "umx": "Open-Unmix",
    "mdx": "MDX-Net",
    "demucs": "Demucs",
    "convtasnet": "Conv-TasNet",
    "mmdenselstm": "MMDenseLSTM",
    "bsroformer_l12": "BS-RoFormer-L12",
    "bsroformer_l6": "BS-RoFormer-L6",
    "bsrnn": "BSRNN-opt",
    "bsrnn_large": "BSRNN-large",
    "bsrnn_simo": "BSRNN-SIMO",
    "bsrnn_all": "BSRNN-all(4ckpt)",
    "bsrnn_large_all": "BSRNN-large-all(4ckpt)",
    "rpca": "RPCA",
    "dprnn": "DPRNN",
}

# models that count as "the 12 finished ones" (everything except rpca + the two
# 4-single-target-ckpt BSRNN variants that never produced a single output)
TARGET12 = [k for k in LABEL if k not in ("rpca", "bsrnn_all", "bsrnn_large_all")]


def load(name):
    with open(os.path.join(CMP, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    led = load("test_sweep_ledger.json")
    runs = load("model_runs_run.json")["runs"]
    man = load("musdb_test_manifest.json")

    # ---- song durations -------------------------------------------------
    dur = {}
    if isinstance(man, dict):
        items = man.get("songs") or man.get("items") or man
        if isinstance(items, dict):
            for k, v in items.items():
                if isinstance(v, dict):
                    dur[k] = v.get("duration_s") or v.get("duration")
        elif isinstance(items, list):
            for v in items:
                nm = v.get("song") or v.get("name")
                d = v.get("dur_s") or v.get("duration_s") or v.get("duration")
                if nm and d:
                    dur[nm] = d

    # ---- output file inventory -----------------------------------------
    files = defaultdict(dict)   # model -> song -> {track: bytes}
    for d in sorted(os.listdir(OUT_ROOT)):
        p = os.path.join(OUT_ROOT, d)
        if not os.path.isdir(p):
            continue
        for s in sorted(os.listdir(p)):
            sp = os.path.join(p, s)
            if not os.path.isdir(sp):
                continue
            got = {}
            for f in os.listdir(sp):
                fp = os.path.join(sp, f)
                if os.path.isfile(fp):
                    got[os.path.splitext(f)[0]] = os.path.getsize(fp)
            if got:
                files[d][s] = got

    # ---- merge ----------------------------------------------------------
    models = {}
    per_song_rows = []
    for m, songs in led["attempts"].items():
        rec = models.setdefault(m, {})
        rec["label"] = LABEL.get(m, m)
        rec["n_attempt"] = len(songs)
        rec["timeouts"] = led["timeouts"].get(m, 0)
        rec["disabled"] = m in led["disabled"]

        wall, infer, rtf, sdr_mean, ntracks, nbytes = [], [], [], [], [], []
        timeout_wait = 0.0
        eff_status = {}
        for song, a in songs.items():
            r = runs.get(m, {}).get(song, {})
            w = a.get("seconds")
            files_present = files.get(rec["label"], {}).get(song, {})
            tracks = sorted(files_present.keys())
            nb = sum(files_present.values())
            # the benchmark's own result JSON is authoritative: a combo the
            # scheduler recorded as FAIL may have been re-run successfully
            # afterwards without the ledger being updated.
            stt = r.get("status") or a.get("status")
            eff_status[song] = stt
            if stt == "TIMEOUT":
                if w is not None:
                    timeout_wait += w
                continue
            if stt == "PASS" and w is not None:
                wall.append(w)
            i = r.get("infer_s")
            if i is not None:
                infer.append(i)
            if r.get("rtf") is not None:
                rtf.append(r["rtf"])
            sdr = r.get("sdr") or {}
            vals = [v for v in sdr.values() if isinstance(v, (int, float))]
            sm = round(sum(vals) / len(vals), 3) if vals else None
            if sm is not None:
                sdr_mean.append(sm)
            if tracks:
                ntracks.append(len(tracks))
                nbytes.append(nb)
            per_song_rows.append({
                "model_key": m,
                "model": rec["label"],
                "song": song,
                "status": stt,
                "ledger_status": a.get("status"),
                "wall_s": w,
                "infer_s": i,
                "rtf": r.get("rtf"),
                "overhead_s": (round(w - i, 2) if (w is not None and i is not None) else None),
                "song_dur_s": dur.get(song) or r.get("duration_s"),
                "n_tracks": len(tracks),
                "tracks": "+".join(tracks),
                "out_mb": round(nb / 1e6, 2) if nb else 0,
                "sdr_mean": sm,
                "sdr_vocals": sdr.get("vocals"),
                "sdr_drums": sdr.get("drums"),
                "sdr_bass": sdr.get("bass"),
                "sdr_other": sdr.get("other"),
            })

        def agg(x, nd=2):
            if not x:
                return {k: None for k in ("n", "mean", "median", "min", "max", "stdev", "total")}
            r = {
                "n": len(x),
                "mean": round(st.mean(x), nd),
                "median": round(st.median(x), nd),
                "min": round(min(x), nd),
                "max": round(max(x), nd),
                "stdev": round(st.pstdev(x), nd) if len(x) > 1 else 0.0,
                "total": round(sum(x), nd),
            }
            return r

        rec["status_counts"] = dict(Counter(eff_status.values()))
        rec["ledger_status_counts"] = dict(Counter(v.get("status") for v in songs.values()))
        rec["reconciled"] = [s_ for s_, st in eff_status.items()
                             if st != songs[s_].get("status")]
        rec["wall_s"] = agg(wall)
        rec["n_pass"] = rec["status_counts"].get("PASS", 0)
        rec["timeout_wait_s"] = round(timeout_wait, 1)
        rec["infer_s"] = agg(infer, 3)
        rec["rtf"] = agg(rtf, 4)
        rec["sdr_mean"] = agg(sdr_mean, 3)
        rec["n_tracks"] = dict(Counter(ntracks)) if ntracks else {}
        rec["out_bytes_total"] = sum(nbytes)
        rec["out_mb_total"] = round(sum(nbytes) / 1e6, 1)
        rec["out_songs"] = len(files.get(rec["label"], {}))
        rec["complete_50"] = (rec["out_songs"] == 50 and rec["status_counts"].get("PASS") == 50)
        rec["overhead_s_total"] = (round(rec["wall_s"]["total"] - rec["infer_s"]["total"], 1)
                                   if rec["wall_s"]["total"] and rec["infer_s"]["total"] else None)
        if rec["wall_s"]["total"] and rec["infer_s"]["total"]:
            rec["infer_share_pct"] = round(100.0 * rec["infer_s"]["total"] / rec["wall_s"]["total"], 1)
            rec["overhead_share_pct"] = round(100.0 - rec["infer_share_pct"], 1)

    # ---- finished set = 50/50 PASS & 50 song folders --------------------
    finished = [m for m, r in models.items() if r.get("complete_50")]
    unfinished = [m for m in models if m not in finished]

    summary = {
        "n_models_total": len(models),
        "finished": finished,
        "unfinished": unfinished,
        "disabled": led["disabled"],
        "ledger_updated": led["updated"],
        "models": models,
    }
    with open(os.path.join(CMP, "test_run_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- CSV ------------------------------------------------------------
    cols = ["model_key", "model", "song", "status", "ledger_status", "song_dur_s", "wall_s", "infer_s",
            "overhead_s", "rtf", "n_tracks", "tracks", "out_mb",
            "sdr_mean", "sdr_vocals", "sdr_drums", "sdr_bass", "sdr_other"]
    with open(os.path.join(CMP, "test_run_per_song.csv"), "w", encoding="utf-8-sig") as f:
        f.write(",".join(cols) + "\n")
        for r in per_song_rows:
            f.write(",".join(
                ('"%s"' % r[c] if isinstance(r[c], str) else str(r[c] if r[c] is not None else ""))
                for c in cols) + "\n")

    mcols = ["model_key", "model", "complete_50", "status_PASS", "status_TIMEOUT", "status_FAIL",
             "timeouts", "disabled", "out_songs", "n_tracks_set",
             "wall_median_s", "wall_mean_s", "wall_min_s", "wall_max_s", "wall_total_s",
             "infer_median_s", "infer_total_s", "overhead_total_s", "infer_share_pct",
             "rtf_median", "sdr_mean_median", "out_mb_total"]
    with open(os.path.join(CMP, "test_run_model_summary.csv"), "w", encoding="utf-8-sig") as f:
        f.write(",".join(mcols) + "\n")
        for m, r in sorted(models.items(), key=lambda kv: (not kv[1].get("complete_50"),
                                                          kv[1]["wall_s"]["median"] or 9e9)):
            row = {
                "model_key": m, "model": r["label"],
                "complete_50": r.get("complete_50"),
                "status_PASS": r["status_counts"].get("PASS", 0),
                "status_TIMEOUT": r["status_counts"].get("TIMEOUT", 0),
                "status_FAIL": r["status_counts"].get("FAIL", 0),
                "timeouts": r["timeouts"], "disabled": r["disabled"],
                "out_songs": r["out_songs"],
                "n_tracks_set": "|".join(f"{k}:{v}" for k, v in sorted(r["n_tracks"].items())),
                "wall_median_s": r["wall_s"]["median"], "wall_mean_s": r["wall_s"]["mean"],
                "wall_min_s": r["wall_s"]["min"], "wall_max_s": r["wall_s"]["max"],
                "wall_total_s": r["wall_s"]["total"],
                "infer_median_s": r["infer_s"]["median"], "infer_total_s": r["infer_s"]["total"],
                "overhead_total_s": r.get("overhead_s_total"),
                "infer_share_pct": r.get("infer_share_pct"),
                "rtf_median": r["rtf"]["median"],
                "sdr_mean_median": r["sdr_mean"]["median"],
                "out_mb_total": r["out_mb_total"],
            }
            f.write(",".join(
                ('"%s"' % row[c] if isinstance(row[c], str) else str(row[c] if row[c] is not None else ""))
                for c in mcols) + "\n")

    # ---- console --------------------------------------------------------
    print("finished (%d):" % len(finished), ", ".join(LABEL[m] for m in finished))
    print("unfinished   :", ", ".join(f"{LABEL[m]}({models[m]['status_counts']})" for m in unfinished))
    print()
    hdr = f"{'model':22s} {'songs':>5s} {'tracks':>7s} {'wall_med':>9s} {'wall_max':>9s} {'infer_med':>10s} {'ovh%':>6s} {'rtf':>7s} {'sdr':>7s} {'MB':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in sorted(models.items(), key=lambda kv: kv[1]["wall_s"]["median"] or 9e9):
        tset = sorted(r["n_tracks"]) 
        print(f"{r['label']:22s} {r['out_songs']:5d} {(str(tset[0]) if tset else '-'):>7s} "
              f"{(r['wall_s']['median'] or 0):9.1f} {(r['wall_s']['max'] or 0):9.1f} "
              f"{(r['infer_s']['median'] or 0):10.2f} {(r.get('overhead_share_pct') if r.get('overhead_share_pct') is not None else 0):6.1f} "
              f"{(r['rtf']['median'] or 0):7.4f} {(r['sdr_mean']['median'] or 0):7.2f} {r['out_mb_total']:7.1f}")
    print()
    print("files ->", CMP)


if __name__ == "__main__":
    main()
