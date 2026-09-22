# -*- coding: utf-8 -*-
"""
================================================================================
 分离方法横向对比实验  —  古典方法 vs 深度模型
--------------------------------------------------------------------------------
 数据 : MUSDB18-HQ / train / <song>  (默认第一首 A Classic Education - NightOwl)
 目标 : 同一段音频，对比各方法分离「人声(vocals)」与「鼓(drums)」的质量
 指标 : SDR / ISR / SAR (BSSEval v4 = museval, 1s 窗, 中位数聚合 —— MUSDB18 官方口径)
        + SI-SDR (scale-invariant)
        + ΔSDR = SDR(方法) - SDR(不分离基线)  ← 跨歌/跨数据集可比的关键指标
 评估 : 单声道 (mono), 单目标。以「不做分离(直接用 mixture)」为基线标定。
--------------------------------------------------------------------------------
 参与对比:
   [基线]          mixture (不分离)
   [古典/无监督]   ICA(FastICA) / NMF / RPCA(Inexact ALM) / HPSS(谐波&打击)
   [深度/预训练]   Open-Unmix (UMX-HQ)
--------------------------------------------------------------------------------
 运行:
   python tools/compare_separation_methods.py
   python tools/compare_separation_methods.py --songs 10 --save-audio

 产出 (outputs/):
   outputs/<模型>/<歌曲>/<目标>.wav   每个模型独立产出的分离音频
   outputs/<模型>/metrics.csv         每个模型自己的结果 (自包含)
   outputs/comparison/                同一数据的跨模型对比 (csv/json/图)
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

# librosa 在本机被 Smart App Control 间接拦截（librosa.stft -> numba -> llvmlite.dll）。
# 注意 `import librosa` 本身会成功（子模块惰性加载），真正炸的是 librosa.stft，
# 所以这里必须做「功能性探测」而不是「导入探测」。
# 古典方法实际只用到 stft/istft，改用 torch 后端的等价垫片即可。
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _probe_librosa():
    try:
        import librosa as _lb
        _probe = np.zeros(4096, dtype=np.float32)
        _X = _lb.stft(_probe, n_fft=2048, hop_length=512)
        _y = _lb.istft(_X, hop_length=512, length=4096)
        assert _y.shape[0] == 4096
        return _lb, "real"
    except Exception as _exc:  # noqa: BLE001
        from _librosa_shim import librosa_stub
        return librosa_stub, f"stub(torch) [{type(_exc).__name__}: {str(_exc)[:60]}]"


librosa, LIBROSA_KIND = _probe_librosa()

try:
    from sklearn.decomposition import NMF, FastICA
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    import museval
    HAVE_MUSEVAL = True
except Exception:
    HAVE_MUSEVAL = False

try:
    import mir_eval
    HAVE_MIREVAL = True
except Exception:
    HAVE_MIREVAL = False


DATA_ROOT = _paths.MUSDB18_ROOT

# ------------------------------------------------------------------ #
# 输出结构（每个模型独立成文件夹 + 一个跨模型比较文件夹）
#
#   outputs/
#   ├── Baseline/<song>/<target>.wav      ← 各模型"独立产出"的分离音频
#   ├── ICA/<song>/*.wav
#   ├── NMF/<song>/*.wav
#   ├── RPCA/<song>/*.wav
#   ├── HPSS/<song>/*.wav
#   ├── Open-Unmix/<song>/*.wav
#   ├── <每个模型>/metrics.csv            ← 该模型自己的结果（自包含）
#   └── comparison/                       ← 同一数据的跨模型对比
#       ├── comparison_results.csv / .json
#       ├── fig_compare_sdr.png / fig_compare_dsdr.png
#       └── realtime_benchmark.json
# ------------------------------------------------------------------ #
OUTPUTS_ROOT = _paths.OUTPUTS
COMPARISON_DIR = _paths.comparison_dir()   # → 04_reports/separation/data/comparison

SR = 44100
EPS = 1e-10
SEED = 0
UMX_ORDER = ["vocals", "drums", "bass", "other"]
BASELINE_NAME = "Baseline: mixture (no separation)"
BASELINE_ONLY = {BASELINE_NAME}

# 模型文件夹（顺序即展示顺序）
MODEL_FOLDERS = ["Baseline", "ICA", "NMF", "RPCA", "HPSS", "Open-Unmix"]


def model_of(method: str) -> str:
    """把方法名映射到所属模型文件夹名"""
    if method.startswith("Baseline"):
        return "Baseline"
    if method.startswith("ICA"):
        return "ICA"
    if method.startswith("NMF"):
        return "NMF"
    if method.startswith("RPCA"):
        return "RPCA"
    if method.startswith("HPSS"):
        return "HPSS"
    if method.startswith("Open-Unmix"):
        return "Open-Unmix"
    return "Other"


def out_stem(method: str, target: str) -> str:
    """模型文件夹内的音频文件名（区分同模型的不同变体）"""
    low = method.lower()
    if "oracle" in low:
        return f"{target}_oracle"
    if "harmonic" in low:
        return f"{target}_harmonic"
    return target


# 基线 SDR 低于此值视为"退化样本"（参考轨过稀疏/静音，museval 失真），聚合时剔除
DEGENERATE_BASELINE_SDR = -60.0


def degenerate_pairs(rows):
    """返回基线不可用或退化的 (song, target) 集合"""
    bad = set()
    for r in rows:
        if r["method"] == BASELINE_NAME:
            s = (r["metrics"] or {}).get("sdr")
            if s is None or s < DEGENERATE_BASELINE_SDR:
                bad.add((r["song"], r["target"]))
    return bad


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    n/a"
    return f"{v:7.2f}"


# ------------------------------------------------------------------ #
# 数据
# ------------------------------------------------------------------ #
def list_songs(subset="train"):
    d = DATA_ROOT / subset
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / "mixture.wav").is_file())


def load_song(song, subset, duration):
    d = DATA_ROOT / subset / song
    n = int(SR * duration)

    def rd(name, stereo=False):
        y, _ = sf.read(str(d / f"{name}.wav"), dtype="float32", always_2d=True)
        y = y[:n]
        return y.T.astype(np.float32) if stereo else y.mean(axis=1).astype(np.float32)

    return rd("mixture"), rd("vocals"), rd("drums"), rd("mixture", stereo=True)


# ------------------------------------------------------------------ #
# 指标 (BSSEval v4 / museval)
# ------------------------------------------------------------------ #
def si_sdr(ref, est):
    ref = np.asarray(ref, float).ravel()
    est = np.asarray(est, float).ravel()
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    t = a * ref
    n = est - t
    return 10 * np.log10((np.sum(t ** 2) + EPS) / (np.sum(n ** 2) + EPS))


def evaluate(ref_mono, est_mono):
    out = {"sdr": None, "isr": None, "sir": None, "sar": None, "si_sdr": None}
    ref = np.asarray(ref_mono, np.float32).reshape(1, -1, 1)
    est = np.asarray(est_mono, np.float32).reshape(1, -1, 1)
    if HAVE_MUSEVAL:
        try:
            sdr, isr, sir, sar, _ = museval.metrics.bss_eval(
                ref, est, compute_permutation=False)
            med = lambda x: float(np.nanmedian(np.atleast_1d(x).astype(float)))
            out["sdr"], out["isr"] = med(sdr), med(isr)
            out["sir"], out["sar"] = med(sir), med(sar)
        except Exception as exc:
            print(f"    [WARN] museval 失败: {exc}")
    elif HAVE_MIREVAL:
        try:
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                ref.reshape(1, -1), est.reshape(1, -1), compute_permutation=False)
            out["sdr"], out["sir"], out["sar"] = map(float, (sdr[0], sir[0], sar[0]))
        except Exception as exc:
            print(f"    [WARN] mir_eval 失败: {exc}")
    out["si_sdr"] = float(si_sdr(ref_mono, est_mono))
    return out


# ------------------------------------------------------------------ #
# 古典方法
# ------------------------------------------------------------------ #
def sep_ica(mix_stereo, sr):
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn 不可用")
    ica = FastICA(n_components=2, random_state=SEED, max_iter=1000, tol=1e-4)
    S = ica.fit_transform(mix_stereo.T)
    return S.T.astype(np.float32)


def _gini(v):
    v = np.sort(np.abs(v))
    n = len(v)
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) / (n * v.sum())) - (n + 1) / n)


def _nmf_fit(mix, sr, k=50, n_iter=250):
    X = librosa.stft(mix, n_fft=2048, hop_length=512)
    mag = np.abs(X)
    nmf = NMF(n_components=k, init="nndsvda", max_iter=n_iter,
              random_state=SEED, solver="mu")
    W = nmf.fit_transform(mag)
    H = nmf.components_
    return X, mag, W, H


def _nmf_mask_to_audio(X, W, H, mag, idx, length):
    num = np.zeros_like(mag)
    for k in idx:
        num += np.outer(W[:, k], H[k, :])
    mask = np.clip(num / (W @ H + EPS), 0.0, 1.0)
    return librosa.istft(mask * X, hop_length=512, length=length).astype(np.float32)


def sep_nmf(mix, sr, k=50):
    X, mag, W, H = _nmf_fit(mix, sr, k)
    Wn = W / (W.sum(axis=0, keepdims=True) + EPS)
    gini = np.array([_gini(H[j]) for j in range(k)]).reshape(-1, 1)
    cent = np.array([np.sum(np.arange(Wn.shape[0]) * Wn[:, j]) / Wn.shape[0]
                     for j in range(k)]).reshape(-1, 1)
    lab = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(
        np.hstack([Wn.T, gini, cent])).labels_
    g = [gini[lab == c].mean() if np.any(lab == c) else -1 for c in (0, 1)]
    voc = 0 if g[0] >= g[1] else 1
    est = _nmf_mask_to_audio(X, W, H, mag, np.where(lab == voc)[0], len(mix))
    return est, (X, W, H, mag, lab, len(mix))


def sep_nmf_oracle(mix, sr, state, ref):
    X, W, H, mag, lab, length = state
    best = None
    for c in (0, 1):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        est = _nmf_mask_to_audio(X, W, H, mag, idx, length)
        s = si_sdr(ref, est)
        if best is None or s > best[0]:
            best = (s, est)
    return best[1]


def _rpca_ialm(M, lam=None, max_iter=120, tol=1e-6):
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    nrm = max(np.linalg.norm(M, 2), EPS)
    Y, mu, rho = M / nrm, 1.25 / nrm, 1.5
    L, S = np.zeros_like(M), np.zeros_like(M)
    fM = np.linalg.norm(M, "fro") + EPS
    for _ in range(max_iter):
        U, sig, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        L = (U * np.maximum(sig - 1.0 / mu, 0.0)) @ Vt
        T = M - L + Y / mu
        S = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0.0)
        Z = M - L - S
        Y = Y + mu * Z
        if np.linalg.norm(Z, "fro") / fM < tol:
            break
        mu *= rho
    return L, S


def sep_rpca(mix, sr):
    X = librosa.stft(mix, n_fft=2048, hop_length=512)
    mag = np.abs(X)
    L, S = _rpca_ialm(mag)
    Sm, Lm = np.abs(S), np.abs(L)
    mask = np.clip(Sm / (Sm + Lm + EPS), 0.0, 1.0)
    return librosa.istft(mask * X, hop_length=512, length=len(mix)).astype(np.float32)


def sep_hpss(mix, sr, which="percussive"):
    h, p = librosa.effects.hpss(mix, kernel_size=31)
    return (p if which == "percussive" else h).astype(np.float32)


# ------------------------------------------------------------------ #
# Open-Unmix
# ------------------------------------------------------------------ #
_UMX = {}


def _umx_all(mix_stereo, sr):
    if "out" in _UMX:
        return _UMX["out"]
    try:
        import torch
        from openunmix import predict as umx_predict
        audio = torch.as_tensor(np.ascontiguousarray(mix_stereo), dtype=torch.float32)
        est = umx_predict.separate(audio=audio, rate=sr, targets=UMX_ORDER,
                                   model_str_or_path="umxhq", device="cpu",
                                   filterbank="torch")
        # openunmix 1.3 返回 dict{target: tensor}, 旧版返回 ndarray(ntarget, ch, n)
        if isinstance(est, dict):
            out = {}
            for t, v in est.items():
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()
                out[t] = np.asarray(v, dtype=np.float32)
            _UMX["out"] = out
        else:
            if hasattr(est, "detach"):
                est = est.detach().cpu().numpy()
            arr = np.asarray(est, dtype=np.float32)
            _UMX["out"] = {t: arr[i] for i, t in enumerate(UMX_ORDER)}
    except Exception as exc:
        print(f"  [SKIP] Open-Unmix 不可用: {type(exc).__name__}: {exc}")
        _UMX["out"] = None
    return _UMX["out"]


def _umx(target):
    def fn(mix_stereo, sr):
        out = _umx_all(mix_stereo, sr)
        if out is None or target not in out:
            raise RuntimeError("Open-Unmix 权重不可用")
        y = out[target]
        return y if y.ndim == 1 else y.mean(axis=0)
    return fn


# ------------------------------------------------------------------ #
def _safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def run(name, est_or_fn, gt, target, results, song, save_audio=False, note=""):
    row = {"song": song, "method": name, "target": target, "note": note,
           "runtime_s": None, "metrics": None, "error": None,
           "sdr_improvement": None, "si_sdr_improvement": None}
    t0 = time.time()
    try:
        est = est_or_fn() if callable(est_or_fn) else est_or_fn
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [FAIL] {name}: {row['error']}")
        results.append(row)
        return row
    est = np.asarray(est, dtype=np.float32)
    if est.ndim == 2:
        est = est[int(np.argmax([si_sdr(gt[target], e) for e in est]))]
    row["runtime_s"] = round(time.time() - t0, 2)
    m = evaluate(gt[target], est)
    row["metrics"] = m
    print(f"  [{name} -> {target}] SDR={_f(m['sdr'])} ISR={_f(m['isr'])} "
          f"SAR={_f(m['sar'])} SI-SDR={_f(m['si_sdr'])} ({row['runtime_s']}s)")
    if save_audio:
        d = OUTPUTS_ROOT / model_of(name) / _safe(song)
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{out_stem(name, target)}.wav"), est, SR, subtype="PCM_16")
    results.append(row)
    return row


def run_song(song, subset, duration, save_audio):
    """对一首歌跑完所有方法，返回行列表"""
    _UMX.clear()
    mix, voc, drm, mix_st = load_song(song, subset, duration)
    gt = {"vocals": voc, "drums": drm}
    print("mixture:", mix.shape, "| vocals:", voc.shape, "| drums:", drm.shape)

    rows = []
    run(BASELINE_NAME, mix, gt, "vocals", rows, song, save_audio,
        note="直接拿混音当估计, 用于标定各方法是否真的有效")
    run(BASELINE_NAME, mix, gt, "drums", rows, song, save_audio, note="同上的鼓目标基线")

    if HAVE_SKLEARN:
        run("ICA (FastICA, best of 2)", lambda: sep_ica(mix_st, SR), gt, "vocals",
            rows, song, save_audio, note="盲源分离: 立体声 2ch -> 2 分量")
    else:
        print("  [SKIP] scikit-learn 缺失")

    nmf_state = None
    if HAVE_SKLEARN:
        def _nmf():
            nonlocal nmf_state
            e, nmf_state = sep_nmf(mix, SR)
            return e
        run("NMF (unsupervised, k=50)", _nmf, gt, "vocals", rows, song, save_audio,
            note="分量按时域稀疏度聚 2 组, 取更间歇者为 vocals")
        if nmf_state is not None:
            run("NMF (oracle grouping)", lambda: sep_nmf_oracle(mix, SR, nmf_state, voc),
                gt, "vocals", rows, song, save_audio, note="分组用真值挑选 -> 上界参考")

    run("RPCA (Inexact ALM)", lambda: sep_rpca(mix, SR), gt, "vocals",
        rows, song, save_audio, note="低秩=伴奏, 稀疏=人声")

    run("HPSS (percussive)", lambda: sep_hpss(mix, SR, "percussive"), gt, "drums",
        rows, song, save_audio, note="谐波-打击分离, 取打击分量")
    run("HPSS (harmonic)", lambda: sep_hpss(mix, SR, "harmonic"), gt, "vocals",
        rows, song, save_audio, note="取谐波分量当人声 (对照)")

    for tgt in ("vocals", "drums"):
        run(f"Open-Unmix umxhq ({tgt})", lambda t=tgt: _umx(t)(mix_st, SR), gt, tgt,
            rows, song, save_audio, note="预训练深度模型 (CPU 推理, umxhq)")
    return rows


def add_improvement(rows):
    base = {(r["song"], r["target"]): r["metrics"] for r in rows
            if r["method"] == BASELINE_NAME and r["metrics"]}
    for r in rows:
        b = base.get((r["song"], r["target"]))
        if not b or not r["metrics"]:
            continue
        if r["metrics"]["sdr"] is not None and b["sdr"] is not None:
            r["sdr_improvement"] = round(r["metrics"]["sdr"] - b["sdr"], 2)
        if r["metrics"]["si_sdr"] is not None and b["si_sdr"] is not None:
            r["si_sdr_improvement"] = round(r["metrics"]["si_sdr"] - b["si_sdr"], 2)


def print_summary(rows, multi):
    banner("汇总表")
    if not multi:
        hdr = (f"{'方法':<34}{'目标':<8}{'SDR':>8}{'ΔSDR':>8}{'ISR':>8}{'SAR':>8}"
               f"{'SI-SDR':>9}{'ΔSI-SDR':>9}{'耗时s':>8}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            if r["error"]:
                print(f"{r['method']:<34}{r['target']:<8}  FAILED: {r['error']}")
                continue
            m = r["metrics"]
            print(f"{r['method']:<34}{r['target']:<8}{_f(m['sdr']):>8}"
                  f"{_f(r['sdr_improvement']):>8}{_f(m['isr']):>8}{_f(m['sar']):>8}"
                  f"{_f(m['si_sdr']):>9}{_f(r['si_sdr_improvement']):>9}"
                  f"{r['runtime_s']:>8}")
        return

    bad = degenerate_pairs(rows)
    keys, agg = [], {}
    for r in rows:
        if r["method"] == BASELINE_NAME:
            continue
        if (r["song"], r["target"]) in bad:
            continue
        k = (r["method"], r["target"])
        if k not in agg:
            agg[k] = {"sdr": [], "dsdr": [], "sisdr": [], "rt": []}
            keys.append(k)
        if r["metrics"] and r["sdr_improvement"] is not None:
            agg[k]["sdr"].append(r["metrics"]["sdr"])
            agg[k]["dsdr"].append(r["sdr_improvement"])
            agg[k]["sisdr"].append(r["metrics"]["si_sdr"])
            agg[k]["rt"].append(r["runtime_s"])

    if bad:
        print("[注意] 已剔除退化样本 (基线不可用/过稀疏): "
              + ", ".join(f"{s} / {t}" for s, t in sorted(bad)) + "\n")

    hdr = (f"{'方法':<34}{'目标':<8}{'SDR(median)':>13}{'ΔSDR(median)':>14}"
           f"{'ΔSDR(mean)':>12}{'sd':>7}{'n':>4}{'耗时s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        a = agg[k]
        if not a["sdr"]:
            print(f"{k[0]:<34}{k[1]:<8}  (无有效结果)")
            continue
        s = np.array(a["dsdr"])
        print(f"{k[0]:<34}{k[1]:<8}{np.median(a['sdr']):>13.2f}"
              f"{np.median(a['dsdr']):>14.2f}{np.mean(a['dsdr']):>12.2f}"
              f"{s.std():>7.2f}{len(a['dsdr']):>4d}{np.mean(a['rt']):>8.1f}")


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["song", "method", "target", "SDR", "delta_SDR", "ISR", "SAR",
                    "SI_SDR", "delta_SI_SDR", "runtime_s", "note"])
        for r in rows:
            m = r["metrics"] or {}
            w.writerow([r["song"], r["method"], r["target"], m.get("sdr"),
                        r.get("sdr_improvement"), m.get("isr"), m.get("sar"),
                        m.get("si_sdr"), r.get("si_sdr_improvement"),
                        r["runtime_s"], r.get("note", "")])


def save_outputs(rows, songs, args):
    payload = {"songs": songs, "subset": args.subset, "duration": args.duration,
               "sr": SR, "metric": "museval BSSEval v4 (1s window, median), mono",
               "results": rows}
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_DIR / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_csv(COMPARISON_DIR / "comparison_results.csv", rows)

    # 每个模型自己的结果（自包含，方便单独查看某个模型）
    for model in MODEL_FOLDERS:
        sub = [r for r in rows if model_of(r["method"]) == model]
        if not sub:
            continue
        d = OUTPUTS_ROOT / model
        d.mkdir(parents=True, exist_ok=True)
        _write_csv(d / "metrics.csv", sub)


def plot_bars(rows, multi):
    if multi:
        bad = degenerate_pairs(rows)
        seen = {}
        for r in rows:
            if r["method"] == BASELINE_NAME:
                continue
            if (r["song"], r["target"]) in bad:
                continue
            k = (r["method"], r["target"])
            if r["metrics"] and r["sdr_improvement"] is not None:
                seen.setdefault(k, []).append(r["sdr_improvement"])
        keys, dsdr = [], []
        for k, v in seen.items():
            keys.append(k)
            dsdr.append(float(np.median(v)))
        if not keys:
            return
        order = np.argsort(dsdr)
        labels = [f"{keys[i][0]}" for i in order]
        vals = [dsdr[i] for i in order]
        xlabel = "ΔSDR vs no-separation baseline (dB)  —  higher is better"
        title = (f"Median ΔSDR over up to {len(set(r['song'] for r in rows))} songs\n"
                 f"(MUSDB18-HQ train, 30-s clips, museval BSSEval v4)")
    else:
        got = [(r["method"], r["metrics"]["sdr"]) for r in rows
               if r["metrics"] and r["metrics"].get("sdr") is not None]
        if not got:
            return
        order = np.argsort([v for _, v in got])
        labels = [got[i][0] for i in order]
        vals = [got[i][1] for i in order]
        xlabel = "SDR (dB)  —  higher is better"
        title = ("Separation quality on the same 30-s clip\n"
                 "(MUSDB18-HQ, A Classic Education - NightOwl, museval BSSEval v4)")

    fig, ax = plt.subplots(figsize=(11, max(4.2, 0.52 * len(labels) + 1.9)))
    colors = ["#C44E52" if v < 0 else ("#DD8452" if v < 3 else "#55A868") for v in vals]
    ax.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + (0.06 if v >= 0 else -0.06), i, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = COMPARISON_DIR / ("fig_compare_dsdr.png" if multi else "fig_compare_sdr.png")
    fig.savefig(out, dpi=150)
    plt.close("all")
    print(f"[OK] 对比图 -> {out}")


# ================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=0, help="起始歌曲序号 (0-based)")
    ap.add_argument("--songs", type=int, default=1, help="跑多少首歌 (>=2 时输出均值±标准差)")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--subset", default="train")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--reanalyze", action="store_true",
                    help="不重跑, 直接读已有 comparison_results.json 重算汇总与图")
    args = ap.parse_args()

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    if args.reanalyze:
        jpath = COMPARISON_DIR / "comparison_results.json"
        with open(jpath, encoding="utf-8") as f:
            payload = json.load(f)
        all_rows = payload["results"]
        add_improvement(all_rows)
        multi = len(payload.get("songs", [])) > 1
        print_summary(all_rows, multi=multi)
        save_outputs(all_rows, payload.get("songs", []), args)
        try:
            plot_bars(all_rows, multi=multi)
        except Exception:
            traceback.print_exc()
        print("\n[reanalyze] 已按 中位数聚合 + 剔除退化样本 重算，未重新计算音频")
        return 0

    songs = list_songs(args.subset)[args.song: args.song + max(1, args.songs)]
    print("指标口径:", "museval (BSSEval v4)" if HAVE_MUSEVAL else
          ("mir_eval" if HAVE_MIREVAL else "仅 SI-SDR"))

    all_rows = []
    for i, song in enumerate(songs):
        banner(f"[{i + 1}/{len(songs)}] {song}  (前 {args.duration} 秒, sr={SR})")
        all_rows.extend(run_song(song, args.subset, args.duration, args.save_audio))

    add_improvement(all_rows)
    print_summary(all_rows, multi=len(songs) > 1)
    save_outputs(all_rows, songs, args)
    try:
        plot_bars(all_rows, multi=len(songs) > 1)
    except Exception:
        traceback.print_exc()
    print(f"\n各模型产出 -> {OUTPUTS_ROOT}\\<模型>\\")
    print(f"跨模型对比 -> {COMPARISON_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
