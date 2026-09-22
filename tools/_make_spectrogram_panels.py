# -*- coding: utf-8 -*-
"""阶段 3 · 频谱面板（T3.1 入口脚本，覆盖 T3.2 / T3.3）

读 : 02_databases/MUSDB18-HQ/test/<Song>/{mixture,vocals,drums,bass,other}.wav
     03_outputs/<Model>/test/<Song>/*.flac        （整曲分离结果，**不重推理**）
写 : 04_reports/separation/figures/spectrograms/panel_*.png
     04_reports/separation/data/comparison/panel_decomp_checks.json

产出三类图
----------
1. `panel_<song>_decomp.png`      BSS-Eval 四分量分解（2×3：Mixture/Target/Estimate/
                                  Interference/Artifacts/滤波失真）
2. `panel_<song>_multimodel.png`  多模型并排（四轨组 / 单目标组分开）
3. `panel_<song>_mask.png`        机制图：模型有效掩码 vs 理想比值掩码 + 频带能量画像

口径
----
* 分解用 `tools/_bss_decompose.py`（复用 museval 内部实现），与报告 SDR **同源**。
* 每张图落盘前都过一遍恒等式残差 + 与 `museval.bss_eval` 的 SDR 交叉验证，
  两者都不达标就写 `status=FAIL` —— 判成功只看 JSON 的 status。
* 片段避开前奏：按「四轨能量占比的最小值」自动挑最均衡的窗口。

用法
----
python tools/_make_spectrogram_panels.py                    # 默认 3 首
python tools/_make_spectrogram_panels.py --dur 30 --songs "Georgia Wonder - Siren"
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths

_paths.setup_env()

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from _bss_decompose import component_energy_db, crit, decompose

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"

SR = 44100
STEMS = ["vocals", "drums", "bass", "other"]
N_FFT = 2048
HOP = 512
FILTERS_LEN = 512

# 选歌：**数据驱动**，由 tools/_select_panel_songs.py 产出，规则写在 JSON 的 criteria 里。
# ⚠️ 不要再手选。原先手选的 3 首里有 `PR - Oh No`，实测其 vocals 只占全曲 **1.1%** 能量
#    （drums 64.6%），对「vocals 为目标」的分解/掩码面板是**退化样本**：
#    参考轨近乎静音 → SDR 掉到 0.45 dB、ISR 0.57 dB，图跑通了但数字没有意义
#    （对应本项目第 2 类陷阱：片段能量陷阱）。
SELECTION_JSON = _paths.DATA_COMPARISON / "panel_song_selection.json"
FALLBACK_SONGS = [
    "The Easton Ellises - Falcon 69",
    "Bobby Nobody - Stitch Up",
    "Raft Monk - Tiring",
]


def load_selected_songs(default_stem="vocals"):
    """优先读选歌 JSON；缺失/为空时退回已核实的 3 首。"""
    if SELECTION_JSON.exists():
        try:
            d = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
            picked = [s for s in (d.get("picked") or []) if s]
            if picked and d.get("criteria", {}).get("stem") == default_stem:
                return picked, d.get("criteria", {})
            if picked:
                print(f"  [warn] 选歌 JSON 的 stem={d.get('criteria',{}).get('stem')} "
                      f"与当前 {default_stem} 不符，退回内置列表")
        except Exception as e:
            print(f"  [warn] 读选歌 JSON 失败: {e}")
    return list(FALLBACK_SONGS), {}

# 面板分组：四轨模型与单目标模型的 SDR 不可直接比（本项目跨组禁令），图上分开画
GROUP_OF = {
    "Demucs": "4-stem", "MDX-Net": "4-stem", "Open-Unmix": "4-stem",
    "Oracle-IRM": "4-stem", "Conv-TasNet": "4-stem", "MMDenseLSTM": "4-stem",
    "BS-RoFormer-L12": "1-target", "BS-RoFormer-L6": "1-target",
    "BSRNN-opt": "1-target", "BSRNN-large": "1-target", "BSRNN-SIMO": "1-target",
}
# DPRNN 已被准入规则排除、RPCA 只跑过 2 首 → 默认不进并排图
EXCLUDE = {"DPRNN", "RPCA"}


# --------------------------------------------------------------------------- 读
def verify_frames(p: Path, expect_frames: int | None = None):
    """读之前先查帧数。

    半成品 FLAC 的 `total_samples=0` 会让 soundfile 按一个天文数字去分配数组
    （报 `array is too big`），本项目已踩过一次（BSRNN-SIMO / Siren，5.7 MB 的假产出）。
    这里在**打开文件之前**就把不合格的挡掉。
    """
    info = sf.info(str(p))
    if info.samplerate != SR:
        raise ValueError(f"采样率 {info.samplerate} != {SR}")
    if info.frames <= 0:
        raise ValueError(f"frames={info.frames}（半成品/未收尾文件）")
    if expect_frames is not None and info.frames != expect_frames:
        raise ValueError(
            f"frames={info.frames} != 源 {expect_frames} "
            f"(ratio={info.frames / expect_frames:.4f})")
    return info


def read_audio(p: Path, expect_frames: int | None = None) -> np.ndarray:
    verify_frames(p, expect_frames)
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{p} 采样率 {sr} != {SR}")
    return y


def load_gt(song: str):
    d = _paths.MUSDB18_ROOT / "test" / song
    mix = read_audio(d / "mixture.wav")
    n = len(mix)
    # 真值四轨必须与 mixture 逐帧对齐，否则后面所有切片都会错位
    stems = {s: read_audio(d / f"{s}.wav", expect_frames=n) for s in STEMS}
    return stems, mix, n


def read_clip(p: Path, s0: int, end: int, expect_frames: int | None = None) -> np.ndarray:
    """只读需要的片段（整曲 430 s 的模型输出没必要整段读进内存）。"""
    verify_frames(p, expect_frames)
    y, sr = sf.read(str(p), start=s0, stop=end, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{p} 采样率 {sr} != {SR}")
    return y


def pick_balanced_offset(stems, dur, target="vocals", target_min=12.0, step=5.0):
    """按「四轨能量占比最小值」挑最均衡的窗口，避开前奏低频引子。

    额外要求**目标轨占比 ≥ target_min**：参考轨太稀疏时 SDR/ISR 会塌成退化值
    （PR - Oh No 的 vocals 只占全曲 1.1%，SDR 0.45 dB），这种窗不能拿来做面板。
    返回 (offset, min_share, shares, qualified)。
    """
    n = min(len(v) for v in stems.values())
    win = int(dur * SR)
    best_ok = (None, -1.0, None)
    best_any = (None, -1.0, None)
    for off in np.arange(0, max(1.0, n / SR - dur), step):
        s0 = int(off * SR)
        if s0 + win > n:
            break
        p = {k: float(np.mean(v[s0:s0 + win] ** 2)) + 1e-20 for k, v in stems.items()}
        tot = sum(p.values())
        sh = {k: 100.0 * v / tot for k, v in p.items()}
        mn = min(sh.values())
        if mn > best_any[1]:
            best_any = (float(off), mn, sh)
        if sh.get(target, 0.0) >= target_min and mn > best_ok[1]:
            best_ok = (float(off), mn, sh)
    pick = best_ok if best_ok[0] is not None else best_any
    return pick[0], pick[1], pick[2], best_ok[0] is not None


def discover_models(song: str, stem: str, expect_frames: int | None = None):
    """扫 03_outputs，返回该歌曲下**帧数合格**的模型 -> 音频路径，以及被跳过的原因。

    只判断「文件存在」是不够的（本项目踩过「有目录 ≠ 成功」）：必须帧数与源 mixture
    一致才准进图，否则会把半成品当成一个模型画进去。
    """
    found, skipped = {}, []
    for mdir in sorted(_paths.OUTPUTS.iterdir()):
        if not mdir.is_dir() or mdir.name in EXCLUDE:
            continue
        f = mdir / "test" / song / f"{stem}.flac"
        if not f.is_file():
            continue
        try:
            verify_frames(f, expect_frames)
        except Exception as e:
            skipped.append({"model": mdir.name, "reason": str(e)})
            print(f"  [skip] {mdir.name}: {e}")
            continue
        found[mdir.name] = f
    return found, skipped


# ------------------------------------------------------------------------- 处理
def mono(x):
    return x.mean(axis=1)


def spec_db(x, ref=None):
    """对数幅度谱（dB）。ref 给定时相对 ref 的峰值归一，便于跨图对比。"""
    import librosa

    X = librosa.stft(x, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(X)
    refv = (np.abs(ref).max() if ref is not None else mag.max()) or 1e-12
    return 20.0 * np.log10(mag / refv + 1e-8)


def render_spec(ax, S, title, vmin=-80, vmax=0, cmap="magma"):
    im = ax.imshow(S, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _freq_ticks(ax, n_freq, sr=SR, n_fft=N_FFT):
    ax.set_yticks([int(n_freq * f / (sr / 2)) for f in (500, 2000, 8000)])
    ax.set_yticklabels(["500", "2k", "8k"], fontsize=7)


# --------------------------------------------------------------------------- 图
def fig_decomp(song, slug, stem, clip, parts, info, outdir):
    refc, estc, mixt = clip["ref"], clip["est"], clip["mix"]
    j = STEMS.index(stem)
    V = dict(vmin=-80, vmax=0)

    panels = [
        (mixt, f"Mixture（输入混音）"),
        (refc[j], f"Target = GT {stem}（真值）"),
        (estc, f"Estimate（模型输出）"),
        (parts["e_interf"], "Interference（干扰）"),
        (parts["e_artif"], "Artifacts（伪影）"),
        (parts["e_spat"], "Filtering（滤波失真 e_spat）"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
    for ax, (x, ttl) in zip(axes.ravel(), panels):
        S = spec_db(mono(x), ref=mono(refc[j]))
        render_spec(ax, S, ttl, **V)
        _freq_ticks(ax, S.shape[0])
    axes[0, 0].set_ylabel("Hz", fontsize=8)

    edb = info["energy_db"]
    share = info["share_pct"]
    fig.suptitle(
        f"{song}  ·  {stem}  ·  BSS-Eval v4 四分量分解   "
        f"[SDR {info['sdr']:.2f} dB | SIR {info['sir']:.2f} | SAR {info['sar']:.2f}]\n"
        f"片段 offset={info['offset']:.0f}s 时长 {info['dur']:.0f}s  "
        f"能量占比  目标 {share['s_true']:.1f}% / 滤波 {share['e_spat']:.1f}% / "
        f"干扰 {share['e_interf']:.1f}% / 伪影 {share['e_artif']:.1f}%   "
        f"恒等式残差 {info['residual_rel']:.1e}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = outdir / f"panel_{slug}_decomp.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_multimodel(song, slug, stem, ref_mono, models_specs, otext, outdir):
    groups = {}
    for name, S in models_specs.items():
        groups.setdefault(GROUP_OF.get(name, "other"), []).append((name, S))

    order = ["4-stem", "1-target", "other"]
    rows = [(g, n, S) for g in order for (n, S) in sorted(groups.get(g, []))]
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 1, figsize=(12.5, max(2.0, 1.15 * len(rows))))
    axes = np.atleast_1d(axes)
    for ax, (g, name, S) in zip(axes, rows):
        render_spec(ax, S, f"{name}   [{g}]", vmin=-80, vmax=0)
        _freq_ticks(ax, S.shape[0])
    fig.suptitle(
        f"{song}  ·  {stem}  ·  多模型并排（同一片段，同一色标）\n"
        f"{otext}\n※ 四轨组与单目标组的 SDR 不可直接比较，仅作视觉对照",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = outdir / f"panel_{slug}_multimodel.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_mask(song, slug, stem, clip, outdir):
    """T3.3 机制图：模型有效掩码 vs 理想比值掩码 + 频带能量画像。"""
    import librosa

    j = STEMS.index(stem)
    X_mix = np.abs(librosa.stft(mono(clip["mix"]), n_fft=N_FFT, hop_length=HOP))
    X_est = np.abs(librosa.stft(mono(clip["est"]), n_fft=N_FFT, hop_length=HOP))
    X_tgt = np.abs(librosa.stft(mono(clip["ref"][j]), n_fft=N_FFT, hop_length=HOP))
    X_all = sum(
        np.abs(librosa.stft(mono(clip["ref"][k]), n_fft=N_FFT, hop_length=HOP))
        for k in range(len(STEMS))
    )

    eps = 1e-8
    m_model = X_est / (X_mix + eps)
    m_ideal = X_tgt / (X_all + eps)
    diff = m_model - m_ideal

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, M, ttl in (
        (axes[0], m_ideal, "理想比值掩码  |target| / Σ|stems|"),
        (axes[1], m_model, "模型有效掩码  |estimate| / |mixture|"),
        (axes[2], diff, "差值（模型 − 理想）"),
    ):
        vmin, vmax = (0, 1) if M is not diff else (-0.5, 0.5)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="coolwarm" if M is diff else "viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); _freq_ticks(ax, M.shape[0])
        fig.colorbar(im, ax=ax, fraction=0.046)

    # 频带能量画像：看模型把能量分到哪些频段
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    edges = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, SR / 2]
    prof_t, prof_e = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        prof_t.append(float(np.mean(X_tgt[sel] ** 2)))
        prof_e.append(float(np.mean(X_est[sel] ** 2)))
    prof_t = 10 * np.log10(np.array(prof_t) + 1e-20)
    prof_e = 10 * np.log10(np.array(prof_e) + 1e-20)
    labels = [f"{int(a/1000)}k" if a >= 1000 else str(int(a)) for a in edges[:-1]]

    fig.suptitle(
        f"{song}  ·  {stem}  ·  掩码机制图（模型如何「切」出目标）\n"
        f"理想掩码均值 {m_ideal.mean():.3f}   模型掩码均值 {m_model.mean():.3f}   "
        f"相关性 {np.corrcoef(m_ideal.ravel(), m_model.ravel())[0,1]:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = outdir / f"panel_{slug}_mask.png"
    fig.savefig(p)
    plt.close(fig)

    # 频带画像单独一张
    fig2, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(labels))
    ax.plot(x, prof_t, "o-", label="GT target", lw=2)
    ax.plot(x, prof_e, "s--", label="Estimate", lw=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("频带下限 (Hz)"); ax.set_ylabel("平均功率 (dB)")
    ax.set_title(f"{song} · {stem} · 频带能量画像", fontsize=9)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    p2 = outdir / f"panel_{slug}_bands.png"
    fig2.savefig(p2)
    plt.close(fig2)
    return p, p2


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None,
                    help="不给则读 panel_song_selection.json（数据驱动选歌）")
    ap.add_argument("--stem", default="vocals")
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--offset", type=float, default=None, help="不给则自动挑最均衡窗口")
    ap.add_argument("--target-min", type=float, default=12.0,
                    help="目标轨能量占比下限(%%)；低于此值视为退化窗")
    ap.add_argument("--ref-model", default="Demucs", help="四分量分解图用哪个模型")
    args = ap.parse_args()

    if args.songs:
        songs = args.songs
        sel_criteria = {"source": "命令行指定"}
    else:
        songs, sel_criteria = load_selected_songs(args.stem)
        sel_criteria = {"source": "panel_song_selection.json", **sel_criteria}

    figdir = _paths.SEPARATION / "figures" / "spectrograms"
    datadir = _paths.DATA_COMPARISON
    figdir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stem": args.stem, "dur": args.dur,
        "filters_len": FILTERS_LEN, "n_fft": N_FFT, "hop": HOP,
        "metric_source": "museval BSSEval v4 内部实现（与报告 SDR 同源）",
        "song_selection": sel_criteria,
        "target_min_pct": args.target_min,
        "songs_list": list(songs),
        "songs": [],
        "status": "PASS",
    }
    t0 = time.time()

    for song in songs:
        stem = args.stem
        slug = song.replace(" ", "_").replace("/", "-")
        print(f"\n=== {song} ===")
        try:
            stems, mix, n = load_gt(song)
        except Exception as e:
            print(f"  [ERR] 读 GT 失败: {e}")
            report["songs"].append({"song": song, "status": "FAIL", "error": str(e)})
            report["status"] = "FAIL"
            continue

        if args.offset is not None:
            off, bal, sh, ok_bal = args.offset, float("nan"), {}, True
        else:
            off, bal, sh, ok_bal = pick_balanced_offset(
                stems, args.dur, target=stem, target_min=args.target_min)
        print(f"  [pick] offset={off:.1f}s  最小轨道占比={bal:.1f}%  "
              f"目标轨({stem})={sh.get(stem, float('nan')):.1f}%  "
              + ("均衡达标" if ok_bal else "*** 无达标窗：目标轨过稀疏，数字将退化 ***"))

        s0 = int(off * SR)
        win = int(args.dur * SR)
        end = min(s0 + win, n)
        sl = slice(s0, end)
        ref = np.stack([stems[s][sl] for s in STEMS], 0).astype(np.float64)
        mixt = mix[sl].astype(np.float64)

        models, skipped = discover_models(song, stem, expect_frames=n)
        print(f"  [models] {len(models)} 个: {', '.join(sorted(models))}")
        if args.ref_model not in models:
            pick = sorted(models)[0] if models else None
            print(f"  [warn] 参考模型 {args.ref_model} 无产出，改用 {pick}")
            ref_model = pick
        else:
            ref_model = args.ref_model

        entry = {
            "song": song, "offset": float(off), "dur": float(args.dur),
            "min_stem_share_pct": None if not sh else float(bal),
            "stem_share_pct": {k: round(v, 2) for k, v in (sh or {}).items()},
            "models": sorted(models), "skipped": skipped,
            "degenerate": not ok_bal,
            "status": "PASS", "figures": [], "checks": {},
        }
        if not ok_bal:
            # 目标轨过稀疏 —— 面板能画出来，但 SDR/ISR 是退化值，不能当结论用
            entry["status"] = "FAIL"
            report["status"] = "FAIL"
            print("  [ERR] 退化窗口 -> 标记 FAIL（需换歌或显式指定 --offset）")

        if not ref_model:
            print("  [ERR] 无任何模型产出")
            entry["status"] = "FAIL"; report["status"] = "FAIL"
            report["songs"].append(entry); continue

        # ---- 参考模型的四分量分解 ----
        est = read_clip(models[ref_model], s0, end, expect_frames=n).astype(np.float64)
        m = min(ref.shape[1], est.shape[0])
        ref_c, est_c = ref[:, :m, :], est[:m, :]
        est_c = est_c[: ref_c.shape[1]]
        ref_c = ref_c[:, : est_c.shape[0], :]

        try:
            parts = decompose(ref_c, est_c, j=STEMS.index(stem), filters_len=FILTERS_LEN)
        except Exception as e:
            print(f"  [ERR] 分解失败: {e}")
            entry["status"] = "FAIL"; report["status"] = "FAIL"
            report["songs"].append(entry); continue

        sdr, isr, sir, sar = [float(np.squeeze(v)) for v in crit(parts)]
        edb = component_energy_db(parts)
        e_lin = {k: 10 ** (v / 10) for k, v in edb.items()}
        tot = sum(e_lin.values())
        share = {k: 100.0 * v / tot for k, v in e_lin.items()}

        info = {"sdr": sdr, "sir": sir, "sar": sar, "isr": isr,
                "energy_db": edb, "share_pct": share,
                "offset": off, "dur": m / SR, "residual_rel": parts["residual_rel"]}

        # ---- 门禁 1：恒等式残差 ----
        ok_resid = parts["residual_rel"] < 1e-9
        # ---- 门禁 2：与 museval.bss_eval 单窗交叉验证 ----
        import museval.metrics as _M

        est_all = ref_c.copy()
        est_all[STEMS.index(stem)] = est_c
        SDR_x, _, _, _, _ = _M.bss_eval(
            ref_c, est_all, window=est_c.shape[0], hop=est_c.shape[0],
            filters_len=FILTERS_LEN)
        sdr_x = float(SDR_x[STEMS.index(stem)][0])
        d_sdr = abs(sdr_x - sdr)
        ok_x = d_sdr < 1e-6

        entry["checks"] = {
            "residual_rel": parts["residual_rel"], "residual_ok": bool(ok_resid),
            "sdr_ours": sdr, "sdr_museval_single_window": sdr_x,
            "sdr_abs_diff": d_sdr, "sdr_crosscheck_ok": bool(ok_x),
            "sir": sir, "sar": sar, "isr": isr,
            "share_pct": {k: round(v, 3) for k, v in share.items()},
        }
        print(f"  [check] 残差 {parts['residual_rel']:.1e} | "
              f"SDR 自算 {sdr:.4f} vs museval {sdr_x:.4f} (diff {d_sdr:.2e})")
        if not (ok_resid and ok_x):
            entry["status"] = "FAIL"; report["status"] = "FAIL"
            print("  [ERR] 门禁未过 -> 本曲标记 FAIL")

        otext = (f"offset={off:.0f}s 时长 {m/SR:.0f}s | 参考模型 {ref_model} "
                 f"(SDR {sdr:.2f} dB)")

        # ---- 出图 ----
        clip = {"ref": ref_c, "est": est_c, "mix": mixt[: est_c.shape[0]]}
        try:
            p1 = fig_decomp(song, slug, stem, clip, parts, info, figdir)
            entry["figures"].append(str(p1))
            print(f"  [fig] {p1.name}")

            specs = {}
            for name, path in sorted(models.items()):
                yy = read_clip(path, s0, end, expect_frames=n).astype(np.float64)
                yy = yy[: est_c.shape[0]]
                specs[name] = spec_db(mono(yy), ref=mono(ref_c[STEMS.index(stem)]))
            r = fig_multimodel(song, slug, stem, mono(ref_c[STEMS.index(stem)]),
                               specs, otext, figdir)
            if r:
                entry["figures"].append(str(r))
                print(f"  [fig] {r.name}  ({len(specs)} 模型)")

            pm, pb = fig_mask(song, slug, stem, clip, figdir)
            entry["figures"] += [str(pm), str(pb)]
            print(f"  [fig] {pm.name}, {pb.name}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERR] 出图失败: {e}")
            entry["status"] = "FAIL"; report["status"] = "FAIL"

        report["songs"].append(entry)

    report["elapsed_s"] = round(time.time() - t0, 1)
    out = datadir / "panel_decomp_checks.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"status = {report['status']}   耗时 {report['elapsed_s']}s")
    print(f"[out] {out}")
    for e in report["songs"]:
        sk = len(e.get("skipped", []))
        print(f"  {e['song']:36s} {e['status']:4s}  {len(e.get('figures', []))} figs"
              + (f"  [skip {sk}]" if sk else ""))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    _sys.exit(main())
