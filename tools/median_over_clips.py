# -*- coding: utf-8 -*-
"""把 model_runs.json 里「多曲」的结果聚合成中位数表。

为什么必须用中位数
------------------
个别曲子的某个 stem 可能极稀疏甚至静音，museval 会给出 nan 或退化值（曾把标准差
拉到 ±30 dB）。用均值会被这种离群点污染，所以一律取中位数，并同时报告参与统计的曲数 n。

用法：
    python tools/median_over_clips.py                 # 用 bench_multisong.SONGS 的片段集
    python tools/median_over_clips.py --all-songs     # 用 JSON 里出现的所有曲目
    python tools/median_over_clips.py --write PATH    # 写 md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
ROOT = _paths.PROJECT_ROOT
RUNS = _paths.comparison_dir() / "model_runs.json"
OUT_JSON = _paths.comparison_dir() / "model_runs_median.json"

sys.path.insert(0, str(_paths.TOOLS))
from bench_multisong import SONGS as BALANCED  # noqa: E402

STEMS = ["vocals", "drums", "bass", "other"]
LABELS = {
    "oracle": "IRM/IBM Oracle", "rpca": "RPCA", "umx": "Open-Unmix", "mdx": "MDX-Net",
    "convtasnet": "Conv-TasNet", "mmdenselstm": "MMDenseLSTM",
    "bsroformer_l12": "BS-RoFormer L12", "bsroformer_l6": "BS-RoFormer L6",
    "bsrnn_all": "oBSRNN (4 ckpt)", "bsrnn_large_all": "BSRNN large (4 ckpt)",
    "bsrnn_simo": "oBSRNN-SIMO", "bsrnn": "oBSRNN (vocals ckpt)",
    "bsrnn_large": "BSRNN large (vocals ckpt)", "demucs": "Demucs (htdemucs)",
    "dprnn": "DPRNN (自训, 4-stem)",
}
ORDER = ["dprnn", "mmdenselstm", "convtasnet", "demucs", "umx", "simo_placeholder", "mdx",
         "bsrnn_simo", "bsrnn_all", "bsrnn_large_all", "bsroformer_l6", "bsroformer_l12",
         "oracle", "rpca"]
TGT = {"bsroformer_l12": "vocals", "bsroformer_l6": "vocals+other"}


def med(vals):
    v = sorted(x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return None, 0
    n = len(v)
    return (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])), n


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-songs", action="store_true")
    ap.add_argument("--write", default=None)
    args = ap.parse_args()

    data = json.loads(RUNS.read_text(encoding="utf-8"))["runs"]
    if args.all_songs:
        want = None
        want_off = {}
    else:
        want = {s for s, _ in BALANCED}
        want_off = {s: o for s, o in BALANCED}

    # 🔴 offset 校验（2026-09-17 补）
    # `model_runs.json` 是「按 model→song 覆盖」写的，**同一个歌名下只会留最后一次的
    # offset**。聚合时若只按歌名挑选，某首歌被在 offset=0 上重跑过一次，就会把
    # 「开头低频引子」的片段混进均衡片段集合，而屏幕上完全看不出来 —— 这正是本项目
    # 反复踩到的「跑通但指标静默错」。这里显式比对 offset，不一致就剔除并高亮报警。
    mismatch = []

    agg = {}
    for model, songs in data.items():
        recs = []
        for s, r in songs.items():
            if want is not None and s not in want:
                continue
            if s in want_off:
                got = r.get("offset_s")
                if got is not None and abs(float(got) - float(want_off[s])) > 0.5:
                    mismatch.append((model, s, want_off[s], float(got)))
                    continue
            recs.append(r)
        if not recs:
            continue
        entry = {"label": LABELS.get(model, model), "n_clips": len(recs),
                 "clips": sorted(r["song"] for r in recs), "sdr": {}, "si_sdr": {}, "rtf": None}
        all_stems = set()
        for r in recs:
            recs_sdr = r.get("sdr") or {}
            if model in TGT:
                all_stems.add(TGT[model])
            else:
                all_stems |= set(recs_sdr) | set(r.get("si_sdr") or {})
        for st in sorted(all_stems):
            m, n = med([(r.get("sdr") or {}).get(st) for r in recs])
            entry["sdr"][st] = {"median": m, "n": n}
            m2, n2 = med([(r.get("si_sdr") or {}).get(st) for r in recs])
            entry["si_sdr"][st] = {"median": m2, "n": n2}
        m, _ = med([r.get("rtf") for r in recs])
        entry["rtf"] = m
        # 四轨均值（只对出全 4 stem 的模型有意义）
        vals = [(entry["sdr"].get(s) or {}).get("median") for s in STEMS]
        entry["sdr_avg4"] = (sum(vals) / 4) if all(v is not None for v in vals) else None
        # 单目标模型（BS-RoFormer）：目标名不在 STEMS 里，单独记一笔，避免整行空值
        if model in TGT:
            t = TGT[model]
            entry["single_target"] = {
                "name": t,
                "sdr_median": (entry["sdr"].get(t) or {}).get("median"),
                "si_sdr_median": (entry["si_sdr"].get(t) or {}).get("median"),
            }
        agg[model] = entry

    OUT_JSON.write_text(json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 渲染 markdown ----
    out = []
    n_clip = next((v["n_clips"] for v in agg.values()), 0)
    keys = [k for k in ORDER if k in agg] + [k for k in agg if k not in ORDER]
    four = [k for k in keys if not agg[k].get("single_target")]
    single = [k for k in keys if agg[k].get("single_target")]

    out.append(f"### 多片段中位数（museval SDR，{n_clip} 个均衡片段）\n")
    out.append("| 模型 | 人声 | 鼓 | 贝斯 | 其他 | 四轨均值 | RTF (中位) | n |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k in four:
        e = agg[k]
        vals = []
        for s in STEMS:
            d = e["sdr"].get(s)
            vals.append(d["median"] if d else None)
        cells = [fmt(v) for v in vals]
        got = [v for v in vals if v is not None]
        avg = fmt(sum(got) / len(got)) if len(got) == 4 else "—"
        out.append(f"| {e['label']} | " + " | ".join(cells) + f" | {avg} | {fmt(e['rtf'])} | {e['n_clips']} |")
    out.append("")
    out.append("> `四轨均值`只对出全 4 个 stem 的模型计算（RPCA / BS-RoFormer 不产出完整 4 stem，故不参与）。")
    if single:
        out.append("")
        out.append("**单目标模型**（不产出 4 stem，单独列出）：\n")
        out.append("| 模型 | 目标 | museval SDR | SI-SDR | RTF (中位) | n |")
        out.append("|---|---|---|---|---|---|")
        for k in single:
            e = agg[k]
            st = e["single_target"]
            out.append(f"| {e['label']} | `{st['name']}` | {fmt(st['sdr_median'])} | "
                       f"{fmt(st['si_sdr_median'])} | {fmt(e['rtf'])} | {e['n_clips']} |")

    out.append("")
    out.append("### 多片段中位数（SI-SDR）\n")
    out.append("| 模型 | 人声 | 鼓 | 贝斯 | 其他 |")
    out.append("|---|---|---|---|---|")
    for k in four:
        e = agg[k]
        cells = []
        for s in STEMS:
            d = e["si_sdr"].get(s)
            cells.append(fmt(d["median"]) if d else "—")
        out.append(f"| {e['label']} | " + " | ".join(cells) + " |")

    text = "\n".join(out)
    print(text)
    if mismatch:
        print("\n" + "!" * 68)
        print("⚠️  有记录因 offset 与均衡片段不符被剔除（结果里不含这些片段）：")
        for model, s, exp, got in mismatch:
            print(f"    {model:18s} {s:32s} 期望 offset={exp:g}s，实际 {got:g}s")
        print("    处理：把该曲目按正确 offset 重跑，或接受当前聚合口径。")
        print("!" * 68)
    if args.write:
        p = Path(args.write)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"\n[写出] {p}")
    print(f"\n[JSON] {OUT_JSON}")


if __name__ == "__main__":
    main()
