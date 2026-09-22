# -*- coding: utf-8 -*-
"""把 04_reports/separation/data/comparison/model_runs.json 渲染成报告用的 Markdown 表格。

用法：
    python tools/summarize_model_runs.py                # 全部
    python tools/summarize_model_runs.py --stems        # 额外输出「按 stem 的 SDR 矩阵」
    python tools/summarize_model_runs.py --write PATH   # 写文件
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
ROOT = _paths.PROJECT_ROOT
JSON_PATH = _paths.comparison_dir() / "model_runs.json"

STEM_ORDER = ["vocals", "drums", "bass", "other"]
STEM_ZH = {"vocals": "人声", "drums": "鼓", "bass": "贝斯", "other": "其他"}


def fmt(v, nd=2, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}{suffix}"
    return str(v)


def load():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def pick(run_map: dict) -> dict:
    """取该模型最新一次（唯一）运行记录。"""
    if not run_map:
        return {}
    song = next(iter(run_map))
    return run_map[song]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", action="store_true", help="额外输出按 stem 分列的 SDR 矩阵")
    ap.add_argument("--write", default=None, help="写到指定 md 文件")
    args = ap.parse_args()

    data = load()
    rows, per_stem = [], {}
    for name, run_map in data["runs"].items():
        r = pick(run_map)
        if not r:
            continue
        si = r.get("si_sdr") or {}
        sdr = r.get("sdr") or {}
        tgt = r.get("targets") or []
        key = "+".join(tgt) if len(tgt) > 1 else (tgt[0] if tgt else "?")
        rows.append({
            "key": name,
            "label": r.get("label", name),
            "status": r.get("status"),
            "params": (r.get("info") or {}).get("params"),
            "load_s": r.get("load_s"),
            "infer_s": r.get("infer_s"),
            "rtf": r.get("rtf"),
            "target": key,
            "sdr": sdr,
            "si_sdr": si,
            "ref_used": r.get("ref_used") or {},
            "error": r.get("error"),
        })
        for st in tgt:
            per_stem.setdefault(st, {})[r.get("label", name)] = (sdr.get(st), si.get(st))

    out = []
    out.append("| 模型 | 状态 | 参数量(M) | 加载(s) | 推理(s) | RTF | 输出轨道 |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        pm = f"{r['params']/1e6:.2f}" if isinstance(r["params"], (int, float)) else "—"
        out.append(
            f"| {r['label']} | {r['status']} | {pm} | {fmt(r['load_s'])} | "
            f"{fmt(r['infer_s'])} | {fmt(r['rtf'], 3)} | {r['target']} |"
        )

    out.append("")
    out.append("### museval SDR / SI-SDR（单目标，mono）")
    out.append("")
    out.append("| 模型 | 轨道 | museval SDR | SI-SDR | 参考信号 |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        for st in (r["sdr"].keys() or r["si_sdr"].keys()):
            ref = r["ref_used"].get(st, st)
            ref_s = ref if ref == st else f"{ref} ← {st}"
            out.append(
                f"| {r['label']} | {st} | {fmt(r['sdr'].get(st))} | "
                f"{fmt(r['si_sdr'].get(st))} | {ref_s} |"
            )

    if args.stems:
        out.append("")
        out.append("### 按 stem 横向对比（museval SDR）")
        out.append("")
        labels = [r["label"] for r in rows]
        out.append("| stem | " + " | ".join(labels) + " |")
        out.append("|---" * (len(labels) + 1) + "|")
        for st in STEM_ORDER:
            d = per_stem.get(st, {})
            if not d:
                continue
            out.append(f"| {STEM_ZH[st]} ({st}) | " + " | ".join(fmt(d.get(l, (None, None))[0]) for l in labels) + " |")

    text = "\n".join(out)
    print(text)
    if args.write:
        p = Path(args.write)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
