#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型本地部署布局：为每个模型目录补齐 `code/` 目录联接（零拷贝）。

设计原则（2026-09-23 第三次重组后）
--------------------------------
- 代码的**唯一物理落点**是 `01_models/_third_party/<repo>/`；权重的唯一物理落点是
  `01_models/_weights/<...>/`。
- 模型按板块分类：`01_models/<separation|denoising>/<模型>/`，目录内**只有 `code/`**
  （一组指向 `_third_party/<repo>` 的目录联接，零额外占用）。
- `weights/` 联接层与模型卡 `README.md` 已于 2026-09-23 移除：
  脚本一律从 `01_models/_weights/` 直接取权重（见 `paths.py::WEIGHTS`），
  模型说明见 `01_models/<板块>/README.md`。

本脚本 **幂等**：已存在且指向正确的联接会被跳过；指向错误会被重建。

用法::

    python tools/_model_deploy_layout.py            # 建立缺失的联接
    python tools/_model_deploy_layout.py --check    # 只检查不修改
    python tools/_model_deploy_layout.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import paths as _paths

_paths.setup_env()

MODELS = _paths.MODELS          # 01_models
THIRD_PARTY = _paths.THIRD_PARTY
SELFIMPL = MODELS / "_selfimpl"

# 模型目录 -> 它实际使用的第三方仓库（顺序即展示顺序）
MODEL_CODE = {
    # ---- 板 1 · 分离 ----
    "BS-RoFormer-L12": ["Music-Source-Separation-Training", "BS-RoFormer"],
    "BS-RoFormer-L6": ["Music-Source-Separation-Training", "BS-RoFormer"],
    "BSRNN-opt": ["bsrnn"],
    "BSRNN-large": ["bsrnn"],
    "BSRNN-SIMO": ["bsrnn"],
    "Demucs": ["demucs"],
    "MDX-Net": [
        "mdx-net",
        "mdx-net-submission",
        "mdx-net-submission-leaderboard_A",
        "mdx-net-submission-leaderboard_B",
    ],
    "Open-Unmix": ["open-unmix-pytorch"],
    "Conv-TasNet": ["Conv-TasNet", "DNN-based_source_separation"],
    "MMDenseLSTM": ["DNN-based_source_separation"],
    "DPRNN": ["Dual-Path-RNN-Pytorch"],
    "RPCA": [],                       # 自研：tools/compare_separation_methods.py::_rpca_ialm
    "Oracle-IRM": ["sigsep-mus-oracle"],   # + 自研掩码
    # ---- 板 2 · 降噪 / 去混响 / 去回声 ----
    "denoiser-dns48": ["denoiser"],
    "denoiser-dns64": ["denoiser"],
    "denoiser-master64": ["denoiser"],
    "Mel-RoFormer-Denoise": ["Music-Source-Separation-Training"],
    "Mel-RoFormer-Dereverb": ["Music-Source-Separation-Training"],
    "Mel-RoFormer-Dereverb-Echo": ["Music-Source-Separation-Training"],
}

# 无第三方代码、纯自研实现的模型 -> 联接到 _selfimpl 索引
SELFIMPL_MODELS = {"RPCA", "Oracle-IRM"}


def _mklink_j(link: Path, target: Path) -> tuple[bool, str]:
    """建目录联接（junction）。不用 mklink /D（符号链接需管理员/开发者模式）。"""
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True, encoding="mbcs", errors="replace",
    )
    return r.returncode == 0, (r.stdout or r.stderr or "").strip()


def _is_junction_to(p: Path, target: Path) -> bool:
    if not p.exists():
        return False
    try:
        if os.readlink(p) is not None:
            return os.path.samefile(p, target)
    except OSError:
        pass
    return os.path.samefile(p, target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查，不创建")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    args = ap.parse_args()

    report: dict = {
        "models_dir": str(MODELS),
        "third_party": str(THIRD_PARTY),
        "n_models": 0,
        "created": [], "kept": [], "rebuilt": [], "failed": [],
        "models": [],
    }

    if not THIRD_PARTY.is_dir():
        print("ERR  找不到 _third_party：%s" % THIRD_PARTY)
        return 2

    for model, repos in MODEL_CODE.items():
        board = _paths.board_of_model(model)
        mdir = _paths.model_dir(model)
        entry = {"model": model, "board": board, "dir": str(mdir.relative_to(MODELS)),
                 "dir_exists": mdir.is_dir(), "code": [], "status": "PASS"}
        if not mdir.is_dir():
            entry["status"] = "FAIL"
            entry["reason"] = "模型目录不存在（先跑 tools/_reclassify_models_2026-09-23.py）"
            print("ERR  %-32s 模型目录不存在：%s" % (model, mdir))
            report["models"].append(entry)
            report["failed"].append(model)
            continue

        cdir = mdir / "code"
        if not cdir.exists() and not args.check:
            cdir.mkdir(parents=True, exist_ok=True)

        links = []
        if repos:
            links += [(r, THIRD_PARTY / r) for r in repos]
        if model in SELFIMPL_MODELS:
            links.append(("_selfimpl", SELFIMPL))

        for name, target in links:
            link = cdir / name
            if not target.exists():
                entry["code"].append({"repo": name, "status": "MISSING_TARGET",
                                      "target": str(target)})
                entry["status"] = "FAIL"
                print("ERR  %-32s code/%-38s 目标仓库不存在" % (model, name))
                continue

            if link.exists():
                if _is_junction_to(link, target):
                    entry["code"].append({"repo": name, "status": "kept",
                                          "target": str(target)})
                    report["kept"].append("%s/code/%s" % (model, name))
                    print("OK   %-32s code/%-38s [已存在]" % (model, name))
                    continue
                # 指向错误 -> 重建
                if not args.check:
                    try:
                        link.unlink()
                    except OSError:
                        pass
                    ok, msg = _mklink_j(link, target)
                    st = "rebuilt" if ok else "FAIL"
                    report["rebuilt" if ok else "failed"].append("%s/code/%s" % (model, name))
                    if not ok:
                        entry["status"] = "FAIL"
                    entry["code"].append({"repo": name, "status": st, "target": str(target),
                                          "msg": msg})
                    print("%-4s %-32s code/%-38s %s" % ("OK" if ok else "ERR", model, name,
                                                        "[重建]" if ok else msg))
                continue

            if args.check:
                entry["code"].append({"repo": name, "status": "would_create",
                                      "target": str(target)})
                entry["status"] = "PENDING"
                print("--- %-32s code/%-38s 缺失（--check 不创建）" % (model, name))
                continue

            ok, msg = _mklink_j(link, target)
            if ok:
                report["created"].append("%s/code/%s" % (model, name))
                entry["code"].append({"repo": name, "status": "created", "target": str(target)})
                print("OK   %-32s code/%-38s [新建]" % (model, name))
            else:
                report["failed"].append("%s/code/%s" % (model, name))
                entry["status"] = "FAIL"
                entry["code"].append({"repo": name, "status": "FAIL", "target": str(target),
                                      "msg": msg})
                print("ERR  %-32s code/%-38s %s" % (model, name, msg))
        report["models"].append(entry)

    report["n_models"] = len(MODEL_CODE)
    n_ok = sum(1 for m in report["models"] if m["status"] == "PASS")
    n_sep = sum(1 for m in report["models"] if m["board"] == "separation")
    n_den = sum(1 for m in report["models"] if m["board"] == "denoising")
    print()
    print("总计 %d 个模型（separation %d / denoising %d）：PASS=%d  FAIL=%d   "
          "新建联接=%d  重建=%d  已存在=%d  失败=%d"
          % (report["n_models"], n_sep, n_den, n_ok, report["n_models"] - n_ok,
             len(report["created"]), len(report["rebuilt"]), len(report["kept"]),
             len(report["failed"])))

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("报告 -> %s" % args.json)
    return 0 if n_ok == report["n_models"] else 1


if __name__ == "__main__":
    sys.exit(main())
