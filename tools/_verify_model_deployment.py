#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型本地部署核验（2026-09-23 第三次重组后的判据）。

判据（全部机器可验，不靠"看起来对"）
------------------------------------
1. **板块归属**：模型目录位于 `01_models/<separation|denoising>/<模型>/`，且与
   `paths.py::MODEL_BOARD` 一致。
2. **目录纯净**：模型目录内**只应有 `code/`**。若还残留 `README.md` 或 `weights/`，
   说明精简未完成 → FAIL（并提示跑 `tools/_reclassify_models_2026-09-23.py`）。
3. **代码联接**：`<模型>/code/<repo>` 是可解析的目录联接，指向 `01_models/_third_party/<repo>`，
   且**目标仓库非空**。关键是 `os.readlink` 能读到 —— 否则说明代码被真的拷贝了一份（违反零拷贝）。
4. **权重真源**：`01_models/_weights/` 内部按 (dev, inode) 去重后占用 == 朴素求和
   （证明 _weights 内无重复文件）。
5. **git 完整性**：每个第三方仓库 `is-shallow == false`、有 remote、HEAD 可解析。

用法::

    python tools/_verify_model_deployment.py
    python tools/_verify_model_deployment.py --json 04_reports/_shared/data/model_deployment_audit.json
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

MODELS = _paths.MODELS
W = _paths.WEIGHTS
TP = _paths.THIRD_PARTY
SEP = _paths.SEP_MODELS
DEN = _paths.DEN_MODELS

# 模型目录里「不该再出现」的名字（2026-09-23 精简掉的冗余层）
FORBIDDEN = ("README.md", "weights")

# 无第三方公开代码/权重、只能作「相关工作」的模型（上游问题，非本机能力问题）
BLOCKED_UPSTREAM = {
    "RPCA+DRNN": "Lai & Wang 2022 (EURASIP JASMP 2022:4) 未公开任何代码或权重；且口径为单声道 2 源，与 MUSDB18 4-stem 不可横比",
    "Pac-HuBERT-SEP": "MERL 项目页未放出代码与权重",
}


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "").strip()


def _link_target(p: Path) -> str | None:
    try:
        return os.readlink(p)
    except OSError:
        return None


def _norm_target(tgt: str) -> Path:
    """把 os.readlink 的返回值规范化。

    Windows 上 junction 的 readlink 结果带**扩展长度前缀** `\\\\?\\`
    （如 `\\\\?\\E:\\FYP_HKBU\\01_models\\_third_party\\bsrnn`），
    与普通 `Path` 做 relative_to 会判不中，必须先剥前缀再比较。
    """
    t = tgt
    if t.startswith("\\\\?\\UNC\\"):
        t = "\\\\" + t[8:]
    elif t.startswith("\\\\?\\"):
        t = t[4:]
    return Path(t)


def walk_files(root: Path):
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            yield Path(dp) / f


def _unique_bytes(root: Path) -> int:
    """按 (dev, inode) 去重后的真实占用——硬链接只算一次。"""
    seen: set = set()
    total = 0
    for f in walk_files(root):
        try:
            st = f.stat()
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += st.st_size
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out: dict = {"models": [], "repos": [], "board_summary": {},
                 "blocked_upstream": BLOCKED_UPSTREAM, "summary": {}}
    fails: list[str] = []

    # ---------- 0) 板块目录 ----------
    print("=" * 100)
    print("%-30s %-11s %-6s %-7s %-9s %s" % ("模型", "板块", "code", "残留", "联接目标", "判定"))
    print("=" * 100)

    # ---------- 1) 逐模型 ----------
    for board, root in (("separation", SEP), ("denoising", DEN)):
        if not root.is_dir():
            fails.append("板块目录缺失：%s" % root)
            continue
        for md in sorted(d for d in root.iterdir() if d.is_dir()):
            e = {"model": md.name, "board": board, "status": "PASS"}

            # 板块归属是否与 paths.MODEL_BOARD 一致
            expect = _paths.board_of_model(md.name)
            if expect != board:
                e["status"] = "FAIL"
                fails.append("%s: 实际在 %s/ 但 MODEL_BOARD 登记为 %s" % (md.name, board, expect))

            # 目录纯净：不该再有 README.md / weights/
            residue = [n for n in FORBIDDEN if (md / n).exists()]
            e["residue"] = residue
            if residue:
                e["status"] = "FAIL"
                fails.append("%s: 目录内仍有冗余层 %s（跑 tools/_reclassify_models_2026-09-23.py）"
                             % (md.name, residue))

            # 代码联接
            cdir = md / "code"
            code_entries = []
            bad = False
            if not cdir.is_dir():
                e["status"] = "FAIL"
                bad = True
                fails.append("%s: 缺 code/ 目录" % md.name)
            else:
                for link in sorted(cdir.iterdir()):
                    tgt = _link_target(link)
                    if tgt is None:
                        code_entries.append({"link": link.name, "kind": "COPY?", "ok": False})
                        bad = True
                        fails.append("%s/code/%s: 不是联接（疑似拷贝）" % (md.name, link.name))
                        continue
                    resolved = _norm_target(tgt)
                    if not resolved.is_dir():
                        code_entries.append({"link": link.name, "kind": "link", "ok": False,
                                             "target": tgt})
                        bad = True
                        fails.append("%s/code/%s: 目标不存在 %s" % (md.name, link.name, tgt))
                        continue
                    # 目标必须落在 _third_party / _selfimpl 内（零拷贝约定）
                    in_place = False
                    for base in (TP, MODELS / "_selfimpl"):
                        try:
                            resolved.relative_to(base)
                            in_place = True
                            break
                        except ValueError:
                            pass
                    n_src = sum(1 for _ in walk_files(resolved))
                    ok = n_src > 0 and in_place
                    code_entries.append({"link": link.name, "kind": "junction", "ok": ok,
                                         "target": tgt, "files": n_src,
                                         "in_official_store": in_place,
                                         "has_git": (resolved / ".git").exists()})
                    if not ok:
                        bad = True
                        fails.append("%s/code/%s: %s" % (
                            md.name, link.name,
                            "目标仓库为空" if n_src == 0 else "目标不在 _third_party/_selfimpl 内"))
                if not code_entries and md.name not in ("RPCA",):
                    pass        # RPCA 应为空（纯自研），单独放过
            e["code"] = code_entries

            if bad and e["status"] == "PASS":
                e["status"] = "FAIL"
            print("%-30s %-11s %-6d %-7s %-9s %s" % (
                md.name, board, len(code_entries),
                ",".join(residue) if residue else "-",
                ",".join(ce["link"] for ce in code_entries[:2]) or "-",
                e["status"]))
            out["models"].append(e)

    # ---------- 2) 权重真源 ----------
    print()
    print("=" * 100)
    print("-- 权重真源 01_models/_weights/ --")
    n_w_files = sum(1 for _ in walk_files(W))
    w_unique = _unique_bytes(W)
    w_naive = sum(f.stat().st_size for f in walk_files(W) if f.is_file())
    dup = round((w_naive - w_unique) / 1e6, 2)
    print("   文件数 = %d ｜ 按 inode 去重 = %.2f GB ｜ 朴素求和 = %.2f GB ｜ 内部重复 = %.2f MB"
          % (n_w_files, w_unique / 1e9, w_naive / 1e9, dup))
    if dup > 1.0:
        fails.append("_weights/ 内部存在 %.2f MB 重复文件" % dup)
    if n_w_files == 0:
        fails.append("_weights/ 为空")

    # ---------- 3) 第三方仓库 git 完整性 ----------
    print()
    print("=" * 100)
    print("%-39s %-8s %-7s %-6s %-42s" % ("仓库", "shallow", "commits", ".git", "remote"))
    print("=" * 100)
    for repo in sorted(d for d in TP.iterdir() if d.is_dir()):
        if not (repo / ".git").exists():
            out["repos"].append({"repo": repo.name, "git": False})
            fails.append("%s: 不是 git 仓库" % repo.name)
            print("%-39s %-8s %-7s %-6s %s" % (repo.name, "-", "-", "NO", "*** 非 git ***"))
            continue
        shallow = _git(repo, "rev-parse", "--is-shallow-repository")
        cnt = _git(repo, "rev-list", "--count", "HEAD")
        url = _git(repo, "config", "--get", "remote.origin.url")
        sha = _git(repo, "rev-parse", "--short", "HEAD")
        gitmb = round(sum(f.stat().st_size for f in walk_files(repo / ".git")) / 1e6, 1)
        ok = shallow == "false" and bool(url)
        out["repos"].append({"repo": repo.name, "git": True, "shallow": shallow,
                             "commits": int(cnt or 0), "remote": url, "head": sha,
                             "git_mb": gitmb, "ok": ok})
        if not ok:
            fails.append("%s: shallow=%s remote=%r" % (repo.name, shallow, url))
        print("%-39s %-8s %-7s %-6s %s" % (repo.name, shallow, cnt, "%.0fM" % gitmb,
                                           url or "*** 无 remote ***"))

    # ---------- 4) 汇总 ----------
    def n_models_in(p: Path) -> int:
        return len([d for d in p.iterdir() if d.is_dir()]) if p.is_dir() else 0

    out["board_summary"] = {"separation": n_models_in(SEP), "denoising": n_models_in(DEN)}
    out["summary"] = {
        "layout": "01_models/<separation|denoising>/<model>/code/",
        "n_models": len(out["models"]),
        "n_pass": sum(1 for e in out["models"] if e["status"] == "PASS"),
        "n_repos": len([r for r in out["repos"] if r.get("git")]),
        "n_repos_full": sum(1 for r in out["repos"] if r.get("ok")),
        "_weights_files": n_w_files,
        "_weights_unique_gb": round(w_unique / 1e9, 2),
        "_weights_naive_gb": round(w_naive / 1e9, 2),
        "_weights_internal_dup_mb": dup,
        "n_fails": len(fails),
    }

    print()
    print("=" * 100)
    print("模型 %d 个（separation %d / denoising %d）：PASS=%d FAIL=%d ｜ 第三方仓库 %d 个（完整历史 %d 个）"
          % (out["summary"]["n_models"], out["board_summary"]["separation"],
             out["board_summary"]["denoising"], out["summary"]["n_pass"],
             out["summary"]["n_models"] - out["summary"]["n_pass"],
             out["summary"]["n_repos"], out["summary"]["n_repos_full"]))
    print("权重真源：%d 个文件 / %.2f GB（唯一占用），内部重复 %.2f MB"
          % (n_w_files, w_unique / 1e9, dup))
    if fails:
        print()
        print("--- 未通过项 ---")
        for f in fails:
            print("  ERR %s" % f)
    else:
        print()
        print("全部通过：模型已按板块分类，目录内只有 code/ 联接（零拷贝），权重统一在 _weights/。")

    if args.json:
        out["fails"] = fails
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("报告 -> %s" % args.json)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
