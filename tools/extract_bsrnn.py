# -*- coding: utf-8 -*-

"""BSRNN 权重（Zenodo 17516442）解压 + MD5 校验 + 清单。

三个 zip 对应：
  bsrnn-opt.zip      -> weights/BSRNN/bsrnn-opt/       (oBSRNN, 4 个目标各自一个 ckpt)
  bsrnn-large.zip    -> weights/BSRNN/bsrnn-large/
  simo-bsrnn-opt.zip -> weights/BSRNN/simo-bsrnn-opt/  (SIMO 版，单 ckpt：separator.ckpt)
"""
from __future__ import annotations

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()

import hashlib
import sys
import zipfile
from pathlib import Path

W = Path(_paths.WEIGHTS / "BSRNN")

EXPECT = {
    "bsrnn-opt.zip":      dict(size=1827060285, md5="89075aac776f82295a074a56d0428a18", out="bsrnn-opt"),
    "bsrnn-large.zip":    dict(size=1627580581, md5="0d1e298e725bf45577a4f61e0a30c2b1", out="bsrnn-large"),
    "simo-bsrnn-opt.zip": dict(size=1213413202, md5="817e8ce4bcd17c38bd3812167e97db58", out="simo-bsrnn-opt"),
}


def md5_of(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, exp in EXPECT.items():
        if only and only not in name:
            continue
        z = W / name
        print(f"\n===== {name} =====")
        if not z.is_file():
            print("  缺失，跳过")
            continue
        size = z.stat().st_size
        print(f"  大小 {size:,} / {exp['size']:,}  {'OK' if size == exp['size'] else '不匹配'}")
        if size != exp["size"]:
            print("  未下完或损坏，跳过")
            continue
        h = md5_of(z)
        print(f"  md5  {h}  {'OK' if h == exp['md5'] else 'MISMATCH 期望 ' + exp['md5']}")
        if h != exp["md5"]:
            continue

        out = W / exp["out"]
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            print(f"  内含 {len(names)} 个条目")
            for n in names[:40]:
                info = zf.getinfo(n)
                print(f"     {info.file_size:>14,}  {n}")
            if len(names) > 40:
                print(f"     ... 其余 {len(names)-40} 项省略")
            zf.extractall(out)
        print(f"  已解压到 {out}")
        cks = sorted(out.rglob("*.ckpt"))
        print(f"  .ckpt 文件 {len(cks)} 个:")
        for c in cks:
            print(f"     {c.relative_to(out)}  {c.stat().st_size:,} B")


if __name__ == "__main__":
    main()
