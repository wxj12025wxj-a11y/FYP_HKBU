# -*- coding: utf-8 -*-

"""分段并行 + 断点续传下载器（只用标准库，避免污染 venv）。

为什么需要它
------------
`curl -C -` 在本机遇到两个坑：
  1. 与 `--retry` 组合会在重试时把偏移量重置为 0 并截断文件；
  2. 若服务器忽略 Range 返回 200（整文件），会把**整个文件追加**到已有内容后面，
     导致尺寸越界。
单连接实测只有 ~200 KB/s，1.8 GB 要 2.5 小时。

本脚本做法
----------
把文件切成 NSEG 段，每段独立下载到 `<out>.seg<i>`，全部完成后再拼接 + 校验 MD5。
每段内部自带重试与断点续传（段文件已存在多少字节就从那里继续）。
因此：**并行提速 + 任意时刻中断都不丢进度**。

用法
----
    python tools/par_download.py                 # 下载清单里全部
    python tools/par_download.py --only bsrnn-opt.zip --nseg 8
"""
from __future__ import annotations

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()

import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

W = Path(_paths.WEIGHTS / "BSRNN")
LG = Path(_paths.LOGS)

MANIFEST = [
    ("bsrnn-opt.zip",      "https://zenodo.org/api/records/17516442/files/bsrnn-opt.zip/content",      1827060285, "89075aac776f82295a074a56d0428a18"),
    ("bsrnn-large.zip",    "https://zenodo.org/api/records/17516442/files/bsrnn-large.zip/content",    1627580581, "0d1e298e725bf45577a4f61e0a30c2b1"),
    ("simo-bsrnn-opt.zip", "https://zenodo.org/api/records/17516442/files/simo-bsrnn-opt.zip/content", 1213413202, "817e8ce4bcd17c38bd3812167e97db58"),
]

UA = {"User-Agent": "Mozilla/5.0 (par_download.py)"}
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(LG / "_par_download.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def seg_len(total: int, nseg: int) -> int:
    return (total + nseg - 1) // nseg


def download_seg(url: str, out: Path, start: int, stop: int, tag: str, max_rounds: int = 300) -> None:
    """下载 [start, stop]，写入 out（断点续传，无论中途失败多少次都不丢已得字节）。"""
    want = stop - start                    # 该段期望字节数
    stall = 0
    rounds = 0
    while True:
        rounds += 1
        if rounds > max_rounds:
            log(f"  {tag} 放弃（超过 {max_rounds} 轮）")
            return
        have = out.stat().st_size if out.is_file() else 0
        if have >= want:
            return
        if have > want:                    # 越界：删掉重来
            out.unlink(missing_ok=True)
            have = 0

        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start + have}-{stop - 1}"})
        got = 0
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status not in (200, 206):
                    raise urllib.error.HTTPError(url, r.status, "unexpected", r.headers, None)
                mode = "ab" if have else "wb"
                with open(out, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
        except Exception as exc:  # noqa: BLE001
            with _print_lock:
                print(f"  {tag} 第 {rounds} 轮中断: {type(exc).__name__}: {str(exc)[:70]}", flush=True)

        new = out.stat().st_size if out.is_file() else 0
        if new <= have:
            stall += 1
            if stall >= 30:
                log(f"  {tag} 连续 {stall} 轮无增长，放弃")
                return
            time.sleep(2)
        else:
            stall = 0


def md5_of(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def adopt_prefix(name: str, size: int, nseg: int) -> int:
    """把已有的「顺序前缀」进度迁移成分段文件，避免重下。

    本目录里可能已经有：
      - `<name>`            先前用顺序方式下到的前缀（干净前缀）
      - `<name>.part`       顺序下载的继续部分（与主文件首尾相接）
    把二者拼成一个前缀 prefix，再按段切片写进 `<name>.seg<i>`，随后删除原文件。

    返回迁移的字节数。
    """
    dst = W / name
    part = W / (name + ".part")
    if not dst.is_file() and not part.is_file():
        return 0

    prefix = W / (name + ".prefix")
    with open(prefix, "wb") as fo:
        for p in (dst, part):
            if p.is_file():
                with open(p, "rb") as fi:
                    while True:
                        b = fi.read(1 << 22)
                        if not b:
                            break
                        fo.write(b)

    L = prefix.stat().st_size
    if L > size:                      # 可疑（曾被越界追加），只保留前 size 字节
        with open(prefix, "rb") as f:
            data = f.read(size)
        prefix.write_bytes(data)
        L = size

    sl = seg_len(size, nseg)
    adopted = 0
    with open(prefix, "rb") as f:
        for i in range(nseg):
            s = i * sl
            e = min(s + sl, size)
            if s >= L:
                break
            take = min(e, L) - s
            out = W / f"{name}.seg{i}"
            if out.is_file() and out.stat().st_size >= take:
                continue
            f.seek(s)
            with open(out, "wb") as g:
                g.write(f.read(take))
            adopted += take

    prefix.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    part.unlink(missing_ok=True)
    for junk in W.glob(name + ".seg*.new"):
        junk.unlink(missing_ok=True)
    log(f"  {name} 迁移已有进度 {adopted/1e6:.1f} MB -> 分段文件")
    return adopted


def fetch(name: str, url: str, size: int, md5: str, nseg: int) -> bool:
    dst = W / name
    log(f"===== {name}  size={size:,}  nseg={nseg} =====")

    if dst.is_file() and dst.stat().st_size == size:
        h = md5_of(dst)
        if h == md5:
            log(f"  {name} 已存在且 md5 正确，跳过")
            return True
        log(f"  {name} 尺寸对但 md5 不符，删除重下")
        dst.unlink()

    adopt_prefix(name, size, nseg)

    sl = seg_len(size, nseg)
    segs = []
    for i in range(nseg):
        s = i * sl
        e = min(s + sl, size)
        if s < size:
            segs.append((i, s, e))

    threads = []
    for i, s, e in segs:
        out = W / f"{name}.seg{i}"
        t = threading.Thread(target=download_seg, args=(url, out, s, e, f"{name}#{i}"), daemon=True)
        t.start()
        threads.append(t)

    # 进度上报
    total = len(segs)
    while any(t.is_alive() for t in threads):
        time.sleep(60)
        done = 0
        cur = 0
        for i, s, e in segs:
            p = W / f"{name}.seg{i}"
            have = min(p.stat().st_size if p.is_file() else 0, e - s)
            cur += have
            if have >= (e - s):
                done += 1
        log(f"  {name} {100.0*cur/size:.1f}%  段 {done}/{total}")

    for t in threads:
        t.join()

    # 拼接
    cur = sum(min((W / f"{name}.seg{i}").stat().st_size if (W / f"{name}.seg{i}").is_file() else 0, e - s)
              for i, s, e in segs)
    if cur != size:
        log(f"  {name} 段总长 {cur} != {size}，未完成（保留段文件以便续传）")
        return False

    tmp = W / (name + ".assembling")
    with open(tmp, "wb") as fo:
        for i, s, e in segs:
            with open(W / f"{name}.seg{i}", "rb") as fi:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
    h = md5_of(tmp)
    if h != md5:
        log(f"  {name} 拼接后 md5={h} != 期望 {md5}，保留组装文件待查")
        return False
    tmp.replace(dst)
    for i, s, e in segs:
        (W / f"{name}.seg{i}").unlink(missing_ok=True)
    log(f"  {name} ✅ 完成 md5={h}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--nseg", type=int, default=8)
    args = ap.parse_args()

    W.mkdir(parents=True, exist_ok=True)
    LG.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, url, size, md5 in MANIFEST:
        if args.only and args.only != name:
            continue
        ok = fetch(name, url, size, md5, args.nseg) and ok
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
