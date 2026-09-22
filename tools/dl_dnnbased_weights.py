"""下载 tky823/DNN-based_source_separation 中 MUSDB18 预训练权重（Google Drive）。

目标：把权重放到 `build_from_pretrained(root=...)` 期望的路径，使其后续能直接加载：
  <root>/ConvTasNet/musdb18/sr44100/<config>/model/best.pth
  <root>/MMDenseLSTM/musdb18/sr44100/paper/model/best.pth

用法:
  python tools/dl_dnnbased_weights.py [--root E:/FYP_HKBU/01_models/_weights/DNN-based_source_separation]
"""
import argparse
import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path

import gdown

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT

# (模型名, task 路径片段, google drive file id, 说明)
JOBS = [
    ("MMDenseLSTM", "musdb18/sr44100/paper",       "1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV", "MMDenseLSTM MUSDB18 (paper config)"),
    ("ConvTasNet",  "musdb18/sr44100/4sec_L20",    "1A6dIofHZJQCUkyq-vxZ6KbPmEHLcf4WK", "Conv-TasNet MUSDB18 4sec_L20 (默认配置)"),
    ("ConvTasNet",  "musdb18/sr44100/8sec_L20",    "1C4uv2z0w1s4rudIMaErLyEccNprJQWSZ", "Conv-TasNet MUSDB18 8sec_L20"),
    ("ConvTasNet",  "musdb18/sr44100/8sec_L64",    "1paXNGgH8m0kiJTQnn1WH-jEIurCKXwtw", "Conv-TasNet MUSDB18 8sec_L64"),
]


def md5_of(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024.0
    return f"{n:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_paths.WEIGHTS / "DNN-based_source_separation"))
    ap.add_argument("--tmp", default=str(PROJECT_ROOT / ".tmp_dl"))
    args = ap.parse_args()

    root = Path(args.root)
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"root = {root}")
    print(f"tmp  = {tmp}")
    ok = 0
    bad = 0

    for model, sub, fid, desc in JOBS:
        out_dir = root / model / sub
        # 这批权重 zip 内部结构是 model/<target>/{best,last}.pth（4 个目标各自一个模型）
        done_marker = out_dir / "model" / "vocals" / "best.pth"
        print("\n" + "=" * 78)
        print(f"[{model}] {desc}")
        print(f"  gdrive id = {fid}")
        print(f"  -> {out_dir}")

        if done_marker.exists():
            print(f"  已存在 model/vocals/best.pth ({human(done_marker.stat().st_size)})，跳过")
            ok += 1
            continue

        zip_path = tmp / f"{model}_{sub.replace('/', '_')}.zip"
        # 已下好的 zip 直接复用
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            t0 = time.time()
            try:
                # gdown 6.x 已移除 fuzzy 参数；直接以 id 形式调用
                res = gdown.download(id=fid, output=str(zip_path), quiet=False)
            except Exception as e:  # noqa: BLE001
                print(f"  !! gdown 异常: {type(e).__name__}: {e}")
                bad += 1
                continue
            if res is None:
                print("  !! gdown 返回 None（可能无权限/超限/网络不可达）")
                bad += 1
                continue
            dt = time.time() - t0
            sz = zip_path.stat().st_size
            print(f"  下载完成 {human(sz)} / {dt:.1f}s ({sz / max(dt, 1e-9) / 1024:.0f} KB/s)")
        else:
            print(f"  复用已有 zip {human(zip_path.stat().st_size)}  md5={md5_of(zip_path)}")

        try:
            with zipfile.ZipFile(zip_path) as z:
                names = z.namelist()
                print(f"  zip 内 {len(names)} 项，示例: {names[:6]}")
                out_dir.mkdir(parents=True, exist_ok=True)
                z.extractall(out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  !! 解压失败: {type(e).__name__}: {e}")
            bad += 1
            continue

        if done_marker.exists():
            print(f"  OK  -> {done_marker} ({human(done_marker.stat().st_size)})")
            ok += 1
        else:
            print("  !! 解压后未找到 model/vocals/best.pth，实际内容：")
            for p in sorted(out_dir.rglob("*"))[:20]:
                print(f"       {p.relative_to(out_dir)}")
            bad += 1

    print("\n" + "=" * 78)
    print(f"汇总: ok={ok} bad={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
