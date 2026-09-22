"""权重可加载性验证器 —— **只依赖 torch / zipfile**，不触碰被 SAC 拦截的
scipy.signal / numpy.random / museval / sklearn / pandas。

目的：证明下载到的权重文件是真实、完整、可被 torch 反序列化的
（即"权重真的下载到位"），并顺带给出参数量等基本信息。

用法:
  python tools/verify_weights_loadable.py
输出:
  outputs/comparison/weight_loadability.json
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT
WEIGHTS = _paths.WEIGHTS
OUT_DIR = _paths.comparison_dir()

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import torch  # noqa: E402


def human(n) -> str:
    if n is None:
        return "-"
    f = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if f < 1024:
            return f"{f:.1f}{u}"
        f /= 1024.0
    return f"{f:.1f}PB"


def summarize(obj, depth: int = 0):
    """递归概括 torch.load 的返回值结构。"""
    if isinstance(obj, torch.Tensor):
        return {"type": f"Tensor{tuple(obj.shape)}", "dtype": str(obj.dtype), "numel": obj.numel()}
    if isinstance(obj, dict):
        n_tensors = 0
        n_numel = 0
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                n_tensors += 1
                n_numel += v.numel()
        info = {
            "type": "dict",
            "keys": [str(k) for k in list(obj.keys())[:30]],
            "n_keys": len(obj),
            "n_tensors_direct": n_tensors,
            "numel_direct": n_numel,
        }
        # 若像 checkpoint，深入 state_dict
        for cand in ("state_dict", "model", "model_state_dict", "weights", "state"):
            sd = obj.get(cand)
            if isinstance(sd, dict) and sd and all(isinstance(v, torch.Tensor) for v in list(sd.values())[:5]):
                tot = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
                info["state_dict_key"] = cand
                info["state_dict_n_params"] = int(tot)
                info["state_dict_n_tensors"] = len(sd)
                break
        # 顶层全是 Tensor 的情况
        if n_tensors == len(obj) and n_tensors > 0 and "state_dict_n_params" not in info:
            info["state_dict_key"] = "<top-level>"
            info["state_dict_n_params"] = int(n_numel)
            info["state_dict_n_tensors"] = n_tensors
        other = [k for k in obj.keys() if not isinstance(obj[k], torch.Tensor)]
        info["non_tensor_keys"] = [str(k) for k in other[:20]]
        return info
    if isinstance(obj, (list, tuple)):
        return {"type": type(obj).__name__, "len": len(obj)}
    return {"type": type(obj).__name__}


def try_load(p: Path) -> dict:
    rec = {"file": str(p), "size": p.stat().st_size, "ok": False}
    t0 = time.time()
    loaded = None
    errs = []
    for kwargs in ({"weights_only": False}, {"weights_only": True}, {}):
        try:
            loaded = torch.load(str(p), map_location="cpu", **kwargs)
            rec["load_kwargs"] = kwargs
            rec["ok"] = True
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{kwargs}: {type(e).__name__}: {str(e)[:200]}")
    rec["load_s"] = round(time.time() - t0, 2)
    if rec["ok"]:
        try:
            rec["structure"] = summarize(loaded)
        except Exception as e:  # noqa: BLE001
            rec["structure"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        rec["errors"] = errs
    return rec


def check_zip(p: Path) -> dict:
    rec = {"file": str(p), "size": p.stat().st_size, "ok": False}
    try:
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            names = z.namelist()
            rec["ok"] = bad is None
            rec["bad_member"] = bad
            rec["n_members"] = len(names)
            rec["members"] = names[:40]
            rec["uncompressed_total"] = sum(i.file_size for i in z.infolist())
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = []
    for pat in ("**/*.ckpt", "**/*.th", "**/*.pth", "**/*.pt"):
        targets.extend(sorted(WEIGHTS.glob(pat)))
    zips = sorted(WEIGHTS.glob("**/*.zip"))

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"待验证 torch 权重 {len(targets)} 个, zip {len(zips)} 个\n")

    result = {"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()),
              "torch_files": [], "zips": []}

    for p in targets:
        r = try_load(p)
        result["torch_files"].append(r)
        rel = p.relative_to(WEIGHTS)
        if r["ok"]:
            st = r.get("structure", {})
            np_ = st.get("state_dict_n_params")
            extra = f"params={np_/1e6:.2f}M" if isinstance(np_, int) else ""
            print(f"  OK   {human(r['size']):>9}  {rel}  keys={st.get('n_keys')} {extra}")
        else:
            print(f"  FAIL {human(r['size']):>9}  {rel}")
            for e in r.get("errors", []):
                print(f"         {e}")

    print()
    for p in zips:
        r = check_zip(p)
        result["zips"].append(r)
        rel = p.relative_to(WEIGHTS)
        if r["ok"]:
            print(f"  OK   {human(r['size']):>9}  {rel}  {r['n_members']} members, "
                  f"解压后 {human(r['uncompressed_total'])}")
        else:
            print(f"  FAIL {human(r['size']):>9}  {rel}  {r.get('error') or r.get('bad_member')}")

    (OUT_DIR / "weight_loadability.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'weight_loadability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
