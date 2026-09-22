# -*- coding: utf-8 -*-
"""BSRNN (Band-Split RNN) 推理加载器 —— 在 Smart App Control 强制开启的机器上跑通。

被阻塞的依赖链
--------------
官方 `separate.py` 的链路：

    separate.py
      -> hydra (已装, OK)
      -> models.separator -> models.pl_module
             ├─ import lightning.pytorch      ← 与 torch 2.14 私有 API 不兼容
             ├─ import pandas as pd           ← SAC 拦 _philox
             ├─ from helpers.data import rec_audio   ← helpers/data.py import pandas + musdb
             └─ from helpers.eval import compute_sdr ← helpers/eval.py import museval (SAC 拦)
      -> Model.load_from_checkpoint(...)      ← Lightning 类方法

对策（全部在内存里做，不改第三方源码）
--------------------------------------
1. 往 sys.modules 注入 `pandas` / `lightning` / `lightning.pytorch` /
   `helpers.data` / `helpers.eval` 的占位模块。
   - `pandas` 在 pl_module 里只用于 test 阶段的 DataFrame 汇总（第 434 行），推理路径用不到。
   - `lightning.pytorch.LightningModule` 退化为 `torch.nn.Module` + 几个空方法。
2. 绕过 `load_from_checkpoint`，改为手工 `torch.load` + `load_state_dict`。
3. 推理直接用 PLModule 的 `_apply_model_to_track`（纯 torch/torchaudio，无阻塞依赖）。

用法
----
    from _bsrnn_loader import load_bsrnn
    run_fn, load_s, info = load_bsrnn(ckpt_path, conf_name, targets)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths                                       # noqa: E402
_paths.setup_env()

import numpy as np                                           # noqa: E402
import torch                                                 # noqa: E402

BSRNN_REPO = Path(_paths.THIRD_PARTY / "bsrnn")


# ------------------------------------------------------------------ #
# 占位模块
# ------------------------------------------------------------------ #
def _raise_stub(name):
    def _f(*_a, **_k):
        raise NotImplementedError(
            f"{name} 在本机不可用（Smart App Control 拦截 或 版本不兼容），已被占位模块替换。"
            "该功能不在 BSRNN 推理路径上。"
        )
    return _f


def install_stubs() -> list[str]:
    """注入占位模块，返回实际替换了哪些。"""
    replaced = []

    # ---- pandas ----
    if "pandas" not in sys.modules:
        pd = types.ModuleType("pandas")
        pd.__version__ = "stub"
        pd.DataFrame = _raise_stub("pandas.DataFrame")
        pd.read_csv = _raise_stub("pandas.read_csv")
        pd.concat = _raise_stub("pandas.concat")
        sys.modules["pandas"] = pd
        replaced.append("pandas")

    # ---- lightning.pytorch ----
    if "lightning.pytorch" not in sys.modules:
        class _LightningModule(torch.nn.Module):
            """最小替身：只保留 PLModule 构造/推理路径上会用到的方法。"""

            def save_hyperparameters(self, *_a, **_k):
                return None

            def log(self, *_a, **_k):
                return None

            def log_dict(self, *_a, **_k):
                return None

            def configure_optimizers(self, *_a, **_k):
                return None

            def optimizers(self, *_a, **_k):
                return None

        lt = types.ModuleType("lightning")
        lt.__path__ = []
        lt.__version__ = "stub"
        ltp = types.ModuleType("lightning.pytorch")
        ltp.__version__ = "stub(torch.nn.Module)"
        ltp.LightningModule = _LightningModule
        lt.pytorch = ltp
        sys.modules["lightning"] = lt
        sys.modules["lightning.pytorch"] = ltp
        replaced.append("lightning.pytorch")

    # ---- helpers.data / helpers.eval ----
    if str(BSRNN_REPO) not in sys.path:
        sys.path.insert(0, str(BSRNN_REPO))
    import helpers as _helpers_pkg  # 命名空间包（目录下无 __init__.py）

    if "helpers.data" not in sys.modules:
        hd = types.ModuleType("helpers.data")
        hd.rec_audio = _raise_stub("helpers.data.rec_audio")
        hd.__dict__.setdefault("__path__", [])
        sys.modules["helpers.data"] = hd
        _helpers_pkg.data = hd
        replaced.append("helpers.data")

    if "helpers.eval" not in sys.modules:
        he = types.ModuleType("helpers.eval")
        he.compute_sdr = _raise_stub("helpers.eval.compute_sdr")
        he.__dict__.setdefault("__path__", [])
        sys.modules["helpers.eval"] = he
        _helpers_pkg.eval = he
        replaced.append("helpers.eval")

    return replaced


# ------------------------------------------------------------------ #
# 配置
# ------------------------------------------------------------------ #
def _pick_device() -> str:
    """BSRNN 仓库的设备约定（照官方 `separate.py:42`）：
    `cfg.eval.device` 只是一个「请求」，实际能用什么设备由 `torch.cuda.is_available()` 裁决。

    🔴 2026-09-17：本文件原先三处把它**写死成 "cpu"**（无 CUDA 时代留下的），
    导致 BSRNN 三变体在装了 CUDA torch 之后**仍然全在 CPU 上跑**
    （RTF 一动不动，与其它模型 3~21× 的提速形成明显反差）。现已改为动态判定。
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_conf(conf_name: str, device: str | None = None):
    """把 hydra 风格的 conf/ 目录手工合成一个 OmegaConf（hydra compose 太重）。"""
    import yaml
    from omegaconf import OmegaConf

    conf = BSRNN_REPO / "conf"

    def rd(p: Path):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}

    base = rd(conf / "config.yaml")
    base.pop("defaults", None)
    merged = dict(base)
    merged["model"] = rd(conf / "model" / f"{conf_name}.yaml")
    merged["eval"] = rd(conf / "eval" / "default.yaml")
    merged["optim"] = rd(conf / "optim" / "default.yaml")
    merged["scheduler"] = rd(conf / "scheduler" / "step.yaml")
    cfg = OmegaConf.create(merged)
    # 整段一次过（segment_len 与片段等长时 fader 会自动退化为单次前向）
    cfg.eval.device = device or _pick_device()
    return cfg


def _pick_state_dict(ckpt_path: Path):
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("state_dict", "state", "model_state_dict"):
            if k in obj:
                return obj[k], k
    return obj, None


def ckpt_targets(ckpt_path) -> list[str] | None:
    """读 checkpoint 里记录的 targets 顺序。

    ⚠️ 对 SIMO（单模型多 masker）**必须**用这个顺序构造模型：
    state_dict 是按位置对齐的，顺序错位会让 4 个 masker 静默互换，
    表现为「跑通了但指标串味」。
    """
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    hp = obj.get("hyper_parameters") if isinstance(obj, dict) else None
    if not isinstance(hp, dict):
        return None
    t = hp.get("targets")
    if t is None:
        return None
    try:
        seq = list(t)
    except TypeError:
        return None
    seq = [str(x) for x in seq]
    return seq or None


# ------------------------------------------------------------------ #
# 主入口
# ------------------------------------------------------------------ #
def load_bsrnn(ckpt_path, conf_name: str, targets, progress=print):
    """返回 (run_fn, load_seconds, info)；run_fn(mix_stereo:(N,2)) -> {target: (N,2)}"""
    import time

    replaced = install_stubs()
    progress(f"    [BSRNN] stub 替换: {replaced or '无'}")

    t0 = time.time()
    dev = _pick_device()
    cfg = _load_conf(conf_name, dev)

    from models.bsrnn import BSRNN  # noqa: E402  （必须在 stub 之后导入）

    ckpt_path = Path(ckpt_path)
    want = [str(t) for t in targets]

    # 多目标（SIMO）：模型结构与 ckpt 的 masker 顺序强绑定，必须按 ckpt 记录的顺序构造。
    build_targets = want
    ct = ckpt_targets(ckpt_path)
    if len(want) > 1:
        if ct and set(ct) != set(want):
            progress(f"    [BSRNN] ⚠ ckpt targets={ct} 与请求 {want} 不一致，按 ckpt 顺序构造")
        if ct:
            build_targets = ct
    elif ct and len(ct) > 1:
        progress(f"    [BSRNN] ⚠ 单目标请求但 ckpt targets={ct}，按 ckpt 顺序构造再取子集")

    model = BSRNN(cfg.optim, cfg.scheduler, cfg.eval, list(build_targets), **dict(cfg.model))
    sd, wrapper = _pick_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval_device = torch.device(dev)      # `_apply_model_to_track` 会 self.to(eval_device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    progress(f"    [BSRNN] conf={conf_name} ckpt={ckpt_path.name} wrapper={wrapper} "
             f"params={n_params/1e6:.2f}M targets={build_targets}"
             + (f" ckpt_order={ct}" if ct else ""))
    if missing:
        progress(f"    [BSRNN] ⚠ missing keys = {len(missing)} 例: {list(missing)[:3]}")
    if unexpected:
        progress(f"    [BSRNN] ⚠ unexpected keys = {len(unexpected)} 例: {list(unexpected)[:3]}")
    dt = time.time() - t0

    def run(mix):
        x = torch.as_tensor(np.ascontiguousarray(mix.T), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            final, _loss = model._apply_model_to_track(x, None, comp_loss=False)
        final = final.detach().cpu()[0]            # (n_targets, C, T)
        est = {t: final[i].numpy().T for i, t in enumerate(build_targets)}
        return {t: est[t] for t in want if t in est}

    return run, dt, {
        "kind": "deep", "repo": "bsrnn(Paul Magron)", "conf": conf_name,
        "ckpt": ckpt_path.name, "params": n_params, "targets": want,
        "build_targets": list(build_targets), "ckpt_targets": ct,
        "stubbed": replaced, "ckpt_wrapper": wrapper,
        "n_missing_keys": len(missing), "n_unexpected_keys": len(unexpected),
    }


# ------------------------------------------------------------------ #
# 非 SIMO：每个目标一份 ckpt，合并成一个「一次出多轨」的 run_fn
# ------------------------------------------------------------------ #
def load_bsrnn_multi(ckpt_map, conf_name: str, progress=print):
    """ckpt_map: {target: ckpt_path}；返回 (run_fn, load_seconds, info)。

    非 SIMO 的 oBSRNN / bsrnn-large 都是「4 个独立单目标模型」，各自只出一个 stem。
    这里把它们合起来跑，等价于论文里的 4-stem 评测口径，同时逐目标记录参数量与耗时。
    """
    import time

    replaced = install_stubs()
    progress(f"    [BSRNN] stub 替换: {replaced or '无'}")

    from models.bsrnn import BSRNN  # noqa: E402

    t0 = time.time()
    dev = _pick_device()
    models, per_target = {}, {}
    for tgt, ckpt_path in ckpt_map.items():
        ct = ckpt_targets(ckpt_path)
        if ct and ct != [tgt]:
            progress(f"    [BSRNN] ⚠ {Path(ckpt_path).name} 内记录 targets={ct}，与文件名 {tgt} 不符")
        cfg = _load_conf(conf_name, dev)
        model = BSRNN(cfg.optim, cfg.scheduler, cfg.eval, [tgt], **dict(cfg.model))
        sd, wrapper = _pick_state_dict(Path(ckpt_path))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval_device = torch.device(dev)  # 同上：交由模型自行 self.to(eval_device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        models[tgt] = model
        per_target[tgt] = {
            "ckpt": Path(ckpt_path).name, "params": n_params,
            "wrapper": wrapper, "ckpt_targets": ct,
            "n_missing_keys": len(missing), "n_unexpected_keys": len(unexpected),
        }
        if missing or unexpected:
            progress(f"    [BSRNN] ⚠ {tgt}: missing={len(missing)} unexpected={len(unexpected)}")
    dt = time.time() - t0

    tot = sum(v["params"] for v in per_target.values())
    progress(f"    [BSRNN] conf={conf_name} 聚合 {len(models)} 个单目标 ckpt，"
             f"合计 params={tot/1e6:.2f}M "
             f"{ {t: round(v['params']/1e6, 2) for t, v in per_target.items()} }")

    def run(mix):
        x = torch.as_tensor(np.ascontiguousarray(mix.T), dtype=torch.float32).unsqueeze(0)
        out = {}
        for tgt, model in models.items():
            with torch.no_grad():
                final, _loss = model._apply_model_to_track(x, None, comp_loss=False)
            out[tgt] = final.detach().cpu()[0][0].numpy().T
        return out

    return run, dt, {
        "kind": "deep", "repo": "bsrnn(Paul Magron)", "conf": conf_name,
        "params": tot, "targets": list(ckpt_map.keys()),
        "stubbed": replaced, "per_target": per_target, "n_models": len(models),
    }
