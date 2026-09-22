# -*- coding: utf-8 -*-
"""
================================================================================
 通用模型基准骨架:  open-source model zoo 的统一测试入口
--------------------------------------------------------------------------------
 目的 : 用同一套口径测"任意一个分离模型能否在本机跑起来并出音频"。
        每个模型都必须: 加载权重 -> 片段推理 -> 存 stem wav -> 计时 -> 算 SDR。
 统一口径:
   - 数据: MUSDB18-HQ datasets/<subset>/<song>/{mixture,vocals,drums,bass,other}.wav
   - 取样: 默认前 30 秒, 44.1 kHz 立体声
   - 指标: museval BSSEval v4 (1s 窗, 中位数) + SI-SDR, 均对 mono 单目标
   - 产物: outputs/<模型目录>/<歌曲>/<目标>.wav
   - 结果: outputs/comparison/model_runs.json  (按模型名合并更新)
   - 日志: logs/<模型>_<日期>.log
 用法:
   python tools/benchmark_model_universal.py --list
   python tools/benchmark_model_universal.py --model oracle
   python tools/benchmark_model_universal.py --model rpca,oracle --duration 30
   python tools/benchmark_model_universal.py --model all
================================================================================
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT

# 🔴 推理期必须摘掉 `expandable_segments`（2026-09-19 实测）
# `paths.setup_env()` 会设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True —— 那是为
# DPRNN **训练** 的显存碎片治理准备的（见 tools/train_dprnn_musdb.py）。
# 但它会让 **torch.istft** 在长序列上崩：
#     RuntimeError: CUDA driver error: device not ready
#     RuntimeError: !handles_.at(i) INTERNAL ASSERT FAILED ...
# 触发条件可复现：openunmix 整曲 + wiener_win_len=300（>60 s 必崩，50 s 正常）。
# 这是 torch 2.14.0+cu126 + 驱动 616.56 的组合 bug，不是我们代码的问题。
#
# ⚠️ 坑：`paths.setup_env()` 用的是 `setdefault`，而 `compare_separation_methods`
# 在 import 时**又会调用一次 setup_env()**。若我们 `pop` 掉这个键，
# 第二次 setdefault 会把它**重新设回来** → guard 静默失效。
# 所以这里填一个**合法的无害值** `max_split_size_mb`（默认行为），
# 让 setdefault 认为键已存在而不再覆盖。
# （注意：不能填空串或 "None" —— 会让 c10_cuda.dll 初始化失败 WinError 1114。）
_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" in _alloc:
    _rest = [p for p in _alloc.split(",") if "expandable_segments" not in p]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_rest) if _rest else "max_split_size_mb:512"


sys.path.insert(0, str(_paths.TOOLS))
sys.path.insert(0, str(_paths.DEMUCS_SRC))

import numpy as np
import soundfile as sf

from compare_separation_methods import (  # noqa: E402
    SR, UMX_ORDER, evaluate, si_sdr,
)

DATA_ROOT = _paths.MUSDB18_ROOT
OUTPUTS_ROOT = _paths.OUTPUTS
COMPARISON_DIR = _paths.comparison_dir()   # → 04_reports/separation/data/comparison
RUNS_JSON = COMPARISON_DIR / "model_runs.json"
LOGS_DIR = _paths.LOGS

# 🔴 结果隔离（2026-09-19）：`--subset test` 时把结果与音频都导流到独立位置。
# 原因：本文件的结果是**按 model→song 覆盖**写 RUNS_JSON，而 RUNS_JSON 正是
# median_over_clips.py 生成论文主表（model_runs_median.json / MEDIAN_TABLE.md）的数据源。
# 该脚本按**歌名**挑片段 —— 若 test 集的 50 首整曲混进同一个 json，
# 主表的中位数会被静默污染（train 的 3 个均衡片段 + test 的 50 首整曲混在一起聚合）。
# 因此 test 集一律走 model_runs_test.json + 03_outputs/<模型>/_musdb18_test/<歌>/。
RUNS_JSON_BY_SUBSET = {
    "test": COMPARISON_DIR / "model_runs_test.json",
    # 可运行性探测：与正式结果完全隔离，避免探针覆盖真实评测记录
    "probe": COMPARISON_DIR / "model_runs_probe.json",
    # 2026-09-21 用户要求的「新文件夹」批次：产物集中到 03_outputs/_test_run/<模型>/<歌>/
    "run": COMPARISON_DIR / "model_runs_run.json",
}
OUT_SUBDIR_BY_SUBSET = {
    "test": "_musdb18_test",
    "probe": "_probe_tmp",
}
# 走**独立顶层新目录**的 tag（而不是在既有模型目录下开子目录）
NEW_TREE_TAGS = {"run"}
NEW_TREE_NAME = "_test_run"


def _out_dir_for(name: str, tag: str, song: str) -> Path:
    """决定音频落盘目录。

    - tag 在 NEW_TREE_TAGS 里 → `03_outputs/_test_run/<模型>/<歌曲>/`（全新顶层目录）
    - 否则 → `03_outputs/<模型>/<子目录>/<歌曲>/`（沿用既有布局）
    """
    if tag in NEW_TREE_TAGS:
        return OUTPUTS_ROOT / NEW_TREE_NAME / MODEL_DIR.get(name, name) / song
    sub = OUT_SUBDIR_BY_SUBSET.get(tag)
    return OUTPUTS_ROOT / MODEL_DIR.get(name, name) / (sub or "") / song


def _runs_json_for(subset: str) -> Path:
    return RUNS_JSON_BY_SUBSET.get(subset, RUNS_JSON)


DEFAULT_SONG = "A Classic Education - NightOwl"
DEFAULT_SUBSET = "train"
DEFAULT_DURATION = 30

# ---- 整曲分块推理（2026-09-19 实测标定，8 GiB RTX 4070 Laptop）-------------
# 超过 CHUNK_THRESHOLD_S 的输入自动切块。60 s 是实测的最优档：
#   chunk 30 s → RTF 0.0211  peak 1.01 GB
#   chunk 60 s → RTF 0.0204  peak 1.87 GB   ← 选它（RTF 最低且接缝最少）
#   chunk 150s → RTF 0.0212  peak 4.44 GB
# 整曲一次性前向：200 s → RTF 0.302 peak **21.4 GB**（sysmem fallback，不报错）
CHUNK_THRESHOLD_S = 90.0
CHUNK_S = 60.0
CHUNK_OVERLAP_S = 2.0

# 模型名 -> 输出目录名
MODEL_DIR = {
    "oracle": "Oracle-IRM",
    "rpca": "RPCA",
    "umx": "Open-Unmix",
    "mdx": "MDX-Net",
    "mmdenselstm": "MMDenseLSTM",
    "bsroformer_l12": "BS-RoFormer-L12",
    "bsroformer_l6": "BS-RoFormer-L6",
    "bsrnn": "BSRNN-opt",
    "bsrnn_all": "BSRNN-opt",
    "bsrnn_large": "BSRNN-large",
    "bsrnn_large_all": "BSRNN-large",
    "bsrnn_simo": "BSRNN-SIMO",
    "convtasnet": "Conv-TasNet",
    "dprnn": "DPRNN",
    "rpca_drnn": "RPCA-DRNN",
    # Demucs 的产物目录是 03_outputs/Demucs（Windows 大小写不敏感，
    # 不写这一条也能落盘，但显式登记才能让 --list / 报告口径一致）。
    "demucs": "Demucs",
}


def banner(t: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def _to_stereo(a) -> np.ndarray:
    """规范成 (N,2) float32"""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return np.stack([a, a], axis=-1)
    if a.ndim == 2:
        if a.shape[1] == 2:
            return a
        if a.shape[0] == 2:
            return a.T
        m = a.mean(1)
        return np.stack([m, m], axis=-1)
    a = a.reshape(2, -1)
    return a.T


def _mono(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a if a.ndim == 1 else a.mean(axis=-1)


# ------------------------------------------------------------------ #
# 数据
# ------------------------------------------------------------------ #
def load_clip(song: str, subset: str, duration: float | None, offset: float = 0.0):
    """返回 (mix_stereo (N,2), gt: {target: mono}, sr)

    offset 很重要：MUSDB18 很多歌的**开头**是低频主导的引子（实测前 10 s 里
    bass/other 占 79~85% 能量），会让所有模型的 bass SDR 一起虚高到 20 dB 以上。
    取曲子中段能得到各 stem 能量更均衡的片段，结论才可外推。
    """
    d = DATA_ROOT / subset / song
    n = None if duration is None else int(SR * duration)
    s0 = int(SR * offset)

    def rd(name):
        y, _ = sf.read(str(d / f"{name}.wav"), dtype="float32", always_2d=True)
        seg = y[s0:] if n is None else y[s0:s0 + n]
        return seg

    mix = rd("mixture")
    gt = {t: rd(t).mean(axis=1) for t in UMX_ORDER if (d / f"{t}.wav").is_file()}
    return mix.astype(np.float32), gt, SR


# ------------------------------------------------------------------ #
# 各模型适配器: 每个 loader 返回 (run_fn, load_seconds, info)
#   run_fn(mix_stereo:(N,2)) -> dict[target] -> ndarray (N,2) 或 (N,)
# ------------------------------------------------------------------ #
def load_oracle():
    """IRM / IBM oracle —— 解析计算, 无需权重。

    真正的计算在 run_one() 的 needs_gt 分支里调用 _oracle_irm(mix, gt)，
    因为掩码必须用到 GT 频谱；此处仅作为注册表占位的轻量自检。"""
    t0 = time.time()
    stft, istft = _stft_istft()
    x = np.zeros(SR, np.float32)
    y = istft(stft(x), length=len(x))
    dt = time.time() - t0
    assert y.shape[0] == len(x), "STFT/ISTFT 往返长度不一致"
    return (lambda mix: {}), dt, {"kind": "analytic", "stft": "torch.stft"}


def _stft_istft():
    """返回 (stft, istft) 两个可调用对象。

    优先 torch.stft（不依赖 llvmlite/scipy.signal，规避本机 Smart App Control
    对 librosa.stft 与 scipy.signal 的加载拦截）；失败再回退 librosa。
    """
    import torch

    n_fft, hop = 2048, 512
    win = torch.hann_window(n_fft)

    def stft(x):
        t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        return torch.stft(t, n_fft=n_fft, hop_length=hop, window=win,
                          return_complex=True).numpy()

    def istft(S, length):
        t = torch.as_tensor(np.ascontiguousarray(S), dtype=torch.complex64)
        return torch.istft(t, n_fft=n_fft, hop_length=hop, window=win,
                           length=int(length), return_complex=False).numpy()

    return stft, istft


def _oracle_irm(mix, gt):
    """Ideal Ratio Mask, 用 GT 各源幅度谱解析计算。返回 {target: (N,2)}"""
    stft, istft = _stft_istft()
    mix_m = _mono(mix)
    X = stft(mix_m)
    P = {t: np.abs(stft(gt[t])) ** 2 for t in UMX_ORDER if t in gt}
    denom = sum(P.values()) + 1e-10
    out = {}
    for t in UMX_ORDER:
        if t not in P:
            continue
        mask = np.clip(P[t] / denom, 0, 1)
        y = istft(mask * X, length=len(mix_m))
        out[t] = np.stack([y, y], axis=-1).astype(np.float32)
    return out


def _oracle_ibm(mix, gt):
    """Ideal Binary Mask (同 oracle, 用于对照)。"""
    stft, istft = _stft_istft()
    mix_m = _mono(mix)
    X = stft(mix_m)
    P = {t: np.abs(stft(gt[t])) ** 2 for t in UMX_ORDER if t in gt}
    denom = sum(P.values()) + 1e-10
    out = {}
    for t in UMX_ORDER:
        if t not in P:
            continue
        mask = (P[t] / denom >= 0.5).astype(np.float32)
        y = istft(mask * X, length=len(mix_m))
        out[t] = np.stack([y, y], axis=-1).astype(np.float32)
    return out


def load_rpca():
    """RPCA (Inexact-ALM) —— 复用项目已有实现。单输出(稀疏=人声)。"""
    from compare_separation_methods import sep_rpca
    t0 = time.time()
    _ = sep_rpca(np.zeros(SR, np.float32), SR)  # 预热/自检
    dt = time.time() - t0

    def run(mix):
        y = sep_rpca(_mono(mix), SR)
        return {"vocals": np.stack([y, y], axis=-1).astype(np.float32)}
    return run, dt, {"kind": "classical", "targets": ["vocals"]}


def load_umx():
    """Open-Unmix umxhq 4 目标 —— 用作骨架自检基准(已知 6.978 dB)。"""
    import torch
    import openunmix
    from openunmix import predict as umx_predict
    t0 = time.time()
    sep = openunmix.umxhq(targets=UMX_ORDER, device="cpu")
    if hasattr(sep, "eval"):
        sep.eval()
    dt = time.time() - t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        try:
            sep.to("cuda")
        except Exception:
            dev = "cpu"

    def run(mix):
        audio = torch.as_tensor(np.ascontiguousarray(mix.T), dtype=torch.float32)
        est = umx_predict.separate(audio=audio, rate=SR, targets=UMX_ORDER,
                                   separator=sep, device=dev, filterbank="torch")
        out = {}
        if isinstance(est, dict):
            for k, v in est.items():
                v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                out[k] = _to_stereo(v)
        else:
            a = est.detach().cpu().numpy() if hasattr(est, "detach") else np.asarray(est)
            for i, t in enumerate(UMX_ORDER):
                out[t] = _to_stereo(a[i])
        return out
    return run, dt, {"kind": "deep", "device": dev}


def load_mdx():
    """MDX-Net —— 走 demucs 内置 mdx_extra 袋子(4 模型集成)。

    权重固定在 weights/MDX-Net/mdx_extra/ 本地目录（文件名含 sha256 前缀，
    demucs 的 LocalRepo 会做完整性校验），因此不再依赖网络下载。

    兼容性: demucs 4.1.0a2 的 demucs/states.py 里是 `torch.load(path, 'cpu')`，
    而 torch>=2.6 把 weights_only 默认改成了 True，会拒绝反序列化
    `demucs.hdemucs.HDemucs`。这里在**调用点局部**临时把默认值改回 False
    （不改动第三方源码）；由于权重文件名自带 sha256 前缀且 demucs 会校验，
    来源与完整性是可信的。
    """
    import torch
    from demucs.pretrained import get_model
    t0 = time.time()
    local_repo = _paths.WEIGHTS / "MDX-Net" / "mdx_extra"

    _orig_load = torch.load

    def _load_weights_only_false(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)

    torch.load = _load_weights_only_false
    try:
        if (local_repo / "mdx_extra.yaml").is_file():
            model = get_model("mdx_extra", repo=local_repo)
            src = f"local:{local_repo}"
        else:  # 回退到远程（会自动下载并缓存到 TORCH_HOME）
            model = get_model("mdx_extra")
            src = "remote"
    finally:
        torch.load = _orig_load

    model.eval()
    dt = time.time() - t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def run(mix):
        from demucs.apply import apply_model
        wav = torch.as_tensor(np.ascontiguousarray(mix.T), dtype=torch.float32)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            out = apply_model(model, wav[None].to(dev), device=dev,
                              progress=False)[0].cpu()
        out = out * ref.std() + ref.mean()
        return {s: _to_stereo(out[i].numpy()) for i, s in enumerate(model.sources)}
    return run, dt, {"kind": "deep", "device": dev, "bag": "mdx_extra", "weights_src": src}


def load_demucs():
    """Demucs (htdemucs) —— 波形域混合架构，**单模型直接出 4 stem**。

    与 MDX-Net 同源（都走 demucs 运行时），权重来自 demucs 官方 release，
    缓存在 `tools/_scratch/.cache/torch/hub/checkpoints`（paths.setup_env() 指定 TORCH_HOME）。

    ⚠️ Demucs 在 CPU 上**非确定性**：同输入两次实测 8.83 / 8.73 dB（±0.1~0.3 dB）。
       报单曲数字必须标注波动，跨模型比较只能用多片段中位数。
    ⚠️ stem 顺序**从 model.sources 取**，不硬编码 —— 多输出模型的输出通道
       与 stem 名是按位置对齐的，顺序写错会「跑通但指标串味」。
    """
    import torch
    from demucs.pretrained import get_model
    t0 = time.time()
    model = get_model("htdemucs")
    model.eval()
    dt = time.time() - t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    def run(mix):
        from demucs.apply import apply_model
        wav = torch.as_tensor(np.ascontiguousarray(mix.T), dtype=torch.float32)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            out = apply_model(model, wav[None].to(dev), device=dev, progress=False)[0].cpu()
        out = out * ref.std() + ref.mean()
        return {s: _to_stereo(out[i].numpy()) for i, s in enumerate(model.sources)}
    return run, dt, {"kind": "deep", "device": dev, "model": "htdemucs",
                     "params_m": round(n_params / 1e6, 2), "sources": list(model.sources)}


def _tky823_paths():
    """把 tky823/DNN-based_source_separation 加入 import 路径并返回其根目录。"""
    root = _paths.THIRD_PARTY / "DNN-based_source_separation"
    for p in (root / "src", root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return root


def _load_torch_compat():
    """返回一个上下文管理器：临时把 torch.load 的 weights_only 默认改回 False。

    torch>=2.6 默认 weights_only=True，会让 tky823 仓库（`torch.load(path, map_location=...)`）
    读取自带 config 的 .pth 失败。这些权重是我们刚下载并校验过的本地文件，可信。
    """
    import contextlib
    import torch

    @contextlib.contextmanager
    def _cm():
        orig = torch.load

        def _load(*a, **kw):
            kw.setdefault("weights_only", False)
            return orig(*a, **kw)

        torch.load = _load
        try:
            yield
        finally:
            torch.load = orig

    return _cm()


def load_convtasnet():
    """Conv-TasNet (tky823 仓库的 MUSDB18 预训练版)。

    调用约定见 egs/musdb18/conv-tasnet/src/adhoc_driver.py::estimate_all：
      输入 (batch, 1, n_mics, T) -> 输出 (batch, n_sources, n_mics, T)
      （注意 ConvTasNet.extract_latent 里 `assert C_in == 1` 是硬编码的 4 维分支，
       所以必须显式带上那个 "1" 通道，不能直接喂 (B, n_mics, T)。）
      推理前按时间轴做 mean/std 标准化，推理后再反标准化。
    """
    import torch
    _tky823_paths()
    from models.conv_tasnet import ConvTasNet

    t0 = time.time()
    with _load_torch_compat():
        model = ConvTasNet.build_from_pretrained(
            root=str(_paths.WEIGHTS / "DNN-based_source_separation"),
            task="musdb18", sample_rate=SR, config="4sec_L20",
            load_state_dict=True,
        )
    model.eval()
    dt = time.time() - t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    srcs = list(getattr(model, "sources", UMX_ORDER))
    n_params = sum(p.numel() for p in model.parameters())

    def run(mix):
        # (n_mics, T) -> (1, 1, n_mics, T)：第二个 1 是 ConvTasNet 期望的 C_in
        x = torch.as_tensor(np.ascontiguousarray(mix.T), dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0)
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        with torch.no_grad():
            est = model(((x - mean) / (std + 1e-8)).to(dev))   # (1, S, n_mics, T)
        est = (est * std.to(dev) + mean.to(dev)).cpu()[0]      # (S, n_mics, T)
        return {t: _to_stereo(est[i].numpy()) for i, t in enumerate(srcs)}
    return run, dt, {"kind": "deep", "device": dev, "repo": "tky823",
                     "config": "4sec_L20", "params": n_params, "sources": srcs}


def load_mmdenselstm():
    """MMDenseLSTM (tky823 仓库的 MUSDB18 paper 配置)。

    要点（踩过的坑）：
      * 必须用 `ParallelMMDenseLSTM`（无 target 参数，一次建 4 个目标子模型），
        而不是 `MMDenseLSTM`（那是单目标版，forward 签名是 (input)）。
      * 基模型 forward 期望 **幅度谱** (B, 1, C, n_bins, n_frames)；
        因此需用 `ParallelMMDenseLSTM.TimeDomainWrapper` 包一层，它接受波形
        (B, 1, C, T)（那个 "1" 是必须的哑维度），内部完成 STFT + 多通道维纳滤波，
        输出 (B, n_sources, C, T)。
    """
    import torch
    _tky823_paths()
    from models.mm_dense_lstm import ParallelMMDenseLSTM

    t0 = time.time()
    with _load_torch_compat():
        base = ParallelMMDenseLSTM.build_from_pretrained(
            root=str(_paths.WEIGHTS / "DNN-based_source_separation"),
            task="musdb18", sample_rate=SR, load_state_dict=True,
        )
        model = ParallelMMDenseLSTM.TimeDomainWrapper(
            base, base.n_fft, hop_length=base.hop_length, window_fn=base.window_fn,
        )
    model.eval()
    dt = time.time() - t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    srcs = list(base.sources)
    n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"    [MMDenseLSTM] sources={srcs} params={n_params/1e6:.2f}M "
          f"n_fft={base.n_fft} hop={base.hop_length}")

    def run(mix):
        # (n_mics, T) -> (1, 1, n_mics, T)
        x = torch.as_tensor(np.ascontiguousarray(mix.T), dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            # 🔴 CUDA 下的 device mismatch（2026-09-17 实测）
            # tky823 参考实现的 `update_em()`（src/algorithm/frequency_mask.py:295/302/310/316）
            # 用**裸 `torch.eye(n_channels)`** 造单位阵再与 CUDA 张量相加 —— 单位阵默认落在 CPU，
            # CPU 上跑不暴露，一上 GPU 就 `found at least two devices, cuda:0 and cpu`。
            # 不改第三方代码：用「默认设备」上下文把函数内部裸创建的张量也钉到同一设备。
            if dev == "cpu":
                est = model(x)              # (1, n_sources, n_mics, T)
            else:
                with torch.device(dev):
                    est = model(x.to(dev))  # (1, n_sources, n_mics, T)
        est = est.detach().cpu()[0]         # (n_sources, n_mics, T)
        return {t: _to_stereo(est[i].numpy()) for i, t in enumerate(srcs)}
    return run, dt, {"kind": "deep", "device": dev, "repo": "tky823",
                     "config": "paper", "params": n_params, "sources": srcs,
                     "n_fft": int(base.n_fft), "hop_length": int(base.hop_length)}


def _ensure_librosa():
    """确保 `from librosa import filters` 可用。

    本机 Smart App Control 拦截了 llvmlite.dll，导致 librosa.filters -> numba
    在 import 阶段就 OSError。而 ZFTurbo 的 models/bs_roformer/__init__.py 会连带
    导入 MelBandRoformer（模块级 `from librosa import filters`），于是连只想要
    BSRoformer 都会被拖死。

    对策：先尝试真 librosa；失败则注入一个只有 `.filters` 属性的占位模块。
    我们只用 BSRoformer（纯 torch，内部不调用 librosa），占位不会被真正使用。
    返回 True 表示用了占位。
    """
    try:
        import librosa.filters  # noqa: F401
        return False
    except Exception:
        pass
    import types
    if "librosa" not in sys.modules:
        lib = types.ModuleType("librosa")
        lib.__path__ = []
        sys.modules["librosa"] = lib
    else:
        lib = sys.modules["librosa"]
    if "librosa.filters" not in sys.modules:
        filt = types.ModuleType("librosa.filters")
        sys.modules["librosa.filters"] = filt
        lib.filters = filt
    return True


def _load_bsroformer(which: str):
    """BS-RoFormer（lucidrains 架构 + ZFTurbo 官方推理链路）。

    which: "L12" -> ep_317 (dim=512 depth=12, target=vocals)
           "L6"  -> ep_937 (dim=384 depth=12, target=other)

    注意：两个官方权重都是**单目标**模型（各自 yaml 的 num_stems=1 +
    training.target_instrument 指定唯一目标），不是 4-stem 模型。

    直接复用 ZFTurbo/Music-Source-Separation-Training 的官方函数，
    避免自己重写分块 STFT / 归一化 / 重叠相加导致数值口径不一致：
      get_model_from_config -> torch.load(ckpt) -> model.load_state_dict
      -> bigshifts_wrapper
    """
    import torch

    stub_used = _ensure_librosa()
    mss = _paths.THIRD_PARTY / "Music-Source-Separation-Training"
    if str(mss) not in sys.path:
        sys.path.insert(0, str(mss))
    from utils.audio_utils import denormalize_audio, normalize_audio
    from utils.model_utils import bigshifts_wrapper, prefer_target_instrument
    from utils.settings import get_model_from_config

    stem = {
        "L12": "model_bs_roformer_ep_317_sdr_12.9755",
        "L6": "model_bs_roformer_ep_937_sdr_10.5309",
    }[which]
    cfg_path = _paths.WEIGHTS / "BS-RoFormer" / "configs" / f"{stem}.yaml"
    ckpt = _paths.WEIGHTS / "BS-RoFormer" / f"{stem}.ckpt"

    t0 = time.time()
    model, config = get_model_from_config("bs_roformer", str(cfg_path))
    # 官方 load_start_checkpoint(type_="inference") 的 else 分支要求传入已加载的
    # state dict；这里自己 load 并剥掉 wrapper 键更直接。
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state", "state_dict", "model_state_dict"):
            if k in sd:
                sd = sd[k]
                break
    model.load_state_dict(sd)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    use_norm = bool(config.get("inference", {}).get("normalize", False))
    tgts = list(prefer_target_instrument(config))
    print(f"    [BS-RoFormer {which}] params={n_params/1e6:.2f}M ckpt={ckpt.name} "
          f"targets={tgts} dim={config.model.get('dim')} depth={config.model.get('depth')} "
          f"normalize={use_norm} librosa_stub={stub_used}")

    # ep_937 的模型自报 target 是 "other"，但实测输出为 vocals+other（见 REGISTRY 注释）。
    # 输出文件名改成能自解释的名字，避免 "other.wav" 里其实装着 vocals+other。
    rename = {"other": "vocals+other"} if which == "L6" else {}

    def run(mix):
        m = np.ascontiguousarray(mix.T, dtype=np.float32)   # (channels, T)
        norm_params = None
        if use_norm:
            m, norm_params = normalize_audio(m)
        est = bigshifts_wrapper(config, model, torch.as_tensor(m, dtype=torch.float32),
                                torch.device(dev), model_type="bs_roformer",
                                pbar=False, bigshifts=1)
        out = {}
        if isinstance(est, dict):
            items = est.items()
        else:
            items = ((t, est[i]) for i, t in enumerate(tgts))
        for k, v in items:
            v = np.asarray(v, dtype=np.float32)
            if norm_params is not None:
                v = denormalize_audio(v, norm_params)
            out[rename.get(k, k)] = _to_stereo(v)
        return out
    return run, dt, {"kind": "deep", "device": dev, "repo": "ZFTurbo/MSS-Training",
                     "ckpt": ckpt.name, "params": n_params, "normalize": use_norm,
                     "targets": tgts, "script": "bs_roformer", "dim": int(config.model.get("dim")),
                     "depth": int(config.model.get("depth")), "librosa_stub": stub_used}


def load_bsroformer_l12():
    return _load_bsroformer("L12")


def load_bsroformer_l6():
    return _load_bsroformer("L6")


# ------------------------------------------------------------------ #
# BSRNN (Band-Split RNN)
# ------------------------------------------------------------------ #
# Zenodo 17516442 的三个包对应不同的 conf/model/*.yaml：
#   bsrnn-opt.zip      -> bsrnn-opt     (oBSRNN, 4 个单目标 ckpt)
#   bsrnn-large.zip    -> bsrnn-large   (大容量 BSRNN, 4 个单目标 ckpt)
#   simo-bsrnn-opt.zip -> simo-bsrnn-opt (SIMO 版，单模型出 4 目标，权重名 separator.ckpt)
# 注意 SIMO 必须用同名 conf（docs/evaluation.md: `model=simo-bsrnn-opt simo=true`），
# 其 joint_bandsplit=True + 每源一个 masker，与非 SIMO 的 bsrnn-opt 结构不同。
BSRNN_SPECS = {
    "bsrnn":           dict(conf="bsrnn-opt",      subdir="bsrnn-opt",      simo=False, all_targets=False),
    "bsrnn_all":       dict(conf="bsrnn-opt",      subdir="bsrnn-opt",      simo=False, all_targets=True),
    "bsrnn_large":     dict(conf="bsrnn-large",    subdir="bsrnn-large",    simo=False, all_targets=False),
    "bsrnn_large_all": dict(conf="bsrnn-large",    subdir="bsrnn-large",    simo=False, all_targets=True),
    "bsrnn_simo":      dict(conf="simo-bsrnn-opt", subdir="simo-bsrnn-opt", simo=True,  all_targets=True),
}


def _find_bsrnn_ckpt(spec, targets):
    root = _paths.WEIGHTS / "BSRNN" / spec["subdir"]
    if not root.is_dir():
        raise FileNotFoundError(f"BSRNN 权重目录不存在: {root}")
    cands = sorted(root.rglob("*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"{root} 下没有 .ckpt（可能还没解压）")
    if spec["simo"]:
        pool = [c for c in cands if c.name == "separator.ckpt"] or cands
    else:
        per = {c.stem: c for c in cands if c.stem in targets}
        if len(per) == len(targets):
            return [per[t] for t in targets]
        pool = cands
    return pool


def _make_bsrnn_loader(key):
    spec = BSRNN_SPECS[key]

    def loader():
        import _bsrnn_loader as L
        targets = list(UMX_ORDER)
        found = _find_bsrnn_ckpt(spec, targets)

        if spec["simo"]:
            return L.load_bsrnn(found[0], spec["conf"], targets)

        if spec["all_targets"]:
            per = {c.stem: c for c in found if c.stem in targets}
            missing = [t for t in targets if t not in per]
            if missing:
                raise FileNotFoundError(f"{spec['subdir']} 缺少 {missing} 的 ckpt")
            return L.load_bsrnn_multi({t: per[t] for t in targets}, spec["conf"])

        # 单目标模式：只评测第一个目标（人声），避免一次跑 4 个模型把时间拖爆。
        pick = next((c for c in found if c.stem == "vocals"), found[0])
        return L.load_bsrnn(pick, spec["conf"], [pick.stem])

    return loader


def load_bsrnn():
    return _make_bsrnn_loader("bsrnn")()


def load_bsrnn_all():
    return _make_bsrnn_loader("bsrnn_all")()


def load_bsrnn_large():
    return _make_bsrnn_loader("bsrnn_large")()


def load_bsrnn_large_all():
    return _make_bsrnn_loader("bsrnn_large_all")()


def load_bsrnn_simo():
    return _make_bsrnn_loader("bsrnn_simo")()


def load_dprnn():
    """DPRNN —— 本机**自训** ckpt（MUSDB18-HQ 4-stem），原生 11.025 kHz。

    与其余模型的差别（必须如实标注，不许粉饰）：
      * 官方仓库只发布语音权重，音乐权重是 `tools/train_dprnn_musdb.py` 在本机训出来的。
      * 模型**原生 11025 Hz**：先把 44.1 kHz 的 mixture 抽取到 11.025 kHz 推理，
        输出再升回 44.1 kHz 交给统一评测。
        → 11 kHz 以上的带宽损失会**如实计入 SDR**（升回去也不会凭空长出高频）。
      * 4 个输出分支按 ckpt 里记录的 `targets` 顺序映射，绝不按位置猜。
      * `_dprnn_loader.build()` 在 state_dict 不匹配时**直接抛错**，不允许 strict=False 蒙过去。
    """
    import torch
    from scipy.signal import resample_poly

    sys.path.insert(0, str(_paths.TOOLS))
    import _dprnn_loader as D

    t0 = time.time()
    model, meta = D.build("best.pt")
    dt = time.time() - t0

    sr_m = int(meta["sr"])
    dec = SR // sr_m                      # 44100 -> 11025
    if dec < 1 or SR % sr_m:
        raise ValueError(f"原生采样率 {sr_m} 与评测采样率 {SR} 不是整数倍关系")
    targets = list(meta["targets"])
    dev = meta["device"]
    n_params = sum(p.numel() for p in model.parameters())

    def run(mix):
        x = _mono(mix)
        y = resample_poly(x, 1, dec).astype(np.float32)      # 下采到原生率
        hop = max(1, int(meta["cfg"]["kernel_size"]) // 2)   # 编码器 stride
        pad = (-len(y)) % hop                                 # 长度对齐到 stride 整数倍
        if pad:
            y = np.pad(y, (0, pad))
        t = torch.as_tensor(y, dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            outs = model(t)
        out = {}
        for name, o in zip(targets, outs):
            w = o.detach().float().cpu().numpy().ravel()[:len(y)]
            w = resample_poly(w, dec, 1).astype(np.float32)[:len(x)]   # 升回 44.1k
            if len(w) < len(x):
                w = np.pad(w, (0, len(x) - len(w)))
            out[name] = w
        return out

    return run, dt, {"kind": "deep", "device": dev,
                     "repo": "Dual-Path-RNN-Pytorch (自训 ckpt)",
                     "params": n_params, "sources": targets, "native_sr": sr_m,
                     "ckpt": Path(meta["ckpt"]).name, "train_step": meta["step"],
                     "train_val_sisdr": (meta.get("val") or {}).get("mean"),
                     "note": f"原生 {sr_m} Hz 自训；升采样回 {SR} Hz 评测，带宽损失计入 SDR"}


# ------------------------------------------------------------------ #
# 注册表
# ------------------------------------------------------------------ #
REGISTRY = {
    "oracle":         dict(loader=load_oracle,          needs_gt=True,  label="IRM/IBM Oracle"),
    # RPCA 的稀疏分量没有天然 stem 归属：实测在本曲最匹配 bass(+3.96dB)、
    # 对 vocals 是 -7.90dB。用 eval_all_stems 做交叉评估，避免任意映射误导结论。
    "rpca":           dict(loader=load_rpca,            needs_gt=False, label="RPCA (Inexact-ALM)", eval_all_stems=True),
    "umx":            dict(loader=load_umx,             needs_gt=False, label="Open-Unmix (umxhq)"),
    "mdx":            dict(loader=load_mdx,             needs_gt=False, label="MDX-Net (mdx_extra)"),
    "convtasnet":     dict(loader=load_convtasnet,      needs_gt=False, label="Conv-TasNet (musdb18)"),
    "mmdenselstm":    dict(loader=load_mmdenselstm,     needs_gt=False, label="MMDenseLSTM (musdb18)"),
    "bsroformer_l12": dict(loader=load_bsroformer_l12,  needs_gt=False, label="BS-RoFormer L12 (ep_317, vocals)"),
    # ep_937 的 yaml 写 target_instrument: other，但实测其输出是「mixture - drums - bass」
    # 即 vocals+other（与 ZFTurbo justfile 注释 "removes drums and bass only" 一致）。
    # 因此评估参考必须重新组合，否则会误判为「模型失效」(0.83 dB vs 12.13 dB)。
    "bsroformer_l6":  dict(loader=load_bsroformer_l6,   needs_gt=False, label="BS-RoFormer L6 (ep_937, vocals+other)",
                           ref_map={"vocals+other": (("vocals", 1.0), ("other", 1.0))}),
    "bsrnn":          dict(loader=load_bsrnn,           needs_gt=False, label="BSRNN (oBSRNN, vocals)"),
    "bsrnn_all":      dict(loader=load_bsrnn_all,       needs_gt=False, label="BSRNN (oBSRNN, 4-stem)"),
    "bsrnn_large":    dict(loader=load_bsrnn_large,     needs_gt=False, label="BSRNN large (vocals)"),
    "bsrnn_large_all":dict(loader=load_bsrnn_large_all, needs_gt=False, label="BSRNN large (4-stem)"),
    "bsrnn_simo":     dict(loader=load_bsrnn_simo,      needs_gt=False, label="BSRNN SIMO (4-stem)"),
    # Demucs：与其余 11 个模型同一口径（同样的片段、同样的 museval 评价），
    # 使 12 个模型可以直接进同一张中位数表。⚠️ 非确定性，见 load_demucs docstring。
    "demucs":         dict(loader=load_demucs,          needs_gt=False, label="Demucs (htdemucs)"),
    # DPRNN：本机自训（官方只有语音权重）。原生 11.025 kHz，见 load_dprnn docstring。
    "dprnn":          dict(loader=load_dprnn,           needs_gt=False, label="DPRNN (自训, 4-stem)"),
}


# ------------------------------------------------------------------ #
# 主体
# ------------------------------------------------------------------ #
def run_one(name: str, song: str, subset: str, duration: int, offset: float = 0.0,
            out_tag: str | None = None, audio_format: str | None = None) -> dict:
    """跑单模型单曲。

    `out_tag` 与 `subset` 解耦：
    - `subset` 决定**数据从哪读**（`MUSDB18-HQ/<subset>/<song>/`）
    - `out_tag` 决定**结果写哪**（独立 json + 独立音频目录）
    这样「对着 test 集做可运行性探针」既能读到正确数据，又不会覆盖正式结果。
    """
    spec = REGISTRY[name]
    label = spec["label"]
    # test 集走 _musdb18_test 子目录 + model_runs_test.json，避免污染 train 片段主表
    tag = out_tag if out_tag is not None else subset
    out_dir = _out_dir_for(name, tag, song)
    runs_json = _runs_json_for(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{name}_{_dt.date.today().isoformat()}.log"

    banner(f"模型: {label}  ({name})")
    rec = {
        "model": name, "label": label, "song": song, "subset": subset,
        "duration_s": duration, "offset_s": offset, "status": "FAIL", "weights_downloaded": None,
        "load_s": None, "infer_s": None, "rtf": None, "targets": [],
        "sdr": {}, "si_sdr": {}, "error": None,
        "torch_ver": None, "cuda": None, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path), "out_dir": str(out_dir),
    }
    try:
        import torch
        rec["torch_ver"] = torch.__version__
        rec["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        mix, gt, _ = load_clip(song, subset, duration, offset)
        print(f"  片段: {len(mix)/SR:.1f}s (offset {offset:g}s), GT 目标: {sorted(gt)}")

        if spec["needs_gt"]:
            t0 = time.time()
            stems = _oracle_irm(mix, gt)
            infer_s = time.time() - t0
            load_s, info = 0.0, {"kind": "analytic"}
            rec["weights_downloaded"] = False
            n_chunks = 0
        else:
            run_fn, load_s, info = spec["loader"]()
            rec["weights_downloaded"] = True if info.get("kind") == "deep" else None
            t0 = time.time()
            # 🔴 整曲必须分块（2026-09-19）：一次性前向在 8 GiB 卡上要么真 OOM（>240 s），
            # 要么触发 CUDA sysmem fallback —— **不报错但慢 15×**（实测 200 s 整曲
            # max_memory_allocated 21.4 GB，RTF 0.302；分块 60 s 后 RTF 0.0204、峰值 1.87 GB）。
            n_chunks = 0
            if len(mix) > CHUNK_THRESHOLD_S * SR:
                from _chunked_infer import chunked_separate
                _lines: list[str] = []
                stems = chunked_separate(run_fn, mix, sr=SR, chunk_s=CHUNK_S,
                                         overlap_s=CHUNK_OVERLAP_S,
                                         progress=_lines.append)
                n_chunks = len(_lines)
                print(f"  [分块推理] {len(mix)/SR:.0f}s -> {n_chunks} 块 × {CHUNK_S:.0f}s"
                      f"（整曲前向会触发 sysmem fallback，见 _chunked_infer.py）")
            else:
                stems = run_fn(mix)
            infer_s = time.time() - t0

        rec["load_s"] = round(load_s, 3)
        rec["infer_s"] = round(infer_s, 3)
        rec["rtf"] = round(infer_s / (len(mix) / SR), 4)
        rec["info"] = info
        rec["targets"] = sorted(stems)
        if n_chunks:
            rec["chunked"] = {"n_chunks": n_chunks, "chunk_s": CHUNK_S,
                              "overlap_s": CHUNK_OVERLAP_S}

        # 存音频
        # 🔴 磁盘约束（2026-09-21）：test 集 50 首 × 12 模型若存 16-bit WAV 要 98 GB，
        # 而 E: 盘仅剩 66 GB。FLAC 是无损（实测往返误差 0.00e+00、压缩到 24.8%），
        # 故默认改存 FLAC —— 对指标零影响（且指标是在写盘前算的）。
        fmt = (audio_format or "flac").lower()
        if fmt != "none":
            ext = "flac" if fmt == "flac" else "wav"
            for t, y in stems.items():
                y = _to_stereo(y)
                path = out_dir / f"{t}.{ext}"
                sf.write(str(path), np.clip(y, -1.0, 1.0), SR,
                         format="FLAC" if fmt == "flac" else "WAV",
                         subtype="PCM_16")
            rec["audio_format"] = fmt

        # 评估
        ref_map = spec.get("ref_map", {})
        for t, y in stems.items():
            est_m = _mono(y)

            # 单输出古典方法（如 RPCA 的稀疏分量）没有天然的 stem 归属，
            # 强行指定一个会误导结论 —— 这里对全部 GT 目标做交叉评估。
            if spec.get("eval_all_stems"):
                cross = {}
                for g, gsig in gt.items():
                    n = min(len(est_m), len(gsig))
                    cross[g] = {
                        "si_sdr": round(float(si_sdr(gsig[:n], est_m[:n])), 3),
                        "corr": round(float(np.corrcoef(gsig[:n], est_m[:n])[0, 1]), 4),
                    }
                rec.setdefault("cross_stem", {})[t] = cross
                best = max(cross, key=lambda k: cross[k]["si_sdr"])
                rec["targets"] = sorted(stems)
                rec["best_match"] = {t: {"stem": best, **cross[best]}}
                rec["sdr"][t] = None
                rec["si_sdr"][t] = cross.get(t, {}).get("si_sdr")
                rec.setdefault("ref_used", {})[t] = "(cross-stem; 见 cross_stem)"
                continue

            # 参考信号：默认同名 GT；若模型输出与 MUSDB 的 stem 划分不同
            # （例如 ep_937 输出 vocals+other），则用 ref_map 里的线性组合重建。
            if t in ref_map:
                parts = ref_map[t]
                if not all(k in gt for k, _ in parts):
                    print(f"    [SKIP] {t}: 参考信号需要 {[k for k, _ in parts]}，GT 缺失")
                    continue
                ref = sum(w * gt[k] for k, w in parts)
                ref_name = "+".join(k for k, _ in parts)
            elif t in gt:
                ref, ref_name = gt[t], t
            else:
                continue
            n = min(len(est_m), len(ref))
            m = evaluate(ref[:n], est_m[:n])
            rec["sdr"][t] = round(m["sdr"], 3) if m["sdr"] is not None else None
            rec["si_sdr"][t] = round(float(si_sdr(ref[:n], est_m[:n])), 3)
            rec.setdefault("ref_used", {})[t] = ref_name

        rec["status"] = "PASS"
        print(f"  ✅ PASS  load {load_s:.2f}s  infer {infer_s:.2f}s  RTF {rec['rtf']}")
        for t in rec["targets"]:
            rn = rec.get("ref_used", {}).get(t, t)
            extra = "" if rn == t else f"   (参考: {rn})"
            print(f"     {t:<12} SDR {rec['sdr'].get(t)}  SI-SDR {rec['si_sdr'].get(t)}{extra}")
        for t, cross in (rec.get("cross_stem") or {}).items():
            row = "  ".join(f"{k}={v['si_sdr']:+.2f}" for k, v in cross.items())
            print(f"     [{t}] 跨 stem SI-SDR: {row}   最佳匹配={rec['best_match'][t]['stem']}")
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        print(f"  ❌ FAIL  {rec['error']}")
        print(rec["traceback"][-800:])

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 合并写 model_runs.json（或 test 集专用的 model_runs_test.json）
    runs_json.parent.mkdir(parents=True, exist_ok=True)
    all_runs = {}
    if runs_json.is_file():
        try:
            all_runs = json.loads(runs_json.read_text(encoding="utf-8"))
        except Exception:
            all_runs = {}
    all_runs.setdefault("runs", {})
    all_runs["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
    all_runs["subset"] = subset
    all_runs["runs"].setdefault(name, {})[song] = rec
    runs_json.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="模型名 / 逗号分隔 / all")
    ap.add_argument("--song", default=DEFAULT_SONG)
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                    help="片段长度（秒）；**0 = 整曲**（用于 MUSDB18 官方 test 集）")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="从第几秒开始取片段（默认 0；建议取中段以避开低频引子）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-tag", default=None,
                    help="结果落盘标签（与 --subset 解耦）。默认跟随 --subset；"
                         "probe = 可运行性探针（写 model_runs_probe.json + _probe_tmp/）")
    ap.add_argument("--audio-format", default=None, choices=["wav", "flac", "none"],
                    help="输出音频格式。默认 flac（无损，约为 WAV 的 25%%，见磁盘约束注释）")
    args = ap.parse_args()

    if args.list:
        for k, v in REGISTRY.items():
            print(f"  {k:<16} {v['label']}")
        return

    names = list(REGISTRY) if args.model == "all" else [m.strip() for m in args.model.split(",")]
    unknown = [m for m in names if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"未知模型: {unknown}; 可用: {sorted(REGISTRY)}")

    duration = None if args.duration == 0 else args.duration      # 0 → 整曲

    banner("通用模型基准")
    print(f"  歌曲: {args.subset}/{args.song} | 片段: "
          f"{'整曲' if duration is None else f'{duration}s'} @{args.offset:g}s | 模型: {names}")
    results = [run_one(n, args.song, args.subset, duration, args.offset, args.out_tag,
                       args.audio_format) for n in names]

    banner("汇总")
    print(f"  {'模型':<18}{'状态':<8}{'load_s':>9}{'infer_s':>10}{'RTF':>9}")
    for r in results:
        print(f"  {r['label'][:16]:<18}{r['status']:<8}"
              f"{(r['load_s'] if r['load_s'] is not None else float('nan')):>9.2f}"
              f"{(r['infer_s'] if r['infer_s'] is not None else float('nan')):>10.2f}"
              f"{(r['rtf'] if r['rtf'] is not None else float('nan')):>9.3f}")
    print(f"\n  结果 JSON: {_runs_json_for(args.out_tag or args.subset)}")


if __name__ == "__main__":
    main()
