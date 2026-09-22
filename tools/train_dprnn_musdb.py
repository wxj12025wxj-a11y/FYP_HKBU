#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPRNN 自训（MUSDB18-HQ，4 stem）—— 填补 12 模型清单里唯一「只差权重」的空缺。

背景
----
`01_models/_third_party/Dual-Path-RNN-Pytorch`（JusperLee）是**语音**分离仓库：
代码在位、模型结构完整（`Dual_RNN_model` 有 `num_spks` 参数，可直接 4 输出），
但官方只发布 WSJ0 语音权重，且数据管线是 WSJ0 的 `.scp` 格式。
→ 本脚本**不复用它的 WSJ0 数据管线**，只复用 `model/model_rnn.py` 的模型定义，
  自己写 MUSDB18-HQ 的 4-stem 管线，从零训练。

关键设定（与评测口径对齐）
--------------------------
* **采样率 11025 Hz**（44100 的 1/4，整数抽取用 `resample_poly(x,1,4)`，无重采样伪影）。
  这是音乐 DPRNN 类工作的通行训练率。评测在 44.1 kHz 下进行，
  estimate 会被升回 44.1 kHz —— **带宽损失会如实计入 SDR**，不做粉饰。
* 目标顺序固定 `TARGETS`；ckpt 里显式存 `targets`，
  加载端必须按 ckpt 顺序构造模型（多输出模型按位置对齐的静默串味陷阱，见 MODEL_REGISTRY §4）。
* 损失 = 负 SI-SDR（DPRNN 原文口径），对 4 个 stem 取均值，静音 stem 自动屏蔽。

产物
----
    ckpt : 01_models/_weights/dprnn_musdb/{last,best}.pt
    历史 : 04_reports/separation/data/dprnn_training.json
    日志 : tools/_logs/train_dprnn_musdb.txt
    曲线 : 04_reports/separation/figures/dprnn/training_curve.png

用法
----
    PY="C:/Users/jerry/.workbuddy/binaries/python/envs/fyp_audio/Scripts/python.exe"
    $PY tools/train_dprnn_musdb.py --minutes 100          # 跑 100 分钟后自动停并存盘
    $PY tools/train_dprnn_musdb.py --minutes 100 --resume  # 续跑
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths                                            # noqa: E402

_paths.setup_env()

import numpy as np                                                # noqa: E402
import soundfile as sf                                            # noqa: E402
from scipy.signal import resample_poly                            # noqa: E402

# ---- 常量 -------------------------------------------------------- #
SR_DATA = 44100                     # 数据集原生
SR_TRAIN = 11025                    # 训练/推理采样率
DECIM = SR_DATA // SR_TRAIN         # 4
TARGETS = ("vocals", "drums", "bass", "other")

DPRNN_REPO = _paths.THIRD_PARTY / "Dual-Path-RNN-Pytorch"
CKPT_DIR = _paths.WEIGHTS / "dprnn_musdb"
HIST_JSON = _paths.DATA / "dprnn_training.json"
LOG_TXT = _paths.LOGS / "train_dprnn_musdb.txt"
CURVE_PNG = _paths.FIGURES / "dprnn" / "training_curve.png"

# 训练超参（一次性写死，便于复现；跟 ckpt 一起存）
HP = dict(
    kernel_size=16, in_channels=64, out_channels=128, hidden_channels=128,
    rnn_type="LSTM", norm="ln", dropout=0.0, bidirectional=True,
    num_layers=6, K=200, num_spks=len(TARGETS),
)
CHUNK_S = 4.0
# 编码器 stride = kernel_size//2；ConvTranspose 解码后长度 = floor(L/stride)*stride，
# 所以喂进去的波形长度**必须**是 stride 的整数倍，否则 est 会比 ref 短几个采样点而炸掉。
HOP = HP["kernel_size"] // 2

# 验证片段 = 基准里用的 3 个能量均衡片段（同一批，便于和正式结果对照）
VAL_CLIPS = (
    ("ANiMAL - Clinic A", "train", 140.0),
    ("Creepoid - OldTree", "train", 60.0),
    ("Dark Ride - Burning Bridges", "train", 180.0),
)

_log_fh = None


def keep_awake(on: bool = True):
    """Windows: 训练期间阻止系统睡眠/关屏。

    🔴 教训（2026-09-17）：上一轮 `--minutes 110` 的预算被**一次 8 小时系统睡眠**吃掉
    （Kernel-Power EventID 42 @02:08:33，日志从 step 3000 直接跳到「用时 555 min」），
    结果只训了 ~23 分钟真实算力。`powercfg` 改的是电源计划，任何人改回去或换方案就失效；
    这里用进程级 API 兜底，`ES_CONTINUOUS` 会一直生效到本进程退出。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if on
            else ES_CONTINUOUS)
    except Exception:                                                  # noqa: BLE001
        pass


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ---------------------------------------------------------------- 模型
def load_model_cls():
    """复用 `_dprnn_loader` 的加载逻辑（保证训练/评测用同一套模型构造）。"""
    sys.path.insert(0, str(_paths.TOOLS))
    from _dprnn_loader import load_model_cls as _cls
    return _cls()


# ---------------------------------------------------------------- 数据
def _mono_deci(path: Path, start_f: int, n_f: int) -> np.ndarray:
    """读 wav → 单声道 → 44100/4 抽取。返回 float32 (n_f/4,)。"""
    with sf.SoundFile(str(path)) as f:
        f.seek(start_f)
        y = f.read(n_f, dtype="float32", always_2d=True)
    if y.shape[0] == 0:
        raise RuntimeError(f"读到 0 帧: {path}")
    x = y.mean(axis=1)
    return resample_poly(x, 1, DECIM).astype(np.float32)


def _song_len(path: Path) -> int:
    return sf.info(str(path)).frames


def build_index(subset: str = "train"):
    """扫描子集，返回 [(song, frames)]，只保留 4 个 stem 齐全的歌。"""
    root = _paths.MUSDB18_ROOT / subset
    idx = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if all((d / f"{t}.wav").is_file() for t in TARGETS):
            idx.append((d.name, _song_len(d / "vocals.wav")))
    return idx


class MusdbChunks:
    """按步数随机采样的 chunk 数据集（不是按歌划分的标准 epoch）。

    每个 step 随机挑一首歌 + 随机起点，读 4 个 stem → 单声道 11.025k。
    合成 mixture 用 **4 个 stem 之和**（而不是读 mixture.wav），
    这样掩码一致性（masks sum to 1）在数值上严格成立。
    """

    def __init__(self, index, chunk_s: float, steps_per_epoch: int, seed: int = 0):
        self.index = index
        self.n = int(round(chunk_s * SR_TRAIN))
        self.n += (-self.n) % HOP          # 对齐到 stride 整数倍
        self.steps = steps_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps

    def sample(self):
        song, frames = self.index[self.rng.integers(len(self.index))]
        d = _paths.MUSDB18_ROOT / "train" / song
        max_start = max(0, frames - self.n * DECIM - 1)
        s0 = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        stems = []
        for t in TARGETS:
            y = _mono_deci(d / f"{t}.wav", s0, self.n * DECIM)
            if len(y) < self.n:                       # 尾端补齐
                y = np.pad(y, (0, self.n - len(y)))
            stems.append(y[:self.n])
        mix = np.sum(stems, axis=0)
        peak = float(np.max(np.abs(mix))) + 1e-8
        mix = mix / peak
        stems = [s / peak for s in stems]
        return song, mix, np.stack(stems)             # (n,), (4, n)


# ---------------------------------------------------------------- 指标
def si_sdr_torch(est, ref, eps=1e-8):
    """per-(batch,source) SI-SDR（dB）。est/ref: (B,S,T)。"""
    import torch
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    a = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + eps)
    tgt = a * ref
    num = tgt.pow(2).sum(dim=-1) + eps
    den = (est - tgt).pow(2).sum(dim=-1) + eps
    return 10.0 * torch.log10(num / den)


def si_sdr_np(est: np.ndarray, ref: np.ndarray, eps=1e-8) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    a = float((est * ref).sum() / (float((ref ** 2).sum()) + eps))
    tgt = a * ref
    num = float((tgt ** 2).sum()) + eps
    den = float(((est - tgt) ** 2).sum()) + eps
    return 10.0 * math.log10(num / den)


# ---------------------------------------------------------------- 验证
def validate(model, device, max_s: float = 10.0):
    """在 3 个均衡片段上算 per-stem SI-SDR（11.025 kHz 域，仅作训练进度代理）。

    真正的 SDR 用 `benchmark_model_universal.py --model dprnn` 在 44.1 kHz 下算。
    """
    import torch
    per_stem = {t: [] for t in TARGETS}
    model.eval()
    with torch.no_grad():
        for song, subset, off in VAL_CLIPS:
            d = _paths.MUSDB18_ROOT / subset / song
            frames = _song_len(d / "vocals.wav")
            s0 = min(int(off * SR_DATA), max(0, frames - 1))
            n = min(int(max_s * SR_TRAIN), (frames - s0) // DECIM)
            stems = []
            for t in TARGETS:
                y = _mono_deci(d / f"{t}.wav", s0, n * DECIM)
                if len(y) < n:
                    y = np.pad(y, (0, n - len(y)))
                stems.append(y[:n])
            mix = np.sum(stems, axis=0)
            peak = float(np.max(np.abs(mix))) + 1e-8
            x = torch.as_tensor(mix / peak, dtype=torch.float32, device=device)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                outs = model(x)
            for i, t in enumerate(TARGETS):
                est = outs[i].float().cpu().numpy().ravel()[:n]
                per_stem[t].append(si_sdr_np(est, stems[i][:len(est)] / peak))
    model.train()
    med = {t: float(np.median(v)) for t, v in per_stem.items()}
    med["mean"] = float(np.mean(list(med.values())))
    return med


# ---------------------------------------------------------------- 训练
def train(args):
    import torch

    global _log_fh
    keep_awake(True)                       # 训练期间禁止系统睡眠（见函数 docstring）
    _paths.LOGS.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_TXT, "a", encoding="utf-8")
    log("=" * 72)
    log(f"DPRNN 自训启动  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}  "
            f"cap={torch.cuda.get_device_capability(0)}  "
            f"mem={torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    if device.type == "cpu":
        log("⚠️  在 CPU 上训练会极慢，已自动把预算压到 15 分钟。用 --cpu 可强制，--minutes 可覆盖。")
        if args.minutes > 15:
            args.minutes = 15

    index = build_index("train")
    log(f"训练集: {len(index)} 首歌（MUSDB18-HQ train）")

    Model = load_model_cls()
    model = Model(**HP).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"模型: {n_par/1e6:.2f} M 参数  cfg={HP}")
    log(f"chunk {CHUNK_S}s @ {SR_TRAIN} Hz = {int(CHUNK_S*SR_TRAIN)} 采样点，batch={args.batch}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, min_lr=1e-5)

    ds = MusdbChunks(index, CHUNK_S, args.steps_per_epoch, seed=args.seed)
    step = 0
    best = -1e9
    hist = {"hp": HP, "chunk_s": CHUNK_S, "sr": SR_TRAIN, "targets": list(TARGETS),
            "batch": args.batch, "lr": args.lr, "steps": [], "vals": [], "started": None}

    start_ckpt = CKPT_DIR / "last.pt"
    if args.resume and start_ckpt.is_file():
        ck = torch.load(start_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        # 🔴 老 ckpt（2026-09-17 之前）没存 `opt` —— 之前会直接 KeyError，
        #    也就是 `--resume` 其实**从未可用**。现在缺就退化为「只续权重、优化器重来」。
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            log("⚠️  ckpt 内无优化器状态（旧格式），只续权重，Adam 动量从头开始")
        # 🔴 `opt.load_state_dict` 会把 ckpt 里的 param_groups（含 lr）一并恢复，
        #    于是命令行 `--lr` 会被**静默忽略**。改 batch 时必须能同步改 lr，
        #    否则「大 batch + 旧小 lr」= 学得更慢。这里显式覆盖回命令行值。
        for g in opt.param_groups:
            g["lr"] = args.lr
        step = ck.get("step", 0)
        best = ck.get("best", -1e9)
        if HIST_JSON.is_file():
            try:
                hist = json.loads(HIST_JSON.read_text(encoding="utf-8"))
                hist.setdefault("hp", HP)
            except Exception:
                pass
        # 🔴 元数据必须跟着命令行刷新（2026-09-19 修）
        #    原实现只合并了 `steps` / `vals`，但 `batch` / `lr` / `hp` 等**超参元数据**
        #    仍是上一轮的。于是 `--batch 8` 续跑后，json 里 `batch` 还写着 4，
        #    而 `finished` / `final_step` / `best_val_mean` 停在**上一轮**的收尾值
        #    （实际观测：steps 末条 step 31800、vals 末条 31900，而 final_step 写着 3404）
        #    → 一份自相矛盾的训练历史。这里在续跑时就把超参与收尾字段全部重写。
        hist["batch"] = args.batch
        hist["lr"] = args.lr
        hist["hp"] = HP
        hist["chunk_s"] = CHUNK_S
        hist["sr"] = SR_TRAIN
        hist["targets"] = list(TARGETS)
        hist["finished"] = None
        hist["final_step"] = None
        hist["best_val_mean"] = None
        hist["resumed_from_step"] = step
        log(f"续跑: 从 step {step} 开始（best={best:.3f}）")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    hist["started"] = hist.get("started") or time.strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()
    budget = args.minutes * 60.0
    amp = (device.type == "cuda") and not args.no_amp
    log(f"预算 {args.minutes} 分钟；AMP={'开' if amp else '关'}；"
        f"每 {args.log_every} 步打印，每 {args.val_every} 步验证")

    running, run_sisdr, run_n = 0.0, 0.0, 0
    model.train()
    nan_streak = 0

    # 🔴 假 batch 事故（2026-09-17）：原先是
    #       x = torch.as_tensor(mix)[None].expand(args.batch, -1)
    #   `expand` **只是把同一段音频复制 batch 份**，梯度等价于 batch=1。
    #   于是「batch=4 / 129 step/min」看着很美，实际数据吞吐只有 4 s/step，
    #   22 分钟训练只见过约 0.5 个 MUSDB epoch → 均值 SI-SDR 卡在 +0.78 dB。
    #   正确做法：真的采 batch 个**不同**片段再 stack。
    def fetch(n: int):
        xs, ys = [], []
        for _ in range(n):
            _, mix, stems = ds.sample()
            xs.append(mix)
            ys.append(stems)
        return (torch.as_tensor(np.stack(xs), device=device),
                torch.as_tensor(np.stack(ys), device=device))

    # 🔴 8 GiB 显存实测（bf16 autocast，见 tools/_scratch/_dprnn_timing_probe.py）：
    #   batch 4 → 3.03 GiB / 0.254 s·step；6 → 4.45 GiB / 0.386 s；8 → 5.88 GiB / 0.374 s；
    #   12 → 8.73 GiB 起**溢出到系统内存**，步耗时从 0.37 s 暴涨到 3.4 s。
    #   本机 NVIDIA 驱动开了 CUDA sysmem fallback：超显存**不报错**，只是悄悄慢 10~100×，
    #   所以必须靠 OOM 兜底 + 保守降级，不能指望它自己炸出来。
    cur_batch = args.batch
    step0 = step                      # 续跑时 step 是全局值，算速率必须用「本轮增量」
    while time.time() - t_start < budget and step < args.max_steps:
        try:
            x, y = fetch(cur_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                outs = model(x)                                # list of (B,T)
                est = torch.stack(outs, dim=1)                 # (B,4,T)
                L = min(est.shape[-1], y.shape[-1])            # 兜底：长度不许不齐
                loss_mat = si_sdr_torch(est.float()[..., :L], y.float()[..., :L])   # (B,4)
            # 静音 stem 的 SI-SDR 无意义 → 按参考能量屏蔽
            energy = y.float().pow(2).mean(dim=-1)              # (B,4)
            valid = energy > 1e-4
            if valid.sum() == 0:
                continue
            loss = -loss_mat[valid].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur_batch > 2:
                cur_batch //= 2
                log(f"⚠️  CUDA OOM → batch 自动降到 {cur_batch}，继续训练")
                continue
            raise

        lv = float(loss.detach())
        running += lv
        run_sisdr += float(loss_mat[valid].mean().detach())
        run_n += 1
        step += 1

        if not math.isfinite(lv):
            nan_streak += 1
            if nan_streak >= 5 and amp:
                log("⚠️  连续 NaN → 关闭 AMP 重来（当前 step 之前的权重保留）")
                amp = False
                nan_streak = 0
            continue

        if step % args.log_every == 0:
            lr_now = opt.param_groups[0]["lr"]
            el = time.time() - t_start
            log(f"step {step:6d} | loss {running/run_n:+.3f} | SI-SDR "
                f"{run_sisdr/run_n:+.3f} dB | lr {lr_now:.2e} | "
                f"{(step-step0)/el*60:.1f} step/min | 已用 {el/60:.1f} min")
            hist["steps"].append({"step": step, "loss": round(running / run_n, 4),
                                 "si_sdr": round(run_sisdr / run_n, 4),
                                 "lr": lr_now, "min": round(el / 60, 2)})
            running, run_sisdr, run_n = 0.0, 0.0, 0

        if step % args.val_every == 0:
            t0 = time.time()
            med = validate(model, device, args.val_seconds)
            log(f"  ★ 验证 @step {step}: mean SI-SDR {med['mean']:+.3f} dB  " +
                " ".join(f"{t}={med[t]:+.2f}" for t in TARGETS) +
                f"  ({time.time()-t0:.1f}s)")
            hist["vals"].append({"step": step, "mean": round(med["mean"], 4),
                                 **{t: round(med[t], 4) for t in TARGETS}})
            sched.step(med["mean"])
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                        "cfg": HP, "targets": list(TARGETS), "sr": SR_TRAIN,
                        "step": step, "best": max(best, med["mean"]),
                        "val": med}, CKPT_DIR / "last.pt")
            if med["mean"] > best:
                best = med["mean"]
                torch.save({"state_dict": model.state_dict(), "cfg": HP,
                            "targets": list(TARGETS), "sr": SR_TRAIN,
                            "step": step, "best": best, "val": med},
                           CKPT_DIR / "best.pt")
                log(f"     ↑ 新最优，已存 best.pt（mean {best:+.3f} dB）")
            HIST_JSON.parent.mkdir(parents=True, exist_ok=True)
            HIST_JSON.write_text(json.dumps(hist, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    # ---- 收尾：无论如何都存一次 ----
    if step > 0:
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "cfg": HP, "targets": list(TARGETS), "sr": SR_TRAIN,
                    "step": step, "best": best}, CKPT_DIR / "last.pt")
    hist["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    hist["total_min"] = round((time.time() - t_start) / 60, 2)
    hist["final_step"] = step
    hist["best_val_mean"] = round(best, 4)
    HIST_JSON.parent.mkdir(parents=True, exist_ok=True)
    HIST_JSON.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"训练结束: step {step}, 用时 {hist['total_min']} min, best mean SI-SDR {best:+.3f} dB")
    log(f"ckpt → {CKPT_DIR}")

    _plot(hist)
    keep_awake(False)                      # 交还睡眠控制权
    return 0


def _plot(hist):
    """训练曲线 → 04_reports/separation/figures/dprnn/training_curve.png"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        steps = hist.get("steps") or []
        vals = hist.get("vals") or []
        if not steps and not vals:
            return
        CURVE_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        if steps:
            ax = axes[0]
            ax.plot([s["step"] for s in steps], [s["si_sdr"] for s in steps],
                    color="#4C78A8", lw=1.6)
            ax.axhline(0, color="#888", lw=0.8, ls=":")
            ax.set_xlabel("训练步数")
            ax.set_ylabel("batch 平均 SI-SDR (dB)")
            ax.set_title("DPRNN 自训：训练损失", fontweight="bold")
            ax.grid(ls=":", alpha=0.45)

        if vals:
            ax = axes[1]
            ax.plot([v["step"] for v in vals], [v["mean"] for v in vals],
                    color="#C0392B", lw=1.8, marker="o", ms=3.5, label="四轨均值")
            for t, c in zip(TARGETS, ("#2E7D32", "#8E44AD", "#E67E22", "#2980B9")):
                ax.plot([v["step"] for v in vals], [v[t] for v in vals],
                        lw=1.1, alpha=0.85, label=t, color=c)
            ax.set_xlabel("训练步数")
            ax.set_ylabel("验证 SDR 代理 (dB) @11.025 kHz")
            ax.set_title("DPRNN 自训：验证曲线（3 个均衡片段）", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(ls=":", alpha=0.45)

        fig.suptitle("DPRNN（自训，MUSDB18-HQ 4-stem）", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(CURVE_PNG, dpi=140)
        plt.close(fig)
        log(f"曲线图 → {CURVE_PNG}")
    except Exception as e:                                             # noqa: BLE001
        log(f"（画图跳过：{e}）")


def main():
    ap = argparse.ArgumentParser(description="DPRNN 自训：MUSDB18-HQ 4-stem @11.025 kHz")
    ap.add_argument("--minutes", type=float, default=100.0, help="训练时长预算（分钟）")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=100000, help="占位，仅决定 __len__")
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（默认有 GPU 就用 GPU）")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
