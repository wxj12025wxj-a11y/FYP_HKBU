"""逐模型依赖探针：区分「未安装」与「被 Smart App Control 拦截」。

- MISSING : ModuleNotFoundError（可直接 pip 安装解决）
- BLOCKED : 应用控制策略拦截 / DLL 加载失败（须先解除 SAC 或换环境）
- OK      : 可正常导入

用法:
  python tools/probe_model_deps.py
输出:
  outputs/comparison/dependency_probe.json
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
PROJECT_ROOT = _paths.PROJECT_ROOT
OUT_DIR = _paths.comparison_dir()

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

BLOCK_HINTS = ("应用程序控制策略", "DLL load failed", "Cannot find module", "llvmlite")

# 模型 -> [(import 名, 用途, 严重度)]  严重度: core=没有就跑不了 / optional=视分支而定
MODEL_DEPS: dict[str, list[tuple[str, str, str]]] = {
    "BS-RoFormer L12 / L6": [
        ("torch", "张量/模型", "core"),
        ("numpy", "数组", "core"),
        ("librosa", "STFT (ZFTurbo inference 路径)", "core"),
        ("soundfile", "音频 IO", "core"),
        ("tqdm", "进度条", "optional"),
        ("yaml", "读取 config", "core"),
        ("omegaconf", "读取 config (utils.settings)", "core"),
        ("ml_collections", "ConfigDict (utils.settings / model_utils)", "core"),
        ("rotary_embedding_torch", "bs_roformer 架构 (RoPE)", "core"),
        ("beartype", "lucidrains bs_roformer 运行时类型检查", "core"),
        ("einx", "lucidrains bs_roformer", "optional"),
        ("torch_einops_utils", "lucidrains bs_roformer", "optional"),
        ("einops", "reshape 算子", "core"),
        ("scipy.signal", "STFT 备选路径 / 重采样", "optional"),
    ],
    "Band-Split RNN (BSRNN)": [
        ("torch", "张量/模型", "core"),
        ("torchaudio", "音频 IO + resample", "core"),
        ("hydra", "separate.py 入口 (@hydra.main)", "core"),
        ("omegaconf", "config", "core"),
        ("lightning.pytorch", "models/pl_module.py", "core"),
        ("pandas", "pl_module 内 SDR 汇总", "core"),
        ("#helpers", "(仓库内模块 models/helpers, 在 bsrnn/ 下运行即可)", "core"),
        ("museval", "helpers.eval.compute_sdr", "core"),
        ("#bs_roformer", "(仓库可选变体, 需 sys.path)", "optional"),
    ],
    "MDX-Net (onnx Track A)": [
        ("torch", "张量/模型", "core"),
        ("onnxruntime", "加载 onnx_A.zip 内的 onnx 模型", "core"),
        ("onnx", "onnx 图操作", "core"),
        ("numpy", "数组", "core"),
        ("librosa", "STFT", "core"),
        ("soundfile", "音频 IO", "core"),
        ("museval", "评估", "optional"),
    ],
    "MDX-Net (Demucs mdx_extra 路线)": [
        ("torch", "张量/模型", "core"),
        ("einops", "demucs 内部", "core"),
        ("julius", "demucs 内部重采样", "core"),
        ("dora", "demucs 内部 (pip 包名 dora-search)", "core"),
        ("soundfile", "音频 IO", "core"),
        ("numpy", "数组", "core"),
    ],
    "Conv-TasNet (tky823)": [
        ("torch", "张量/模型", "core"),
        ("numpy", "数组", "core"),
        ("scipy", "重采样/滤波", "core"),
        ("scipy.signal", "stft/istft", "core"),
        ("soundfile", "音频 IO", "core"),
        ("librosa", "STFT", "core"),
        ("musdb", "MUSDB18 读取", "optional"),
        ("tqdm", "进度条", "optional"),
    ],
    "MMDenseLSTM (tky823)": [
        ("torch", "张量/模型", "core"),
        ("numpy", "数组", "core"),
        ("scipy", "重采样/滤波", "core"),
        ("scipy.signal", "stft/istft", "core"),
        ("soundfile", "音频 IO", "core"),
        ("librosa", "STFT", "core"),
    ],
    "DPRNN (待自训)": [
        ("torch", "张量/模型/训练", "core"),
        ("numpy", "数组", "core"),
        ("numpy.random", "打乱/初始化 (训练必需)", "core"),
        ("scipy.signal", "stft/istft", "core"),
        ("soundfile", "音频 IO", "core"),
        ("librosa", "STFT", "core"),
        ("musdb", "MUSDB18 读取", "core"),
    ],
    "IRM / IBM oracle": [
        ("torch", "torch.stft 路径", "core"),
        ("numpy", "数组", "core"),
        ("librosa", "参考实现 (sigsep-mus-oracle 依赖它)", "optional"),
        ("scipy.signal", "sigsep-mus-oracle 内部", "optional"),
        ("soundfile", "音频 IO", "core"),
    ],
    "RPCA (自研)": [
        ("numpy", "核心线性代数", "core"),
        ("numpy.linalg", "SVD", "core"),
        ("scipy.signal", "stft/istft", "core"),
        ("soundfile", "音频 IO", "core"),
        ("librosa", "STFT", "core"),
    ],
    "RPCA+DRNN (自研)": [
        ("torch", "DRNN 训练/推理", "core"),
        ("numpy", "数组", "core"),
        ("numpy.random", "初始化/打乱", "core"),
        ("scipy.signal", "stft/istft + 中值滤波 + 形态学", "core"),
        ("scipy.ndimage", "形态学后处理", "core"),
        ("soundfile", "音频 IO", "core"),
    ],
    "评估指标 (全模型共用)": [
        ("museval", "BSSEval v4 SDR/ISR/SAR (MUSDB18 官方口径)", "core"),
        ("pystoi", "STOI", "optional"),
        ("mir_eval", "PESQ/其他", "optional"),
        ("pandas", "汇总表", "optional"),
        ("matplotlib", "绘图", "optional"),
    ],
}


# 函数级探测：顶层 import 能过、但真正调用时被拦的情况
FUNC_PROBES: list[tuple[str, str]] = [
    ("librosa.stft", "import librosa,numpy as np; librosa.stft(np.zeros(4410,dtype='float32'))"),
    ("librosa.effects.hpss", "import librosa,numpy as np; librosa.effects.hpss(np.zeros(4410,dtype='float32'))"),
    ("librosa.resample", "import librosa,numpy as np; librosa.resample(np.zeros(4410,dtype='float32'),orig_sr=44100,target_sr=8000)"),
    ("numpy.random.default_rng", "import numpy as np; np.random.default_rng(0)"),
    ("numpy.random.seed", "import numpy as np; np.random.seed(0)"),
    ("numpy.linalg.svd", "import numpy as np; np.linalg.svd(np.eye(4))"),
    ("scipy.signal.stft", "from scipy.signal import stft; import numpy as np; stft(np.zeros(4410))"),
    ("scipy.signal.medfilt", "from scipy.signal import medfilt; import numpy as np; medfilt(np.zeros(101))"),
    ("scipy.signal.resample", "from scipy.signal import resample; import numpy as np; resample(np.zeros(4410),800)"),
    ("scipy.ndimage.binary_opening", "from scipy import ndimage; import numpy as np; ndimage.binary_opening(np.zeros((5,5),bool))"),
    ("torch.stft", "import torch; torch.stft(torch.zeros(1,4410),2048,512,window=torch.hann_window(2048),return_complex=True)"),
    ("torchaudio.resample", "import torch,torchaudio; torchaudio.functional.resample(torch.zeros(1,4410),44100,8000)"),
    ("torch.save/load 往返", "import torch,io; b=io.BytesIO(); torch.save({'w':torch.zeros(4)},b); b.seek(0); torch.load(b,map_location='cpu',weights_only=True)"),
    ("museval.metrics.bss_eval", "import museval,numpy as np; museval.metrics.bss_eval(np.zeros(1001),np.zeros((1001,1)))"),
    ("soundfile.read", "import soundfile as sf; print(sf.__libsndfile_version__)"),
    ("matplotlib 后端", "import matplotlib; matplotlib.use('Agg'); from matplotlib import pyplot"),
]


def probe(mod: str) -> tuple[str, str]:
    try:
        importlib.import_module(mod)
        return "OK", ""
    except ModuleNotFoundError as e:
        return "MISSING", str(e)[:160]
    except Exception as e:  # noqa: BLE001
        msg = str(e)[:200]
        if any(h in msg for h in BLOCK_HINTS):
            return "BLOCKED", msg
        return "ERROR", f"{type(e).__name__}: {msg}"


def probe_func(code: str) -> tuple[str, str]:
    try:
        exec(code, {})  # noqa: S102
        return "OK", ""
    except ModuleNotFoundError as e:
        return "MISSING", str(e)[:160]
    except Exception as e:  # noqa: BLE001
        msg = str(e)[:200]
        if any(h in msg for h in BLOCK_HINTS):
            return "BLOCKED", msg
        return "ERROR", f"{type(e).__name__}: {msg}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"models": {}, "functional": []}
    tally = {"OK": 0, "MISSING": 0, "BLOCKED": 0, "ERROR": 0, "LOCAL": 0}

    for model, deps in MODEL_DEPS.items():
        print(f"\n=== {model} ===")
        recs = []
        for mod, purpose, sev in deps:
            if mod.startswith("#"):
                st, detail = "LOCAL", "仓库内模块，无需安装"
                tally["LOCAL"] += 1
                recs.append({"module": mod[1:], "purpose": purpose, "severity": sev,
                             "status": st, "detail": detail})
                print(f"  [仓库内] {sev:<8} {mod[1:]:<26} {purpose}")
                continue
            st, detail = probe(mod)
            tally[st] = tally.get(st, 0) + 1
            recs.append({"module": mod, "purpose": purpose, "severity": sev,
                         "status": st, "detail": detail})
            mark = {"OK": "OK  ", "MISSING": "缺包", "BLOCKED": "被拦", "ERROR": "错误"}[st]
            print(f"  [{mark}] {sev:<8} {mod:<26} {purpose}")
            if st != "OK" and detail:
                print(f"           -> {detail}")
        out["models"][model] = recs

    print("\n=== 函数级探测（顶层导入正常但调用时可能被拦） ===")
    for name, code in FUNC_PROBES:
        st, detail = probe_func(code)
        tally[st] = tally.get(st, 0) + 1
        out["functional"].append({"probe": name, "status": st, "detail": detail})
        mark = {"OK": "OK  ", "MISSING": "缺包", "BLOCKED": "被拦", "ERROR": "错误"}[st]
        print(f"  [{mark}] {name}")
        if st != "OK" and detail:
            print(f"           -> {detail}")

    out["tally"] = tally
    (OUT_DIR / "dependency_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("汇总:", tally)
    print(f"-> {OUT_DIR / 'dependency_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
