"""生成 11 个模型的开源 / 权重供给清单（JSON + Markdown 表格）。

- 静态部分：人工核实的开源状态、仓库、许可证、权重来源与官方大小/校验和
- 动态部分：扫描本机实际落盘的文件（存在与否 / 大小 / 是否匹配）

用法:
  python tools/model_provisioning_inventory.py
输出:
  outputs/comparison/model_provisioning.json
  outputs/comparison/model_provisioning.md
"""
from __future__ import annotations

import hashlib
import json
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


def human(n: int | None) -> str:
    if n is None:
        return "-"
    f = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if f < 1024:
            return f"{f:.1f}{u}"
        f /= 1024.0
    return f"{f:.1f}PB"


def md5_of(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 静态元数据
# status: available(权重已可得) / custom_weights(需自训) / no_weights(无需权重) /
#         reimplement(无官方代码，需自研) / unavailable(不可得)

MODELS = [
    dict(
        key="rpca", name="RPCA", family="古典 / 低秩稀疏分解",
        open_source="yes", repos=[],
        license="-", code_note="算法本身公开，本仓库自研实现 `tools/compare_separation_methods.py::_rpca_ialm`",
        weights_status="no_weights",
        weights=[], notes="IALM 求解；已知 P2 性能问题（每轮全秩 SVD）",
    ),
    dict(
        key="rpca_drnn", name="RPCA+DRNN", family="混合（古典 + 深度）",
        open_source="no", repos=[],
        license="-", code_note="**未公开代码**。Lai & Wang, EURASIP JASMP 2022:4, DOI 10.1186/s13636-022-00236-9（CC-BY 4.0），仅放出分离音频附件",
        weights_status="reimplement",
        weights=[],
        notes="原论文为**单声道 2 源**（人声/伴奏）歌唱分离，非 4-stem；须自研复现，且不可与 4-stem 方法直接横比",
    ),
    dict(
        key="conv_tasnet", name="Conv-TasNet", family="时域深度学习",
        open_source="yes",
        repos=[
            {"name": "kaituoxu/Conv-TasNet", "license": "MIT", "local": "third_party/Conv-TasNet"},
            {"name": "tky823/DNN-based_source_separation", "license": "NONE(无 LICENSE 文件)", "local": "third_party/DNN-based_source_separation"},
        ],
        license="MIT / 无许可证（见 repos）",
        code_note="两套独立实现；权重来自 tky823 仓库（可 `ConvTasNet.build_from_pretrained(task='musdb18')`）",
        weights_status="available",
        weights=[
            dict(name="Conv-TasNet MUSDB18 4sec_L20", path="DNN-based_source_separation/ConvTasNet/musdb18/sr44100/4sec_L20/model/vocals/best.pth",
                 source="Google Drive (tky823)", url="https://drive.google.com/uc?id=1A6dIofHZJQCUkyq-vxZ6KbPmEHLcf4WK",
                 size=None, md5=None, note="zip 内含 4 目标 × {best,last}.pth"),
            dict(name="Conv-TasNet MUSDB18 8sec_L20", path="DNN-based_source_separation/ConvTasNet/musdb18/sr44100/8sec_L20/model/vocals/best.pth",
                 source="Google Drive (tky823)", url="https://drive.google.com/uc?id=1C4uv2z0w1s4rudIMaErLyEccNprJQWSZ",
                 size=None, md5=None, note=""),
            dict(name="Conv-TasNet MUSDB18 8sec_L64", path="DNN-based_source_separation/ConvTasNet/musdb18/sr44100/8sec_L64/model/vocals/best.pth",
                 source="Google Drive (tky823)", url="https://drive.google.com/uc?id=1paXNGgH8m0kiJTQnn1WH-jEIurCKXwtw",
                 size=None, md5=None, note=""),
        ],
        notes="",
    ),
    dict(
        key="dprnn", name="DPRNN (DPRNN-TasNet)", family="时域深度学习",
        open_source="yes",
        repos=[
            {"name": "JusperLee/Dual-Path-RNN-Pytorch", "license": "Apache-2.0", "local": "third_party/Dual-Path-RNN-Pytorch"},
            {"name": "tky823/DNN-based_source_separation", "license": "NONE(无 LICENSE 文件)", "local": "third_party/DNN-based_source_separation"},
        ],
        license="Apache-2.0 / 无许可证（见 repos）",
        code_note="tky823 提供 `DPRNNTasNet`，但其 `pretrained_model_ids` **只有 wsj0-mix / librispeech，没有 musdb18**",
        weights_status="custom_weights",
        weights=[],
        notes="**无 MUSDB18 预训练权重** → 按既定方案自训 MUSDB18 版（有界预算 ≤3h）",
    ),
    dict(
        key="mmdenselstm", name="MMDenseLSTM", family="频域深度学习",
        open_source="yes",
        repos=[{"name": "tky823/DNN-based_source_separation", "license": "NONE(无 LICENSE 文件)", "local": "third_party/DNN-based_source_separation"}],
        license="无许可证（仓库无 LICENSE 文件）",
        code_note="官方论文（Takahashi et al.）未放代码，社区实现为 tky823 仓库；可 `MMDenseLSTM.build_from_pretrained(task='musdb18')`",
        weights_status="available",
        weights=[
            dict(name="MMDenseLSTM MUSDB18 (paper)", path="DNN-based_source_separation/MMDenseLSTM/musdb18/sr44100/paper/model/vocals/best.pth",
                 source="Google Drive (tky823)", url="https://drive.google.com/uc?id=1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV",
                 size=5813829, md5=None, note="4 目标 × {best,last}.pth，每份约 5.8MB"),
        ],
        notes="实测 1.37M 参数/目标（4 目标合计 5.49M）；内置配置 n_fft=4096 / hop_length=1024 / sections=[380,644,1025] / num_features=32 → 与 Takahashi et al. 2018 的 MMDenseLSTM 谱图设置一致；in_channels=2（立体声）",
    ),
    dict(
        key="bs_roformer_l12", name="BS-RoFormer L12", family="频域 Transformer（Band-Split RoPE）",
        open_source="yes",
        repos=[
            {"name": "lucidrains/BS-RoFormer", "license": "MIT", "local": "third_party/BS-RoFormer"},
            {"name": "ZFTurbo/Music-Source-Separation-Training", "license": "MIT", "local": "third_party/Music-Source-Separation-Training"},
        ],
        license="MIT",
        code_note="lucidrains 为架构实现；ZFTurbo 为训练/推理框架并携带 viperx 配置",
        weights_status="available",
        weights=[
            dict(name="BS-RoFormer ep_317 (SDR 12.9755)", path="BS-RoFormer/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                 source="GitHub TRvlvr/model_repo releases", url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                 size=639331213, md5=None, note="config dim=512 depth=12 dim_f=1024 → 本清单记为 L12"),
            dict(name="ep_317 配置", path="BS-RoFormer/configs/model_bs_roformer_ep_317_sdr_12.9755.yaml",
                 source="ZFTurbo/MSS-Training configs/viperx", url="(本地副本)", size=2006, md5=None, note=""),
        ],
        notes="",
    ),
    dict(
        key="bs_roformer_l6", name="BS-RoFormer L6", family="频域 Transformer（Band-Split RoPE）",
        open_source="yes",
        repos=[
            {"name": "lucidrains/BS-RoFormer", "license": "MIT", "local": "third_party/BS-RoFormer"},
            {"name": "ZFTurbo/Music-Source-Separation-Training", "license": "MIT", "local": "third_party/Music-Source-Separation-Training"},
        ],
        license="MIT",
        code_note="同 L12，权重与配置不同",
        weights_status="available",
        weights=[
            dict(name="BS-RoFormer ep_937 (SDR 10.5309)", path="BS-RoFormer/model_bs_roformer_ep_937_sdr_10.5309.ckpt",
                 source="GitHub TRvlvr/model_repo releases", url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt",
                 size=393068365, md5=None, note="config dim=384 depth=12 dim_f=1024 → 本清单记为 L6"),
            dict(name="ep_937 配置", path="BS-RoFormer/configs/model_bs_roformer_ep_937_sdr_10.5309.yaml",
                 source="ZFTurbo/MSS-Training configs/viperx", url="(本地副本)", size=2488, md5=None, note=""),
        ],
        notes="",
    ),
    dict(
        key="bsrnn", name="Band-Split RNN (BSRNN)", family="频域 RNN",
        open_source="yes",
        repos=[{"name": "magronp/bsrnn", "license": "MIT", "local": "third_party/bsrnn"}],
        license="MIT",
        code_note="作者官方实现（PyTorch Lightning + Hydra）；权重为作者自称 unofficial 的自训版本",
        weights_status="available",
        weights=[
            dict(name="bsrnn-opt", path="BSRNN/bsrnn-opt.zip",
                 source="Zenodo 17516442", url="https://zenodo.org/api/records/17516442/files/bsrnn-opt.zip/content",
                 size=1827060285, md5="89075aac776f82295a074a56d0428a18", note="优化版，4 个单源 checkpoint"),
            dict(name="bsrnn-large", path="BSRNN/bsrnn-large.zip",
                 source="Zenodo 17516442", url="https://zenodo.org/api/records/17516442/files/bsrnn-large.zip/content",
                 size=1627580581, md5="0d1e298e725bf45577a4f61e0a30c2b1", note="对齐原论文规模，4 个单源 checkpoint"),
            dict(name="simo-bsrnn-opt", path="BSRNN/simo-bsrnn-opt.zip",
                 source="Zenodo 17516442", url="https://zenodo.org/api/records/17516442/files/simo-bsrnn-opt.zip/content",
                 size=1213413202, md5="817e8ce4bcd17c38bd3812167e97db58", note="SIMO 变体，单 checkpoint 出 4 源（性能最好、最轻）"),
        ],
        notes="",
    ),
    dict(
        key="irm_ibm_oracle", name="IRM / IBM oracle", family="oracle 掩码（上界参考）",
        open_source="yes",
        repos=[{"name": "sigsep/sigsep-mus-oracle", "license": "MIT", "local": "third_party/sigsep-mus-oracle"}],
        license="MIT",
        code_note="提供 IBM.py / IRM.py / MWF.py / MIX.py / GT.py；本仓库亦有自研解析实现（`benchmark_model_universal.py::_oracle_irm/_oracle_ibm`）",
        weights_status="no_weights",
        weights=[],
        notes="用户原述 “IRN oracle” 经澄清为 IRM/IBM 笔误；oracle 由 GT 直接计算掩码，无需任何权重",
    ),
    dict(
        key="mdx_net", name="MDX-Net", family="混合时频双流网络",
        open_source="yes",
        repos=[
            {"name": "kuielab/mdx-net", "license": "MIT", "local": "third_party/mdx-net"},
            {"name": "mdx-net-submission", "license": "MIT", "local": "third_party/mdx-net-submission"},
        ],
        license="MIT",
        code_note="官方训练代码；onnx 权重来自 Zenodo；另可经 Demucs 的 `mdx_extra` bag 直接跑 4 模型集成",
        weights_status="available",
        weights=[
            dict(name="KUIELab-MDX-Net Track A (onnx_A.zip)", path="MDX-Net/onnx_A.zip",
                 source="Zenodo 5717356", url="https://zenodo.org/api/records/5717356/files/onnx_A.zip/content",
                 size=110359032, md5="72605d3722ebef18c353f831abdbfc07", note="MDX Challenge 2021 Track A"),
            dict(name="mixer.ckpt", path="MDX-Net/mixer.ckpt",
                 source="Zenodo 5717356", url="https://zenodo.org/api/records/5717356/files/mixer.ckpt/content",
                 size=1208, md5="485113a7e20a0dc5ccc47e7645464aed", note="两流混合权重"),
            dict(name="mdx_extra/e51eebcc", path="MDX-Net/mdx_extra/e51eebcc-c1b80bdd.th",
                 source="dl.fbaipublicfiles.com (Demucs mdx_final)", url="https://dl.fbaipublicfiles.com/demucs/mdx_final/e51eebcc-c1b80bdd.th",
                 size=None, md5=None, note="Demucs mdx_extra bag 成员"),
            dict(name="mdx_extra/a1d90b5c", path="MDX-Net/mdx_extra/a1d90b5c-ae9d2452.th",
                 source="dl.fbaipublicfiles.com (Demucs mdx_final)", url="https://dl.fbaipublicfiles.com/demucs/mdx_final/a1d90b5c-ae9d2452.th",
                 size=None, md5=None, note=""),
            dict(name="mdx_extra/5d2d6c55", path="MDX-Net/mdx_extra/5d2d6c55-db83574e.th",
                 source="dl.fbaipublicfiles.com (Demucs mdx_final)", url="https://dl.fbaipublicfiles.com/demucs/mdx_final/5d2d6c55-db83574e.th",
                 size=None, md5=None, note=""),
            dict(name="mdx_extra/cfa93e08", path="MDX-Net/mdx_extra/cfa93e08-61801ae1.th",
                 source="dl.fbaipublicfiles.com (Demucs mdx_final)", url="https://dl.fbaipublicfiles.com/demucs/mdx_final/cfa93e08-61801ae1.th",
                 size=None, md5=None, note=""),
        ],
        notes="onnx 路线需额外安装 onnxruntime（当前环境未装）",
    ),
    dict(
        key="pac_hubert_sep", name="Pac-HuBERT-SEP", family="自监督预训练 + Res-U-Net",
        open_source="no", repos=[],
        license="-",
        code_note="**无公开代码/权重**。Ke Chen, Gordon Wichern, François Germain, Jonathan Le Roux (MERL) + UCSD, arXiv:2304.02160；MERL 官方下载页未列出该模型",
        weights_status="unavailable",
        weights=[],
        notes="用户原述 “PaG-HuBERT-SEP” 经确认为 Pac-HuBERT 笔误。不可得 → 只能引用论文数字，或在报告中明确标注为不可复现",
    ),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 若已有可加载性报告，则把实测参数量附加进来
    load_map: dict[str, dict] = {}
    load_json = OUT_DIR / "weight_loadability.json"
    if load_json.is_file():
        try:
            _l = json.loads(load_json.read_text(encoding="utf-8"))
            for r in _l.get("torch_files", []):
                if r.get("ok"):
                    load_map[r["file"]] = r
        except Exception:
            pass

    report = {"weights_root": str(WEIGHTS), "models": []}
    for m in MODELS:
        entry = dict(m)
        wl = []
        for w in m["weights"]:
            p = WEIGHTS / w["path"]
            rec = dict(w)
            rec["local_path"] = str(p)
            if p.exists():
                actual = p.stat().st_size
                rec["actual_size"] = actual
                rec["present"] = True
                if w["size"] is not None:
                    rec["size_match"] = (actual == w["size"])
                else:
                    rec["size_match"] = None
                if w["md5"]:
                    rec["md5_actual"] = md5_of(p)
                    rec["md5_match"] = (rec["md5_actual"] == w["md5"])
                else:
                    rec["md5_actual"] = None
                    rec["md5_match"] = None
            else:
                rec["present"] = False
                rec["actual_size"] = None
                rec["size_match"] = None
                rec["md5_actual"] = None
                rec["md5_match"] = None

            # 附上实测参数量 / 是否可被 torch 反序列化
            lr = load_map.get(str(p))
            if lr:
                st = lr.get("structure", {})
                rec["loadable"] = True
                rec["params"] = st.get("state_dict_n_params")
            else:
                rec["loadable"] = False if rec["present"] else None
                rec["params"] = None
            wl.append(rec)
        entry["weights"] = wl

        # 汇总结论
        if not wl:
            entry["provision_state"] = m["weights_status"]
        else:
            n_ok = sum(1 for r in wl if r["present"] and (r["size_match"] in (True, None)) and (r["md5_match"] in (True, None)))
            entry["n_weights"] = len(wl)
            entry["n_weights_ok"] = n_ok
            entry["provision_state"] = "downloaded" if n_ok == len(wl) else f"partial({n_ok}/{len(wl)})"
        report["models"].append(entry)

    (OUT_DIR / "model_provisioning.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- Markdown 摘要 ----
    lines = ["| # | 模型 | 开源 | 许可证 | 权重可得性 | 落盘状态 |", "|---|---|---|---|---|---|"]
    for i, e in enumerate(report["models"], 1):
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            i, e["name"], e["open_source"], e["license"], e["weights_status"], e["provision_state"]))
    lines.append("")
    lines.append("### 权重文件明细")
    lines.append("")
    lines.append("| 模型 | 权重 | 期望大小 | 实际大小 | 大小 | md5 | torch 可加载 | 参数量 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in report["models"]:
        for r in e["weights"]:
            sm = {True: "OK", False: "MISMATCH", None: "-"}[r["size_match"]]
            mm = {True: "OK", False: "MISMATCH", None: "-"}[r["md5_match"]]
            ld = {True: "OK", False: "FAIL", None: "-"}[r.get("loadable")]
            pp = f"{r['params']/1e6:.2f}M" if isinstance(r.get("params"), int) and r["params"] else "-"
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                e["name"], r["name"], human(r["size"]), human(r["actual_size"]), sm, mm, ld, pp))
    (OUT_DIR / "model_provisioning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 控制台 ----
    for e in report["models"]:
        print("{:<22} {:<10} {:<14} {}".format(e["name"], e["open_source"], e["weights_status"], e["provision_state"]))
    print()
    print("-> outputs/comparison/model_provisioning.json")
    print("-> outputs/comparison/model_provisioning.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
