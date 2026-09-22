# -*- coding: utf-8 -*-
"""把产出图表压缩后以 base64 内联进 HTML 模板, 使其完全自包含。

用法: python tools/_embed_images.py
输入: reports/METHOD_VISUAL_COMPARISON.html  (含 {{IMG_xxx}} 占位符)
输出: 同名文件, 占位符替换为 data:image/jpeg;base64,...
"""
import base64
import io
import re
from pathlib import Path

from PIL import Image

# ---- 路径中枢：五大模块布局，见 tools/paths.py ----
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths as _paths
_paths.setup_env()
ROOT = _paths.PROJECT_ROOT
HTML_OUT = _paths.HTML / "METHOD_VISUAL_COMPARISON.html"
BASE_DIR = _paths.FIGURES / "audio_analysis"
DEM_DIR = _paths.FIGURES / "demucs"
MAX_W = 1100
QUALITY = 80

# 编号 -> 文件名（baseline 与 demucs 两套同名）
FILES = {
    1: "fig1_waveform_mixture_vocals.png",
    2: "fig2_waveform_sr_compare.png",
    3: "fig3_spectrogram_mixture.png",
    4: "fig3_spectrogram_vocals.png",
    5: "fig4_spectrogram_win128.png",
    6: "fig4_spectrogram_win1024.png",
    7: "fig5_trim_before_after.png",
    8: "fig6_mfcc_mixture.png",
    9: "fig6_mfcc_vocals.png",
}


def datauri(p: Path) -> str:
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if w > MAX_W:
        im = im.resize((MAX_W, int(h * MAX_W / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=QUALITY, optimize=True)
    b = buf.getvalue()
    print(f"  {p.name:<38} {len(b)//1024:>4} KB")
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def main():
    html = HTML_OUT.read_text(encoding="utf-8")
    mapping = {"{{IMG_DSDR}}": datauri(_paths.FIGURES_COMPARISON / "fig_compare_dsdr.png")}
    for i, f in FILES.items():
        mapping["{{IMG_B" + str(i) + "}}"] = datauri(BASE_DIR / f)
        mapping["{{IMG_D" + str(i) + "}}"] = datauri(DEM_DIR / f)

    for k, v in mapping.items():
        html = html.replace(k, v)

    left = re.findall(r"\{\{IMG_[A-Z0-9_]+\}\}", html)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"\n替换 {len(mapping)} 张图")
    print("残留占位符:", left if left else "无 ✓")
    print(f"输出文件大小: {len(html)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
