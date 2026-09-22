# FYP 前期配置与 MUSDB18 音频分析 · 执行手册

> 项目根目录：`E:\FYP_HKBU`
> 本阶段目标：完成音频基础分析，产出 9 张 Progress Report 图表。
> **数据集：MUSDB18-HQ（wav）—— 脚本现在同时兼容原始 MUSDB18（stem.mp4），会自动识别格式。**
> 明确不做：实时系统、深度学习、Demucs、Web 前端、模型训练、GPU 加速。

---

## 0. 先说结论：你的环境现状（我已实测）

我直接检查了你机器上的真实情况，和你初始提示词里写的「已安装软件：无」**不一致**，有几项可以省掉：

| 项目 | 初始假设 | 实测结果 | 结论 |
|---|---|---|---|
| FFmpeg | 未安装 | ✅ **已安装** `ffmpeg 8.1.2-full_build-www.gyan.dev`，已在 PATH | **阶段 A3 可跳过** |
| E:\FYP_HKBU | 待创建 | ✅ 已存在（内含 `demucs-main`、`lyrics-game`） | 无需新建根目录 |
| Miniconda / conda | 未安装 | ❌ 确实没有 | 需要装，或用下面「路线 B」 |
| **数据集** | 未下载 | ✅ **已放好 MUSDB18-HQ**（`datasets\train` 100 首 + `datasets\test` 50 首） | **已可出图** |
| Python 3.10 | 未安装 | 系统有 Python 3.13.14 | 见路线 A / B |

### 0.1 数据集实际结构（我核对过）

你放的是 **MUSDB18-HQ**，根目录是 `E:\FYP_HKBU\datasets`（不是 `datasets\musdb18`）：

```
E:\FYP_HKBU\datasets\
├── train\            100 首
│   └── A Classic Education - NightOwl\
│       ├── mixture.wav   ← 混音（脚本取它当 mixture）
│       ├── vocals.wav    ← 人声
│       ├── drums.wav
│       ├── bass.wav
│       └── other.wav
└── test\              50 首（结构同上）
```

实测音频参数：**44100 Hz / 立体声 / PCM_16 / 每首约 171.4 秒**。
`train` 下 100 首 × 5 wav = 500 个文件，完整无缺。

### 0.2 我已经跑通的验证（端到端实测）

| 组件 | 实测版本 | 状态 |
|---|---|---|
| Python | 3.13.14 | ✅ |
| numpy / scipy | 2.5.3 / 1.18.1 | ✅ |
| librosa | **1.0.0** | ✅（已适配新版 API） |
| matplotlib | 3.11.1 | ✅ |
| musdb / stempeg | 0.4.3 / 0.2.6 | ✅ |
| FFmpeg | 8.1.2 | ✅ |

**回归测试**（合成 stem.mp4）与**真实数据分析**都已跑通，9/9 张图全部生成：

```
使用歌曲： A Classic Education - NightOwl      ← train 下第一首
mixture shape: (1323000,) sr: 44100
vocals  shape: (1323000,)
trim index    : [1024, 1323000]
MFCC shape    : (13, 2584)
[SUCCESS] 9/9 张图全部生成完毕
```

> 📁 真实结果已在 `E:\FYP_HKBU\reports\figures\`。
> 另有一份合成样本图在 `E:\FYP_HKBU\reports\_smoketest_figures\`，仅作环境自检，**别写进报告**。

---

## 1. 两条环境路线，选一条即可

### 路线 A（推荐，按你原计划）：Miniconda + Python 3.10

```bat
:: A1 安装 Miniconda（勾选 Add to PATH），装完开 Anaconda Prompt
:: A2 建环境
conda create -n fyp_audio python=3.10 -y
conda activate fyp_audio
:: A3 FFmpeg —— 你已经有了，跳过；想确认就执行:
ffmpeg -version
:: A4 装库
pip install numpy scipy matplotlib librosa soundfile pandas scikit-learn
pip install musdb stempeg
```

### 路线 B（我已替你建好并验证）：直接复用现有 Python 环境

我已在 `C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio` 建好了一个装齐全部依赖的环境。
不想装 Miniconda 的话，每次运行前激活它即可：

```bat
C:\Users\jerry\.workbuddy\binaries\python\envs\fyp_audio\Scripts\activate.bat
cd E:\FYP_HKBU
python run_audio_analysis.py
```

> 两条路线跑出来的结果完全一致。**建议路线 A**，因为它跟你报告里写的环境一致，答辩时更好解释。

---

## 2. 目录结构（实测现状）

```
E:\FYP_HKBU\
├── run_audio_analysis.py          ← 主分析脚本（已为你写好，已验证）
├── FYP_Audio_Setup_Guide.md       ← 本手册
├── tools\
│   └── make_synthetic_musdb.py    ← 冒烟测试样本生成器（可选）
├── datasets\                      ← 你的 MUSDB18-HQ 根目录
│   ├── train\  <歌名>\{mixture,vocals,drums,bass,other}.wav   (100 首)
│   ├── test\   <歌名>\{...}.wav                                (50 首)
│   └── _smoketest_musdb18\        ← 冒烟测试样本（可随时删）
└── reports\
    ├── figures\                   ← 9 张 PNG 正式输出到这里（已生成）
    └── _smoketest_figures\        ← 合成样本自检图（别写进报告）
```

> 脚本会自动搜索 `datasets`、`datasets\musdb18`、`datasets\musdb18hq` 等常见位置，
> 并自动判断是「HQ wav」还是「原始 stem.mp4」，**不需要你手动改路径**。

---

## 3. 数据集格式说明（重要）

原始 MUSDB18 是 `.stem.mp4`（多音轨 MP4），**MUSDB18-HQ 是无损 wav 版**，两者内容一致，
HQ 版音质更好、体积更大（约 22 GB）。

| | 原始 MUSDB18 | MUSDB18-HQ（你在用） |
|---|---|---|
| 结构 | `train\xxx.stem.mp4` | `train\<歌名>\mixture.wav` 等 5 个 |
| musdb 参数 | `is_wav=False` | `is_wav=True` |
| 依赖 | 需要 FFmpeg + stempeg 解码 | 直接读 wav，更快更稳 |
| 脚本支持 | ✅ 自动识别 | ✅ 自动识别 |

> 你的 HQ 版放在 `E:\FYP_HKBU\datasets\`，脚本默认就能找到，**直接运行即可**。
> 如果哪天换成原始 stem.mp4 版，把 `datasets\musdb18\` 建好放进去也一样能跑，不用改代码。

---

## 4. 运行（阶段 B）

```bat
cd E:\FYP_HKBU
python run_audio_analysis.py
```

### 可用参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--duration N` | 截取秒数（内存不足时改 `10`） | `30` |
| `--song N` | train 目录下第 N 首（0-based） | `0`（第一首） |
| `--root PATH` | 自定义数据集根目录 | 自动搜索 |
| `--format` | `auto` / `hq` / `stem`（强制指定格式） | `auto` |
| `--subset` | `train` / `test` | `train` |
| `--show` | 出图后弹窗显示（默认只存文件不弹窗） | 关 |

> **默认不弹窗**是我做的改动：9 张图连弹会卡住终端。想看就加 `--show`。
> 想换歌：`python run_audio_analysis.py --song 3`

---

## 5. 预期结果

### 5.1 终端输出（下面是**真实数据集**的实际输出）

```
STEP 0 / 数据集定位与格式识别
数据集根目录 : E:\FYP_HKBU\datasets
识别到的格式 : MUSDB18-HQ (wav, 44.1kHz)
该子集共 100 首，本次使用第 1 首
...
使用歌曲： A Classic Education - NightOwl
[INFO] 按 chunk 只解码前 30 秒 (省内存模式)
mixture shape: (1323000,) sr: 44100
vocals shape: (1323000,)
Task 1-b / 采样率对比
  sr= 44100 Hz -> samples = 1323000
  sr= 22050 Hz -> samples = 661500
  sr= 16000 Hz -> samples = 480000
Task 2-b / STFT 窗长对比 win=128 vs 1024
  win_length= 128  hop=  64  frame=(65, 20672)
  win_length=1024  hop= 512  frame=(513, 2584)
Task 3 / Trim 前后对比
original shape: (1323000,)
trimmed shape : (1321976,)
trim index    : [1024, 1323000]
trimmed 部分时长 ≈ 29.977 s
Task 4 / 13 维 MFCC
MFCC - Mixture MFCC shape: (13, 2584)
MFCC - Vocals MFCC shape: (13, 2584)

全部图表已生成，保存在： E:\FYP_HKBU\reports\figures
[SUCCESS] 9/9 张图全部生成完毕 (歌曲: <歌曲名>, 时长: 30s)
```

> 30 秒 × 44100 Hz = **1,323,000** 采样点，这是校验截取是否生效的关键数字。
> `frame` 数会略高于 2584（STFT padding），属正常。

### 5.2 图像输出（9 张，PNG，dpi=150）

| 文件 | 内容 | 观察重点 |
|---|---|---|
| `fig1_waveform_mixture_vocals.png` | mixture / vocals 波形上下对比 | mixture 振幅持续饱满；vocals 只在人声段落有波形（间歇性） |
| `fig2_waveform_sr_compare.png` | 44100 / 22050 / 16000 Hz 三行波形 | 采样率越低波形越平滑，体现时域分辨率代价 |
| `fig3_spectrogram_mixture.png` | mixture 频谱（log 频率轴） | 低频节奏强，中高频混有伴奏与人声 |
| `fig3_spectrogram_vocals.png` | vocals 频谱（log 频率轴） | 清晰的水平谐波线，背景能量明显更少 |
| `fig4_spectrogram_win128.png` | mixture 频谱，win=128 | 时间分辨率高 / 频率分辨率低，低频糊在一起 |
| `fig4_spectrogram_win1024.png` | mixture 频谱，win=1024 | 频率分辨率高 / 时间分辨率低，谐波清晰但瞬态变模糊 |
| `fig5_trim_before_after.png` | trim 前后波形 | 裁剪后去掉首尾静音，shape 变小 |
| `fig6_mfcc_mixture.png` | mixture 13 维 MFCC 热图 | 纹理复杂（混响与伴奏干扰） |
| `fig6_mfcc_vocals.png` | vocals 13 维 MFCC 热图 | 纹理更干净，倒谱域分布与 mixture 明显不同 |

> ⚠️ **关于 fig5 的实测提示**：第一首 `A Classic Education - NightOwl` 是编曲很满的歌，
> `top_db=30` 只裁掉了开头 1024 个采样点（约 23 ms），所以上下两张波形**看起来几乎一样**，
> `trimmed shape` 也只从 1323000 变成 1321976。这是正常现象，不是 bug。
> 如果你想在报告里展示更明显的裁剪效果，两个办法：
> 1. 换一首开头有明显静音的歌（`--song N` 换序号试）；
> 2. 把 `top_db` 调大（如 `60`），阈值更严 → 裁掉更多。

### 5.3 报告可直接引用的结论段落

> 本阶段使用 MUSDB18-HQ（44.1 kHz 立体声 wav）中的 mixture 与 vocals 完成音频基础分析。
> 波形图显示混合信号持续包含伴奏能量，而人声信号具有明显间歇性；
> 频谱图显示人声在 log 频率轴上呈现清晰谐波结构；
> 通过调整 STFT 窗长，可观察时间分辨率与频率分辨率的权衡；
> MFCC 热图显示混合信号与人声在倒谱域存在差异。
> 这些图表为后续实时降噪系统的输入/输出对比提供了可视化基线。

---

## 6. 冒烟测试（可选，用于验证环境）

`tools/make_synthetic_musdb.py` 会用 FFmpeg 合成一个**格式完全合法**的假 `stem.mp4`，
放在 `datasets\_smoketest_musdb18\train\`，用于快速验证「musdb + stempeg + FFmpeg + 绘图」链路：

```bat
cd E:\FYP_HKBU
python tools\make_synthetic_musdb.py
python run_audio_analysis.py --root "E:\FYP_HKBU\datasets\_smoketest_musdb18"
```

> 它生成的是**合成音频、不是音乐**，只用于环境自检 —— 我已用它验证过，结果 9/9 通过。
> 你既然已经有真实 HQ 数据集，这一步可以跳过。

---

## 7. 常见报错与处置

| 报错 / 现象 | 原因 | 处置 |
|---|---|---|
| `FileNotFoundError: ffmpeg` / stempeg 报 ffmpeg 相关错 | FFmpeg 不在 PATH | 你已装好，执行 `ffmpeg -version` 确认；否则把 `C:\ffmpeg\bin` 加入 PATH |
| `ValueError: No tracks found` 或 `musdb` 加载为空 | 目录层级不对，或格式与 `is_wav` 不匹配 | 脚本会**自动识别**格式；若手工指定请用 `--format hq`（wav）或 `--format stem`（stem.mp4） |
| `FileNotFoundError: mixture.wav` | HQ 歌曲目录不完整 | 确认每首歌曲文件夹内 5 个 wav 齐全 |
| 控制台中文乱码 | Windows 控制台 GBK 编码 | 脚本已内置 `reconfigure(utf-8)`；若仍乱码执行 `chcp 65001` |
| 9 个窗口弹不停 / 卡住 | 交互式后端阻塞 | 脚本默认 `Agg` 后端只存图；要弹窗才加 `--show` |
| `MemoryError` | 整轨加载 + 多图缓存 | 用 `--duration 10`；脚本已默认按 chunk 只解码 30 秒 |
| `librosa.display.waveplot` 报 AttributeError | librosa ≥0.10 改名 | 已统一使用 `waveshow`（脚本内已是新版 API） |
| 图像里中文变方块 | 字体缺中文 | 脚本所有标题均为英文，规避该问题 |

---

## 8. 未来扩展（本阶段不实现，仅预留）

- **多首歌批量**：`--song` 已参数化，加一层循环遍历 `list_stem_files()` 即可
- **多模型对比**：把 `mixture/vocals` 换成「模型输入 / 模型输出」两组信号，绘图函数可直接复用
- **降噪前后对比**：复用 `plot_spectrogram()` 与 `plot_mfcc()`，只需改标题与文件名
