"""Build the offline A/B listening page for three MUSDB18-HQ test songs.

Scans the locally produced stems under 03_outputs/_test_run/<Model>/<Song>/,
parses each file's真实音频参数 (FLAC STREAMINFO / WAV fmt chunk -- no decoder
needed, so Smart App Control cannot break the build), merges the per-song
metrics from test_run_per_song.csv, and emits one self-contained HTML page
that references the audio by relative path.

Deliverables
    04_reports/separation/html/listen_compare.html      <- the page
    04_reports/separation/data/comparison/listen_manifest.json

Everything is local: no network data is fetched or referenced.
"""
from __future__ import annotations

import csv
import json
import os
import struct
import sys
from urllib.parse import quote

ROOT = r"E:\FYP_HKBU"
OUT_DIR = os.path.join(ROOT, "03_outputs", "_test_run")
DB_DIR = os.path.join(ROOT, "02_databases", "MUSDB18-HQ", "test")
OUT_HTML = os.path.join(ROOT, "04_reports", "separation", "html", "listen_compare.html")
OUT_JSON = os.path.join(ROOT, "04_reports", "separation", "data", "comparison", "listen_manifest.json")
CSV_PATH = os.path.join(ROOT, "04_reports", "separation", "data", "comparison", "test_run_per_song.csv")

# ---------------------------------------------------------------- songs ----
SONGS = [
    {
        "key": "Georgia Wonder - Siren",
        "why": "指定样本 1 · 50 首里最长的曲目（430.4 s），四位四轨模型在此拉开最大差距",
    },
    {
        "key": "PR - Oh No",
        "why": "指定样本 2 · 50 首里最短的曲目（76.2 s），几乎全部模型都在这里出现退化",
    },
    {
        "key": "M.E.R.C. Music - Knockout",
        "why": "随机取中位 · 按 50 首单曲平均墙钟排序落在第 26/50 位（正中间），即“歌与歌差距”的中位样本，270.0 s",
    },
]

# --------------------------------------------------------------- models ----
STEM_CN = {"vocals": "人声", "drums": "鼓", "bass": "贝斯",
           "other": "其他乐器", "vocals+other": "人声+其他（单总线）"}

# key, folder, display label, group, filename per target, note
MODELS = [
    ("oracle", "Oracle-IRM", "Oracle-IRM", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "理想掩码上界，不是可部署模型"),
    ("mdx", "MDX-Net", "MDX-Net", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "四轨同台里“快 + 准”折中最优"),
    ("demucs", "Demucs", "Demucs", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "波形域 U-Net"),
    ("umx", "Open-Unmix", "Open-Unmix", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "频谱域基线"),
    ("convtasnet", "Conv-TasNet", "Conv-TasNet", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "端到端时域分离"),
    ("mmdenselstm", "MMDenseLSTM", "MMDenseLSTM", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "多通道 LSTM 系"),
    ("dprnn", "DPRNN", "DPRNN（自训）", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "自训权重，SDR 为负，仅作负面证据"),
    ("bsrnn_simo", "BSRNN-SIMO", "BSRNN-SIMO", "G4",
     {"vocals": "vocals.flac", "drums": "drums.flac", "bass": "bass.flac", "other": "other.flac"},
     "全量 SDR 最高，但本机 180 s 超时上限使这 3 首未产出"),
    ("bsroformer_l6", "BS-RoFormer-L6", "BS-RoFormer-L6", "G2",
     {"vocals+other": "vocals+other.flac"},
     "只输出人声+其他总线"),
    ("bsroformer_l12", "BS-RoFormer-L12", "BS-RoFormer-L12", "G1",
     {"vocals": "vocals.flac"},
     "人声单轨 SDR 最强的模型"),
    ("bsrnn", "BSRNN-opt", "BSRNN-opt", "G1",
     {"vocals": "vocals.flac"},
     "人声单轨"),
    ("bsrnn_large", "BSRNN-large", "BSRNN-large", "G1",
     {"vocals": "vocals.flac"},
     "人声单轨，本机单首耗时最低"),
]

GROUPS = [
    ("G4", "四轨全分离", "4 条轨可同时播放 → 叠出来的就是该模型重建的整曲，可直接与源混音对听。",
     "#2563eb"),
    ("G2", "双总线输出", "只给出 1 条总线，其余成分未建模。", "#7c3aed"),
    ("G1", "仅人声单轨", "只出人声。SDR 只评 1 条轨，不能与四轨模型的四轨均值直接比较。", "#b45309"),
]

# ------------------------------------------------------- audio plumbing ----
def probe(path):
    """Return (sr, ch, frames, subtype) from the container header only."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
            if head[:4] == b"fLaC":
                f.seek(4)
                while True:
                    blk = f.read(4)
                    if len(blk) < 4:
                        return None
                    _, btype, size = blk[0] >> 7, blk[0] & 0x7F, int.from_bytes(blk[1:4], "big")
                    if btype == 0:                      # STREAMINFO
                        si = f.read(size)
                        bits = int.from_bytes(si[10:18], "big")
                        sr = (bits >> 44) & 0xFFFFF
                        ch = ((bits >> 41) & 0x7) + 1
                        bps = ((bits >> 36) & 0x1F) + 1
                        frames = bits & 0xFFFFFFFFF
                        return sr, ch, frames, "PCM_%d" % bps
                    if btype == 127:
                        return None
                    f.seek(size, os.SEEK_CUR)
            if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
                f.seek(12)
                sr = ch = bps = None
                while True:
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        return None
                    cid, csz = hdr[:4], int.from_bytes(hdr[4:8], "little")
                    if cid == b"fmt ":
                        fmt = f.read(csz)
                        ch, sr = struct.unpack_from("<HI", fmt, 2)
                        bps = struct.unpack_from("<H", fmt, 14)[0]
                        f.seek(csz % 2, os.SEEK_CUR)
                    elif cid == b"data":
                        frames = csz // max(1, (ch or 1) * (bps or 2) // 8) if csz else 0
                        return sr, ch, frames, "PCM_%d" % bps
                    else:
                        f.seek(csz + csz % 2, os.SEEK_CUR)
    except OSError:
        return None
    return None


def rel_url(abs_path):
    rel = os.path.relpath(abs_path, os.path.join(ROOT, "04_reports", "separation", "html"))
    return quote(rel.replace("\\", "/"), safe="/+")


def track_rec(abs_path, target, label, sdr=None, extra=None):
    if not os.path.isfile(abs_path):
        return None
    p = probe(abs_path)
    mb = round(os.path.getsize(abs_path) / 1e6, 1)
    sr, ch, frames, sub = p if p else (None, None, None, None)
    rec = {
        "target": target,
        "label": label,
        "src": rel_url(abs_path),
        "mb": mb,
        "sr": sr,
        "ch": ch,
        "dur": round(frames / sr, 2) if (frames and sr) else None,
        "sub": sub,
        "sdr": sdr,
    }
    if extra:
        rec.update(extra)
    return rec


# -------------------------------------------------------------- metrics ----
def load_metrics():
    out = {}
    with open(CSV_PATH, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[(r["song"], r["model_key"])] = r
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build():
    metrics = load_metrics()
    songs = []
    problems = []

    for s in SONGS:
        key = s["key"]
        src_dir = os.path.join(DB_DIR, key)
        song = {"key": key, "why": s["why"], "slug": key.lower().replace(" ", "-"),
                "ref": [], "models": [], "dur_s": None, "missing": []}

        mix = os.path.join(src_dir, "mixture.wav")
        mrec = track_rec(mix, "mixture", "源混音 mixture（原始输入）")
        if mrec is None:
            problems.append("missing mixture: %s" % mix)
        else:
            song["dur_s"] = mrec["dur"]
            song["ref"].append(dict(mrec, role="mixture"))

        for tgt in ("vocals", "drums", "bass", "other"):
            p = os.path.join(src_dir, tgt + ".wav")
            r = track_rec(p, tgt, "GT " + STEM_CN[tgt], extra={"role": "gt"})
            if r is None:
                problems.append("missing GT: %s" % p)
            else:
                song["ref"].append(r)

        mix_frames = probe(mix)[2] if os.path.isfile(mix) else None
        excluded = []
        for mk, folder, label, grp, files, note in MODELS:
            row = metrics.get((key, mk))
            led_status = (row or {}).get("ledger_status")
            rec = {
                "key": mk, "label": label, "grp": grp, "note": note,
                "dir": os.path.join(OUT_DIR, folder, key).replace("\\", "/"),
                "tracks": [], "chips": {}, "status": "PASS", "reason": "",
            }
            tracks = []
            why = ""
            if not row or row.get("status") != "PASS":
                why = {"TIMEOUT": "180 s 超时上限中断，半成品已丢弃",
                       "FAIL": "运行失败"}.get(row and row.get("status"),
                                             "被超时熔断禁用，本曲未运行")
            else:
                for tgt, fn in files.items():
                    p = os.path.join(OUT_DIR, folder, key, fn)
                    sdr = (num(row.get("sdr_" + tgt)) if "+" not in tgt
                           else num(row.get("sdr_mean")))
                    r = track_rec(p, tgt, STEM_CN.get(tgt, tgt), sdr=sdr)
                    if r is None:
                        why = "产出文件缺失"
                        break
                    if mix_frames and (not r["dur"] or r["dur"] * r["sr"] < mix_frames * 0.999):
                        why = "文件不完整（长度不足源混音的 99.9%，超时残留）"
                        break
                    tracks.append(r)
                if not tracks and not why:
                    why = "产出为空"
                n_exp = num(row.get("n_tracks"))
                if tracks and n_exp and len(tracks) != int(n_exp):
                    why = "产出轨数与记录不符（%d/%d）" % (len(tracks), int(n_exp))

            if why:
                # a half-written directory is the single most misleading state
                # this sweep can produce: report the leftovers explicitly.
                d = os.path.join(OUT_DIR, folder, key)
                left = []
                if os.path.isdir(d):
                    for fn in sorted(os.listdir(d)):
                        fp = os.path.join(d, fn)
                        if os.path.isfile(fp):
                            left.append("%s %.1fMB" % (fn, os.path.getsize(fp) / 1e6))
                if left:
                    why += "；目录内残留 %d 个未完成文件（%s），已忽略" % (len(left), "、".join(left))
                rec["status"] = "MISSING"
                rec["reason"] = why
                song["missing"].append(label)
                excluded.append("%s：%s" % (label, why))
            rec["tracks"] = tracks
            if row:
                rec["chips"] = {k: num(row.get(k)) for k in
                                ("wall_s", "infer_s", "overhead_s", "out_mb",
                                 "sdr_mean", "sdr_vocals", "sdr_drums",
                                 "sdr_bass", "sdr_other")}
                rec["chips"]["ledger_status"] = led_status
            song["models"].append(rec)
        song["excluded"] = excluded

        # order inside a group: better SDR first, missing last
        song["models"].sort(key=lambda m: (m["grp"], -(m["chips"].get("sdr_mean") or -99)))
        songs.append(song)

    # merge the audio QC produced by _verify_listen_audio.py (if it has run)
    qc_path = os.path.join(ROOT, "04_reports", "separation", "data", "comparison", "listen_audio_qc.json")
    if os.path.isfile(qc_path):
        with open(qc_path, encoding="utf-8") as fh:
            qc = json.load(fh)
        for song in songs:
            buckets = [song["ref"]] + [m["tracks"] for m in song["models"]]
            for tracks in buckets:
                for t in tracks:
                    q = qc.get(t["src"])
                    if q and q.get("ok"):
                        t["peak"] = q["peak"]
                        t["rms"] = q["rms"]
                        frac = q["clipped"] / max(1.0, q["dur"] * q["sr"])
                        t["clip_frac"] = round(frac, 9)
                        # >0.001% of samples pinned at full scale = audible
                        t["clipped"] = 1 if frac > 1e-5 else 0

    db = {"root": ROOT, "generated_by": "_build_listen_page.py",
          "songs": songs, "groups": [{"id": g, "title": t, "note": n, "color": c}
                                     for g, t, n, c in GROUPS],
          "problems": problems}
    return db


# ----------------------------------------------------------------- html ----
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MUSDB18 test 三首样本 · 逐模型人耳聆听对比</title>
<style>
:root{
  --bg:#eef1f6; --panel:#fff; --ink:#0f172a; --muted:#64748b; --line:#dde3ec;
  --accent:#2563eb; --g4:#2563eb; --g2:#7c3aed; --g1:#b45309; --ref:#0f766e;
  --warn:#b91c1c; --ok:#15803d; --soft:#f3f6fb;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
a{color:var(--accent)}
header.top{background:linear-gradient(120deg,#101a2e,#1e3a8a 60%,#0f766e);color:#fff;
  padding:26px 30px 22px}
header.top h1{margin:0 0 6px;font-size:23px;letter-spacing:.3px}
header.top p{margin:4px 0 0;color:#c7d2e4;font-size:13px;max-width:1000px}
header.top .kpis{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
header.top .kpi{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);
  border-radius:10px;padding:7px 12px;font-size:12.5px}
header.top .kpi b{font-size:15px;display:block;line-height:1.3}
.wrap{max-width:1200px;margin:0 auto;padding:18px 22px 80px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 6px}
.tab{border:1px solid var(--line);background:var(--panel);border-radius:999px;
  padding:8px 16px;cursor:pointer;font-size:13.5px;color:var(--ink);transition:.15s}
.tab:hover{border-color:#c3cddd}
.tab.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.tab small{opacity:.7;margin-left:6px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:16px 18px;margin:12px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}
h2.sec{font-size:17px;margin:22px 0 4px}
p.lead{margin:4px 0 10px;color:var(--muted);font-size:13.5px}
.note{background:#f8fafc;border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
  padding:10px 14px;margin:10px 0;font-size:13px;color:#334155}
.note.warn{background:#fef2f2;border-color:var(--warn)}
.note.ok{background:#f0fdf4;border-color:var(--ok)}
.note ul{margin:6px 0 0;padding-left:18px}
.note li{margin:3px 0}
details.help{border:1px solid var(--line);border-radius:12px;background:var(--panel);
  padding:10px 16px;margin:12px 0}
details.help>summary{cursor:pointer;font-weight:600;font-size:14px}
kbd{background:#f1f5f9;border:1px solid var(--line);border-bottom-width:2px;
  border-radius:5px;padding:1px 6px;font-size:12px;font-family:ui-monospace,Consolas,monospace}
.badge{display:inline-block;border-radius:6px;padding:1px 7px;font-size:11.5px;
  vertical-align:middle;margin-left:6px;border:1px solid transparent}
.b-mix{background:#ccfbf1;color:#0f766e;border-color:#99f6e4}
.b-gt{background:#e0f2fe;color:#075985;border-color:#bae6fd}
.b-oracle{background:#fef9c3;color:#854d0e;border-color:#fde68a}
.b-warn{background:#fee2e2;color:#991b1b;border-color:#fecaca}
.b-plain{background:var(--soft);color:var(--muted);border-color:var(--line)}

/* ---------- transport ---------- */
.transport{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:10px;
  flex-wrap:wrap;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);
  border:1px solid var(--line);border-radius:12px;padding:9px 12px;margin:10px 0 14px;
  box-shadow:0 4px 14px rgba(15,23,42,.07)}
.tbtn{border:1px solid var(--line);background:var(--panel);border-radius:9px;
  padding:6px 12px;cursor:pointer;font-size:13px}
.tbtn:hover{border-color:#c3cddd}
.tbtn.primary{background:var(--accent);color:#fff;border-color:var(--accent);min-width:92px}
.tbtn.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.seek{flex:1;min-width:190px;height:22px;display:flex;align-items:center;cursor:pointer}
.seek .rail{position:relative;width:100%;height:6px;border-radius:3px;background:#e2e8f0}
.seek .rail i{position:absolute;left:0;top:0;bottom:0;width:0;border-radius:3px;background:var(--accent)}
.seek .rail b{position:absolute;top:-4px;width:3px;height:14px;border-radius:2px;background:#0f172a;
  transform:translateX(-1.5px)}
.time{font:12.5px ui-monospace,Consolas,monospace;color:var(--muted);min-width:96px;text-align:center}

/* ---------- model card ---------- */
.mgroup{margin:16px 0 6px;display:flex;align-items:baseline;gap:10px}
.mgroup .dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.mgroup h3{margin:0;font-size:15px}
.mgroup .gn{color:var(--muted);font-size:12.5px}
.mcard{border:1px solid var(--line);border-left-width:4px;border-radius:12px;
  background:var(--panel);margin:10px 0;overflow:hidden}
.mcard.g4{border-left-color:var(--g4)} .mcard.g2{border-left-color:var(--g2)}
.mcard.g1{border-left-color:var(--g1)} .mcard.missing{border-left-color:#cbd5e1;opacity:.72}
.mhead{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 14px;
  background:var(--soft);border-bottom:1px solid var(--line)}
.idx{width:24px;height:24px;border-radius:7px;background:var(--ink);color:#fff;
  display:grid;place-items:center;font-size:12.5px;font-weight:700}
.realname{font-weight:700;font-size:14.5px}
.codename{font-weight:700;font-size:14.5px;display:none;font-family:ui-monospace,Consolas,monospace;
  color:#0f766e}
body.blind .realname{display:none} body.blind .codename{display:inline}
body.blind .chip.sdr,body.blind .chip.sdrmean{visibility:hidden}
body.blind .macts{visibility:hidden}
.chip{background:#fff;border:1px solid var(--line);border-radius:6px;padding:1px 7px;
  font-size:11.5px;color:#475569}
.chip.sdr{background:#eef2ff;border-color:#c7d2fe;color:#3730a3}
.chip.sdrmean{background:#1e293b;border-color:#1e293b;color:#fff;font-weight:700}
.macts{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.act{border:1px solid var(--line);background:#fff;border-radius:8px;padding:4px 10px;
  font-size:12.3px;cursor:pointer}
.act:hover{border-color:#93c5fd;background:#eff6ff}
.mnote{padding:4px 14px 0;font-size:12.3px;color:var(--muted)}

/* ---------- track rows ---------- */
.row{display:flex;align-items:center;gap:10px;padding:7px 14px;border-top:1px solid #f1f5f9}
.row:hover{background:#fafcff}
.row.playing{background:#eff6ff}
.row .pl{width:30px;height:30px;border-radius:8px;border:1px solid var(--line);background:#fff;
  cursor:pointer;font-size:12px;display:grid;place-items:center;flex:0 0 auto}
.row .pl:hover{border-color:#93c5fd}
.row.on .pl{background:var(--accent);color:#fff;border-color:var(--accent)}
.row .nm{min-width:196px;font-size:13.2px}
.row .nm .sub{display:block;color:var(--muted);font-size:11.3px;line-height:1.35}
.row .bar{flex:1;height:20px;display:flex;align-items:center;cursor:pointer;min-width:110px}
.row .bar .rail{position:relative;width:100%;height:5px;border-radius:3px;background:#e6ebf3}
.row .bar .rail i{position:absolute;inset:0 auto 0 0;width:0;border-radius:3px;background:#60a5fa}
.row.on .bar .rail i{background:var(--accent)}
.row .tm{font:11.5px ui-monospace,Consolas,monospace;color:var(--muted);min-width:84px;
  text-align:right}
.row input[type=range]{width:74px;accent-color:var(--accent)}
.row .solo{border:1px solid var(--line);background:#fff;border-radius:7px;padding:3px 8px;
  font-size:11.5px;cursor:pointer}
.row.on .solo{background:#0f766e;color:#fff;border-color:#0f766e}
.derived{color:#b45309}

/* ---------- blind table ---------- */
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left}
th{background:var(--soft);font-weight:600;color:#334155;position:relative}
tbody tr:hover{background:#fafcff}
.hidden{display:none!important}
footer{color:var(--muted);font-size:12.3px;margin-top:26px;text-align:center}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);background:#0f172a;
  color:#fff;padding:9px 16px;border-radius:9px;font-size:13px;opacity:0;pointer-events:none;
  transition:.25s;z-index:99}
#toast.show{opacity:1}
</style>
</head>
<body>
<header class="top">
  <h1>MUSDB18-HQ test · 三首样本逐模型聆听对比</h1>
  <p>每首歌都提供<b>源混音</b>、<b>官方 GT 四轨</b>与<b>每个模型在本机的全部产出</b>。
     音频全部来自本机磁盘，未使用任何网络数据。播放器支持任意拖动定位、
     多轨同时播放（叠听）与盲听模式 —— 用耳朵判断，而不是只看分数。</p>
  <div class="kpis" id="kpis"></div>
</header>

<div class="wrap">

<details class="help" open>
  <summary>怎么用这一页（30 秒读完）</summary>
  <div class="note">
    <ul>
      <li><b>切歌</b>：上方标签页。切歌会停止所有播放。</li>
      <li><b>单轨试听</b>：点某一行左侧的 ▶，只有它出声，并且<b>从当前播放头继续</b>——
          所以连着点几行，听到的是同一秒的不同模型，这才是公平的 A/B。</li>
      <li><b>叠听</b>：打开工具栏的「叠听」后，可以同时点亮多行，它们会自动对齐着一起播。
          把某个模型的 4 条轨全点亮，听到的就是<b>它重建的整曲</b>，直接和「源混音」比对。
          把「GT 四轨」全点亮，应该和源混音几乎一模一样（这是校准用）。</li>
      <li><b>音量</b>：每行独立。四轨叠听时若明显比源混音轻/闷，说明该模型有能量泄漏或相位问题。</li>
      <li><b>快捷键</b>：<kbd>空格</kbd> 播放/暂停 · <kbd>←</kbd><kbd>→</kbd> 前后 5 秒 ·
          <kbd>1</kbd>–<kbd>9</kbd> 在「叠听」下点亮第 N 个模型 · <kbd>0</kbd> 全停。</li>
      <li><b>盲听</b>：工具栏「盲听」会把模型名换成随机代号并打乱顺序，听完再点「揭晓对照表」。</li>
    </ul>
  </div>
</details>

<div class="tabs" id="tabs"></div>
<div class="transport">
  <button class="tbtn primary" id="pp">▶ 播放</button>
  <button class="tbtn" id="back">-5s</button>
  <button class="tbtn" id="fwd">+5s</button>
  <button class="tbtn" id="stop">■ 全停</button>
  <div class="seek" id="seek"><div class="rail"><i></i><b></b></div></div>
  <span class="time" id="time">0:00 / 0:00</span>
  <button class="tbtn" id="loop" title="播完自动回到开头">↻ 循环</button>
  <button class="tbtn" id="mode" title="允许多行同时播放">叠听：关</button>
  <button class="tbtn" id="blind">👁 盲听：关</button>
  <button class="tbtn" id="reveal">揭晓对照表</button>
</div>

<div id="songmeta"></div>
<div id="content"></div>

<div class="card hidden" id="revealbox">
  <h2 class="sec" style="margin-top:0">盲听对照表（听完了再看）</h2>
  <p class="lead">按当前歌曲展开。代号随机分配，与排名无关。</p>
  <div id="revealtable"></div>
</div>

<footer id="foot"></footer>
</div>
<div id="toast"></div>

<script>
/* Any unexpected script error must be visible, never a silently blank page. */
window.addEventListener("error", function(e){
  var b = document.getElementById("errbar");
  if(!b){
    b = document.createElement("div"); b.id = "errbar";
    b.style.cssText = "position:fixed;left:0;right:0;top:0;z-index:999;background:#b91c1c;"
      + "color:#fff;padding:8px 14px;font:13px/1.5 system-ui,sans-serif";
    document.body.appendChild(b);
  }
  b.textContent = "页面脚本出错：" + (e.message || e);
});
const DB = /*__DB__*/;

/* ------------------------------------------------------------ state --- */
const S = {cur:null, playing:false, head:0, sync:false, blind:false, loop:false, raf:null,
           els:new Map(), active:new Set()};
const $  = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>[...r.querySelectorAll(s)];
const fmt = t => (!isFinite(t)||t<0) ? "0:00"
  : Math.floor(t/60)+":"+String(Math.floor(t%60)).padStart(2,"0");
const toast = m => {const t=$("#toast"); t.textContent=m; t.classList.add("show");
  clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove("show"),1600);};

const songOf = k => DB.songs.find(s=>s.key===k);
const modelsOf = s => s.models.filter(m=>m.tracks.length);

/* ------------------------------------------------------- kpis/header -- */
(function kpis(){
  const nModels = new Set(DB.songs.flatMap(s=>modelsOf(s).map(m=>m.key))).size;
  const nClips  = DB.songs.reduce((a,s)=>a+modelsOf(s).reduce((b,m)=>b+m.tracks.length,0)
                                    + s.ref.length, 0);
  const nMB = Math.round(DB.songs.reduce((a,s)=>
      a + s.ref.reduce((x,r)=>x+r.mb,0)
        + modelsOf(s).reduce((b,m)=>b+m.tracks.reduce((x,t)=>x+t.mb,0),0), 0));
  const kpi=[["3","首歌",DB.songs.map(s=>s.key.split(" - ")[0]).join(" / ")],
             [nModels+" × 3","模型 × 歌曲", "这 3 首上有实际产出的模型数"],
             [nClips,"个可播音频", "源混音 + GT 四轨 + 各模型产出"],
             [nMB+" MB","页面引用音频体积", "无损 FLAC / WAV，播放时按需流式读取"]];
  $("#kpis").innerHTML = kpi.map(([a,b,c])=>`<div class="kpi"><b>${a}</b>${b}<br>${c}</div>`).join("");
})();

/* -------------------------------------------------------------- tabs -- */
const tabs = $("#tabs");
DB.songs.forEach((s,i)=>{
  const b=document.createElement("button");
  b.className="tab"; b.dataset.key=s.key;
  b.innerHTML = `${s.key}<small>${s.dur_s?s.dur_s.toFixed(0)+"s":""}</small>`;
  b.onclick=()=>select(s.key);
  tabs.appendChild(b);
});

/* --------------------------------------------------------- rendering -- */
function rowHTML(t, ownerKey, songKey, cls=""){
  const id = `${songKey}|${ownerKey}|${t.target}`;
  const meta = [t.sr? (t.sr/1000).toFixed(1)+"kHz":"?", t.ch?t.ch+"ch":"?", 
                t.dur?t.dur.toFixed(1)+"s":"", t.mb+"MB",
                (t.peak!=null? "峰值 "+t.peak.toFixed(3) : "")].filter(Boolean).join(" · ");
  const sdr = (t.sdr!=null) ? `<span class="chip sdr">SDR ${t.sdr.toFixed(2)} dB</span>` : "";
  const clip = (t.clipped>0)
      ? `<span class="chip" style="background:#fee2e2;border-color:#fecaca;color:#991b1b"
           title="波形峰值触顶，人耳听到的是失真/爆音，而不是模型本身的分辨力">削波警告</span>` : "";
  S.els.set(id, null);
  return `<div class="row ${cls}" data-id="${id}" data-src="${t.src}">
    <button class="pl" title="播放/暂停这一轨">▶</button>
    <div class="nm">${t.label} <span class="chip b-plain">${t.target}</span> ${sdr}${clip}
      <span class="meta">${meta}</span></div>
    <div class="bar"><div class="rail"><i></i></div></div>
    <span class="tm">-:-- / ${t.dur?fmt(t.dur):"-:--"}</span>
    <input type="range" class="vol" min="0" max="100" value="100" title="音量">
    <button class="solo" title="独奏这一轨">S</button>
  </div>`;
}

function chipLine(m){
  const c=m.chips||{}, out=[];
  if(c.wall_s!=null) out.push(`本曲墙钟 <b>${c.wall_s.toFixed(0)}s</b>`);
  if(c.infer_s!=null) out.push(`推理 ${c.infer_s.toFixed(0)}s`);
  if(c.overhead_s!=null) out.push(`开销 ${c.overhead_s.toFixed(0)}s`);
  if(c.out_mb) out.push(`产出 ${c.out_mb.toFixed(0)}MB`);
  if(c.sdr_mean!=null) out.push(`<span class="chip sdrmean">SDR 均 ${c.sdr_mean.toFixed(2)} dB</span>`);
  return out.join(" · ") || "本曲无实测指标";
}

function renderSong(sk){
  const s = songOf(sk), host = $("#content");
  let h = "";

  h += `<div class="card"><h2 class="sec" style="margin-top:0">${s.key}</h2>
        <p class="lead">${s.why}　·　时长 ${s.dur_s?s.dur_s.toFixed(1):"?"} s　·　
        44.1 kHz / 立体声 / 16 bit。${s.missing.length
          ?`<b style="color:#b45309">本曲未纳入对比：${s.missing.join("、")}</b>`:""}</p>`;
  if(s.excluded && s.excluded.length){
    h += `<div class="note warn"><b>为什么有些模型没有播放器</b>（这些产出被完整性校验挡掉了，不放进对比，
          否则你会听到一段断掉的音频还以为模型很差）：<ul>`
       + s.excluded.map(e=>`<li>${e}</li>`).join("") + `</ul></div>`;
  }
  h += `<div class="mgroup" style="margin-top:6px"><span class="dot" style="background:var(--ref)"></span>
        <h3>参考：源混音与官方 GT 分轨</h3>
        <span class="gn">先把这两组听熟，后面的差异才有参照系</span></div>`;
  h += `<div class="mcard" style="border-left-color:var(--ref)">`;
  s.ref.forEach(t=>{ h += rowHTML(t, "REF", sk); });
  h += `<div class="mnote">把「GT 人声/鼓/贝斯/其他」四行在叠听模式下同时点亮，
        得到的就是源混音本身（可用来校准音量与你的听觉）。</div></div>`;

  for(const g of DB.groups){
    const ms = s.models.filter(m=>m.grp===g.id);
    if(!ms.length) continue;
    h += `<div class="grpwrap" data-g="${g.id}">
          <div class="mgroup"><span class="dot" style="background:${g.color}"></span>
          <h3>${g.title}</h3><span class="gn">${g.note}</span></div>`;
    ms.forEach((m,gi)=>{
      const miss = m.tracks.length===0;
      h += `<div class="mcard ${g.id.toLowerCase()} ${miss?"missing":""}" data-model="${m.key}"
              data-code="${codeFor(sk,m.key)}" data-idx="${gi}">
        <div class="mhead">
          <span class="idx">${gi+1}</span>
          <span class="realname">${m.label}</span>
          <span class="codename">${codeFor(sk,m.key)}</span>
          <span class="chip">${m.tracks.length?m.tracks.length+" 轨输出":"无产出"}</span>
          ${m.note?`<span class="chip">${m.note}</span>`:""}
          <div class="macts">
            ${miss?"":`<button class="act" data-a="all" data-m="${m.key}">全轨叠播</button>`}
            ${m.tracks.some(t=>t.target==="vocals"||t.target==="vocals+other")
                ?`<button class="act" data-a="voc" data-m="${m.key}">仅人声</button>`:""}
            ${m.tracks.length>1?`<button class="act" data-a="inst" data-m="${m.key}">仅伴奏</button>`:""}
          </div>
        </div>
        <div class="mnote">${miss? "<b style='color:#b91c1c'>本曲无有效产出</b> —— "+m.reason : chipLine(m)}</div>
        ${m.tracks.map(t=>rowHTML(t, m.key, sk)).join("")}
      </div>`;
    });
    h += `</div>`;
  }
  host.innerHTML = h + "</div>";
}

function codeFor(sk, mk){
  /* deterministic per-song shuffle: same load -> same code, but the code and
     the card order carry no information about the model's identity or score. */
  const keys = modelsOf(songOf(sk)).map(m=>m.key);
  let seed = 2166136261;
  for(const ch of sk) seed = ((seed ^ ch.charCodeAt(0)) * 16777619) >>> 0;
  const rnd = n => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed % n; };
  const order = keys.slice();
  for(let i=order.length-1;i>0;i--){ const j=rnd(i+1); [order[i],order[j]]=[order[j],order[i]]; }
  const pos = order.indexOf(mk);
  return "样本 " + (pos<0 ? "?" : String.fromCharCode(65+pos));
}

/* ----------------------------------------------------------- audio ---- */
function audioFor(id){
  let a = S.els.get(id);
  if(a) return a;
  const row = $(`.row[data-id="${CSS.escape(id)}"]`);
  if(!row) return null;
  a = new Audio();
  a.preload = "metadata";
  a.src = row.dataset.src;
  a.volume = 1;
  a.addEventListener("loadedmetadata", ()=>{
    row.querySelector(".tm").textContent = "-:-- / " + fmt(a.duration);
  });
  a.addEventListener("error", ()=>{
    toast("音频加载失败：" + row.dataset.src.split("/").pop() + "（请用 start_listen.bat 打开）");
    row.querySelector(".nm").insertAdjacentHTML("beforeend",
      ' <span class="badge b-warn">加载失败</span>');
  });
  a.addEventListener("play", ()=>row.classList.add("playing"));
  a.addEventListener("pause", ()=>row.classList.remove("playing"));
  S.els.set(id, a);
  return a;
}

function activeList(){ return [...S.active].map(id=>({id, a:audioFor(id)})).filter(x=>x.a); }

function ensureRAF(){
  if(S.raf) return;
  const tick = ()=>{
    const live = activeList().filter(x=>!x.a.paused);
    if(!live.length){ S.raf=null; S.playing=false; paintUI(); return; }
    const lead = live[0].a;
    S.head = lead.currentTime;
    live.forEach(({a})=>{
      if(a===lead) return;
      if(Math.abs(a.currentTime - S.head) > 0.12) a.currentTime = S.head;
    });
    paintUI();
    if(S.loop&&lead.duration&&lead.currentTime>=lead.duration-0.06){
      S.head=0; live.forEach(({a})=>a.currentTime=0);
    }
    S.raf = requestAnimationFrame(tick);
  };
  S.raf = requestAnimationFrame(tick);
}

function playActive(){
  const list = activeList();
  if(!list.length){ toast("先点亮一首轨（▶ 或 全轨叠播）"); return; }
  list.forEach(({a})=>{
    if(Math.abs(a.currentTime - S.head) > 0.15) a.currentTime = S.head;
    const pr = a.play();
    if(pr&&pr.catch) pr.catch(()=>{});
  });
  S.playing = true; ensureRAF(); paintUI();
}

function pauseAll(){
  activeList().forEach(({a})=>a.pause());
  S.playing=false; if(S.raf){cancelAnimationFrame(S.raf); S.raf=null;} paintUI();
}

function hardStop(){
  pauseAll();
  [...S.els.values()].forEach(a=>{ if(a){ a.pause(); a.currentTime=0; } });
  S.active.clear(); S.head=0;
  $$(".row").forEach(r=>{
    r.classList.remove("on","playing");
    const f=r.querySelector(".bar .rail i"); if(f) f.style.width="0%";
  });
  paintUI();
}

function toggleRow(id, exclusive){
  const a = audioFor(id); if(!a) return;
  if(!S.sync || exclusive){
    const wasActive = !!S.els.get(id) && S.active.has(id) && !a.paused;
    S.active.clear();
    if(wasActive && exclusive){ a.pause(); S.playing=false; syncRows(); paintUI(); return; }
    S.active.add(id);
  } else {
    if(S.active.has(id)){ S.active.delete(id); a.pause(); syncRows(); paintUI();
      if(!activeList().some(x=>!x.a.paused)) pauseAll(); return; }
    S.active.add(id);
  }
  a.currentTime = S.head;
  playActive(); syncRows();
}

function syncRows(){
  $$(".row").forEach(r=>r.classList.toggle("on", S.active.has(r.dataset.id)));
}

function paintUI(){
  const s = songOf(S.cur);
  const dur = s ? (s.dur_s||0) : 0;
  $("#time").textContent = fmt(S.head)+" / "+fmt(dur);
  const pct = dur? (S.head/dur*100) : 0;
  $$(".seek .rail i").forEach(e=>e.style.width=pct+"%");
  const k = $(".seek .rail b"); if(k) k.style.left=pct+"%";
  $("#pp").textContent = S.playing ? "⏸ 暂停" : "▶ 播放";
  /* per-row bars */
  activeList().forEach(({id,a})=>{
    const row=$(`.row[data-id="${CSS.escape(id)}"]`); if(!row) return;
    const d=a.duration||dur||1;
    row.querySelector(".bar .rail i").style.width = (a.currentTime/d*100)+"%";
    row.querySelector(".tm").textContent = fmt(a.currentTime)+" / "+fmt(a.duration||dur);
  });
  $("#mode").textContent = (S.sync?"▣ 叠听：开":"▢ 叠听：关");
  $("#mode").classList.toggle("on", S.sync);
  $("#loop").classList.toggle("on", S.loop);
  $("#blind").textContent = (S.blind?"👁 盲听：开":"👁 盲听：关");
  $("#blind").classList.toggle("on", S.blind);
}

function seekTo(t){
  const s=songOf(S.cur); const dur=(s&&s.dur_s)||0;
  S.head = Math.max(0, Math.min(t, dur? dur-0.05 : t));
  activeList().forEach(({a})=>{ a.currentTime=S.head; });
  paintUI();
}

/* ------------------------------------------------------------ events -- */
function select(sk){
  hardStop();
  S.els.clear(); S.active.clear();
  S.cur = sk;
  document.body.classList.toggle("blind", S.blind);
  $$(".tab").forEach(t=>t.classList.toggle("on", t.dataset.key===sk));
  renderSong(sk);
  reorderCards();
  syncRows(); paintUI();
  if(S.blind){ renderReveal(); } else { $("#revealbox").classList.add("hidden"); }
}

$("#content").addEventListener("click", ev=>{
  const row = ev.target.closest(".row");
  if(row && (ev.target.closest(".pl")||ev.target.closest(".solo")||ev.target.closest(".nm"))){
    toggleRow(row.dataset.id, !!ev.target.closest(".solo") || !S.sync); return;
  }
  if(row && ev.target.closest(".bar")){
    const r=ev.target.closest(".bar").getBoundingClientRect();
    const frac=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));
    const s=songOf(S.cur); seekTo(frac*((s&&s.dur_s)||0)); return;
  }
  const act = ev.target.closest(".act");
  if(act){
    const m = act.dataset.m, k = act.dataset.a;
    const rec = songOf(S.cur).models.find(x=>x.key===m); if(!rec) return;
    let pick = rec.tracks;
    if(k==="voc") pick = rec.tracks.filter(t=>t.target.includes("vocals"));
    if(k==="inst") pick = rec.tracks.filter(t=>t.target!=="vocals");
    if(!S.sync){ S.sync=true; }
    S.active.clear();
    pick.forEach(t=>S.active.add(`${S.cur}|${m}|${t.target}`));
    activeList().forEach(({a})=>a.pause());
    S.head = Math.max(0, S.head);
    playActive(); syncRows(); paintUI();
    return;
  }
});
$("#content").addEventListener("input", ev=>{
  if(ev.target.classList.contains("vol")){
    const id = ev.target.closest(".row").dataset.id;
    const a = audioFor(id); if(a) a.volume = ev.target.value/100;
  }
});

$("#pp").onclick = ()=>S.playing? pauseAll() : playActive();
$("#stop").onclick = ()=>hardStop();
$("#back").onclick = ()=>seekTo(S.head-5);
$("#fwd").onclick  = ()=>seekTo(S.head+5);
$("#loop").onclick = ()=>{S.loop=!S.loop; paintUI();};
$("#mode").onclick = ()=>{S.sync=!S.sync; if(!S.sync){ pauseAll(); }
  toast(S.sync?"叠听已开：可同时点亮多行":"叠听已关：一次只播一行"); paintUI();};
function reorderCards(){
  $$(".grpwrap").forEach(w=>{
    const cards=[...$$(".mcard", w)];
    const keyOf = c => S.blind ? c.dataset.code : String(+c.dataset.idx).padStart(3,"0");
    cards.sort((a,b)=> keyOf(a).localeCompare(keyOf(b)));
    cards.forEach(c=>w.appendChild(c));
    cards.forEach((c,i)=>{ const ix=c.querySelector(".idx"); if(ix) ix.textContent=i+1; });
  });
}

$("#blind").onclick = ()=>{
  S.blind=!S.blind; document.body.classList.toggle("blind", S.blind);
  hardStop();
  reorderCards();
  $("#revealbox").classList.add("hidden");
  if(S.blind) renderReveal();
  paintUI();
  toast(S.blind?"盲听已开：模型名换成代号，组内顺序已打乱":"盲听已关：已恢复模型名与原始顺序");
};
$("#reveal").onclick = ()=>{
  if(!$("#revealtable").innerHTML) renderReveal();
  $("#revealbox").classList.toggle("hidden");
};
function renderReveal(){
  const s=songOf(S.cur);
  const rows = modelsOf(s).map(m=>`<tr><td><b>${codeFor(S.cur,m.key)}</b></td>
      <td>${m.label}</td><td>${m.tracks.length} 轨</td>
      <td>${m.chips.sdr_mean!=null?m.chips.sdr_mean.toFixed(2)+" dB":"—"}</td></tr>`).join("");
  $("#revealtable").innerHTML = `<table><thead><tr><th>代号</th><th>模型</th>
      <th>输出轨数</th><th>本曲 SDR 均</th></tr></thead><tbody>${rows}</tbody></table>`;
}

$("#seek").onclick = ev=>{
  const r=$("#seek").getBoundingClientRect();
  const frac=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));
  const s=songOf(S.cur); seekTo(frac*((s&&s.dur_s)||0));
};
document.addEventListener("keydown", ev=>{
  const t = ev.target;
  if(t && t.tagName && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
  if(ev.code==="Space"){ ev.preventDefault(); S.playing?pauseAll():playActive(); }
  else if(ev.key==="ArrowLeft"){ seekTo(S.head-5); }
  else if(ev.key==="ArrowRight"){ seekTo(S.head+5); }
  else if(ev.key==="0"){ hardStop(); }
  else if(/^[1-9]$/.test(ev.key)){
    const cards=$$(".mcard").filter(c=>!c.classList.contains("missing"));
    const c=cards[+ev.key-1]; if(!c) return;
    const m=c.dataset.model;
    S.sync=true;
    S.active.clear();
    songOf(S.cur).models.find(x=>x.key===m).tracks
      .forEach(t=>S.active.add(`${S.cur}|${m}|${t.target}`));
    activeList().forEach(({a})=>a.pause());
    playActive(); syncRows(); paintUI();
  }
});

$("#foot").innerHTML = "数据来源：本机 03_outputs/_test_run/ 与 02_databases/MUSDB18-HQ/test/ 的实测产出 · "
  + "未使用任何网络数据 · 页面生成于 " + new Date().toLocaleString("zh-CN")
  + "<br>若音频无法播放，请双击 E:\\FYP_HKBU\\start_listen.bat 后访问 http://127.0.0.1:8123/04_reports/separation/html/listen_compare.html";

select(DB.songs[0].key);
</script>
</body>
</html>
"""


def main():
    db = build()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(db, fh, ensure_ascii=False, indent=1)

    html = HTML.replace("/*__DB__*/", json.dumps(db, ensure_ascii=False))
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    print("html :", OUT_HTML, round(os.path.getsize(OUT_HTML) / 1024, 1), "KB")
    print("json :", OUT_JSON)
    for s in db["songs"]:
        print("\n== %s  dur=%s" % (s["key"], s["dur_s"]))
        for r in s["ref"]:
            print("   [ref ] %-28s %s" % (r["label"], r["src"]))
        for m in s["models"]:
            got = len(m["tracks"])
            src = m["tracks"][0]["src"] if got else "-"
            print("   [%-11s] %-18s %d track(s)  %s" % (m["grp"], m["label"], got, src))
    if db["problems"]:
        print("\n!! problems (%d)" % len(db["problems"]))
        for p in db["problems"][:40]:
            print("   ", p)
    else:
        print("\nall referenced files exist")


if __name__ == "__main__":
    sys.exit(main())
