# -*- coding: utf-8 -*-
"""BSRNN 管线探针（不需要权重）：验证 stub + conf 合成 + 建模 + fader 前向。

目的：把「BSRNN 跑不起来」的风险前置拆掉。权重到位后只需验证 ckpt 加载。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _bsrnn_loader import BSRNN_REPO, install_stubs, _load_conf  # noqa: E402


def main():
    conf_name = sys.argv[1] if len(sys.argv) > 1 else "bsrnn-opt"
    print(f"[probe] repo = {BSRNN_REPO}")
    replaced = install_stubs()
    print(f"[probe] stub 替换 = {replaced}")

    cfg = _load_conf(conf_name)
    print(f"[probe] conf 合成 OK: model.name={cfg.model.name} feature_dim={cfg.model.feature_dim} "
          f"num_repeat={cfg.model.num_repeat} n_att_head={cfg.model.n_att_head} "
          f"stereo={cfg.model.get('stereo')} sample_rate={cfg.model.sample_rate} "
          f"n_fft={cfg.model.n_fft} n_hop={cfg.model.n_hop} nb_channels={cfg.model.nb_channels}")
    print(f"[probe] eval: segment_len={cfg.eval.segment_len} overlap={cfg.eval.overlap} "
          f"hop_size={cfg.eval.hop_size}")

    t0 = time.time()
    from models.bsrnn import BSRNN
    targets = ["vocals", "bass", "drums", "other"]
    model = BSRNN(cfg.optim, cfg.scheduler, cfg.eval, targets, **dict(cfg.model))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[probe] BSRNN 构建 OK  {time.time()-t0:.2f}s  params={n_params/1e6:.2f}M  targets={model.targets}")

    model.eval_device = "cpu"
    model.eval()

    t1 = time.time()
    x = torch.randn(1, 2, 44100 * 10) * 0.05
    with torch.no_grad():
        final, loss = model._apply_model_to_track(x, None, comp_loss=False)
    dt = time.time() - t1
    print(f"[probe] fader 前向 OK  {dt:.2f}s  RTF={dt/10:.3f}  out={tuple(final.shape)}  loss={loss}")
    print(f"[probe] 每源 rms = {[round(float(final[0,i].pow(2).mean().sqrt()),5) for i in range(final.shape[1])]}")

    # 短片段（< segment_len，触发 fader 分块路径）也测一下
    t2 = time.time()
    x2 = torch.randn(1, 2, 44100 * 4) * 0.05
    with torch.no_grad():
        final2, _ = model._apply_model_to_track(x2, None, comp_loss=False)
    print(f"[probe] 短片段(4s) 前向 OK  {time.time()-t2:.2f}s  out={tuple(final2.shape)}")
    print("[probe] ALL OK")


if __name__ == "__main__":
    main()
