# -*- coding: utf-8 -*-
"""BSS-Eval v4 四分量分解 —— 直接复用 museval 内部实现。

为什么不用自己实现
------------------
本项目报告里的 SDR 一律由 `museval.metrics.bss_eval`（BSSEval v4）产出。
若面板改用"自己写的瞬时投影"，面板与报告就会**口径分裂**：
图上看着对，数字却对不上。这里直接调用 museval 的
`_compute_reference_correlations` / `_compute_projection_filters` / `_bss_decomp_mtifilt`，
使面板的分解与报告 SDR **同源**。

四分量（BSS-Eval v4，见 Vincent 2006 / Liutkus 2018）
----------------------------------------------------
估计 = s_true（目标） + e_spat（滤波/空间失真） + e_interf（干扰） + e_artif（伪影）

恒等式：`s_true + e_spat + e_interf + e_artif` 的前 nsampl 个样本 == 估计（逐样本精确）。
本模块的 `decompose()` 会顺带把这个残差算出来，供调用方断言。

用法
----
>>> ref = np.stack([vocals, drums, bass, other], 0)   # (4, nsampl, nchan)
>>> est = model_vocals                                # (nsampl, nchan)
>>> r = decompose(ref, est, j=0, filters_len=512)
>>> r["residual_rel"]            # ~1e-12，恒等式校验
>>> crit(r)                      # SDR/ISR/SIR/SAR，与 bss_eval 同口径
"""

from __future__ import annotations

import numpy as np

DEFAULT_FILTERS_LEN = 512


def decompose(reference_sources, estimated_source, j=0, filters_len=DEFAULT_FILTERS_LEN):
    """把「估计」拆成四个分量。

    Parameters
    ----------
    reference_sources : (nsrc, nsampl, nchan) 真值多轨（不含 mixture）
    estimated_source  : (nsampl, nchan) 待分解的估计（单轨）
    j                 : 该估计对应的真值轨下标
    filters_len       : 失真滤波器长度，与 bss_eval 默认保持一致（512）

    Returns
    -------
    dict(s_true, e_spat, e_interf, e_artif, residual_abs, residual_rel,
         nsrc, nsampl, nchan, filters_len)
    """
    import museval.metrics as M

    ref = np.ascontiguousarray(np.asarray(reference_sources, dtype=np.float64))
    est = np.ascontiguousarray(np.asarray(estimated_source, dtype=np.float64))
    if ref.ndim != 3:
        raise ValueError(f"reference_sources 需为 (nsrc, nsampl, nchan)，实得 {ref.shape}")
    if est.ndim != 2:
        raise ValueError(f"estimated_source 需为 (nsampl, nchan)，实得 {est.shape}")
    nsrc, nsampl, nchan = ref.shape
    if est.shape[0] != nsampl:
        raise ValueError(f"长度不一致：ref {nsampl} vs est {est.shape[0]}")
    if not (0 <= j < nsrc):
        raise ValueError(f"j={j} 越界（nsrc={nsrc}）")

    # 与 bss_eval 内部 compute_GsfC / compute_Cj 完全一致的调用序列
    G, sf = M._compute_reference_correlations(ref, filters_len)
    C = M._compute_projection_filters(G, sf, est)          # 全源投影（含干扰）
    Cj = M._compute_projection_filters(G[j, j], sf[j], est)  # 仅目标轨投影（含滤波失真）

    s_true, e_spat, e_interf, e_artif = M._bss_decomp_mtifilt(ref, est, j, C, Cj)

    # 恒等式校验：四分量之和的前 nsampl 个样本应当逐样本等于估计
    total = s_true + e_spat + e_interf + e_artif
    resid = total[:nsampl] - est
    residual_abs = float(np.max(np.abs(resid)))
    denom = float(np.max(np.abs(est))) or 1.0
    residual_rel = residual_abs / denom

    return {
        "s_true": s_true,
        "e_spat": e_spat,
        "e_interf": e_interf,
        "e_artif": e_artif,
        "residual_abs": residual_abs,
        "residual_rel": residual_rel,
        "nsrc": int(nsrc),
        "nsampl": int(nsampl),
        "nchan": int(nchan),
        "filters_len": int(filters_len),
    }


def crit(parts, bsseval_sources_version=False):
    """由四分量算 SDR/ISR/SIR/SAR（与 museval.bss_eval 同一判据函数）。"""
    import museval.metrics as M

    return M._bss_crit(
        parts["s_true"], parts["e_spat"], parts["e_interf"], parts["e_artif"],
        bsseval_sources_version,
    )


def component_energy_db(parts):
    """四分量各自的能量（dB），用于面板上的占比标注。"""
    out = {}
    for k in ("s_true", "e_spat", "e_interf", "e_artif"):
        e = float(np.sum(parts[k] ** 2))
        out[k] = 10.0 * np.log10(e + 1e-30)
    return out


def self_test(verbose=True):
    """合成信号的单元自检：构造已知答案，验证分解与判据自洽。"""
    rng = np.random.default_rng(0)
    n = 44100
    ref = rng.standard_normal((4, n, 2)).astype(np.float64)
    # 估计 = 0.8 * target + 0.2 * interferer（无伪影），应拿到高 SIR
    est = 0.8 * ref[0] + 0.2 * ref[1]
    parts = decompose(ref, est, j=0, filters_len=32)
    sdr, isr, sir, sar = crit(parts)
    ok_resid = parts["residual_rel"] < 1e-9
    if verbose:
        print("[selftest] residual_rel = %.3e  (需 < 1e-9)" % parts["residual_rel"])
        print("[selftest] SDR=%.2f dB  ISR=%.2f  SIR=%.2f  SAR=%.2f"
              % (np.squeeze(sdr), np.squeeze(isr), np.squeeze(sir), np.squeeze(sar)))
    return ok_resid


if __name__ == "__main__":
    print("OK " if self_test() else "ERR selftest failed")
