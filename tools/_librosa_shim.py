# -*- coding: utf-8 -*-
"""最小 librosa 垫片 —— 用于绕过 Smart App Control 对 numba/llvmlite 的拦截。

背景
----
本机 Smart App Control 处于强制开启状态，`llvmlite.dll` 被 Windows 代码完整性策略
拦截（WinError 4551）。于是：

    librosa/__init__.py -> librosa.filters -> from numba import jit -> llvmlite.binding
                                                                            ^^^^^^^^^^ OSError

只要进程里 `import librosa` 就会崩。而 RPCA / NMF 等古典方法其实只需要
`librosa.stft` / `librosa.istft`（默认 hann 窗、center=True），这两件事
`torch.stft` / `torch.istft` 可以给出**完全等价**的结果。

本模块提供一个鸭子类型对象 `librosa_stub`，接口与用到的 librosa API 一致：
    .stft(y, n_fft=2048, hop_length=512)
    .istft(S, hop_length=512, length=None)
    .effects.hpss(...)   -> 明确抛 NotImplementedError（本机无法等价复现）
    __version__          -> "stub(torch)"

用法
----
    try:
        import librosa
    except Exception:
        from _librosa_shim import librosa_stub as librosa

注意：这只替代 STFT/ISTFT。任何真正依赖 numba JIT 的 librosa 功能
（如 effects.hpss 的 medfilt、mel 滤波器组）都不会被静默地给出错误结果，
而是显式抛 NotImplementedError。
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class _Effects:
    """librosa.effects 的占位：HPSS 依赖 numba 的 median filter，本机无法等价复现。"""

    @staticmethod
    def hpss(*_a, **_kw):
        raise NotImplementedError(
            "HPSS 需要 librosa.effects.hpss（内部依赖 numba 的 median filter），"
            "在本机 Smart App Control 下不可用。"
        )


class _Display:
    @staticmethod
    def specshow(*_a, **_kw):
        raise NotImplementedError("librosa.display 在本机不可用（垫片未实现）")


class _LibrosaStub:
    """torch 后端的 librosa 鸭子类型替身。"""

    __version__ = "stub(torch.stft)"
    is_stub = True

    effects = _Effects()
    display = _Display()

    # ---------------- STFT ----------------
    @staticmethod
    def stft(y, n_fft=2048, hop_length=None, win_length=None, window="hann",
             center=True, pad_mode="constant", dtype=None):
        """等价于 librosa.stft 的常用调用（hann 窗、center=True）。

        返回复数数组，形状与 librosa 一致：(1 + n_fft//2, T) 或 (..., 1+n_fft//2, T)。
        """
        if not _HAS_TORCH:
            raise RuntimeError("librosa 垫片需要 torch")
        hop = 512 if hop_length is None else hop_length
        win_len = n_fft if win_length is None else win_length
        if isinstance(window, str):
            if window != "hann":
                raise NotImplementedError(f"垫片仅实现 hann 窗，收到 window={window!r}")
            win = torch.hann_window(win_len)
        else:
            win = torch.as_tensor(np.asarray(window), dtype=torch.float32)

        t = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32)
        out = torch.stft(t, n_fft=n_fft, hop_length=hop, win_length=win_len, window=win,
                         center=center, pad_mode=pad_mode, return_complex=True)
        arr = out.numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    # ---------------- iSTFT ----------------
    @staticmethod
    def istft(S, hop_length=None, n_fft=None, win_length=None, window="hann",
              center=True, length=None):
        """等价于 librosa.istft 的常用调用。"""
        if not _HAS_TORCH:
            raise RuntimeError("librosa 垫片需要 torch")
        S = np.asarray(S)
        if n_fft is None:
            n_fft = 2 * (S.shape[-2] - 1)
        hop = 512 if hop_length is None else hop_length
        win_len = n_fft if win_length is None else win_length
        if isinstance(window, str):
            if window != "hann":
                raise NotImplementedError(f"垫片仅实现 hann 窗，收到 window={window!r}")
            win = torch.hann_window(win_len)
        else:
            win = torch.as_tensor(np.asarray(window), dtype=torch.float32)

        t = torch.as_tensor(np.ascontiguousarray(S), dtype=torch.complex64)
        out = torch.istft(t, n_fft=n_fft, hop_length=hop, win_length=win_len, window=win,
                          center=center, length=None if length is None else int(length),
                          return_complex=False)
        return out.numpy()

    # ---------------- 其它常用 ----------------
    @staticmethod
    def load(*_a, **_kw):
        raise NotImplementedError("请直接用 soundfile 读音频")


librosa_stub = _LibrosaStub()
