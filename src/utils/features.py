from __future__ import annotations
import math
import numpy as np
import torch


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None
) -> torch.Tensor:
    if fmax is None:
        fmax = sr / 2.0

    m_min = _hz_to_mel(fmin)
    m_max = _hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = np.array([_mel_to_hz(m) for m in m_pts])

    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (right - k) / (right - center)

    return torch.from_numpy(fb)


def log_mel_spectrogram(
    wav: torch.Tensor,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 80,
    eps: float = 1e-10
) -> torch.Tensor:
    if wav.dim() != 1:
        wav = wav.view(-1)

    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    stft = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (stft.real ** 2 + stft.imag ** 2)

    fb = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(power.device)
    mel = torch.matmul(fb, power)
    logmel = torch.log(mel + eps).transpose(0, 1).contiguous()

    mean = logmel.mean(dim=0, keepdim=True)
    std = logmel.std(dim=0, keepdim=True)
    logmel = (logmel - mean) / (std + 1e-5)

    return logmel
