from __future__ import annotations

import random
import torch
import torch.nn.functional as F


class FrequencyMask:
    def __init__(self, max_width: int = 15):
        self.max_width = int(max_width)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [T, M]
        T, M = spec.shape
        w = random.randint(0, min(self.max_width, M))
        if w == 0:
            return spec
        f0 = random.randint(0, max(0, M - w))
        spec = spec.clone()
        spec[:, f0:f0 + w] = 0.0
        return spec


class TimeMask:
    def __init__(self, max_width: int = 50):
        self.max_width = int(max_width)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        T, M = spec.shape
        w = random.randint(0, min(self.max_width, T))
        if w == 0:
            return spec
        t0 = random.randint(0, max(0, T - w))
        spec = spec.clone()
        spec[t0:t0 + w, :] = 0.0
        return spec


class AddGaussianNoise:
    def __init__(self, sigma: float = 0.01):
        self.sigma = float(sigma)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return wav
        return wav + torch.randn_like(wav) * self.sigma


class TimeStretch:
    """
    Простая time-stretch аугментация на waveform через линейную интерполяцию.
    Делается через torch.nn.functional.interpolate (везде доступно).
    """
    def __init__(self, min_rate: float = 0.9, max_rate: float = 1.1):
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        rate = random.uniform(self.min_rate, self.max_rate)
        if abs(rate - 1.0) < 1e-6:
            return wav

        wav = wav.view(-1).float()
        T = int(wav.shape[0])
        new_T = max(1, int(T / rate))

        # F.interpolate expects [N, C, L]
        x = wav.view(1, 1, T)
        y = F.interpolate(x, size=new_T, mode="linear", align_corners=True)
        return y.view(new_T)
