from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSubsampling(nn.Module):
    def __init__(self, in_feats: int, out_ch: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # СЛОЙ 1: stride=1 (без сжатия)
            nn.Conv2d(1, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # СЛОЙ 2: stride=1 (ТОЖЕ без сжатия)
            # Раньше тут было 2, и это убивало длинные фразы
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F] -> [B, 1, F, T]
        x = x.permute(0, 2, 1).unsqueeze(1)
        y = self.conv(x)
        b, c, f_new, t_new = y.shape
        y = y.permute(0, 3, 1, 2).contiguous().view(b, t_new, c * f_new)

        # Длина не меняется, так как stride=1 везде
        return y, x_len

class CTCBiLSTM(nn.Module):
    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        enc_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        subsample_ch: int = 32,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.sub = ConvSubsampling(in_feats=n_mels, out_ch=subsample_ch)

        # Авторасчет размерности
        with torch.no_grad():
            dummy = torch.zeros(1, 100, n_mels)
            dummy_l = torch.tensor([100])
            out, _ = self.sub(dummy, dummy_l)
            conv_dim = out.shape[-1]

        self.pre = nn.Linear(conv_dim, enc_dim)

        self.rnn = nn.LSTM(
            input_size=enc_dim,
            hidden_size=enc_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(enc_dim, vocab_size)
        self.blank_id = int(blank_id)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.classifier.bias is not None:
             self.classifier.bias.data[self.blank_id] = -1.0

    def forward(self, feats: torch.Tensor, feats_len: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, x_len = self.sub(feats, feats_len)
        x = self.pre(x)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "log_probs_len": x_len}
