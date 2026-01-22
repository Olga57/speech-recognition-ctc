from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from src.utils.features import log_mel_spectrogram


def pad_2d(specs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([s.shape[0] for s in specs], dtype=torch.long)
    max_len = int(lens.max().item()) if len(specs) > 0 else 0
    feat = int(specs[0].shape[1]) if len(specs) > 0 else 0
    out = torch.zeros((len(specs), max_len, feat), dtype=torch.float32)
    for i, s in enumerate(specs):
        out[i, : s.shape[0], :] = s
    return out, lens


def pad_tokens(seqs: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lens.max().item()) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) > 0:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lens


def make_batch(
    batch: List[Dict],
    tokenizer,
    sr: int = 16000,
    n_fft: int = 400,
    hop: int = 160,
    win: int = 400,
    n_mels: int = 80,
) -> Dict:
    utt_ids = [b["utt_id"] for b in batch]
    wavs = [b["wav"].float() for b in batch]
    texts = [b.get("text", None) for b in batch]

    # Генерируем спектрограммы
    specs = [
        log_mel_spectrogram(
            w, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win, n_mels=n_mels
        )
        for w in wavs
    ]
    feats, feat_lens = pad_2d(specs)

    token_seqs: List[List[int]] = []
    for t in texts:
        token_seqs.append([] if t is None else tokenizer.encode(t))
    tokens, token_lens = pad_tokens(token_seqs, pad_id=0)

    return {
        "utt_id": utt_ids,
        "feats": feats,
        "feats_len": feat_lens,
        "tokens": tokens,
        "tokens_len": token_lens,
        "text": texts,
        "wav": wavs,  # Добавляем само аудио
        "sr": [sr] * len(wavs),  # Добавляем частоту дискретизации
    }
