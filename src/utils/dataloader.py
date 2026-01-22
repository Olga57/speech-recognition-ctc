from __future__ import annotations
from typing import Callable, Dict, List, Optional
import json

import torch
from torch.utils.data import DataLoader

from src.utils.collate import make_batch


class ASRCollator:
    def __init__(
        self,
        tokenizer,
        sr: int = 16000,
        n_fft: int = 400,
        hop: int = 160,
        win: int = 400,
        n_mels: int = 80,
        wav_augs: Optional[List[Callable]] = None,
        spec_augs: Optional[List[Callable]] = None,
        return_aug_debug: bool = False,
        aug_debug_max_items: int = 2,
    ):
        self.tokenizer = tokenizer
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = int(win)
        self.n_mels = int(n_mels)
        self.wav_augs = wav_augs or []
        self.spec_augs = spec_augs or []
        self.return_aug_debug = bool(return_aug_debug)
        self.aug_debug_max_items = int(aug_debug_max_items)

    def __call__(self, batch: List[Dict]) -> Dict:
        aug_items: List[Dict] = []

        # 1) Waveform augs + сохраняем wav_before / wav_after
        for bi, b in enumerate(batch):
            wav_before = b["wav"].float()
            wav = wav_before
            for aug in self.wav_augs:
                wav = aug(wav)

            b["wav"] = wav

            if self.return_aug_debug and len(aug_items) < self.aug_debug_max_items:
                aug_items.append(
                    {
                        "utt_id": b["utt_id"],
                        "_batch_idx": bi,
                        "wav_before": wav_before.clone(),
                        "wav_after": wav.clone(),
                    }
                )

        # 2) make_batch -> feats/tokens/len/etc
        out = make_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            sr=self.sr,
            n_fft=self.n_fft,
            hop=self.hop,
            win=self.win,
            n_mels=self.n_mels,
        )

        # 3) Spec augs + сохраняем spec_before/spec_after
        feats = out["feats"]
        dbg_map = {it["_batch_idx"]: it for it in aug_items} if aug_items else {}

        for i in range(feats.shape[0]):
            spec = feats[i]

            if i in dbg_map:
                dbg_map[i]["spec_before"] = spec.clone()

            for aug in self.spec_augs:
                spec = aug(spec)

            if i in dbg_map:
                dbg_map[i]["spec_after"] = spec.clone()

            feats[i] = spec

        out["feats"] = feats

        if self.return_aug_debug and len(aug_items) > 0:
            for it in aug_items:
                it.pop("_batch_idx", None)
            out["aug_debug"] = aug_items

        return out


class LibrispeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str, target_sr: int = 16000, max_items=None, **kwargs):
        self.data: List[Dict] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

        if max_items is not None:
            self.data = self.data[: int(max_items)]
        self.target_sr = int(target_sr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import soundfile as sf
        import numpy as np

        item = self.data[int(idx)]

        wav, sr = sf.read(item["audio_filepath"])  # wav: np.ndarray, sr: int


        if isinstance(wav, np.ndarray) and wav.ndim == 2:
            wav = wav.mean(axis=1)

        wav = wav.astype(np.float32, copy=False)
        sr = int(sr)

        if sr != self.target_sr:
            # resample из scipy (без torchaudio)
            from scipy.signal import resample

            num_samples = int(len(wav) * self.target_sr / sr)
            if num_samples <= 0:
                num_samples = 1
            wav = resample(wav, num_samples).astype(np.float32, copy=False)

        return {
            "wav": torch.from_numpy(wav).float(),
            "text": str(item.get("text", "")).lower(),
            "utt_id": str(item.get("utt_id", f"utt_{idx}")),
        }


def build_loader(dataset, collate_fn, batch_size: int, shuffle: bool, num_workers: int = 2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
