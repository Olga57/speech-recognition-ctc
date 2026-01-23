from __future__ import annotations
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SUPPORTED_AUDIO_EXT = (".wav", ".flac", ".mp3")

class CustomDirDataset(Dataset):
    """
    root_dir/
      audio/            *.wav|*.flac|*.mp3
      transcriptions/   <same_id>.txt
    """
    def __init__(self, root_dir, target_sr=16000, **kwargs) -> None:
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audio")
        self.tr_dir = os.path.join(root_dir, "transcriptions")

        self.target_sr = int(target_sr)

        max_items = kwargs.get("max_items", None)
        self.max_items = max_items

        lowercase = kwargs.get("lowercase", True)
        self.lowercase = bool(lowercase)

        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"audio dir not found: {self.audio_dir}")

        self.items = self._index()

    def _index(self):
        files = [f for f in os.listdir(self.audio_dir) if f.lower().endswith(SUPPORTED_AUDIO_EXT)]
        files.sort()
        out = []
        for f in files:
            uid = os.path.splitext(f)[0]
            ap = os.path.join(self.audio_dir, f)
            tp = os.path.join(self.tr_dir, f"{uid}.txt")
            if not os.path.isdir(self.tr_dir) or not os.path.isfile(tp):
                tp = None
            out.append((uid, ap, tp))
            if self.max_items is not None and len(out) >= int(self.max_items):
                break
        return out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        uid, ap, tp = self.items[idx]

        # audio load: librosa only inside try/except
        try:
            import librosa
            wav, _ = librosa.load(ap, sr=self.target_sr, mono=True)
        except Exception:
            # Fallback (например, если librosa не стоит)
            wav = np.zeros(self.target_sr, dtype=np.float32)

        text = None
        if tp is not None:
            with open(tp, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if self.lowercase and text is not None:
                text = text.lower()

        wav_t = torch.tensor(wav, dtype=torch.float32)

        return {
            "wav": wav_t,
            "audio": wav_t,
            "text": text,
            "utt_id": uid,
            "sr": self.target_sr
        }
