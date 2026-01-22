from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa


import sys
if "datasets" in sys.modules:
    del sys.modules["datasets"]
import datasets as hf_datasets


class LibriSpeechHF(Dataset):
    def __init__(
        self,
        split: str = "train.clean.100",
        target_sr: int = 16000,
        max_items: Optional[int] = None,
    ) -> None:
        self.target_sr = int(target_sr)


        ds = hf_datasets.load_dataset("librispeech_asr", split=split, trust_remote_code=True)


        self.ds = ds.decode(False)

        if max_items is not None:
            limit = min(int(max_items), len(ds))
            self.ds = self.ds.select(range(limit))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        item = self.ds[idx]
        audio_dict = item["audio"]

        import io

        try:
            if isinstance(audio_dict, dict) and audio_dict.get("bytes"):
                wav, _ = librosa.load(io.BytesIO(audio_dict["bytes"]), sr=self.target_sr)
            else:
                wav, _ = librosa.load(audio_dict["path"], sr=self.target_sr)
        except Exception:

            wav = np.zeros(self.target_sr, dtype=np.float32)

        wav = torch.from_numpy(wav).float()
        if wav.ndim > 1:
            wav = wav.mean(dim=0)

        text = (item.get("text", "") or "").strip().lower()
        uid = str(item.get("id", idx))
        return {"utt_id": uid, "wav": wav, "sr": self.target_sr, "text": text}
