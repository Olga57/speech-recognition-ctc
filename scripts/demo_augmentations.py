from __future__ import annotations

import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from asr_datasets.librispeech_hf import LibriSpeechHF
from asr_datasets.custom_dir_dataset import CustomDirDataset
from utils.features import log_mel_spectrogram
from augmentations.spec_augs import TimeStretch, AddGaussianNoise, TimeMask, FrequencyMask


def plot_spec(spec: torch.Tensor, title: str, out_path: str) -> None:
    # spec: [T, M]
    plt.figure()
    plt.imshow(spec.transpose(0, 1).cpu().numpy(), aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="librispeech", choices=["librispeech", "customdir"])
    ap.add_argument("--split", type=str, default="validation.clean")
    ap.add_argument("--custom_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="outputs/augs_demo")
    ap.add_argument("--num_items", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.source == "librispeech":
        ds = LibriSpeechHF(split=args.split, target_sr=16000, max_items=args.num_items)
    else:
        if args.custom_root is None:
            raise ValueError("--custom_root required for customdir")
        ds = CustomDirDataset(root_dir=args.custom_root, target_sr=16000, max_items=args.num_items)

    wav_augs = [
        AddGaussianNoise(sigma=0.005),
        TimeStretch(min_rate=0.9, max_rate=1.1),
    ]
    spec_augs = [
        FrequencyMask(max_width=12),
        TimeMask(max_width=40),
    ]

    for i in range(min(args.num_items, len(ds))):
        item = ds[i]
        uid = item["utt_id"]
        wav = item["wav"].float()

        spec_before = log_mel_spectrogram(wav, sr=16000)
        plot_spec(spec_before, f"{uid} - BEFORE", os.path.join(args.out_dir, f"{uid}_before.png"))

        wav_aug = wav
        for a in wav_augs:
            wav_aug = a(wav_aug)

        spec_after = log_mel_spectrogram(wav_aug, sr=16000)
        for a in spec_augs:
            spec_after = a(spec_after)

        plot_spec(spec_after, f"{uid} - AFTER", os.path.join(args.out_dir, f"{uid}_after.png"))

    print("Saved demo images to:", args.out_dir)


if __name__ == "__main__":
    main()
