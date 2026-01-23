from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# === ИСПРАВЛЕНО: src.datasets вместо src.asr_datasets ===
from src.datasets.librispeech_hf import LibriSpeechHF
from src.datasets.custom_dir_dataset import CustomDirDataset

from src.text.char_tokenizer import CharTokenizer
from src.model.ctc_model import CTCBiLSTM
from src.utils.dataloader import ASRCollator
from src.trainer.ctc_trainer import ctc_greedy_decode
from src.text.ctc_beam_search import ctc_prefix_beam_search


def save_txt(out_dir: str, utt_id: str, text: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{utt_id}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
@torch.no_grad()
def main(cfg: DictConfig):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    ap.add_argument("--checkpoint", type=str, default="weights/best.pt")
    ap.add_argument("--out_dir", type=str, default="predictions")
    ap.add_argument("--decode", type=str, default="beam", choices=["beam", "greedy"])
    ap.add_argument("--beam_size", type=int, default=10)


    args, _ = ap.parse_known_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    tokenizer = CharTokenizer()


    source = cfg.dataset.train_source

    if source == "librispeech":

        ds = LibriSpeechHF(
            split=cfg.dataset.valid_split,
            target_sr=16000,
            max_items=getattr(cfg.dataset, "max_valid_items", 200),
        )
    else:

        custom_root = cfg.dataset.get("custom_root", None)
        if custom_root is None:
            raise ValueError("В конфиге Hydra (dataset.custom_root) не указан путь к данным!")
        ds = CustomDirDataset(root_dir=custom_root, target_sr=16000)

    collate = ASRCollator(tokenizer, wav_augs=[], spec_augs=[], return_aug_debug=False)
    dl = DataLoader(ds, batch_size=getattr(cfg.training, "batch_size", 8), shuffle=False, num_workers=0, collate_fn=collate)

    model = CTCBiLSTM(
        n_mels=80,
        vocab_size=tokenizer.vocab_size,
        enc_dim=256,
        num_layers=4,
        dropout=0.1,
        subsample_ch=128,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(torch.device(args.device))
    model.eval()

    n_saved = 0
    for batch in dl:
        feats = batch["feats"].to(args.device)
        feats_len = batch["feats_len"].to(args.device)
        utt_ids = batch["utt_id"]

        out = model(feats, feats_len)
        lp = out["log_probs"]
        ll = out["log_probs_len"]

        for i in range(lp.shape[0]):
            T = int(ll[i].item())
            lp_i = lp[i, :T, :].contiguous()  # [T, V]

            if args.decode == "greedy":
                ids = ctc_greedy_decode(lp_i, blank_id=0)
            else:
                ids = ctc_prefix_beam_search(lp_i, beam_size=args.beam_size, blank_id=0)

            text = tokenizer.decode(ids)
            save_txt(args.out_dir, utt_ids[i], text)
            n_saved += 1

    print("Saved:", n_saved, "files to", args.out_dir)


if __name__ == "__main__":
    main()
