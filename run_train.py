from __future__ import annotations

import os
import sys
import argparse

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = "/content/asr_project"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.text.char_tokenizer import CharTokenizer
from src.model.ctc_model import CTCBiLSTM
from src.trainer.ctc_trainer import CTCTrainer
from src.utils.dataloader import ASRCollator, build_loader
from src.augmentations.spec_augs import TimeStretch, AddGaussianNoise, TimeMask, FrequencyMask

@hydra.main(version_base=None, config_path="/content/asr_project/src/configs", config_name="config")
def main(cfg: DictConfig):

    args = argparse.Namespace(**OmegaConf.to_container(cfg.training, resolve=True))

    device = str(getattr(args, "device", "cpu"))
    print(f"Using device: {device}")

    save_dir = getattr(args, "save_dir", "weights")
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = CharTokenizer()


    train_ds = hydra.utils.instantiate(cfg.dataset)
    val_ds = hydra.utils.instantiate(cfg.dataset, max_items=400)


    wav_augs = [AddGaussianNoise(sigma=0.005), TimeStretch(min_rate=0.9, max_rate=1.1)]
    spec_augs = [FrequencyMask(max_width=20), TimeMask(max_width=60)]

    use_wandb = bool(getattr(args, "use_wandb", False))
    log_aug_examples = bool(getattr(args, "log_aug_examples", False))

    collate_train = ASRCollator(
        tokenizer,
        wav_augs=wav_augs,
        spec_augs=spec_augs,
        return_aug_debug=(use_wandb and log_aug_examples),
        aug_debug_max_items=4,
    )
    collate_valid = ASRCollator(tokenizer, wav_augs=[], spec_augs=[])

    batch_size = int(getattr(args, "batch_size", 16))

    train_loader = build_loader(
        train_ds, collate_train,
        batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    valid_loader = build_loader(
        val_ds, collate_valid,
        batch_size=batch_size,
        shuffle=False, num_workers=2
    )


    model = CTCBiLSTM(
        n_mels=80,
        vocab_size=tokenizer.vocab_size,
        enc_dim=cfg.model.enc_dim,      # 256
        num_layers=cfg.model.num_layers, # 4
        dropout=cfg.model.dropout,       # 0.2
        subsample_ch=cfg.model.subsample_ch, # 128
        blank_id=0,
    )


    epochs = int(getattr(args, "epochs", 50))
    max_lr = 3e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Scheduler: OneCycleLR with max_lr={max_lr}, epochs={epochs}")

    run = None
    if use_wandb:
        run = wandb.init(
            project="asr_project",
            name=str(getattr(args, "run_name", "ctc_big_onecycle")),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = CTCTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        device=device,
        blank_id=0,
        use_wandb=use_wandb,
        grad_clip=10.0,
        log_aug_examples=log_aug_examples,
    )

    best_wer = 1e9

    for epoch in range(1, epochs + 1):
        tr = trainer.train_one_epoch(
            train_loader, epoch=epoch, wandb_run=run,
            log_every=int(getattr(args, "log_every", 50)),
            scheduler=scheduler
        )


        va = trainer.validate(
            valid_loader, epoch=epoch, wandb_run=run,
            log_samples=int(getattr(args, "log_samples", 3))
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={tr['loss']:.4f} | valid_loss={va['loss']:.4f} | "
            f"WER_G={va['wer_greedy']:.4f} | WER_B={va['wer_beam']:.4f} | "
            f"CER_G={va['cer_greedy']:.4f} | CER_B={va['cer_beam']:.4f}"
        )

        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "valid": va}, os.path.join(save_dir, "last.pt"))
        if va["wer_beam"] < best_wer:
            best_wer = va["wer_beam"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "valid": va}, os.path.join(save_dir, "best.pt"))

    if run: run.finish()

if __name__ == "__main__":
    if "-f" in sys.argv:
        try:
            idx = sys.argv.index("-f")
            del sys.argv[idx:idx + 2]
        except ValueError: pass
    main()
