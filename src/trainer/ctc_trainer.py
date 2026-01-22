from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from src.utils.metrics import get_cer, get_wer

def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int) -> List[int]:
    ids = torch.argmax(log_probs, dim=-1).tolist()
    out: List[int] = []
    prev = -1
    for i in ids:
        if i != blank_id and i != prev:
            out.append(i)
        prev = i
    return out

def _mask(lengths: torch.Tensor, t: int) -> torch.Tensor:
    return torch.arange(t, device=lengths.device)[None, :] < lengths[:, None]

class CTCTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tokenizer,
        device: str = "cuda",
        blank_id: int = 0,
        use_wandb: bool = True,
        grad_clip: float = 1.0,
        log_aug_examples: bool = True,
        # Аргументы для совместимости
        entropy_weight: float = 0.0,
        blank_prior_weight: float = 0.0,
        blank_prior_target: float = 0.0,
        space_prior_weight: float = 0.0,
        space_id: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.blank_id = int(blank_id)
        self.use_wandb = bool(use_wandb)
        self.grad_clip = float(grad_clip)
        self.log_aug_examples = bool(log_aug_examples)

        self.ctc = torch.nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)
        self.model.to(self.device)

        self.full_validation_history = []

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is None: continue
            g = p.grad.data.norm(2).item()
            total += g * g
        return float(total ** 0.5)

    def train_one_epoch(self, loader, epoch: int, wandb_run=None, log_every: int = 50, scheduler=None) -> Dict[str, float]:
        self.model.train()
        losses = []
        blank_ratios = []

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"train e{epoch}", leave=False)
        for step, batch in pbar:
            feats = batch["feats"].to(self.device)
            feats_len = batch["feats_len"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            tokens_len = batch["tokens_len"].to(self.device)


            if feats.shape[1] > 3000:
                continue

            out = self.model(feats, feats_len)
            log_probs = out["log_probs"]
            log_probs_len = out["log_probs_len"].to(self.device)




            max_target_len = tokens_len.max().item()

            current_input_len = log_probs.shape[1]


            needed_len = max_target_len + 2

            if current_input_len < needed_len:
                pad_amount = needed_len - current_input_len

                log_probs = F.pad(log_probs, (0, 0, 0, pad_amount), value=-10000.0)


                log_probs[:, current_input_len:, self.blank_id] = 0.0


                log_probs_len = torch.max(log_probs_len, tokens_len + 2)

            lp = log_probs.transpose(0, 1).contiguous()

            targets = []
            for i in range(tokens.shape[0]):
                targets.append(tokens[i, : int(tokens_len[i].item())].contiguous())
            targets_1d = torch.cat(targets, dim=0) if len(targets) else torch.zeros((0,), dtype=torch.long, device=self.device)

            loss = self.ctc(lp, targets_1d, log_probs_len, tokens_len)


            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                print(f"[WARN] Step {step} loss explosion: {loss.item()}. Skipping update.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            with torch.no_grad():
                B, T, V = log_probs.shape
                m = _mask(log_probs_len, T)
                denom = m.float().sum().clamp_min(1.0)
                ids = torch.argmax(log_probs, dim=-1)
                br = float((((ids == self.blank_id) & m).float().sum() / denom).item())

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            gnorm = self._grad_norm()
            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            lval = float(loss.item())
            losses.append(lval)
            blank_ratios.append(br)
            pbar.set_postfix(loss=lval, blank=br)

            if self.use_wandb and wandb_run is not None and (step % log_every == 0):
                lr = float(self.optimizer.param_groups[0]["lr"])
                log_dict = {
                    "train/loss": lval,
                    "train/lr": lr,
                    "train/grad_norm": gnorm,
                    "train/blank_ratio": br,
                    "epoch": epoch,
                }

                if self.log_aug_examples and "aug_debug" in batch:
                    import wandb
                    aug_data = batch["aug_debug"]
                    for idx, item in enumerate(aug_data):
                        wav_bef = item["wav_before"].cpu().numpy()
                        wav_aft = item["wav_after"].cpu().numpy()
                        log_dict[f"aug_vis/sample_{idx}_audio_CLEAN"] = wandb.Audio(
                            wav_bef, sample_rate=16000, caption=f"Clean {item['utt_id']}"
                        )
                        log_dict[f"aug_vis/sample_{idx}_audio_NOISY"] = wandb.Audio(
                            wav_aft, sample_rate=16000, caption=f"Augmented {item['utt_id']}"
                        )
                        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                        s_bef = item["spec_before"].transpose(0, 1).cpu().numpy()
                        axes[0].imshow(s_bef, origin="lower", aspect="auto")
                        axes[0].set_title("Clean Spec")
                        axes[0].axis("off")
                        s_aft = item["spec_after"].transpose(0, 1).cpu().numpy()
                        axes[1].imshow(s_aft, origin="lower", aspect="auto")
                        axes[1].set_title("Augmented Spec (Masked)")
                        axes[1].axis("off")
                        plt.tight_layout()
                        log_dict[f"aug_vis/sample_{idx}_specs_compare"] = wandb.Image(fig)
                        plt.close(fig)

                wandb_run.log(log_dict, commit=True)

        return {"loss": float(np.mean(losses)) if len(losses) else 0.0}

    @torch.no_grad()
    def validate(self, loader, epoch: int, wandb_run=None, log_samples: int = 5) -> Dict[str, float]:
        from src.text.ctc_beam_search import ctc_prefix_beam_search
        self.model.eval()
        losses, wers_greedy, wers_beam, cers_greedy, cers_beam = [], [], [], [], []
        blank_ratios, hyp_chars_g, hyp_chars_b = [], [], []
        current_epoch_logs = []

        for batch in tqdm(loader, desc=f"valid e{epoch}", leave=False):
            feats = batch["feats"].to(self.device)
            feats_len = batch["feats_len"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            tokens_len = batch["tokens_len"].to(self.device)
            texts = batch["text"]

            out = self.model(feats, feats_len)
            log_probs = out["log_probs"]
            log_probs_len = out["log_probs_len"].to(self.device)


            max_target_len = tokens_len.max().item()
            current_input_len = log_probs.shape[1]
            needed_len = max_target_len + 2
            if current_input_len < needed_len:
                pad_amount = needed_len - current_input_len
                log_probs = F.pad(log_probs, (0, 0, 0, pad_amount), value=-10000.0)
                log_probs[:, current_input_len:, self.blank_id] = 0.0
                log_probs_len = torch.max(log_probs_len, tokens_len + 2)
            # --------------------------------------------

            lp = log_probs.transpose(0, 1).contiguous()
            targets = []
            for i in range(tokens.shape[0]):
                targets.append(tokens[i, : int(tokens_len[i].item())].contiguous())
            targets_1d = torch.cat(targets, dim=0) if len(targets) else torch.zeros((0,), dtype=torch.long, device=self.device)

            loss = self.ctc(lp, targets_1d, log_probs_len, tokens_len)
            losses.append(float(loss.item()))

            B, T, _ = log_probs.shape
            m = _mask(log_probs_len, T)
            ids = torch.argmax(log_probs, dim=-1)
            denom = m.float().sum().clamp_min(1.0)
            blank_ratios.append(float((((ids == self.blank_id) & m).float().sum() / denom).item()))

            for i in range(B):
                ref = texts[i] if texts[i] is not None else ""
                Ti = int(log_probs_len[i].item())
                lp_sample = log_probs[i, :Ti, :]

                hyp_ids_g = ctc_greedy_decode(lp_sample, blank_id=self.blank_id)
                hyp_greedy = self.tokenizer.decode(hyp_ids_g)

                hyp_beam = hyp_greedy
                if len(current_epoch_logs) < log_samples:
                    try:
                        hyp_ids_b = ctc_prefix_beam_search(lp_sample, 10, self.blank_id)
                        hyp_beam = self.tokenizer.decode(hyp_ids_b)
                    except Exception: pass

                hyp_chars_g.append(len(hyp_greedy.replace(" ", "")))
                hyp_chars_b.append(len(hyp_beam.replace(" ", "")))
                wers_greedy.append(get_wer(ref, hyp_greedy))
                wers_beam.append(get_wer(ref, hyp_beam))
                cers_greedy.append(get_cer(ref, hyp_greedy))
                cers_beam.append(get_cer(ref, hyp_beam))

                if len(current_epoch_logs) < log_samples:
                    audio_obj = None
                    if "wav" in batch:
                        audio = batch["wav"][i]
                        if torch.is_tensor(audio): audio = audio.detach().cpu().numpy()
                        audio_np = audio.astype(np.float32, copy=False)
                        if self.use_wandb and wandb_run:
                            import wandb
                            audio_obj = wandb.Audio(audio_np, sample_rate=16000, caption=f"Ref: {ref}")
                    current_epoch_logs.append([epoch, ref, hyp_greedy, hyp_beam, wers_greedy[-1], wers_beam[-1], cers_greedy[-1], cers_beam[-1], audio_obj])

        res = {
            "loss": float(np.mean(losses)) if len(losses) else 0.0,
            "wer_greedy": float(np.mean(wers_greedy)) if len(wers_greedy) else 0.0,
            "wer_beam": float(np.mean(wers_beam)) if len(wers_beam) else 0.0,
            "cer_greedy": float(np.mean(cers_greedy)) if len(cers_greedy) else 0.0,
            "cer_beam": float(np.mean(cers_beam)) if len(cers_beam) else 0.0,
            "blank_ratio": float(np.mean(blank_ratios)) if len(blank_ratios) else 0.0,
            "avg_hyp_chars_g": float(np.mean(hyp_chars_g)) if len(hyp_chars_g) else 0.0,
            "avg_hyp_chars_b": float(np.mean(hyp_chars_b)) if len(hyp_chars_b) else 0.0,
        }

        if self.use_wandb and wandb_run:
            import wandb
            wandb_run.log({
                "valid/loss": res["loss"], "valid/wer_greedy": res["wer_greedy"], "valid/wer_beam": res["wer_beam"],
                "valid/cer_greedy": res["cer_greedy"], "valid/cer_beam": res["cer_beam"], "valid/blank_ratio": res["blank_ratio"],
                "valid/avg_hyp_chars_g": res["avg_hyp_chars_g"], "valid/avg_hyp_chars_b": res["avg_hyp_chars_b"], "epoch": epoch
            }, commit=True)
            self.full_validation_history.extend(current_epoch_logs)
            wandb_run.log({"validation_table": wandb.Table(columns=["Epoch", "Ref", "Greedy", "Beam Search", "WER (G)", "WER (B)", "CER (G)", "CER (B)", "Audio"], data=self.full_validation_history)}, commit=True)

        return res
