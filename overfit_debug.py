import torch
import hydra
import sys
import os

ROOT = "/content/asr_project"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.text.char_tokenizer import CharTokenizer
from src.model.ctc_model import CTCBiLSTM
from src.utils.dataloader import ASRCollator, build_loader

@hydra.main(version_base=None, config_path="/content/asr_project/src/configs", config_name="config")
def main(cfg):
    print("=== ФИНАЛЬНЫЙ ТЕСТ: ДОЖИМАЕМ ДО ТЕКСТА ===")
    device = "cpu"


    dataset = hydra.utils.instantiate(cfg.dataset)
    first_item = dataset[0]
    target_text = first_item['text']
    print(f"\n[ЦЕЛЬ]: {target_text}")

    class SingleItemDataset(torch.utils.data.Dataset):
        def __init__(self, item): self.item = item
        def __len__(self): return 4
        def __getitem__(self, idx): return self.item

    tokenizer = CharTokenizer()
    loader = build_loader(SingleItemDataset(first_item), ASRCollator(tokenizer), batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))


    model = CTCBiLSTM(
        n_mels=80,
        vocab_size=tokenizer.vocab_size,
        enc_dim=64,
        num_layers=2,
        dropout=0.0,
        subsample_ch=32,
        blank_id=0,
    ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()

    for i in range(1, 301):
        feats = batch["feats"].to(device)
        feats_len = batch["feats_len"].to(device)
        tokens = batch["tokens"].to(device)
        tokens_len = batch["tokens_len"].to(device)

        optimizer.zero_grad()
        out = model(feats, feats_len)
        lp = out["log_probs"].transpose(0, 1)

        targets = torch.cat([tokens[j, : int(tokens_len[j])] for j in range(tokens.shape[0])])

        loss = ctc_loss(lp, targets, out["log_probs_len"], tokens_len)
        loss.backward()
        optimizer.step()

        if i % 20 == 0:

            with torch.no_grad():
                probs = out["log_probs"][0].exp()
                ids = torch.argmax(probs, dim=-1).tolist()
                decoded = []
                prev = -1
                for idx in ids:
                    if idx != 0 and idx != prev: decoded.append(idx)
                    prev = idx
                text = tokenizer.decode(decoded)

                print(f"Step {i} | Loss: {loss.item():.4f} | Pred: {text[:50]}...")

                if loss.item() < 0.5:
                    print("\n[УСПЕХ] Loss упал ниже 0.5! Модель выучила пример.")
                    break

if __name__ == "__main__":
    main()
