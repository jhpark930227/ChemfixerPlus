from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .batching import decoder_batch, encoder_batch, move_batch
from .model import ChemFixerPlusModel
from .seed import dataloader_generator, seed_everything
from .tokenizer import SmilesTokenizer, Vocabulary


class MaskedSmilesDataset(Dataset):
    def __init__(
        self,
        smiles: Sequence[str],
        tokenizer: SmilesTokenizer,
        masking_probability: float = 0.10,
        seed: int = 0,
    ) -> None:
        self.smiles = list(smiles)
        self.tokenizer = tokenizer
        self.masking_probability = masking_probability
        self.seed = seed
        self.tokens = [tokenizer.tokenize(s) for s in self.smiles]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        original = self.tokens[idx]
        masked = ["<MASK>" if rng.random() < self.masking_probability else t for t in original]
        if original and masked == original:
            masked[rng.randrange(len(masked))] = "<MASK>"
        return {"masked": masked, "target": original}


def make_pretrain_batch(items: Sequence[dict], vocab: Vocabulary) -> dict:
    enc = [["<BOS>", *item["masked"], "<EOS>"] for item in items]
    tgt = [item["target"] for item in items]
    return {
        "encoder": encoder_batch(enc, vocab),
        "decoder": decoder_batch(tgt, vocab),
    }


def _scheduler(optimizer, updates: int, name: str, min_lr: float):
    name = name.lower()
    if name in {"none", "constant"}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates), eta_min=min_lr
    )


def pretrain_steps(
    model: ChemFixerPlusModel,
    dataset: MaskedSmilesDataset,
    vocab: Vocabulary,
    steps: int,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cpu",
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    scheduler: str = "cosine",
    min_lr: float = 0.0,
    seed: int = 0,
    deterministic: bool = False,
) -> list[dict[str, float]]:
    """Masked full-sequence reconstruction for the shared base checkpoint."""
    seed_everything(seed, deterministic=deterministic)
    dev = torch.device(device)
    model.to(dev)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=dataloader_generator(seed),
        collate_fn=lambda xs: make_pretrain_batch(xs, vocab),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _scheduler(opt, steps, scheduler, min_lr)
    history: list[dict[str, float]] = []
    it = iter(loader)

    for update in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _micro in range(grad_accum_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = move_batch(batch, dev)
            enc = batch["encoder"]
            dec_in, dec_mask, labels = batch["decoder"]
            # Shared base pretraining does not use ChemFixer+-specific cues/heads.
            memory = model.encode(**enc, use_structure=False)
            logits = model.decode(dec_in, dec_mask, memory, enc["padding_mask"])
            loss = model.sequence_ce(logits, labels)
            (loss / grad_accum_steps).backward()
            loss_sum += float(loss.detach().cpu())

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        current_lr = float(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step()
        history.append({"step": float(update + 1), "loss": loss_sum / grad_accum_steps, "lr": current_lr})
    return history
