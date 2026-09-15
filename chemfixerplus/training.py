from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .batching import make_training_batch, move_batch
from .data import ChemFixerPairDataset
from .model import ChemFixerPlusModel
from .seed import dataloader_generator, seed_everything
from .tokenizer import Vocabulary


def compute_losses(
    model: ChemFixerPlusModel,
    batch: dict,
    token_pos_weight: float,
    gap_pos_weight: float,
) -> dict[str, torch.Tensor]:
    # Localization on the original invalid input.
    xp = batch["x_plain"]
    mem_plain = model.encode(**xp)
    tok_logits, gap_logits = model.localization_logits(mem_plain, batch["source_lengths"])
    l_tok = model.masked_bce_with_logits(
        tok_logits, batch["token_labels"], batch["token_label_mask"], token_pos_weight
    )
    l_gap = model.masked_bce_with_logits(
        gap_logits, batch["gap_labels"], batch["gap_label_mask"], gap_pos_weight
    )
    l_loc = l_tok + l_gap

    # Global full-sequence correction on every pair, conditioned on original X.
    xg = batch["x_global"]
    mem_global = model.encode(**xg)
    g_in, g_mask, g_labels = batch["global_decoder"]
    g_logits = model.decode(g_in, g_mask, mem_global, xg["padding_mask"])
    l_global = model.sequence_ce(g_logits, g_labels)

    # Local repair-only decoding only for locally eligible aligned pairs.
    if batch["x_local"] is not None:
        xl = batch["x_local"]
        mem_local = model.encode(**xl)
        l_in, l_mask, l_labels = batch["local_decoder"]
        l_logits = model.decode(l_in, l_mask, mem_local, xl["padding_mask"])
        eligible_mean = model.sequence_ce(l_logits, l_labels)
        # Manuscript: ineligible pairs contribute zero L_rep, then per-sequence
        # losses are averaged over the full batch.
        l_repair = eligible_mean * (len(batch["local_indices"]) / len(batch["items"]))
    else:
        l_repair = l_loc.new_zeros(())

    total = l_loc + l_repair + model.cfg.lambda_global * l_global
    return {
        "total": total,
        "loc": l_loc,
        "repair": l_repair,
        "global": l_global,
        "token_loc": l_tok,
        "gap_loc": l_gap,
    }


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    updates: int,
    scheduler: str,
    min_lr: float,
):
    name = scheduler.lower()
    if name in {"none", "constant"}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, updates),
        eta_min=min_lr,
    )


def train_steps(
    model: ChemFixerPlusModel,
    dataset: ChemFixerPairDataset,
    vocab: Vocabulary,
    steps: int,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    scheduler: str = "constant",
    min_lr: float = 0.0,
    seed: int = 0,
    deterministic: bool = False,
    scheduler_updates: int | None = None,
    checkpoint_every: int = 0,
    checkpoint_prefix: str | Path | None = None,
) -> list[dict[str, float]]:
    """Run `steps` optimizer updates.

    `batch_size` is the micro-batch size.  Effective batch size is
    `batch_size * grad_accum_steps`, matching the manuscript's effective-batch
    wording while remaining practical on 24-GB GPUs.
    """
    if steps <= 0:
        return []
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    seed_everything(seed, deterministic=deterministic)
    dev = torch.device(device)
    model.to(dev)
    token_pw, gap_pw = dataset.localization_pos_weights()
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=dataloader_generator(seed),
        collate_fn=lambda xs: make_training_batch(xs, vocab),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _cosine_scheduler(
        opt,
        scheduler_updates or steps,
        scheduler,
        min_lr,
    )

    history: list[dict[str, float]] = []
    iterator = iter(loader)
    for update in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}

        for _micro in range(grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, dev)
            losses = compute_losses(model, batch, token_pw, gap_pw)
            (losses["total"] / grad_accum_steps).backward()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        current_lr = float(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step()

        row = {k: v / grad_accum_steps for k, v in sums.items()}
        row["step"] = float(update + 1)
        row["lr"] = current_lr
        if not math.isfinite(row["total"]):
            raise RuntimeError("Non-finite training loss")
        history.append(row)

        # Periodic model snapshot.
        if (
            checkpoint_every > 0
            and checkpoint_prefix is not None
            and (
                (update + 1) % checkpoint_every == 0
                or (update + 1) == steps
            )
        ):
            prefix = Path(checkpoint_prefix)
            suffix = prefix.suffix or ".pt"
            ckpt = prefix.with_name(
                f"{prefix.stem}_step{update + 1:05d}{suffix}"
            )
            save_checkpoint(model, vocab, ckpt)
            print(f"checkpoint={ckpt.resolve()}", flush=True)

        # Live training progress.
        if (update + 1) == 1 or (update + 1) % 10 == 0 or (update + 1) == steps:
            seen = (update + 1) * batch_size * grad_accum_steps
            approx_epoch = seen / len(dataset)

            print(
                f"[{update + 1:5d}/{steps}] "
                f"epoch~{approx_epoch:.2f} "
                f"total={row['total']:.4f} "
                f"loc={row['loc']:.4f} "
                f"repair={row['repair']:.4f} "
                f"global={row['global']:.4f} "
                f"lr={row['lr']:.3e}",
                flush=True,
            )

    return history


def save_checkpoint(model: ChemFixerPlusModel, vocab: Vocabulary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.config_dict(),
        "vocab": vocab.token_to_id,
    }, path)
