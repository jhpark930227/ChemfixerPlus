#!/usr/bin/env python

import argparse
import math
import re
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from chemfixerplus.batching import make_training_batch, move_batch
from chemfixerplus.checkpoint import load_checkpoint
from chemfixerplus.data import ChemFixerPairDataset, read_pairs_jsonl
from chemfixerplus.seed import dataloader_generator, seed_everything
from chemfixerplus.tokenizer import SmilesTokenizer
from chemfixerplus.training import compute_losses


def save_full(path, model, vocab, optimizer, global_step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config_dict(),
        "vocab": vocab.token_to_id,
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
    }

    # Atomic save: write completely first, then rename.
    tmp_path = path.with_name(path.name + ".tmp")

    torch.save(
        payload,
        tmp_path,
    )

    tmp_path.replace(path)

    print(
        f"checkpoint={path.resolve()}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--pairs", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", required=True)

    p.add_argument("--target-step", type=int, required=True)
    p.add_argument("--scheduler-steps", type=int, default=20000)
    p.add_argument("--checkpoint-every", type=int, default=188)

    p.add_argument("--output-prefix", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2027)

    args = p.parse_args()

    seed_everything(args.seed)

    # --------------------------------------------------------
    # Load model + vocab
    # --------------------------------------------------------
    model, vocab = load_checkpoint(
        args.resume,
        map_location="cpu",
    )

    raw = torch.load(
        args.resume,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # Determine completed global step
    # --------------------------------------------------------
    if "global_step" in raw:
        start_step = int(raw["global_step"])
    else:
        m = re.search(
            r"_step(\d+)",
            Path(args.resume).stem,
        )

        if not m:
            raise RuntimeError(
                "Could not infer starting step from checkpoint filename."
            )

        start_step = int(m.group(1))

    if start_step >= args.target_step:
        raise RuntimeError(
            f"Checkpoint already at step {start_step}, "
            f"target is {args.target_step}."
        )

    print("=" * 72)
    print("RESUME TRAINING")
    print("=" * 72)
    print("checkpoint   :", args.resume)
    print("start step   :", start_step)
    print("target step  :", args.target_step)
    print("remaining    :", args.target_step - start_step)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    pairs = read_pairs_jsonl(args.pairs)

    dataset = ChemFixerPairDataset(
        pairs,
        SmilesTokenizer(),
    )

    cfg = yaml.safe_load(
        Path(args.config).read_text()
    )

    tr = cfg["training"]

    micro_batch = int(
        tr.get(
            "micro_batch_size",
            tr.get("batch_size", 64),
        )
    )

    grad_accum = int(
        tr.get("grad_accum_steps", 1)
    )

    effective = micro_batch * grad_accum

    print("pairs        :", len(dataset))
    print("micro batch  :", micro_batch)
    print("grad accum   :", grad_accum)
    print("effective    :", effective)
    print("vocab        :", len(vocab))

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        generator=dataloader_generator(
            args.seed + start_step
        ),
        collate_fn=lambda xs:
            make_training_batch(xs, vocab),
    )

    iterator = iter(loader)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    device = torch.device(args.device)
    model.to(device)

    base_lr = float(tr["lr"])
    min_lr = float(tr.get("min_lr", 0.0))
    grad_clip = float(tr.get("grad_clip", 1.0))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
    )

    # If this is a full resume checkpoint, restore Adam state.
    if "optimizer_state" in raw:
        optimizer.load_state_dict(
            raw["optimizer_state"]
        )
        print("optimizer    : RESTORED")
    else:
        print(
            "optimizer    : FRESH "
            "(old checkpoint contained weights only)"
        )

    token_pw, gap_pw = (
        dataset.localization_pos_weights()
    )

    # --------------------------------------------------------
    # Continue to target global step.
    #
    # LR is calculated from the ORIGINAL 20K cosine horizon,
    # not restarted from zero.
    # --------------------------------------------------------
    for global_step in range(
        start_step + 1,
        args.target_step + 1,
    ):
        model.train()

        # Original cosine schedule position.
        t = global_step - 1

        lr = (
            min_lr
            + 0.5
            * (base_lr - min_lr)
            * (
                1.0
                + math.cos(
                    math.pi
                    * t
                    / args.scheduler_steps
                )
            )
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(
            set_to_none=True
        )

        sums = {}

        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            batch = move_batch(
                batch,
                device,
            )

            losses = compute_losses(
                model,
                batch,
                token_pw,
                gap_pw,
            )

            (
                losses["total"]
                / grad_accum
            ).backward()

            for key, value in losses.items():
                sums[key] = (
                    sums.get(key, 0.0)
                    + float(
                        value.detach().cpu()
                    )
                )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )

        optimizer.step()

        row = {
            k: v / grad_accum
            for k, v in sums.items()
        }

        approx_epoch = (
            global_step
            * effective
            / len(dataset)
        )

        if (
            global_step == start_step + 1
            or global_step % 10 == 0
            or global_step == args.target_step
        ):
            print(
                f"[{global_step:5d}/{args.target_step}] "
                f"epoch~{approx_epoch:.2f} "
                f"total={row['total']:.4f} "
                f"loc={row['loc']:.4f} "
                f"repair={row['repair']:.4f} "
                f"global={row['global']:.4f} "
                f"lr={lr:.3e}",
                flush=True,
            )

        if (
            global_step % args.checkpoint_every == 0
            or global_step == args.target_step
        ):
            out = (
                f"{args.output_prefix}"
                f"_step{global_step:05d}.pt"
            )

            save_full(
                out,
                model,
                vocab,
                optimizer,
                global_step,
            )

    final_path = (
        f"{args.output_prefix}_final.pt"
    )

    save_full(
        final_path,
        model,
        vocab,
        optimizer,
        args.target_step,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print("final step :", args.target_step)
    print("saved      :", final_path)


if __name__ == "__main__":
    main()
