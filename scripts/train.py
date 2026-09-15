#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from chemfixerplus.checkpoint import load_pretrained_for_finetune
from chemfixerplus.data import ChemFixerPairDataset, read_pairs_jsonl
from chemfixerplus.model import ChemFixerPlusModel, ModelConfig
from chemfixerplus.seed import seed_everything
from chemfixerplus.tokenizer import SmilesTokenizer, Vocabulary
from chemfixerplus.training import save_checkpoint, train_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="Training correction pairs (JSONL)")
    p.add_argument(
        "--validation-pairs",
        default=None,
        help=(
            "Optional fixed validation correction pairs (JSONL) used only "
            "for checkpoint selection."
        ),
    )
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument(
        "--output",
        default="checkpoints/chemfixerplus_last.pt",
        help="Final-update checkpoint.",
    )
    p.add_argument(
        "--best-output",
        default="checkpoints/chemfixerplus_best.pt",
        help="Lowest-validation-loss checkpoint when --validation-pairs is used.",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--legacy-vocab", default=None)
    p.add_argument(
        "--pretrained",
        default=None,
        help="Optional masked-pretraining checkpoint",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--scheduler-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--checkpoint-prefix", default=None)
    p.add_argument(
        "--validation-every",
        type=int,
        default=None,
        help=(
            "Validation interval in optimizer updates. Defaults to "
            "training.validation_every from the config."
        ),
    )
    p.add_argument(
        "--validation-batch-size",
        type=int,
        default=None,
        help="Validation batch size. Defaults to the training micro-batch size.",
    )
    args = p.parse_args()

    cfg_obj = yaml.safe_load(Path(args.config).read_text())
    model_cfg = ModelConfig(**cfg_obj["model"])
    tr_cfg = cfg_obj["training"]
    seed = int(
        tr_cfg.get("seed", 0)
        if args.seed is None
        else args.seed
    )
    seed_everything(seed, deterministic=args.deterministic)

    tok = SmilesTokenizer()

    train_pairs = read_pairs_jsonl(args.pairs)
    dataset = ChemFixerPairDataset(train_pairs, tok)

    validation_dataset = None
    if args.validation_pairs:
        validation_pairs = read_pairs_jsonl(args.validation_pairs)
        validation_dataset = ChemFixerPairDataset(
            validation_pairs,
            tok,
        )
        if len(validation_dataset) == 0:
            raise SystemExit("Validation pair file is empty.")

    if args.pretrained:
        model, vocab = load_pretrained_for_finetune(
            args.pretrained,
            dataset.all_token_sequences(),
            seed=seed,
            map_location="cpu",
        )
        # Paper-size paired fine-tuning should retain the architecture of the
        # shared checkpoint. A mismatched config is flagged instead of silently
        # changing the backbone.
        if model.config_dict() != model_cfg.__dict__:
            raise SystemExit(
                "Config/model mismatch: paired fine-tuning must use the same "
                "backbone architecture as the masked-pretraining checkpoint."
            )
    else:
        legacy = (
            Vocabulary.load(args.legacy_vocab)
            if args.legacy_vocab
            else None
        )
        vocab = Vocabulary.build(
            dataset.all_token_sequences(),
            legacy_vocab=legacy,
            include_plus_tokens=True,
        )
        model = ChemFixerPlusModel(vocab, model_cfg)

    micro_batch = int(
        tr_cfg.get(
            "micro_batch_size",
            tr_cfg.get("batch_size", 64),
        )
    )
    grad_accum = int(tr_cfg.get("grad_accum_steps", 1))
    expected_effective = int(
        tr_cfg.get(
            "effective_batch_size",
            micro_batch * grad_accum,
        )
    )
    if micro_batch * grad_accum != expected_effective:
        raise SystemExit(
            "effective_batch_size mismatch: "
            f"{micro_batch} x {grad_accum} != "
            f"{expected_effective}"
        )

    validation_every = 0
    validation_batch_size = None
    best_output = None

    if validation_dataset is not None:
        validation_every = int(
            args.validation_every
            if args.validation_every is not None
            else tr_cfg.get("validation_every", 5000)
        )
        if validation_every <= 0:
            raise SystemExit(
                "validation_every must be >= 1 when "
                "--validation-pairs is provided."
            )

        validation_batch_size = int(
            args.validation_batch_size
            if args.validation_batch_size is not None
            else tr_cfg.get(
                "validation_batch_size",
                micro_batch,
            )
        )
        if validation_batch_size <= 0:
            raise SystemExit(
                "validation_batch_size must be >= 1."
            )

        if Path(args.best_output).resolve() == Path(args.output).resolve():
            raise SystemExit(
                "--best-output and --output must be different paths."
            )
        best_output = args.best_output

    history = train_steps(
        model,
        dataset,
        vocab,
        steps=args.steps or int(tr_cfg["updates"]),
        batch_size=micro_batch,
        lr=float(tr_cfg["lr"]),
        device=args.device,
        grad_clip=float(tr_cfg.get("grad_clip", 1.0)),
        grad_accum_steps=grad_accum,
        scheduler=str(tr_cfg.get("scheduler", "cosine")),
        min_lr=float(tr_cfg.get("min_lr", 0.0)),
        seed=seed,
        deterministic=args.deterministic,
        scheduler_updates=args.scheduler_steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_prefix=args.checkpoint_prefix,
        validation_dataset=validation_dataset,
        validation_every=validation_every,
        validation_batch_size=validation_batch_size,
        best_checkpoint=best_output,
    )

    # Always preserve the final-update checkpoint separately from the
    # validation-selected checkpoint.
    save_checkpoint(model, vocab, args.output)

    print(
        f"train_pairs={len(dataset)} "
        f"validation_pairs="
        f"{0 if validation_dataset is None else len(validation_dataset)} "
        f"vocab={len(vocab)} seed={seed} "
        f"micro_batch={micro_batch} grad_accum={grad_accum} "
        f"effective_batch={expected_effective}"
    )
    print(
        f"first_loss={history[0]['total']:.4f} "
        f"last_loss={history[-1]['total']:.4f}"
    )
    print(
        f"first_lr={history[0]['lr']:.8g} "
        f"last_lr={history[-1]['lr']:.8g}"
    )
    print(f"last_checkpoint={Path(args.output).resolve()}")
    if best_output is not None:
        print(f"best_checkpoint={Path(best_output).resolve()}")


if __name__ == "__main__":
    main()
