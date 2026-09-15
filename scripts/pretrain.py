#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from chemfixerplus.chemistry import is_valid_smiles
from chemfixerplus.data import read_smiles_file
from chemfixerplus.model import ChemFixerPlusModel, ModelConfig
from chemfixerplus.pretraining import MaskedSmilesDataset, pretrain_steps
from chemfixerplus.seed import seed_everything
from chemfixerplus.tokenizer import SmilesTokenizer, Vocabulary
from chemfixerplus.training import save_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smiles", required=True, help="Local valid-SMILES CSV/TXT file")
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--output", default="checkpoints/pretrained.pt")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    cfg_obj = yaml.safe_load(Path(args.config).read_text())
    model_cfg = ModelConfig(**cfg_obj["model"])
    tr_cfg = cfg_obj["training"]
    seed = int(tr_cfg.get("seed", 0) if args.seed is None else args.seed)
    seed_everything(seed, deterministic=args.deterministic)

    tok = SmilesTokenizer()
    smiles = [
        s for s in read_smiles_file(args.smiles)
        if tok.is_lossless(s) and is_valid_smiles(s) and len(tok.tokenize(s)) <= 254
    ]
    if not smiles:
        raise SystemExit("No valid <=254-token SMILES found")
    seqs = [tok.tokenize(s) for s in smiles]
    # Base checkpoint excludes Plus-only mode/sentinel tokens. They are appended
    # during paired fine-tuning while preserving all existing token IDs.
    vocab = Vocabulary.build(seqs, include_plus_tokens=False)
    dataset = MaskedSmilesDataset(
        smiles, tok,
        masking_probability=float(tr_cfg.get("masking_probability", 0.10)),
        seed=seed,
    )
    model = ChemFixerPlusModel(vocab, model_cfg)
    hist = pretrain_steps(
        model, dataset, vocab,
        steps=args.steps or int(tr_cfg["updates"]),
        batch_size=int(tr_cfg.get("micro_batch_size", tr_cfg.get("batch_size", 64))),
        lr=float(tr_cfg["lr"]),
        device=args.device,
        grad_clip=float(tr_cfg.get("grad_clip", 1.0)),
        grad_accum_steps=int(tr_cfg.get("grad_accum_steps", 1)),
        scheduler=str(tr_cfg.get("scheduler", "cosine")),
        min_lr=float(tr_cfg.get("min_lr", 0.0)),
        seed=seed,
        deterministic=args.deterministic,
    )
    save_checkpoint(model, vocab, args.output)
    print(f"valid_smiles={len(smiles)} vocab={len(vocab)} seed={seed}")
    print(f"first_loss={hist[0]['loss']:.4f} last_loss={hist[-1]['loss']:.4f}")
    print(f"first_lr={hist[0]['lr']:.8g} last_lr={hist[-1]['lr']:.8g}")
    print(f"saved={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
