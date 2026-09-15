#!/usr/bin/env python
"""Tiny end-to-end integration test on the worked example from Fig. 1.

This intentionally overfits one pair. It is a software test, not an experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from chemfixerplus.data import ChemFixerPairDataset, PairExample
from chemfixerplus.inference import ChemFixerPredictor
from chemfixerplus.model import ChemFixerPlusModel, ModelConfig
from chemfixerplus.tokenizer import SmilesTokenizer, Vocabulary
from chemfixerplus.training import train_steps

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(0)
    invalid = "CCC(=O)O)CC1CCCCC"
    valid = "CCC(=O)OCC1CCCCC1"
    tok = SmilesTokenizer()
    ds = ChemFixerPairDataset([PairExample(invalid, valid, {"source": "paper_figure_1"})], tok)
    vocab = Vocabulary.build(ds.all_token_sequences())
    model = ChemFixerPlusModel(
        vocab,
        ModelConfig(
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            max_positions=260,
        ),
    )
    hist = train_steps(model, ds, vocab, steps=140, batch_size=1, lr=3e-3, device="cpu")
    pred = ChemFixerPredictor(model, vocab, device="cpu", beam_size=5).predict(invalid)
    report = {
        "input": invalid,
        "expected": valid,
        "output": pred.output_smiles,
        "route": pred.route,
        "resolved": pred.resolved,
        "first_loss": hist[0]["total"],
        "last_loss": hist[-1]["total"],
        "exact_match": pred.output_smiles == valid,
    }
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "paper_example_overfit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not (pred.resolved and pred.route == "LOCAL" and pred.output_smiles == valid):
        raise SystemExit("End-to-end paper-example overfit test failed")


if __name__ == "__main__":
    main()
