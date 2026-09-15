#!/usr/bin/env python
from __future__ import annotations

import argparse

from chemfixerplus.checkpoint import load_checkpoint
from chemfixerplus.inference import ChemFixerPredictor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--smiles", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model, vocab = load_checkpoint(args.checkpoint, map_location=args.device)
    predictor = ChemFixerPredictor(model, vocab, device=args.device)
    pred = predictor.predict(args.smiles)
    print(f"input={pred.input_smiles}")
    print(f"route={pred.route}")
    print(f"resolved={pred.resolved}")
    print(f"output={pred.output_smiles}")


if __name__ == "__main__":
    main()
