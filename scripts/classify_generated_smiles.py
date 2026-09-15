#!/usr/bin/env python
"""Split generated SMILES into RDKit-valid and invalid outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chemfixerplus.chemistry import is_valid_smiles
from chemfixerplus.data import read_smiles_file


def write_csv(path: Path, smiles: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for s in smiles:
            writer.writerow([s])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify generated SMILES using RDKit validation."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--valid-output", required=True)
    parser.add_argument("--invalid-output", required=True)

    args = parser.parse_args()

    smiles = read_smiles_file(args.input)

    valid = []
    invalid = []

    for s in smiles:
        if is_valid_smiles(s):
            valid.append(s)
        else:
            invalid.append(s)

    write_csv(Path(args.valid_output), valid)
    write_csv(Path(args.invalid_output), invalid)

    total = len(smiles)

    print("total   :", total)
    print("valid   :", len(valid))
    print("invalid :", len(invalid))
    print(
        "validity:",
        f"{len(valid) / max(total, 1):.6f}",
    )


if __name__ == "__main__":
    main()
