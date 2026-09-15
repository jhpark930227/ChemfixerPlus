#!/usr/bin/env python3
"""Prepare a one-SMILES-per-line corpus for ChemFixer+ pretraining."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Iterator


COMMON_SMILES_COLUMNS = (
    "smiles",
    "SMILES",
    "canonical_smiles",
    "canonical_smiles_std",
    "molecule_smiles",
    "mol_smiles",
)


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def effective_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]

    for suffix in (".parquet", ".smiles", ".csv", ".tsv", ".smi", ".txt"):
        if name.endswith(suffix):
            return suffix

    return Path(name).suffix


def resolve_smiles_column(fieldnames, requested):
    if not fieldnames:
        raise ValueError("Input table has no header.")

    if requested is not None:
        if requested not in fieldnames:
            raise ValueError(
                f"SMILES column {requested!r} not found. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return requested

    for name in COMMON_SMILES_COLUMNS:
        if name in fieldnames:
            return name

    raise ValueError(
        "Could not infer the SMILES column. "
        "Use --smiles-column explicitly. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def iter_plain(path: Path) -> Iterator[str]:
    with open_text(path) as handle:
        for line in handle:
            value = line.strip()
            if value:
                yield value


def iter_delimited(path: Path, delimiter: str, smiles_column):
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        column = resolve_smiles_column(reader.fieldnames, smiles_column)

        for row in reader:
            value = row.get(column)
            if value is None:
                continue
            value = value.strip()
            if value:
                yield value


def iter_parquet(path: Path, smiles_column):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Parquet input requires pandas and pyarrow."
        ) from exc

    frame = pd.read_parquet(path)
    column = resolve_smiles_column(list(frame.columns), smiles_column)

    for value in frame[column]:
        if value is None:
            continue

        value = str(value).strip()

        if value and value.lower() != "nan":
            yield value


def iter_smiles(path: Path, smiles_column):
    suffix = effective_suffix(path)

    if suffix in {".txt", ".smi", ".smiles"}:
        return iter_plain(path)

    if suffix == ".csv":
        return iter_delimited(path, ",", smiles_column)

    if suffix == ".tsv":
        return iter_delimited(path, "\t", smiles_column)

    if suffix == ".parquet":
        return iter_parquet(path, smiles_column)

    raise ValueError(
        f"Unsupported format: {path.name}. "
        "Supported: TXT, SMI, SMILES, CSV, TSV, Parquet."
    )


def rdkit_valid(smiles: str) -> bool:
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles) is not None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a one-SMILES-per-line corpus "
            "for ChemFixer+ masked pretraining."
        )
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default=None)

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate SMILES while preserving first occurrence.",
    )

    parser.add_argument(
        "--validate-rdkit",
        action="store_true",
        help="Keep only RDKit-parseable SMILES. No canonicalization is applied.",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    invalid = 0
    duplicates = 0

    seen = set()

    with args.output.open("w", encoding="utf-8") as out:
        for smiles in iter_smiles(args.input, args.smiles_column):
            total += 1

            if args.validate_rdkit and not rdkit_valid(smiles):
                invalid += 1
                continue

            if args.deduplicate:
                if smiles in seen:
                    duplicates += 1
                    continue
                seen.add(smiles)

            out.write(smiles + "\n")
            written += 1

    print("ChemFixer+ corpus preparation")
    print(f"  input:              {args.input}")
    print(f"  output:             {args.output}")
    print(f"  non-empty rows:     {total}")
    print(f"  written SMILES:     {written}")

    if args.validate_rdkit:
        print(f"  RDKit-invalid:      {invalid}")

    if args.deduplicate:
        print(f"  duplicates removed: {duplicates}")

    print("Done.")


if __name__ == "__main__":
    main()
