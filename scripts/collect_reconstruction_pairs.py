#!/usr/bin/env python
"""Collect generator-derived invalid/valid SMILES pairs.

ChemFixer+ is fine-tuned on invalid autoregressive reconstructions X
paired with their valid reference SMILES Y.

Example
-------
python scripts/collect_reconstruction_pairs.py \
    --input epoch_008_reconstructions.csv epoch_010_reconstructions.csv \
    --target-column target_smiles \
    --prediction-column predicted_smiles \
    --id-column chembl_id \
    --deduplicate \
    --output data/processed/correction_pairs.jsonl
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chemfixerplus.chemistry import is_valid_smiles
from chemfixerplus.data import PairExample, write_pairs_jsonl
from chemfixerplus.tokenizer import SmilesTokenizer


_NULLS = {"", "nan", "none", "null"}


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value.lower() in _NULLS:
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect invalid autoregressive reconstructions and pair "
            "them with their valid reference SMILES."
        )
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more reconstruction CSV files.",
    )
    parser.add_argument(
        "--target-column",
        default="target_smiles",
    )
    parser.add_argument(
        "--prediction-column",
        default="predicted_smiles",
    )
    parser.add_argument(
        "--id-column",
        default=None,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output invalid/valid pair JSONL.",
    )
    parser.add_argument(
        "--max-smiles-tokens",
        type=int,
        default=254,
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
    )

    args = parser.parse_args()

    tokenizer = SmilesTokenizer()

    pairs: list[PairExample] = []
    seen: set[tuple[str, str]] = set()

    total = 0
    valid_predictions = 0
    invalid_predictions = 0
    invalid_reference = 0
    unsupported = 0

    for filename in args.input:
        path = Path(filename)

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise RuntimeError(
                    f"No CSV header found in {path}"
                )

            required = {
                args.target_column,
                args.prediction_column,
            }

            missing = required - set(reader.fieldnames)

            if missing:
                raise RuntimeError(
                    f"{path}: missing columns "
                    f"{sorted(missing)}"
                )

            for row_index, row in enumerate(
                reader,
                start=2,
            ):
                total += 1

                target = clean_text(
                    row.get(args.target_column)
                )

                prediction = clean_text(
                    row.get(args.prediction_column)
                )

                if not target or not is_valid_smiles(target):
                    invalid_reference += 1
                    continue

                # A valid reconstruction is not a correction pair.
                if prediction and is_valid_smiles(prediction):
                    valid_predictions += 1
                    continue

                invalid_predictions += 1

                if not prediction:
                    unsupported += 1
                    continue

                try:
                    x_tokens = tokenizer.tokenize(prediction)
                    y_tokens = tokenizer.tokenize(target)
                except Exception:
                    unsupported += 1
                    continue

                if (
                    not x_tokens
                    or not y_tokens
                    or len(x_tokens) > args.max_smiles_tokens
                    or len(y_tokens) > args.max_smiles_tokens
                ):
                    unsupported += 1
                    continue

                key = (prediction, target)

                if args.deduplicate and key in seen:
                    continue

                seen.add(key)

                metadata = {
                    "source": "generator_reconstruction",
                    "source_file": path.name,
                    "source_row": row_index,
                }

                if args.id_column:
                    value = clean_text(
                        row.get(args.id_column)
                    )
                    if value:
                        metadata["reference_id"] = value

                pairs.append(
                    PairExample(
                        invalid=prediction,
                        valid=target,
                        metadata=metadata,
                    )
                )

    write_pairs_jsonl(
        pairs,
        args.output,
    )

    print("=" * 72)
    print("GENERATOR-DERIVED PAIR COLLECTION")
    print("=" * 72)
    print("input rows         :", total)
    print("valid predictions  :", valid_predictions)
    print("invalid predictions:", invalid_predictions)
    print("invalid references :", invalid_reference)
    print("unsupported/skipped:", unsupported)
    print("correction pairs   :", len(pairs))
    print("saved              :", Path(args.output).resolve())

    if not pairs:
        raise SystemExit(
            "No usable invalid/valid reconstruction pairs were collected."
        )


if __name__ == "__main__":
    main()
