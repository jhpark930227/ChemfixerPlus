from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from torch.utils.data import Dataset

from .alignment import AlignmentResult, align_tokens
from .repair import is_local_route, serialize_repair
from .tokenizer import SmilesTokenizer, Vocabulary


@dataclass
class PairExample:
    invalid: str
    valid: str
    metadata: dict


def read_smiles_file(path: str | Path) -> list[str]:
    """Read common MOSES-style CSV or one-SMILES-per-line text files."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # CSV with common SMILES header.
    if "," in lines[0] or lines[0].strip().lower() in {"smiles", "smile"}:
        reader = csv.DictReader(lines)
        if reader.fieldnames:
            lower = {name.lower(): name for name in reader.fieldnames}
            for candidate in ("smiles", "smile", "mol", "molecule"):
                if candidate in lower:
                    key = lower[candidate]
                    return [row[key].strip() for row in reader if row.get(key, "").strip()]
        # Fallback: first column after header.
        raw = list(csv.reader(lines))
        return [row[0].strip() for row in raw[1:] if row and row[0].strip()]

    return [line.strip() for line in lines]


def write_pairs_jsonl(pairs: Iterable[PairExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps({"invalid": pair.invalid, "valid": pair.valid, **pair.metadata}) + "\n")


def read_pairs_jsonl(path: str | Path) -> list[PairExample]:
    out: list[PairExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        invalid = obj.pop("invalid")
        valid = obj.pop("valid")
        out.append(PairExample(invalid, valid, obj))
    return out


class ChemFixerPairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PairExample],
        tokenizer: SmilesTokenizer,
        max_repair_tokens: int = 32,
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_repair_tokens = max_repair_tokens
        self.cache: list[dict] = []

        # Dataset alignment progress.
        total_pairs = len(self.pairs)
        align_start = time.time()

        if total_pairs >= 1000:
            print(
                f"[dataset] building token alignments for "
                f"{total_pairs:,} pairs...",
                flush=True,
            )

        for pair_idx, pair in enumerate(self.pairs, 1):
            x = tokenizer.tokenize(pair.invalid)
            y = tokenizer.tokenize(pair.valid)
            aln = align_tokens(x, y)
            ser = serialize_repair(x, aln.spans) if len(aln.spans) <= 3 else None
            local = (
                ser is not None
                and is_local_route(len(x), aln.spans)
                and len(ser.repair_target) <= max_repair_tokens
            )
            self.cache.append({
                "invalid": pair.invalid,
                "valid": pair.valid,
                "x_tokens": x,
                "y_tokens": y,
                "alignment": aln,
                "local_eligible": local,
                "marked_tokens": ser.marked_source if ser is not None else [],
                "repair_tokens": ser.repair_target if ser is not None else [],
            })

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> dict:
        return self.cache[idx]

    def all_token_sequences(self) -> Iterator[list[str]]:
        for item in self.cache:
            yield item["x_tokens"]
            yield item["y_tokens"]
            if item["marked_tokens"]:
                yield item["marked_tokens"]
            if item["repair_tokens"]:
                yield item["repair_tokens"]

    def localization_pos_weights(self) -> tuple[float, float]:
        tok_pos = tok_total = gap_pos = gap_total = 0
        for item in self.cache:
            aln: AlignmentResult = item["alignment"]
            tok_pos += int(sum(aln.token_labels))
            tok_total += len(aln.token_labels)
            gap_pos += int(sum(aln.gap_labels))
            gap_total += len(aln.gap_labels)
        tok_neg = max(tok_total - tok_pos, 1)
        gap_neg = max(gap_total - gap_pos, 1)
        return tok_neg / max(tok_pos, 1), gap_neg / max(gap_pos, 1)
