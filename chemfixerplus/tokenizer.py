from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

BASE_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]
PLUS_SPECIAL_TOKENS = ["<LOCAL>", "<GLOBAL>", "<e1>", "<e2>", "<e3>", "<e4>"]
SPECIAL_TOKENS = BASE_SPECIAL_TOKENS + PLUS_SPECIAL_TOKENS

# Longest/specific patterns first. Bracket expressions stay intact so stereo/charge
# can be represented as lexical roles without requiring RDKit parsing.
TOKEN_RE = re.compile(
    r"(<(?:PAD|UNK|BOS|EOS|MASK|LOCAL|GLOBAL|e[1-4])>)"
    r"|(\[[^\]]+\])"
    r"|(%\d{2,3})"
    r"|(@@?)"
    r"|(Br|Cl|Si|Se|Na|Li|Ca|Al|Mg|Zn|Sn|Ag|Fe|Cu|Mn|Hg|Pb|Bi)"
    r"|(.)"
)


class SmilesTokenizer:
    """Parser-free, lossless SMILES tokenizer.

    The tokenizer intentionally does not call a chemistry parser because ChemFixer+
    must tokenize invalid inputs. Bracket atoms, multi-digit ring labels, directional
    bonds, stereo markers and ordinary one-character SMILES symbols are preserved.
    """

    def tokenize(self, smiles: str) -> list[str]:
        if smiles == "":
            return []
        tokens: list[str] = []
        pos = 0
        for match in TOKEN_RE.finditer(smiles):
            if match.start() != pos:
                raise ValueError(f"Tokenizer gap at character {pos}: {smiles!r}")
            tok = next(g for g in match.groups() if g is not None)
            tokens.append(tok)
            pos = match.end()
        if pos != len(smiles):
            raise ValueError(f"Tokenizer stopped at {pos}/{len(smiles)}: {smiles!r}")
        if self.detokenize(tokens) != smiles:
            raise ValueError(f"Non-lossless tokenization: {smiles!r} -> {tokens!r}")
        return tokens

    @staticmethod
    def detokenize(tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def is_lossless(self, smiles: str) -> bool:
        try:
            return self.detokenize(self.tokenize(smiles)) == smiles
        except Exception:
            return False


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @property
    def id_to_token(self) -> list[str]:
        out = [None] * len(self.token_to_id)
        for tok, idx in self.token_to_id.items():
            out[idx] = tok
        return out  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __getitem__(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    def token(self, idx: int) -> str:
        return self.id_to_token[idx]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self[t] for t in tokens]

    def decode(self, ids: Sequence[int]) -> list[str]:
        table = self.id_to_token
        return [table[i] for i in ids]

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        legacy_vocab: "Vocabulary | None" = None,
        include_plus_tokens: bool = True,
    ) -> "Vocabulary":
        """Build vocabulary while preserving legacy IDs when supplied.

        This supports the checkpoint-transfer rule described in the paper: existing
        token IDs can remain unchanged and new tokens are appended.
        """
        if legacy_vocab is not None:
            mapping = dict(legacy_vocab.token_to_id)
            next_id = max(mapping.values(), default=-1) + 1
            specials = SPECIAL_TOKENS if include_plus_tokens else BASE_SPECIAL_TOKENS
            for tok in specials:
                if tok not in mapping:
                    mapping[tok] = next_id
                    next_id += 1
        else:
            specials = SPECIAL_TOKENS if include_plus_tokens else BASE_SPECIAL_TOKENS
            mapping = {tok: i for i, tok in enumerate(specials)}
            next_id = len(mapping)

        observed = sorted({tok for seq in token_sequences for tok in seq})
        for tok in observed:
            if tok not in mapping:
                mapping[tok] = next_id
                next_id += 1
        return cls(mapping)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
