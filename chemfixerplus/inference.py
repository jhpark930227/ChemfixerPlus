from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .alignment import EditSpan, spans_from_predictions
from .batching import encoder_batch
from .chemistry import is_valid_smiles
from .model import ChemFixerPlusModel
from .repair import (
    assemble_from_replacements,
    is_local_route,
    parse_repair_sequence,
    serialize_repair,
)
from .tokenizer import SmilesTokenizer, Vocabulary


@dataclass
class Prediction:
    input_smiles: str
    output_smiles: str
    route: str
    resolved: bool
    token_scores: list[float]
    gap_scores: list[float]
    spans: list[EditSpan]


@dataclass
class _Beam:
    ids: list[int]
    logprob: float
    finished: bool

    def score(self) -> float:
        # length-normalized log-likelihood, excluding BOS
        return self.logprob / max(1, len(self.ids) - 1)


class ChemFixerPredictor:
    def __init__(
        self,
        model: ChemFixerPlusModel,
        vocab: Vocabulary,
        tokenizer: SmilesTokenizer | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        beam_size: int = 5,
        max_repair_tokens: int = 32,
        max_smiles_tokens: int = 254,
    ) -> None:
        self.model = model.eval().to(device)
        self.vocab = vocab
        self.tokenizer = tokenizer or SmilesTokenizer()
        self.device = torch.device(device)
        self.threshold = threshold
        self.beam_size = beam_size
        self.max_repair_tokens = max_repair_tokens
        self.max_smiles_tokens = max_smiles_tokens

        self.control_ids = {
            vocab[t]
            for t in ["<PAD>", "<UNK>", "<BOS>", "<MASK>", "<LOCAL>", "<GLOBAL>"]
        }
        self.sentinel_ids = {vocab[f"<e{i}>"] for i in range(1, 5)}

    @torch.no_grad()
    def _encode(self, tokens: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = encoder_batch([tokens], self.vocab)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        memory = self.model.encode(**batch)
        return memory, batch["padding_mask"]

    @torch.no_grad()
    def _localization(self, source_tokens: Sequence[str]) -> tuple[list[float], list[float]]:
        serialized = ["<BOS>", *source_tokens, "<EOS>"]
        memory, _ = self._encode(serialized)
        lengths = torch.tensor([len(source_tokens)], device=self.device)
        tlog, glog = self.model.localization_logits(memory, lengths)
        return (
            torch.sigmoid(tlog[0, : len(source_tokens)]).cpu().tolist(),
            torch.sigmoid(glog[0, : len(source_tokens) + 1]).cpu().tolist(),
        )

    def _allowed_local_ids(self, generated_tokens: Sequence[str], k: int) -> set[int]:
        # generated_tokens excludes BOS and EOS.
        expected = [f"<e{i}>" for i in range(1, k + 2)]
        if not generated_tokens:
            return {self.vocab[expected[0]]}
        # Find last sentinel position and verify order prefix.
        seen = [t for t in generated_tokens if t.startswith("<e")]
        expected_prefix = expected[: len(seen)]
        if seen != expected_prefix:
            return set()
        if len(seen) == k + 1:
            return {self.vocab["<EOS>"]}

        next_sentinel = self.vocab[expected[len(seen)]]
        allowed = set(range(len(self.vocab)))
        allowed -= self.control_ids
        allowed -= self.sentinel_ids
        allowed.discard(self.vocab["<EOS>"])
        allowed.add(next_sentinel)
        return allowed

    def _allowed_global_ids(self) -> set[int]:
        allowed = set(range(len(self.vocab)))
        allowed -= self.control_ids
        allowed -= self.sentinel_ids
        allowed.add(self.vocab["<EOS>"])
        return allowed

    @torch.no_grad()
    def _beam_decode(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        max_tokens: int,
        local_k: int | None,
    ) -> list[tuple[list[str], float]]:
        beams = [_Beam([self.vocab["<BOS>"]], 0.0, False)]
        for _ in range(max_tokens + 1):
            expanded: list[_Beam] = []
            all_finished = True
            for beam in beams:
                if beam.finished:
                    expanded.append(beam)
                    continue
                all_finished = False
                inp = torch.tensor([beam.ids], device=self.device)
                pad = torch.zeros_like(inp, dtype=torch.bool)
                logits = self.model.decode(inp, pad, memory, memory_mask)[0, -1]
                logp = F.log_softmax(logits, dim=-1)
                generated = self.vocab.decode(beam.ids[1:])
                if len(generated) >= max_tokens:
                    # Budgets exclude decoder BOS/EOS. Once the budget is reached,
                    # only a legal termination is permitted.
                    if local_k is None:
                        allowed = {self.vocab["<EOS>"]}
                    else:
                        final = f"<e{local_k + 1}>"
                        allowed = {self.vocab["<EOS>"]} if generated and generated[-1] == final else set()
                else:
                    allowed = (
                        self._allowed_local_ids(generated, local_k)
                        if local_k is not None
                        else self._allowed_global_ids()
                    )
                if not allowed:
                    continue
                idx = torch.tensor(sorted(allowed), device=self.device)
                vals = logp[idx]
                top_k = min(self.beam_size, vals.numel())
                top_vals, top_pos = torch.topk(vals, top_k)
                for value, pos in zip(top_vals.tolist(), top_pos.tolist()):
                    tok_id = int(idx[pos].item())
                    ids = [*beam.ids, tok_id]
                    expanded.append(
                        _Beam(ids, beam.logprob + float(value), tok_id == self.vocab["<EOS>"])
                    )
            if all_finished or not expanded:
                break
            expanded.sort(key=lambda b: b.score(), reverse=True)
            beams = expanded[: self.beam_size]

        out: list[tuple[list[str], float]] = []
        for beam in sorted(beams, key=lambda b: b.score(), reverse=True):
            tokens = self.vocab.decode(beam.ids[1:])
            if tokens and tokens[-1] == "<EOS>":
                tokens = tokens[:-1]
            out.append((tokens, beam.score()))
        return out

    def _best_valid_global(self, source_tokens: Sequence[str]) -> str | None:
        memory, mem_mask = self._encode(["<GLOBAL>", *source_tokens, "<EOS>"])
        for tokens, _score in self._beam_decode(memory, mem_mask, self.max_smiles_tokens, local_k=None):
            if len(tokens) > self.max_smiles_tokens:
                continue
            smiles = self.tokenizer.detokenize(tokens)
            if smiles and is_valid_smiles(smiles):
                return smiles
        return None

    def _best_valid_local(self, source_tokens: Sequence[str], spans: Sequence[EditSpan]) -> str | None:
        ser = serialize_repair(source_tokens, spans)
        if len(ser.marked_source) + 2 > 256:
            return None
        memory, mem_mask = self._encode(["<LOCAL>", *ser.marked_source, "<EOS>"])
        for tokens, _score in self._beam_decode(memory, mem_mask, self.max_repair_tokens, local_k=len(spans)):
            replacements = parse_repair_sequence(tokens, len(spans))
            if replacements is None:
                continue
            candidate_tokens = assemble_from_replacements(source_tokens, spans, replacements)
            if len(candidate_tokens) > self.max_smiles_tokens:
                continue
            smiles = self.tokenizer.detokenize(candidate_tokens)
            if smiles and is_valid_smiles(smiles):
                return smiles
        return None

    @torch.no_grad()
    def predict(self, smiles: str) -> Prediction:
        if is_valid_smiles(smiles):
            return Prediction(smiles, smiles, "BYPASS", True, [], [], [])
        if not smiles or not self.tokenizer.is_lossless(smiles):
            return Prediction(smiles, smiles, "UNRESOLVED", False, [], [], [])
        source = self.tokenizer.tokenize(smiles)
        if len(source) > self.max_smiles_tokens:
            return Prediction(smiles, smiles, "UNRESOLVED", False, [], [], [])

        token_scores, gap_scores = self._localization(source)
        spans = spans_from_predictions(
            [s >= self.threshold for s in token_scores],
            [s >= self.threshold for s in gap_scores],
        )
        route_local = is_local_route(len(source), spans)
        if route_local:
            local = self._best_valid_local(source, spans)
            if local is not None:
                return Prediction(smiles, local, "LOCAL", True, token_scores, gap_scores, list(spans))

        global_out = self._best_valid_global(source)
        if global_out is not None:
            return Prediction(smiles, global_out, "GLOBAL", True, token_scores, gap_scores, list(spans))
        return Prediction(smiles, smiles, "UNRESOLVED", False, token_scores, gap_scores, list(spans))
