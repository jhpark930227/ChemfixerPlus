from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .alignment import EditSpan


@dataclass
class RepairSerialization:
    marked_source: list[str]
    repair_target: list[str]
    contexts: list[list[str]]


def serialize_repair(source: Sequence[str], spans: Sequence[EditSpan]) -> RepairSerialization:
    if len(spans) > 3:
        raise ValueError("Local ChemFixer+ supports at most three spans")
    marked: list[str] = []
    repair: list[str] = []
    contexts: list[list[str]] = []
    cursor = 0
    for k, span in enumerate(spans, start=1):
        if span.source_start < cursor:
            raise ValueError("Overlapping spans")
        context = list(source[cursor:span.source_start])
        contexts.append(context)
        marked.extend(context)
        marked.append(f"<e{k}>")
        repair.append(f"<e{k}>")
        repair.extend(span.replacement)
        cursor = span.source_end
    tail = list(source[cursor:])
    contexts.append(tail)
    marked.extend(tail)
    repair.append(f"<e{len(spans) + 1}>")
    return RepairSerialization(marked, repair, contexts)


def parse_repair_sequence(tokens: Sequence[str], k: int) -> list[list[str]] | None:
    expected = [f"<e{i}>" for i in range(1, k + 2)]
    if not tokens or tokens[0] != expected[0]:
        return None
    replacements: list[list[str]] = [[] for _ in range(k)]
    current = 0
    for tok in tokens[1:]:
        if current == k:
            # Anything after final sentinel is invalid here; EOS is stripped earlier.
            return None
        next_sentinel = expected[current + 1]
        if tok == next_sentinel:
            current += 1
            if current == k:
                # final e_{K+1} consumed
                continue
        elif tok.startswith("<e"):
            return None
        else:
            replacements[current].append(tok)
    return replacements if current == k else None


def assemble_from_replacements(
    source: Sequence[str],
    spans: Sequence[EditSpan],
    replacements: Sequence[Sequence[str]],
) -> list[str]:
    if len(spans) != len(replacements):
        raise ValueError("Span/replacement count mismatch")
    out: list[str] = []
    cursor = 0
    for span, repl in zip(spans, replacements):
        out.extend(source[cursor:span.source_start])
        out.extend(repl)
        cursor = span.source_end
    out.extend(source[cursor:])
    return out


def routing_stats(source_len: int, spans: Sequence[EditSpan]) -> tuple[float, int]:
    if source_len <= 0:
        raise ValueError("source_len must be positive")
    source_selected = sum(span.source_len for span in spans)
    insertion_gaps = len({g for span in spans for g in span.insertion_gaps})
    rho = (source_selected + insertion_gaps) / source_len
    enc_len = source_len - source_selected + len(spans) + 2
    return rho, enc_len


def is_local_route(
    source_len: int,
    spans: Sequence[EditSpan],
    max_spans: int = 3,
    max_density: float = 0.20,
    max_encoder_len: int = 256,
) -> bool:
    if not (1 <= len(spans) <= max_spans):
        return False
    rho, enc_len = routing_stats(source_len, spans)
    return rho <= max_density and enc_len <= max_encoder_len
