from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Literal, Sequence

OpName = Literal["KEEP", "REPLACE", "DELETE", "INSERT"]


@dataclass(frozen=True)
class EditOp:
    op: OpName
    src_index: int
    tgt_index: int
    src_token: str | None
    tgt_token: str | None


@dataclass
class EditSpan:
    source_start: int
    source_end: int  # exclusive
    replacement: list[str]
    insertion_gaps: set[int]

    @property
    def source_len(self) -> int:
        return self.source_end - self.source_start


@dataclass
class AlignmentResult:
    ops: list[EditOp]
    token_labels: list[float]
    gap_labels: list[float]
    spans: list[EditSpan]


@dataclass
class _Cell:
    score: tuple[int, int, int]  # edit cost, negative keeps, span count
    prev: tuple[int, int, int] | None
    op: EditOp | None


def align_tokens(source: Sequence[str], target: Sequence[str]) -> AlignmentResult:
    """Token-level Levenshtein alignment with ChemFixer+ tie-breaking.

    Primary objective: minimum edit cost.
    Ties: maximize KEEP operations, then minimize non-KEEP span count.
    Remaining exact ties are deterministic by transition order.
    """
    n, m = len(source), len(target)
    # state: (i, j, last_was_edit)
    dp: dict[tuple[int, int, int], _Cell] = {
        (0, 0, 0): _Cell((0, 0, 0), None, None)
    }

    def relax(
        cur_key: tuple[int, int, int],
        nxt_key: tuple[int, int, int],
        op: EditOp,
        edit_cost: int,
        keep_inc: int,
        edit_now: bool,
    ) -> None:
        cur = dp[cur_key]
        prev_edit = bool(cur_key[2])
        score = (
            cur.score[0] + edit_cost,
            cur.score[1] - keep_inc,
            cur.score[2] + (1 if edit_now and not prev_edit else 0),
        )
        old = dp.get(nxt_key)
        if old is None or score < old.score:
            dp[nxt_key] = _Cell(score, cur_key, op)

    # Iterative dynamic program. Transition order gives deterministic final ties:
    # KEEP, REPLACE, DELETE, INSERT.
    for total in range(n + m + 1):
        states = [k for k in list(dp.keys()) if k[0] + k[1] == total]
        states.sort()
        for i, j, last in states:
            key = (i, j, last)
            if i < n and j < m and source[i] == target[j]:
                relax(
                    key, (i + 1, j + 1, 0),
                    EditOp("KEEP", i, j, source[i], target[j]),
                    0, 1, False,
                )
            if i < n and j < m and source[i] != target[j]:
                relax(
                    key, (i + 1, j + 1, 1),
                    EditOp("REPLACE", i, j, source[i], target[j]),
                    1, 0, True,
                )
            if i < n:
                relax(
                    key, (i + 1, j, 1),
                    EditOp("DELETE", i, j, source[i], None),
                    1, 0, True,
                )
            if j < m:
                relax(
                    key, (i, j + 1, 1),
                    EditOp("INSERT", i, j, None, target[j]),
                    1, 0, True,
                )

    finals = [((n, m, last), dp[(n, m, last)]) for last in (0, 1) if (n, m, last) in dp]
    if not finals:
        raise RuntimeError("Alignment DP failed")
    final_key, _ = min(finals, key=lambda kv: kv[1].score)

    ops_rev: list[EditOp] = []
    key = final_key
    while True:
        cell = dp[key]
        if cell.prev is None:
            break
        assert cell.op is not None
        ops_rev.append(cell.op)
        key = cell.prev
    ops = list(reversed(ops_rev))

    token_labels = [0.0] * n
    gap_labels = [0.0] * (n + 1)
    for op in ops:
        if op.op in {"DELETE", "REPLACE"}:
            token_labels[op.src_index] = 1.0
        elif op.op == "INSERT":
            gap_labels[op.src_index] = 1.0

    spans = ops_to_spans(ops)
    return AlignmentResult(ops, token_labels, gap_labels, spans)


def ops_to_spans(ops: Sequence[EditOp]) -> list[EditSpan]:
    spans: list[EditSpan] = []
    current_ops: list[EditOp] = []

    def flush() -> None:
        nonlocal current_ops
        if not current_ops:
            return
        consumed = [o.src_index for o in current_ops if o.op in {"DELETE", "REPLACE"}]
        inserted_gaps = {o.src_index for o in current_ops if o.op == "INSERT"}
        if consumed:
            start = min(consumed)
            end = max(consumed) + 1
        else:
            assert inserted_gaps
            start = end = min(inserted_gaps)
        replacement = [o.tgt_token for o in current_ops if o.op in {"REPLACE", "INSERT"}]
        spans.append(
            EditSpan(
                source_start=start,
                source_end=end,
                replacement=[t for t in replacement if t is not None],
                insertion_gaps=inserted_gaps,
            )
        )
        current_ops = []

    for op in ops:
        if op.op == "KEEP":
            flush()
        else:
            current_ops.append(op)
    flush()
    return spans


def reconstruct_target(source: Sequence[str], spans: Sequence[EditSpan]) -> list[str]:
    out: list[str] = []
    cursor = 0
    for span in spans:
        if span.source_start < cursor:
            raise ValueError("Overlapping spans")
        out.extend(source[cursor:span.source_start])
        out.extend(span.replacement)
        cursor = span.source_end
    out.extend(source[cursor:])
    return out


def spans_from_predictions(
    token_selected: Sequence[bool],
    gap_selected: Sequence[bool],
) -> list[EditSpan]:
    """Merge selected token sites and touching insertion gaps into spans.

    A gap is connected only to an adjacent selected token. Two selected gaps do not
    merge across an unselected token, preventing accidental absorption of context.
    """
    n = len(token_selected)
    if len(gap_selected) != n + 1:
        raise ValueError("gap_selected must have n+1 elements")

    token_nodes = {i for i, flag in enumerate(token_selected) if flag}
    gap_nodes = {g for g, flag in enumerate(gap_selected) if flag}

    # Build small connected components over selected tokens/gaps.
    nodes = [("t", i) for i in sorted(token_nodes)] + [("g", g) for g in sorted(gap_nodes)]
    parent = {node: node for node in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in token_nodes:
        if i + 1 in token_nodes:
            union(("t", i), ("t", i + 1))
        if i in gap_nodes:
            union(("t", i), ("g", i))
        if i + 1 in gap_nodes:
            union(("t", i), ("g", i + 1))

    comps: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in nodes:
        comps.setdefault(find(node), []).append(node)

    spans: list[EditSpan] = []
    for comp in comps.values():
        toks = [idx for kind, idx in comp if kind == "t"]
        gaps = {idx for kind, idx in comp if kind == "g"}
        if toks:
            start, end = min(toks), max(toks) + 1
        else:
            assert gaps
            start = end = min(gaps)
        spans.append(EditSpan(start, end, [], gaps))

    spans.sort(key=lambda s: (s.source_start, s.source_end))
    return spans
