from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

ROLE_NAMES = ["atom", "bond", "ring", "branch", "stereo", "charge", "special"]
ROLE_TO_ID = {name: i for i, name in enumerate(ROLE_NAMES)}

RING_NON = 0
RING_OPEN = 1
RING_CLOSE = 2
RING_UNMATCHED = 3

BONDS = {"-", "=", "#", ":", "/", "\\", "."}
BRANCH = {"(", ")"}
AROMATIC = {"b", "c", "n", "o", "p", "s"}
ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "I", "Cl", "Br", "Si", "Se", "*"}
SPECIAL_RE = re.compile(r"^<.*>$")
RING_RE = re.compile(r"^(?:\d|%\d{2,3})$")


@dataclass
class StructuralFeatures:
    role_multihot: list[list[float]]
    depth_ids: list[int]
    ring_ids: list[int]


def token_roles(token: str) -> set[str]:
    roles: set[str] = set()
    if SPECIAL_RE.match(token):
        return {"special"}
    if token in BONDS:
        roles.add("bond")
        # Directional bond symbols carry E/Z stereochemical information.
        if token in {"/", "\\"}:
            roles.add("stereo")
    if token in BRANCH:
        roles.add("branch")
    if RING_RE.match(token):
        roles.add("ring")
    if token in {"@", "@@"}:
        roles.add("stereo")
    if token in ORGANIC or token in AROMATIC:
        roles.add("atom")
    if token.startswith("[") and token.endswith("]"):
        roles.add("atom")
        if "@" in token:
            roles.add("stereo")
        if "+" in token or "-" in token:
            roles.add("charge")
    # Standalone charge symbols are uncommon but remain lexically identifiable.
    if token == "+":
        roles.add("charge")
    if not roles:
        roles.add("special")
    return roles


def branch_depth(tokens: Sequence[str], clip: int = 16) -> list[int]:
    depth = 0
    values: list[int] = []
    for tok in tokens:
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        depth = max(-clip, min(clip, depth))
        values.append(depth + clip)  # 0..32 embedding IDs
    return values


def lexical_ring_states(tokens: Sequence[str]) -> list[int]:
    states = [RING_NON] * len(tokens)
    positions: dict[str, list[int]] = {}
    for i, tok in enumerate(tokens):
        if RING_RE.match(tok):
            positions.setdefault(tok, []).append(i)

    for poss in positions.values():
        paired_count = (len(poss) // 2) * 2
        for k in range(0, paired_count, 2):
            states[poss[k]] = RING_OPEN
            states[poss[k + 1]] = RING_CLOSE
        if len(poss) % 2:
            states[poss[-1]] = RING_UNMATCHED
    return states


def structural_features(tokens: Sequence[str]) -> StructuralFeatures:
    role_multihot: list[list[float]] = []
    for tok in tokens:
        vec = [0.0] * len(ROLE_NAMES)
        for role in token_roles(tok):
            vec[ROLE_TO_ID[role]] = 1.0
        role_multihot.append(vec)
    return StructuralFeatures(
        role_multihot=role_multihot,
        depth_ids=branch_depth(tokens),
        ring_ids=lexical_ring_states(tokens),
    )
