from __future__ import annotations

from typing import Sequence

import torch

from .structure import ROLE_NAMES, structural_features
from .tokenizer import Vocabulary


def pad_2d(seqs: Sequence[Sequence[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in seqs), default=0)
    ids = torch.full((len(seqs), max_len), pad, dtype=torch.long)
    mask = torch.ones((len(seqs), max_len), dtype=torch.bool)  # True = padding
    for i, seq in enumerate(seqs):
        if seq:
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = False
    return ids, mask


def encoder_batch(token_lists: Sequence[Sequence[str]], vocab: Vocabulary) -> dict[str, torch.Tensor]:
    ids_list = [vocab.encode(seq) for seq in token_lists]
    ids, pad_mask = pad_2d(ids_list, vocab["<PAD>"])
    bsz, max_len = ids.shape
    roles = torch.zeros((bsz, max_len, len(ROLE_NAMES)), dtype=torch.float32)
    depth = torch.full((bsz, max_len), 16, dtype=torch.long)
    ring = torch.zeros((bsz, max_len), dtype=torch.long)
    for b, seq in enumerate(token_lists):
        feat = structural_features(seq)
        if seq:
            roles[b, : len(seq)] = torch.tensor(feat.role_multihot, dtype=torch.float32)
            depth[b, : len(seq)] = torch.tensor(feat.depth_ids, dtype=torch.long)
            ring[b, : len(seq)] = torch.tensor(feat.ring_ids, dtype=torch.long)
    return {
        "input_ids": ids,
        "padding_mask": pad_mask,
        "role_multihot": roles,
        "depth_ids": depth,
        "ring_ids": ring,
    }


def decoder_batch(target_lists: Sequence[Sequence[str]], vocab: Vocabulary) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = [["<BOS>", *seq] for seq in target_lists]
    labels = [[*seq, "<EOS>"] for seq in target_lists]
    input_ids, pad_mask = pad_2d([vocab.encode(s) for s in inputs], vocab["<PAD>"])
    label_ids, _ = pad_2d([vocab.encode(s) for s in labels], vocab["<PAD>"])
    return input_ids, pad_mask, label_ids


def make_training_batch(items: Sequence[dict], vocab: Vocabulary) -> dict:
    x_plain = [["<BOS>", *item["x_tokens"], "<EOS>"] for item in items]
    x_global = [["<GLOBAL>", *item["x_tokens"], "<EOS>"] for item in items]
    y_global = [item["y_tokens"] for item in items]

    batch: dict = {
        "items": items,
        "x_plain": encoder_batch(x_plain, vocab),
        "x_global": encoder_batch(x_global, vocab),
        "global_decoder": decoder_batch(y_global, vocab),
        "source_lengths": torch.tensor([len(item["x_tokens"]) for item in items], dtype=torch.long),
    }

    max_n = max(len(item["x_tokens"]) for item in items)
    tok_labels = torch.zeros((len(items), max_n), dtype=torch.float32)
    tok_mask = torch.zeros((len(items), max_n), dtype=torch.bool)
    gap_labels = torch.zeros((len(items), max_n + 1), dtype=torch.float32)
    gap_mask = torch.zeros((len(items), max_n + 1), dtype=torch.bool)
    for b, item in enumerate(items):
        aln = item["alignment"]
        n = len(aln.token_labels)
        tok_labels[b, :n] = torch.tensor(aln.token_labels)
        tok_mask[b, :n] = True
        gap_labels[b, : n + 1] = torch.tensor(aln.gap_labels)
        gap_mask[b, : n + 1] = True
    batch.update({
        "token_labels": tok_labels,
        "token_label_mask": tok_mask,
        "gap_labels": gap_labels,
        "gap_label_mask": gap_mask,
    })

    local_indices = [i for i, item in enumerate(items) if item["local_eligible"]]
    batch["local_indices"] = torch.tensor(local_indices, dtype=torch.long)
    if local_indices:
        local_enc = [["<LOCAL>", *items[i]["marked_tokens"], "<EOS>"] for i in local_indices]
        local_tgt = [items[i]["repair_tokens"] for i in local_indices]
        batch["x_local"] = encoder_batch(local_enc, vocab)
        batch["local_decoder"] = decoder_batch(local_tgt, vocab)
    else:
        batch["x_local"] = None
        batch["local_decoder"] = None
    return batch


def move_batch(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_batch(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_batch(v, device) for v in obj)
    return obj
