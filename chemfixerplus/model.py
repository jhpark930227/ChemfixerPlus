from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structure import ROLE_NAMES
from .tokenizer import Vocabulary


@dataclass
class ModelConfig:
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.25
    max_positions: int = 260
    lambda_global: float = 0.25


class ChemFixerPlusModel(nn.Module):
    def __init__(self, vocab: Vocabulary, cfg: ModelConfig) -> None:
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        v = len(vocab)
        d = cfg.d_model

        self.token_embedding = nn.Embedding(v, d, padding_idx=vocab["<PAD>"])
        self.position_embedding = nn.Embedding(cfg.max_positions, d)
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), d)
        self.depth_embedding = nn.Embedding(33, d)
        self.ring_embedding = nn.Embedding(4, d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.num_decoder_layers)
        self.output_projection = nn.Linear(d, v, bias=False)

        self.token_locator = nn.Linear(d, 1)
        self.gap_locator = nn.Linear(2 * d, 1)
        self.left_boundary = nn.Parameter(torch.zeros(d))
        self.right_boundary = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.left_boundary, std=0.02)
        nn.init.normal_(self.right_boundary, std=0.02)

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def _positions(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.cfg.max_positions:
            raise ValueError(f"Sequence length {length} exceeds max_positions={self.cfg.max_positions}")
        return torch.arange(length, device=device)

    def encode(
        self,
        input_ids: torch.Tensor,
        role_multihot: torch.Tensor,
        depth_ids: torch.Tensor,
        ring_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        use_structure: bool = True,
    ) -> torch.Tensor:
        _, length = input_ids.shape
        pos = self.position_embedding(self._positions(length, input_ids.device))[None, :, :]
        x = self.token_embedding(input_ids) + pos
        if use_structure:
            role = role_multihot @ self.role_embedding.weight
            x = x + role + self.depth_embedding(depth_ids) + self.ring_embedding(ring_ids)
        return self.encoder(x, src_key_padding_mask=padding_mask)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, length = decoder_input_ids.shape
        pos = self.position_embedding(self._positions(length, decoder_input_ids.device))[None, :, :]
        y = self.token_embedding(decoder_input_ids) + pos
        causal = torch.triu(
            torch.ones((length, length), device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        out = self.decoder(
            y,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_projection(out)

    def localization_logits(
        self,
        memory: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded token logits [B,max_n] and gap logits [B,max_n+1].

        `memory` is produced from <BOS> X <EOS>, so source token states begin at 1.
        """
        bsz = memory.shape[0]
        max_n = int(source_lengths.max().item())
        tok_logits = memory.new_zeros((bsz, max_n))
        gap_logits = memory.new_zeros((bsz, max_n + 1))
        for b in range(bsz):
            n = int(source_lengths[b].item())
            h = memory[b, 1 : 1 + n]
            tok_logits[b, :n] = self.token_locator(h).squeeze(-1)
            if n == 0:
                pair = torch.cat([self.left_boundary, self.right_boundary])
                gap_logits[b, 0] = self.gap_locator(pair).squeeze(-1)
                continue
            gaps = []
            gaps.append(torch.cat([self.left_boundary, h[0]], dim=-1))
            for i in range(1, n):
                gaps.append(torch.cat([h[i - 1], h[i]], dim=-1))
            gaps.append(torch.cat([h[-1], self.right_boundary], dim=-1))
            gap_logits[b, : n + 1] = self.gap_locator(torch.stack(gaps)).squeeze(-1)
        return tok_logits, gap_logits

    @staticmethod
    def masked_bce_with_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        pos_weight: float,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=torch.tensor(pos_weight, device=logits.device),
        )
        mask_f = mask.float()
        per_seq = (loss * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        valid_seq = mask.any(dim=1)
        return per_seq[valid_seq].mean() if valid_seq.any() else loss.new_zeros(())

    def sequence_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_tok = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=self.vocab["<PAD>"],
            reduction="none",
        )
        mask = labels.ne(self.vocab["<PAD>"]).float()
        per_seq = (per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        valid_seq = mask.sum(dim=1).gt(0)
        return per_seq[valid_seq].mean() if valid_seq.any() else per_tok.new_zeros(())
