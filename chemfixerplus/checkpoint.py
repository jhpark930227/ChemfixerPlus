from __future__ import annotations

from pathlib import Path

import torch

from .model import ChemFixerPlusModel, ModelConfig
from .seed import seed_everything
from .tokenizer import Vocabulary


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[ChemFixerPlusModel, Vocabulary]:
    obj = torch.load(path, map_location=map_location, weights_only=False)
    vocab = Vocabulary(obj["vocab"])
    cfg = ModelConfig(**obj["model_config"])
    model = ChemFixerPlusModel(vocab, cfg)
    model.load_state_dict(obj["model_state"])
    return model, vocab


def load_pretrained_for_finetune(
    path: str | Path,
    token_sequences,
    seed: int,
    map_location: str = "cpu",
) -> tuple[ChemFixerPlusModel, Vocabulary]:
    """Transfer the shared masked-pretraining backbone into ChemFixer+.

    The manuscript describes a shared extended masked-pretraining checkpoint,
    followed by paired fine-tuning where ChemFixer+ adds structural embeddings,
    token/gap heads, sentinels, and local/global mode tokens.  This loader keeps
    all overlapping pretrained token IDs/weights and initializes Plus-specific
    modules/tokens at the requested fine-tuning seed.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    old_vocab = Vocabulary(obj["vocab"])
    cfg = ModelConfig(**obj["model_config"])

    # Preserve all old IDs and append Plus tokens / any correction-only tokens.
    vocab = Vocabulary.build(
        token_sequences,
        legacy_vocab=old_vocab,
        include_plus_tokens=True,
    )

    seed_everything(seed)
    model = ChemFixerPlusModel(vocab, cfg)
    old_state = obj["model_state"]
    new_state = model.state_dict()

    # Shared backbone weights transfer exactly where shapes agree.
    transferable_prefixes = (
        "position_embedding.",
        "encoder.",
        "decoder.",
    )
    for key, value in old_state.items():
        if key.startswith(transferable_prefixes) and key in new_state and new_state[key].shape == value.shape:
            new_state[key] = value

    # Token/output rows are copied token-by-token because the paired vocabulary
    # may append sentinels, mode tokens, or newly observed SMILES expressions.
    old_tok = old_state["token_embedding.weight"]
    old_out = old_state["output_projection.weight"]
    for token, old_id in old_vocab.token_to_id.items():
        if token in vocab.token_to_id:
            new_id = vocab.token_to_id[token]
            new_state["token_embedding.weight"][new_id] = old_tok[old_id]
            new_state["output_projection.weight"][new_id] = old_out[old_id]

    # Do not copy role/depth/ring embeddings, locators, or learned boundary
    # vectors: these are ChemFixer+-specific modules introduced at fine-tuning.
    model.load_state_dict(new_state)
    return model, vocab
