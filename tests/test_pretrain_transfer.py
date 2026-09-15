from pathlib import Path

import torch

from chemfixerplus.checkpoint import load_pretrained_for_finetune
from chemfixerplus.model import ChemFixerPlusModel, ModelConfig
from chemfixerplus.tokenizer import SmilesTokenizer, Vocabulary
from chemfixerplus.training import save_checkpoint


def test_pretrain_transfer_preserves_ids_and_adds_plus_modules(tmp_path: Path):
    tok = SmilesTokenizer()
    seqs = [tok.tokenize("CCO"), tok.tokenize("C1CCCCC1")]
    base_vocab = Vocabulary.build(seqs, include_plus_tokens=False)
    assert "<LOCAL>" not in base_vocab.token_to_id
    torch.manual_seed(123)
    base = ChemFixerPlusModel(
        base_vocab,
        ModelConfig(d_model=16, nhead=4, num_encoder_layers=1,
                    num_decoder_layers=1, dim_feedforward=32, dropout=0.0),
    )
    ckpt = tmp_path / "pre.pt"
    save_checkpoint(base, base_vocab, ckpt)

    paired_sequences = [*seqs, ["<e1>", "N", "<e2>"]]
    plus, plus_vocab = load_pretrained_for_finetune(
        ckpt, paired_sequences, seed=7, map_location="cpu"
    )

    for token, old_id in base_vocab.token_to_id.items():
        assert plus_vocab.token_to_id[token] == old_id
        assert torch.equal(
            plus.token_embedding.weight[old_id],
            base.token_embedding.weight[old_id],
        )
    assert "<LOCAL>" in plus_vocab.token_to_id
    assert "<GLOBAL>" in plus_vocab.token_to_id
    assert "<e1>" in plus_vocab.token_to_id
    assert plus.token_embedding.weight.shape[0] > base.token_embedding.weight.shape[0]
