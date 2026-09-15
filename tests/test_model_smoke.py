import torch

from chemfixerplus.data import ChemFixerPairDataset, PairExample
from chemfixerplus.model import ChemFixerPlusModel, ModelConfig
from chemfixerplus.tokenizer import SmilesTokenizer, Vocabulary
from chemfixerplus.training import train_steps


def _fixed_pairs():
    return [
        PairExample(
            "CCC(=O)O)CC1CCCCC",
            "CCC(=O)OCC1CCCCC1",
            {"source": "paper_figure_1"},
        ),
        PairExample(
            "C1CCCCC",
            "C1CCCCC1",
            {"source": "unit_test"},
        ),
        PairExample(
            "CC(=O))O",
            "CC(=O)O",
            {"source": "unit_test"},
        ),
    ]


def test_model_forward_backward_smoke():
    torch.manual_seed(0)

    tok = SmilesTokenizer()
    ds = ChemFixerPairDataset(
        _fixed_pairs(),
        tok,
    )

    vocab = Vocabulary.build(
        ds.all_token_sequences()
    )

    cfg = ModelConfig(
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_positions=260,
    )

    model = ChemFixerPlusModel(
        vocab,
        cfg,
    )

    hist = train_steps(
        model,
        ds,
        vocab,
        steps=2,
        batch_size=len(ds),
        lr=1e-3,
    )

    assert len(hist) == 2

    assert all(
        torch.isfinite(
            torch.tensor(h["total"])
        )
        for h in hist
    )
