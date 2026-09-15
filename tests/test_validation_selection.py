import torch

from chemfixerplus.checkpoint import load_checkpoint
from chemfixerplus.data import ChemFixerPairDataset, PairExample
from chemfixerplus.model import ChemFixerPlusModel, ModelConfig
from chemfixerplus.tokenizer import SmilesTokenizer, Vocabulary
from chemfixerplus.training import train_steps


def _train_pairs():
    return [
        PairExample("CCC(=O)O)", "CCC(=O)O", {"split": "train"}),
        PairExample("C1CCCCC", "C1CCCCC1", {"split": "train"}),
        PairExample("CC(=O))O", "CC(=O)O", {"split": "train"}),
        PairExample("C(C", "CCO", {"split": "train"}),
    ]


def _validation_pairs():
    return [
        PairExample("C(N", "CCN", {"split": "validation"}),
        PairExample("C(CC", "CCC", {"split": "validation"}),
    ]


def test_validation_best_checkpoint(tmp_path):
    torch.manual_seed(0)

    tok = SmilesTokenizer()

    train_ds = ChemFixerPairDataset(
        _train_pairs(),
        tok,
    )
    validation_ds = ChemFixerPairDataset(
        _validation_pairs(),
        tok,
    )

    vocab = Vocabulary.build(
        train_ds.all_token_sequences()
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

    best_path = tmp_path / "best.pt"

    history = train_steps(
        model,
        train_ds,
        vocab,
        steps=3,
        batch_size=2,
        lr=1e-3,
        deterministic=True,
        validation_dataset=validation_ds,
        validation_every=1,
        validation_batch_size=2,
        best_checkpoint=best_path,
    )

    assert len(history) == 3
    assert best_path.exists()

    validation_rows = [
        row for row in history
        if "val_total" in row
    ]

    assert len(validation_rows) == 3
    assert all(
        torch.isfinite(
            torch.tensor(row["val_total"])
        )
        for row in validation_rows
    )

    loaded_model, loaded_vocab = load_checkpoint(
        best_path,
        map_location="cpu",
    )

    assert loaded_model.cfg.d_model == 32
    assert len(loaded_vocab) == len(vocab)
