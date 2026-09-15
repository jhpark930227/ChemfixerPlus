from chemfixerplus.data import ChemFixerPairDataset, PairExample
from chemfixerplus.model import ChemFixerPlusModel, ModelConfig
from chemfixerplus.seed import seed_everything
from chemfixerplus.tokenizer import SmilesTokenizer, Vocabulary
from chemfixerplus.training import train_steps


def _tiny_dataset():
    tok = SmilesTokenizer()

    pairs = [
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

    ds = ChemFixerPairDataset(
        pairs,
        tok,
    )

    vocab = Vocabulary.build(
        ds.all_token_sequences()
    )

    return ds, vocab


def test_cosine_scheduler_decays_and_seed_is_repeatable():
    ds, vocab = _tiny_dataset()

    cfg = ModelConfig(
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )

    histories = []

    for _ in range(2):
        seed_everything(42)

        model = ChemFixerPlusModel(
            vocab,
            cfg,
        )

        hist = train_steps(
            model,
            ds,
            vocab,
            steps=3,
            batch_size=2,
            lr=1e-3,
            scheduler="cosine",
            seed=42,
        )

        histories.append(hist)

    assert histories[0][0]["lr"] > histories[0][-1]["lr"]

    assert (
        [round(x["total"], 7) for x in histories[0]]
        ==
        [round(x["total"], 7) for x in histories[1]]
    )
