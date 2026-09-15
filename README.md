# ChemFixer+

Reference implementation of **ChemFixer+: Copy-Preserving Multi-Span Editing for Robust SMILES Correction**.

## Overview

ChemFixer+ is a structure-aware SMILES correction framework that separates **where to edit** from **what to generate**. Sparse error regions are localized at token and gap level, while unselected context is copied directly. Inputs that are unsuitable for local repair, or unsuccessful local repairs, are handled by a shared global correction path.

The implementation includes:

- extended SMILES tokenization with stereochemistry, charges, directional bonds, and multi-digit ring labels;
- lexical role, branch-depth, and ring-state structural features;
- token- and gap-level error localization with deterministic alignment;
- copy-preserving multi-span repair with local/global routing;
- paper-aligned masked pretraining, paired fine-tuning, and inference.

ChemFixer+ builds on our previous **ChemFixer** framework while introducing structure-aware error localization and copy-preserving multi-span repair.

Previous work: [ChemFixer, IEEE JBHI 2026](https://doi.org/10.1109/JBHI.2025.3593825)


## Installation

Python 3.10+ is recommended.

Install a PyTorch build appropriate for your system first. One tested configuration is PyTorch 2.7.1 with CUDA 11.8.

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e '.[dev]'
```

Run the test suite:

```bash
python -m pytest -q
```

All tests should pass before training or inference.

## Data preparation

ChemFixer+ masked pretraining uses a one-SMILES-per-line corpus. The provided `prepare_smiles_corpus.py` utility converts common molecular dataset formats, including TXT, SMI, SMILES, CSV, TSV, and Parquet, into this format.

By default, the original SMILES strings are preserved without canonicalization. Duplicate removal and RDKit validity filtering are optional.

### MOSES example

For a MOSES-style CSV file:

```bash
python scripts/prepare_smiles_corpus.py \
  --input /path/to/moses/train.csv \
  --smiles-column SMILES \
  --output data/raw/moses_train.txt
```

### ChEMBL37 example

For a ChEMBL37 Parquet export:

```bash
python scripts/prepare_smiles_corpus.py \
  --input /path/to/chembl37.parquet \
  --smiles-column smiles \
  --output data/raw/chembl37_train.txt
```

The `--smiles-column` argument can be changed to match the downloaded dataset. Optional `--deduplicate` and `--validate-rdkit` flags are available when those preprocessing steps are desired.

## Correction-pair construction

ChemFixer+ is fine-tuned on invalid autoregressive reconstructions paired with their valid reference SMILES.

Given reconstruction CSV files containing reference and predicted SMILES:

```bash
python scripts/collect_reconstruction_pairs.py \
  --input reconstruction_epoch_*.csv \
  --target-column target_smiles \
  --prediction-column predicted_smiles \
  --id-column chembl_id \
  --deduplicate \
  --output data/processed/correction_pairs.jsonl
```

Valid reconstructions are excluded. The output JSONL contains generator-derived invalid/valid correction pairs that can be passed directly to `scripts/train.py`.

For free-generation outputs that do not have paired references, RDKit-valid and invalid outputs can be separated with:

```bash
python scripts/classify_generated_smiles.py \
  --input generated_smiles.csv \
  --valid-output outputs/valid.csv \
  --invalid-output outputs/invalid.csv
```

## Training

### Masked pretraining

```bash
python scripts/pretrain.py \
  --smiles /path/to/valid_smiles.csv \
  --config configs/paper.yaml \
  --seed 0 \
  --device cuda \
  --output checkpoints/pretrained.pt
```

Masked pretraining performs full-sequence reconstruction with masking probability 0.10.

The ChemFixer+-specific localization heads, structural embeddings, mode tokens, and repair sentinels are introduced for paired fine-tuning while preserving pretrained token IDs and transferable model parameters.

### Paired fine-tuning

```bash
python scripts/train.py \
  --pairs data/processed/correction_pairs_train.jsonl \
  --validation-pairs data/processed/correction_pairs_valid.jsonl \
  --config configs/paper.yaml \
  --pretrained checkpoints/pretrained.pt \
  --seed 0 \
  --device cuda \
  --best-output checkpoints/chemfixerplus_best.pt \
  --output checkpoints/chemfixerplus_last.pt
```

With the paper configuration, validation is evaluated every 1,000 optimizer updates. The checkpoint with the lowest validation joint loss is written to `--best-output`, while `--output` preserves the final-update checkpoint. Held-out test benchmarks are not used for checkpoint selection.

The paper configuration uses:

| Component | Setting |
|---|---:|
| Encoder layers | 6 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Model dimension | 512 |
| Feed-forward dimension | 2048 |
| Dropout | 0.25 |
| Mask probability | 0.10 |
| Peak learning rate | `1e-4` |
| Effective batch size | 64 |
| Optimizer updates | 20,000 |
| Validation interval | 1,000 updates |
| Checkpoint selection | lowest validation joint loss |
| LR schedule | cosine decay |
| Gradient clipping | 1.0 |
| Global-loss weight | 0.25 |
| Beam size | 5 |

For multiple fine-tuning seeds:

```bash
for SEED in 0 1 2; do
  python scripts/train.py \
    --pairs data/processed/correction_pairs_train.jsonl \
    --validation-pairs data/processed/correction_pairs_valid.jsonl \
    --config configs/paper.yaml \
    --pretrained checkpoints/pretrained.pt \
    --seed "$SEED" \
    --device cuda \
    --best-output "checkpoints/chemfixerplus_seed${SEED}_best.pt" \
    --output "checkpoints/chemfixerplus_seed${SEED}_last.pt"
done
```

## Inference

```bash
python scripts/predict.py \
  --checkpoint checkpoints/chemfixerplus_best.pt \
  --device cuda \
  --smiles 'CCC(=O)O)CC1CCCCC'
```

Inference first predicts token and gap edit sites. Eligible inputs are routed through copy-preserving local repair. Inputs that do not satisfy local-routing constraints, or local repairs that fail validation, are processed by global correction. If no acceptable candidate is found, the original input is returned unresolved.

## Integration test

The Fig. 1 example provides a lightweight end-to-end software test:

```bash
python scripts/verify_paper_example.py


The example verifies alignment, localization, local routing, repair decoding, copy-preserving assembly, and RDKit validation.

It is an integration test, not a reported benchmark experiment.

## Implementation and experiment artifacts

This repository provides the ChemFixer+ implementation, data utilities,
and training/inference scripts. The paper-aligned configuration is available
in `configs/paper.yaml`.

The current release contains source code and configuration files.
Paper-trained checkpoints, fixed experimental datasets, and table/figure
reproduction artifacts are not bundled with this release.


## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
