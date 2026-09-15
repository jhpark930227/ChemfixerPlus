# Data

No generator-derived correction dataset or trained checkpoint is distributed
with this repository at this stage.

ChemFixer+ is fine-tuned on pairs consisting of:

- an invalid autoregressive reconstruction produced by an upstream molecular
  generator, and
- its corresponding valid reference SMILES.

A reconstruction CSV can be converted into ChemFixer+ correction pairs with:

```bash
python scripts/collect_reconstruction_pairs.py \
  --input reconstruction_epoch_*.csv \
  --target-column target_smiles \
  --prediction-column predicted_smiles \
  --id-column chembl_id \
  --deduplicate \
  --output data/processed/correction_pairs.jsonl

`CONTRIBUTING.md`도 synthetic 문구 제거:

```bash
cat > CONTRIBUTING.md <<'EOF'
# Contributing

This repository accompanies the ChemFixer+ manuscript.

Please do not commit unreleased generator-derived correction datasets,
trained checkpoints, experiment logs, or private benchmark outputs.

Contributions to the core implementation, documentation, tests, and
general-purpose data interfaces are welcome.
