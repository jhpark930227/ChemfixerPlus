from __future__ import annotations

from rdkit import Chem, rdBase


def mol_from_smiles(smiles: str):
    if not smiles:
        return None
    try:
        # Invalid strings are expected inputs in ChemFixer+, so suppress parser noise
        # locally while preserving the validity result.
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return mol_from_smiles(smiles) is not None


def canonical_isomeric_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_nonisomeric_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
