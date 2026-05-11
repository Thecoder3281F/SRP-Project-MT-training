#!/usr/bin/env python3
"""Compute pairwise Tanimoto similarities between unique sulfonamides in test and train CSVs.

Output CSV columns: test_smiles, train_smiles, tanimoto
"""
import argparse
import sys
from typing import Optional, List, Tuple

import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception:
    print("RDKit is required. Install RDKit in your environment.")
    raise


SULFONAMIDE_SMARTS = "S(=O)(=O)N"


def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if "smile" in c.lower() or "smiles" in c.lower() or "sulfonamide" in c.lower()]
    if candidates:
        return candidates[0]
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        for v in sample.iloc[:5]:
            if Chem.MolFromSmiles(v) is not None:
                return col
    return None


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def unique_sulfonamides(df: pd.DataFrame, smiles_col: str) -> List[Tuple[str, int]]:
    """Return list of (canonical_smiles, first_row_index) for unique sulfonamides in df."""
    pat = Chem.MolFromSmarts(SULFONAMIDE_SMARTS)
    seen = {}
    for idx, smi in df[smiles_col].fillna("").astype(str).items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if not m.HasSubstructMatch(pat):
            continue
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        if can not in seen:
            seen[can] = int(idx) if str(idx).isdigit() else idx
    return [(s, seen[s]) for s in seen]


def mols_from_smiles_list(smiles_list: List[str]):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return mols


def fps_from_mols(mols, radius=2, n_bits=2048):
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]


def compute_pairwise(test_smiles_list, train_smiles_list, radius=2, n_bits=2048):
    test_mols = mols_from_smiles_list(test_smiles_list)
    train_mols = mols_from_smiles_list(train_smiles_list)
    test_fps = fps_from_mols(test_mols, radius=radius, n_bits=n_bits)
    train_fps = fps_from_mols(train_mols, radius=radius, n_bits=n_bits)

    rows = []
    for i, t_fp in enumerate(test_fps):
        sims = DataStructs.BulkTanimotoSimilarity(t_fp, train_fps)
        for j, s in enumerate(sims):
            rows.append((test_smiles_list[i], train_smiles_list[j], float(s)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--smiles-col", default=None)
    p.add_argument("--out", default="mt_training/tanimoto_pairwise.csv")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    args = p.parse_args()

    df_train = load_csv(args.train)
    df_test = load_csv(args.test)

    smiles_col = args.smiles_col or detect_smiles_column(df_train) or detect_smiles_column(df_test)
    if smiles_col is None:
        print("Could not detect SMILES column. Provide --smiles-col.")
        sys.exit(1)

    print(f"Using SMILES column: {smiles_col}")

    train_unique = unique_sulfonamides(df_train, smiles_col)
    test_unique = unique_sulfonamides(df_test, smiles_col)

    if not train_unique:
        print("No sulfonamides found in train set.")
        sys.exit(1)
    if not test_unique:
        print("No sulfonamides found in test set.")
        sys.exit(1)

    train_smiles = [s for s, _ in train_unique]
    test_smiles = [s for s, _ in test_unique]

    print(f"Found {len(train_smiles)} unique sulfonamides in train and {len(test_smiles)} in test")

    rows = compute_pairwise(test_smiles, train_smiles, radius=args.radius, n_bits=args.n_bits)

    out_df = pd.DataFrame(rows, columns=["test_smiles", "train_smiles", "tanimoto"])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
