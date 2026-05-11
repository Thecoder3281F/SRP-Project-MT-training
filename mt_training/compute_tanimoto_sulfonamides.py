#!/usr/bin/env python3
"""Compute Tanimoto similarities between sulfonamides in test and train CSVs.

Usage example:
  python mt_training/compute_tanimoto_sulfonamides.py \
    --train datasets_final/chanlam_final/train_final.csv \
    --test datasets_final/chanlam_final/test_final.csv \
    --smiles-col input_sulfonamide --topk 5 --out mt_training/tanimoto_results.csv
"""
import argparse
import sys
from typing import Optional

import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception as e:
    print("RDKit is required to run this script. Install RDKit in your environment.")
    raise


SULFONAMIDE_SMARTS = "S(=O)(=O)N"


def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    # Common column names
    candidates = [c for c in df.columns if "smile" in c.lower() or "smiles" in c.lower() or "sulfonamide" in c.lower()]
    if candidates:
        return candidates[0]

    # Try parsing columns to see if any yields valid RDKit molecules
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        # try a few values
        for v in sample.iloc[:5]:
            if Chem.MolFromSmiles(v) is not None:
                return col
    return None


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def filter_sulfonamides(df: pd.DataFrame, smiles_col: str):
    pat = Chem.MolFromSmarts(SULFONAMIDE_SMARTS)
    smiles = df[smiles_col].fillna("").astype(str)
    mols = []
    rows = []
    for idx, smi in smiles.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if m.HasSubstructMatch(pat):
            mols.append(m)
            rows.append((idx, smi))
    return rows, mols


def mols_to_fps(mols, radius=2, n_bits=2048):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]
    return fps


def compute_topk(test_rows, test_fps, train_rows, train_fps, topk=5):
    results = []
    for t_idx, t_fp in enumerate(test_fps):
        sims = DataStructs.BulkTanimotoSimilarity(t_fp, train_fps)
        # get topk indices
        topk_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:topk]
        for i in topk_idx:
            train_row_idx, train_smi = train_rows[i]
            test_row_idx, test_smi = test_rows[t_idx]
            results.append({
                "test_row": test_row_idx,
                "test_smiles": test_smi,
                "train_row": train_row_idx,
                "train_smiles": train_smi,
                "tanimoto": sims[i],
            })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train CSV")
    p.add_argument("--test", required=True, help="Path to test CSV")
    p.add_argument("--smiles-col", default=None, help="SMILES column name (optional). If omitted, auto-detected.")
    p.add_argument("--topk", type=int, default=5, help="Number of top matches to keep per test sulfonamide")
    p.add_argument("--out", default="mt_training/tanimoto_results.csv", help="Output CSV path")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    args = p.parse_args()

    df_train = load_csv(args.train)
    df_test = load_csv(args.test)

    smiles_col = args.smiles_col
    if smiles_col is None:
        smiles_col = detect_smiles_column(df_train) or detect_smiles_column(df_test)
        if smiles_col is None:
            print("Could not detect a SMILES column. Provide --smiles-col.")
            sys.exit(1)

    print(f"Using SMILES column: {smiles_col}")

    train_rows, train_mols = filter_sulfonamides(df_train, smiles_col)
    test_rows, test_mols = filter_sulfonamides(df_test, smiles_col)

    if not train_mols:
        print("No sulfonamides found in train set.")
        sys.exit(1)
    if not test_mols:
        print("No sulfonamides found in test set.")
        sys.exit(1)

    train_fps = mols_to_fps(train_mols, radius=args.radius, n_bits=args.n_bits)
    test_fps = mols_to_fps(test_mols, radius=args.radius, n_bits=args.n_bits)

    results = compute_topk(test_rows, test_fps, train_rows, train_fps, topk=args.topk)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
