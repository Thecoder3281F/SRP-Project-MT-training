#!/usr/bin/env python3
"""Combine reactant and reagent columns into `input_reactants` and `input_reagents`.

Usage:
  python mt_training/combine_reactants_reagents.py \
    --input path/to/data.csv \
    --reactant-cols input_boronic_acid,input_sulfonamide \
    --reagent-cols input_catalyst,input_base,input_solvent \
    --out output.csv

If `--reactant-cols` or `--reagent-cols` are omitted the script attempts
simple auto-detection by column name keywords.
"""
import argparse
import os
from typing import List

import pandas as pd

try:
    from rdkit import Chem
except Exception:
    Chem = None


def canonicalise_smiles(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    if Chem is None:
        return s
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return s


def join_components(vals: List[str], canonicalise: bool = True) -> str:
    parts = []
    for v in vals:
        if pd.isna(v):
            continue
        v = str(v).strip()
        if not v:
            continue
        # split if columns contain multiple SMILES separated by '.'
        if '.' in v and not v.lower().startswith('inchi='):
            pieces = [p.strip() for p in v.split('.') if p.strip()]
        else:
            pieces = [v]
        for p in pieces:
            pcan = canonicalise_smiles(p) if canonicalise else p
            if pcan:
                parts.append(pcan)
    # deduplicate while preserving order
    seen = set()
    out = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return '.'.join(out)


def autodetect_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    reactant_keywords = ['reactant', 'substrate', 'boronic', 'sulfonamide', 'electrophile']
    reagent_keywords = ['catalyst', 'base', 'solvent', 'reagent', 'additive']

    reactant_cols = [c for c in cols if any(k in c.lower() for k in reactant_keywords)]
    reagent_cols = [c for c in cols if any(k in c.lower() for k in reagent_keywords)]

    return reactant_cols, reagent_cols


def combine_columns(df: pd.DataFrame, reactant_cols: List[str], reagent_cols: List[str], canonicalise=True):
    df = df.copy()
    if reactant_cols:
        df['input_reactants'] = df[reactant_cols].apply(lambda row: join_components(row.tolist(), canonicalise), axis=1)
    else:
        df['input_reactants'] = ''

    if reagent_cols:
        df['input_reagents'] = df[reagent_cols].apply(lambda row: join_components(row.tolist(), canonicalise), axis=1)
    else:
        df['input_reagents'] = ''

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input CSV file')
    p.add_argument('--reactant-cols', default=None, help='Comma-separated reactant columns')
    p.add_argument('--reagent-cols', default=None, help='Comma-separated reagent columns')
    p.add_argument('--out', default=None, help='Output CSV path (defaults to input with _combined suffix)')
    p.add_argument('--no-canonicalise', action='store_true', help='Do not canonicalise SMILES')
    args = p.parse_args()

    df = pd.read_csv(args.input)

    if args.reactant_cols:
        reactant_cols = [c.strip() for c in args.reactant_cols.split(',') if c.strip()]
    else:
        reactant_cols, _ = autodetect_columns(df)

    if args.reagent_cols:
        reagent_cols = [c.strip() for c in args.reagent_cols.split(',') if c.strip()]
    else:
        _, reagent_cols = autodetect_columns(df)

    missing = [c for c in (reactant_cols + reagent_cols) if c and c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found in input: {missing}")

    df2 = combine_columns(df, reactant_cols, reagent_cols, canonicalise=not args.no_canonicalise)

    # Ensure `input_reactants` and `input_reagents` appear before product columns,
    # and move original reactant/reagent source columns to the end (keep them).
    # Reorder columns so that:
    # 1) non-product/core columns (preserving their original order)
    # 2) `input_reactants`, `input_reagents` (if present)
    # 3) product columns (preserving original order)
    # 4) original reactant/reagent source columns (preserving original order)
    cols_orig = list(df2.columns)
    product_cols = [c for c in cols_orig if 'product' in c.lower()]
    src_cols = [c for c in (reactant_cols + reagent_cols) if c and c in cols_orig]
    inputs = [c for c in ('input_reactants', 'input_reagents') if c in cols_orig]

    core_cols = [c for c in cols_orig if c not in product_cols + src_cols + inputs]

    new_order = core_cols + inputs + product_cols + src_cols
    # Deduplicate while preserving order (defensive)
    seen = set()
    final_cols = []
    for c in new_order:
        if c not in seen and c in df2.columns:
            final_cols.append(c)
            seen.add(c)

    df2 = df2[final_cols]
    print(f"Reordered columns. Products moved to end. Final column order saved.")

    out_path = args.out or os.path.splitext(args.input)[0] + '_combined.csv'
    df2.to_csv(out_path, index=False)
    print(f'Wrote combined file to {out_path}')
    print(f'Detected reactant columns: {reactant_cols}')
    print(f'Detected reagent columns: {reagent_cols}')


if __name__ == '__main__':
    main()
