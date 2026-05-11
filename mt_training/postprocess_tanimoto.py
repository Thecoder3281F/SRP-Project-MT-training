#!/usr/bin/env python3
"""Postprocess raw Tanimoto CSV to select top-5 unique train sulfonamides per test sulfonamide."""
import pandas as pd
import sys


def unique_topk(raw_path: str, out_path: str, topk: int = 5):
    df = pd.read_csv(raw_path)
    out_rows = []
    for test_smiles, group in df.groupby('test_smiles'):
        group_sorted = group.sort_values('tanimoto', ascending=False)
        seen = set()
        rank = 1
        for _, row in group_sorted.iterrows():
            train_smi = row['train_smiles']
            if train_smi in seen:
                continue
            seen.add(train_smi)
            out_rows.append({
                'test_smiles': test_smiles,
                'rank': rank,
                'train_smiles': train_smi,
                'train_row': row.get('train_row'),
                'tanimoto': row['tanimoto']
            })
            rank += 1
            if rank > topk:
                break
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)
    print(f'Wrote {len(out_df)} rows to {out_path}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: postprocess_tanimoto.py <raw_csv> <out_csv> [topk]')
        sys.exit(1)
    raw = sys.argv[1]
    out = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    unique_topk(raw, out, topk=k)
