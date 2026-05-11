#!/usr/bin/env python3
"""Group pairwise Tanimoto CSV by `test_smiles`, sort descending, add rank, and write output."""
import argparse
import pandas as pd


def group_and_rank(in_path: str, out_path: str):
    df = pd.read_csv(in_path)
    df['tanimoto'] = pd.to_numeric(df['tanimoto'], errors='coerce').fillna(0.0)
    df_sorted = df.sort_values(['test_smiles', 'tanimoto'], ascending=[True, False])
    df_sorted['rank'] = df_sorted.groupby('test_smiles')['tanimoto'].rank(method='first', ascending=False).astype(int)
    cols = ['test_smiles', 'rank', 'train_smiles', 'tanimoto']
    df_sorted.to_csv(out_path, index=False, columns=cols)
    print(f'Wrote {len(df_sorted)} rows to {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default='mt_training/tanimoto_pairwise.csv')
    p.add_argument('--out', dest='out_path', default='mt_training/tanimoto_grouped.csv')
    args = p.parse_args()
    group_and_rank(args.in_path, args.out_path)


if __name__ == '__main__':
    main()
