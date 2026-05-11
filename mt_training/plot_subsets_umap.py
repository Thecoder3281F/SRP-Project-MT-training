#!/usr/bin/env python3
"""Plot UMAP of fingerprint vectors for progressive subset CSVs.

Usage examples:
  python mt_training/plot_subsets_umap.py \
    --dir datasets_final/chanlam_final/progressive_spread_rows \
    --out plot_subsets_umap.png \
    --smiles-col product_1_canonical_smiles \
    --max-per-file 200

Requirements: rdkit, umap-learn, numpy, pandas, matplotlib
"""
import argparse
import glob
import os
import sys
from typing import List

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except Exception:
    Chem = None
    AllChem = None
    DataStructs = None

try:
    import umap
except Exception:
    umap = None

import matplotlib.pyplot as plt
import matplotlib


def find_subset_files(dirpath: str) -> List[str]:
    files = sorted(
        glob.glob(os.path.join(dirpath, "train_rows_*.csv")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
    )
    return files


def compute_fp_array(smiles_list: List[str], n_bits: int = 2048, radius: int = 2):
    """Return numpy array shape (n_samples, n_bits) of 0/1 fingerprints.
    Rows with unparseable SMILES will have all zeros.
    """
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    if Chem is None or AllChem is None or DataStructs is None:
        raise SystemExit("RDKit is required for fingerprint calculation. Please install rdkit.")

    arr = None
    for i, smi in enumerate(smiles_list):
        try:
            if pd.isna(smi):
                continue
            m = Chem.MolFromSmiles(str(smi))
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            vec = np.zeros((n_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, vec)
            X[i, :] = vec
        except Exception:
            continue
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Directory containing train_groups_*.csv files")
    p.add_argument("--out", default="umap_subsets.png", help="Output image path")
    p.add_argument("--out-csv", default="", help="Optional output CSV with 2D coords appended")
    p.add_argument("--smiles-col", default="product_1_canonical_smiles")
    p.add_argument("--max-per-file", type=int, default=500, help="Max rows to sample per subset file")
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    files = find_subset_files(args.dir)
    if not files:
        raise SystemExit(f"No subset files found in {args.dir}")

    frames = []
    labels = []
    sources = []

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        if args.max_per_file and len(df) > args.max_per_file:
            df = df.sample(n=args.max_per_file, random_state=args.random_state)
        df = df.reset_index(drop=True)
        df['_subset_file'] = os.path.basename(f)
        df['_subset_order'] = i
        frames.append(df)

    bigdf = pd.concat(frames, ignore_index=True)
    smiles = bigdf[args.smiles_col].fillna("").tolist()

    if umap is None:
        print("umap-learn not installed. Install with: pip install umap-learn")
        raise SystemExit(1)

    print(f"Computing fingerprints for {len(smiles)} molecules...")
    X = compute_fp_array(smiles)
    nonzero = X.sum(axis=1) > 0
    if not nonzero.any():
        raise SystemExit("No valid fingerprints computed; check SMILES column and RDKit installation")

    Xnz = X[nonzero]
    meta = bigdf.loc[nonzero].reset_index(drop=True)

    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="jaccard",
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(Xnz)

    meta['umap_x'] = embedding[:, 0]
    meta['umap_y'] = embedding[:, 1]

    # plot by subset file size: draw larger subsets first, smaller last (so smaller are on top)
    sizes_series = meta.groupby('_subset_file').size()
    order = list(sizes_series.sort_values(ascending=False).index)

    cmap = matplotlib.cm.get_cmap('tab10')
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*']

    plt.figure(figsize=(9, 7), dpi=200)

    # scale marker size and alpha so smaller subsets are larger and more opaque
    min_count = int(sizes_series.min())
    max_count = int(sizes_series.max()) if int(sizes_series.max()) > 0 else 1

    for idx, name in enumerate(order):
        g = meta[meta['_subset_file'] == name]
        col = cmap(idx % 10)
        marker = markers[idx % len(markers)]
        # smaller groups -> larger markers
        size = int(np.interp(len(g), [min_count, max_count], [80, 8]))
        alpha = float(np.interp(len(g), [min_count, max_count], [0.95, 0.35]))
        plt.scatter(
            g['umap_x'],
            g['umap_y'],
            s=size,
            color=col,
            marker=marker,
            label=name,
            alpha=alpha,
            edgecolors='black',
            linewidths=0.3,
        )

    plt.legend(markerscale=1.5, fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title('UMAP of subsets')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(args.out)
    print('Saved plot to', args.out)

    if args.out_csv:
        meta.to_csv(args.out_csv, index=False)
        print('Saved coords to', args.out_csv)


if __name__ == '__main__':
    main()
