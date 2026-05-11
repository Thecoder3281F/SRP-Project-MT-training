#!/usr/bin/env python3
"""
Load a Hugging Face dataset (chanlam), concatenate splits, shuffle with a fixed seed,
and write to CSV for cross-checking reactants/reagents.

Usage examples:
  python "mt_training/make_chanlam_csv.py" TheCoder3281/chanlam --out data_chanlam.csv
  python "mt_training/make_chanlam_csv.py" TheCoder3281/chanlam --split all --out data_chanlam.csv
"""
import argparse
import os
from datasets import load_dataset
import pandas as pd


def load_and_concat(ds, split_arg):
    if split_arg == "all":
        frames = []
        for s in ds.keys():
            try:
                frames.append(ds[s].to_pandas())
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    else:
        if split_arg not in ds.keys():
            raise ValueError(f"Split '{split_arg}' not found in dataset. Available: {list(ds.keys())}")
        return ds[split_arg].to_pandas()


def main():
    p = argparse.ArgumentParser(description="Dump HF dataset splits to a single shuffled CSV (seedable)")
    p.add_argument("dataset", help="Hugging Face dataset id (e.g. TheCoder3281/chanlam)")
    p.add_argument("--split", default="all", help="Split to use or 'all' to concatenate all splits")
    p.add_argument("--out", default="chanlam_shuffled.csv", help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    p.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the dataset (default: shuffle)")
    args = p.parse_args()

    print(f"Loading dataset '{args.dataset}' from Hugging Face...")
    ds = load_dataset(args.dataset)
    print(f"Available splits: {list(ds.keys())}")

    print(f"Building dataframe from split='{args.split}'...")
    df = load_and_concat(ds, args.split)
    if df.empty:
        print("No data loaded — exiting.")
        return

    if args.no_shuffle:
        print(f"Loaded {len(df)} rows — not shuffling (seed ignored).")
    else:
        print(f"Loaded {len(df)} rows — shuffling with seed {args.seed}...")
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote shuffled CSV to: {args.out}")


if __name__ == "__main__":
    main()
