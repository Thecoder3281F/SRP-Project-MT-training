#!/usr/bin/env python3
"""Group-based 80/10/10 splitter by sulfonamide.

Usage example:
  python mt_training/group_split_by_sulfonamide.py \
    --input datasets_final/MIT_final/MIT_mixed_final/mit_all.csv \
    --group-col input_sulfonamide \
    --out-dir datasets_final/MIT_final/MIT_mixed_final/splits \
    --seed 42

The script groups rows by canonicalised `--group-col` (sulfonamide) and assigns
whole groups to train/val/test to avoid group duplication across splits.
"""
import argparse
import os
import random
from collections import defaultdict

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


def build_groups(df: pd.DataFrame, group_col: str, canonicalise: bool = True):
    df = df.copy()
    if canonicalise:
        df['_group_key'] = df[group_col].apply(canonicalise_smiles)
    else:
        df['_group_key'] = df[group_col].astype(str).str.strip()
    groups = defaultdict(list)
    for idx, key in df['_group_key'].items():
        groups[key].append(idx)
    return df, groups


def assign_groups_to_splits(groups, total_rows, fracs, seed=0):
    keys = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(keys)

    targets = {k: fracs[k] * total_rows for k in fracs}
    counts = {k: 0 for k in fracs}
    assignment = {}

    for key in keys:
        gsize = len(groups[key])
        # compute deficits (target - current), prefer the largest deficit
        deficits = {k: targets[k] - counts[k] for k in fracs}
        best = max(deficits.items(), key=lambda x: x[1])[0]
        # if all deficits are <= 0, place into split with smallest relative occupancy
        if deficits[best] <= 0:
            rel = {k: (counts[k] / max(1, targets[k])) for k in fracs}
            best = min(rel.items(), key=lambda x: x[1])[0]

        assignment[key] = best
        counts[best] += gsize

    return assignment, counts


def save_splits(df, groups, assignment, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    split_indices = defaultdict(list)
    for key, split in assignment.items():
        split_indices[split].extend(groups[key])

    for split, indices in split_indices.items():
        out_path = os.path.join(out_dir, f"{split}.csv")
        df.loc[indices].to_csv(out_path, index=False)


def validate_no_group_overlap(assignment):
    # groups map to exactly one split by construction; just sanity-check
    groups_per_split = defaultdict(list)
    for g, s in assignment.items():
        groups_per_split[s].append(g)
    overlaps = False
    # ensure group keys are unique across splits
    all_groups = set()
    for s, gl in groups_per_split.items():
        for g in gl:
            if g in all_groups:
                overlaps = True
            all_groups.add(g)
    return overlaps, {s: len(gl) for s, gl in groups_per_split.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input CSV with MIT data (all rows)')
    p.add_argument('--group-col', default='input_sulfonamide', help='Column to group by (sulfonamide)')
    p.add_argument('--out-dir', default='splits', help='Output directory for CSV splits')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--train-frac', type=float, default=0.8)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--test-frac', type=float, default=0.1)
    p.add_argument('--no-canonicalise', action='store_true', help='Do not canonicalise SMILES when grouping')
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if args.group_col not in df.columns:
        raise SystemExit(f"Group column '{args.group_col}' not found in {args.input}")

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if abs(frac_sum - 1.0) > 1e-6:
        raise SystemExit('train/val/test fractions must sum to 1.0')

    fracs = {'train': args.train_frac, 'val': args.val_frac, 'test': args.test_frac}

    df2, groups = build_groups(df, args.group_col, canonicalise=not args.no_canonicalise)
    total_rows = len(df2)

    assignment, counts = assign_groups_to_splits(groups, total_rows, fracs, seed=args.seed)

    save_splits(df2, groups, assignment, args.out_dir)

    overlaps, groups_per_split = validate_no_group_overlap(assignment)

    print('Split summary:')
    for s in ('train', 'val', 'test'):
        n_rows = counts.get(s, 0)
        n_groups = groups_per_split.get(s, 0)
        print(f"  {s}: rows={n_rows}, groups={n_groups}, fraction={n_rows/total_rows:.4f}")

    if overlaps:
        print('Warning: group overlap detected across splits (unexpected)')
    else:
        print('Validation: no group duplication across splits (OK)')

    # Save mapping for inspection
    mapping_path = os.path.join(args.out_dir, 'group_assignment.csv')
    with open(mapping_path, 'w', encoding='utf8') as fh:
        fh.write('group_key,split,group_size\n')
        for g, split in assignment.items():
            fh.write(f'"{g}",{split},{len(groups[g])}\n')

    print(f'Wrote splits to {args.out_dir} and mapping to {mapping_path}')


if __name__ == '__main__':
    main()
