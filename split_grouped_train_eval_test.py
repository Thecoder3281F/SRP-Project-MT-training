#!/usr/bin/env python3
"""Create a fair 70/15/15 grouped train/eval/test split.

This script preserves all rows belonging to the same group in the same split,
which helps avoid data leakage when a single reaction input appears multiple times
across the dataset.

Example:
    python split_grouped_train_eval_test.py \
        --input _datasets_new_cleaned/buchwald_hartwig/buchwald_hartwig_combined_cleaned_canonical.csv \
        --out-dir _datasets_new_cleaned/buchwald_hartwig/splits_grouped_70_15_15 \
        --group-col reactants
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def choose_group_col(df: pd.DataFrame, group_col: str | None) -> str:
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in CSV.")
        return group_col

    preferred = [
        "reactants",
        "input_reactants",
        "reactant_1",
        "input_sulfonamide",
    ]
    for candidate in preferred:
        if candidate in df.columns:
            return candidate

    if "reactant_1" in df.columns and "reactant_2" in df.columns:
        return "reactant_1"

    raise ValueError(
        "No suitable grouping column found. Please pass --group-col explicitly."
    )


def build_group_key(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col in ["reactant_1", "reactant_2"]:
        return df[group_col].astype(str).str.strip()

    if group_col == "reactants" and "reactants" in df.columns:
        return df["reactants"].astype(str).str.strip()

    if group_col in df.columns:
        return df[group_col].astype(str).str.strip()

    if "reactant_1" in df.columns and "reactant_2" in df.columns:
        combined = (
            df["reactant_1"].fillna("").astype(str).str.strip()
            + "||"
            + df["reactant_2"].fillna("").astype(str).str.strip()
        )
        return combined

    raise ValueError(f"Could not derive a group key from column '{group_col}'.")


def split_grouped_70_15_15(df: pd.DataFrame, group_col: str, seed: int = 42):
    groups = build_group_key(df, group_col)
    unique_groups = groups.nunique()

    if unique_groups < 3:
        raise ValueError(
            f"Need at least 3 unique groups for a 70/15/15 split, got {unique_groups}."
        )

    # 70/30 split for train vs remainder.
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    temp_groups = build_group_key(temp_df, group_col)

    # 50/50 split of the remainder to get 15/15 percentages.
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    eval_idx, test_idx = next(splitter2.split(temp_df, groups=temp_groups))

    eval_df = temp_df.iloc[eval_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

    return train_df, eval_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Create a 70/15/15 grouped train/eval/test split.")
    parser.add_argument("--input", type=str, required=True, help="CSV file to split.")
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write train.csv, eval.csv, and test.csv",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Column to use as the grouping key (default: reactants if present)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    group_col = choose_group_col(df, args.group_col)
    train_df, eval_df, test_df = split_grouped_70_15_15(df, group_col=group_col, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    eval_path = out_dir / "eval.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Group column: {group_col}")
    print(f"Train rows: {len(train_df)}")
    print(f"Eval rows: {len(eval_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
