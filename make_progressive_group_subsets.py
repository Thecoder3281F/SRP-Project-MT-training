#!/usr/bin/env python3
"""Create progressively larger group-aware CSV subsets.

Typical use case:
- You have one training CSV (e.g. Chan-Lam train split)
- You want nested subsets like 10, 50, 100, ... up to full data
- Sampling should be by a group key so each group appears at most once

By default this script writes one representative row per group (`--sample-mode representative`),
which ensures no duplicate group keys (e.g. no duplicate reactants) in each subset.

Example:
  python mt_training/make_progressive_group_subsets.py \
    --input datasets_final/chanlam_final/train_final.csv \
    --out-dir datasets_final/chanlam_final/progressive_by_reactants \
    --group-col input_reactants \
    --sizes 10,50,100,250,500,1000 \
    --seed 42
"""

import argparse
import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
except Exception:
    Chem = None


DEFAULT_SIZES = [10, 50, 100, 250, 500, 1000, 2000, 5000]


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


def parse_sizes(spec: str | None) -> List[int]:
    if not spec or not spec.strip():
        return DEFAULT_SIZES.copy()

    vals: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError as exc:
            raise SystemExit(f"Invalid size '{tok}' in --sizes") from exc
        if v <= 0:
            raise SystemExit("All --sizes values must be positive integers")
        vals.append(v)

    if not vals:
        raise SystemExit("No valid sizes parsed from --sizes")
    return vals


def detect_group_col(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise SystemExit(f"Group column '{user_col}' not found in input CSV")
        return user_col

    for candidate in ["_group_key", "input_reactants", "input_sulfonamide"]:
        if candidate in df.columns:
            return candidate

    raise SystemExit(
        "Could not auto-detect group column. Please pass --group-col explicitly."
    )


def build_group_key_series(df: pd.DataFrame, group_col: str, canonicalise: bool) -> pd.Series:
    if canonicalise:
        return df[group_col].apply(canonicalise_smiles)
    return df[group_col].astype(str).str.strip()


def build_groups(group_key_series: pd.Series) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, key in group_key_series.items():
        groups[key].append(idx)
    return groups


def choose_representative_indices(groups: Dict[str, List[int]], ordered_keys: List[str], seed: int) -> List[int]:
    rnd = random.Random(seed + 999)
    out = []
    for key in ordered_keys:
        out.append(rnd.choice(groups[key]))
    return out


def _compute_fp(smiles: str):
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None
    if Chem is None:
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        from rdkit.Chem import AllChem

        return AllChem.GetMorganFingerprintAsBitVect(m, 2)
    except Exception:
        return None


def choose_diverse_indices(
    groups: Dict[str, List[int]],
    ordered_keys: List[str],
    df: pd.DataFrame,
    diversity_col: str,
    per_group: int,
    seed: int,
) -> List[int]:
    """
    Greedy max-min selection: for each group in ordered_keys, pick up to `per_group` rows
    whose fingerprint maximises the minimum distance to already selected items.
    Falls back to random choice if fingerprints cannot be computed.
    """
    rnd = random.Random(seed + 1234)
    selected_indices: List[int] = []

    # cache fingerprints for candidate rows
    fp_cache = {}
    for key in ordered_keys:
        for idx in groups[key]:
            val = df.at[idx, diversity_col] if diversity_col in df.columns else ""
            if pd.isna(val):
                val = ""
            fp_cache[idx] = _compute_fp(val)

    # helper to compute similarity
    try:
        from rdkit import DataStructs as _DS
    except Exception:
        _DS = None

    for key in ordered_keys:
        candidates = list(groups[key])
    pbar_diverse = tqdm(total=len(ordered_keys), desc="  Selecting diverse rows per group", leave=False)
    for key in ordered_keys:
        candidates = list(groups[key])
        pbar_diverse.update(1)
        for _ in range(min(per_group, len(candidates))):
            # evaluate each remaining candidate
            best_idx = None
            best_score = -1.0
            for idx in candidates:
                fp = fp_cache.get(idx)
                if fp is None or Chem is None or _DS is None:
                    # no fingerprint -> random tie-breaker
                    score = rnd.random()
                else:
                    if not selected_indices:
                        # no selected yet: prefer variety via random seed
                        score = rnd.random()
                    else:
                        # compute minimum similarity to selected set
                        sims = []
                        for sidx in selected_indices:
                            sfp = fp_cache.get(sidx)
                            if sfp is None:
                                continue
                            try:
                                sims.append(_DS.TanimotoSimilarity(fp, sfp))
                            except Exception:
                                continue
                        if not sims:
                            score = rnd.random()
                        else:
                            min_sim = min(sims)
                            # lower similarity means more diverse -> higher score
                            score = 1.0 - min_sim

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                # fallback
                best_idx = rnd.choice(candidates)

            selected_indices.append(best_idx)
            candidates.remove(best_idx)

    return selected_indices


def choose_spread_indices(groups, ordered_keys, df, diversity_col, per_group, target_rows, seed=None):
    """Round-robin spread selection across groups until target_rows reached.
    For each group in cycles, pick one best candidate (diversity-aware if possible)
    until we reach target_rows or exhaust candidates.
    """
    rng = random.Random(seed)
    fp_cache = {}

    # cache fingerprints for candidate rows
    for key in ordered_keys:
        for idx in groups.get(key, []):
            val = df.at[idx, diversity_col] if diversity_col in df.columns else None
            fp_cache[idx] = _compute_fp(val)

    # try importing DataStructs for similarity checks
    try:
        from rdkit import DataStructs as _DS
    except Exception:
        _DS = None

    # prepare per-group candidate lists (shuffled)
    group_candidates = {}
    for k in ordered_keys:
        rows = list(groups.get(k, []))
        rng.shuffle(rows)
        group_candidates[k] = rows

    selected = []
    picked_per_group = {k: 0 for k in ordered_keys}

    # helper to pick next candidate for a group
    def _best_candidate_for_group(k):
        candidates = group_candidates.get(k, [])
        best_idx = None
        if _DS is not None and selected:
            # choose candidate with minimal max similarity to selected (more diverse)
            best_score = None
            for idx in candidates:
                if idx in selected:
                    continue
                fp_idx = fp_cache.get(idx)
                if fp_idx is None:
                    return idx
                sims = []
                for s in selected:
                    fp_s = fp_cache.get(s)
                    if fp_s is None:
                        sims.append(0.0)
                    else:
                        try:
                            sims.append(_DS.TanimotoSimilarity(fp_idx, fp_s))
                        except Exception:
                            sims.append(0.0)
                max_sim = max(sims) if sims else 0.0
                # prefer lower max_sim
                if best_score is None or max_sim < best_score:
                    best_score = max_sim
                    best_idx = idx
            return best_idx
        else:
            # pick first unseen candidate
            for idx in candidates:
                if idx not in selected:
                    return idx
            return None

    # round-robin until target reached or no candidates remain
    exhausted = False
    pbar = tqdm(total=target_rows, desc="  Spread selection progress", leave=False)
    while len(selected) < target_rows and not exhausted:
        exhausted = True
        for k in ordered_keys:
            if len(selected) >= target_rows:
                break
            if picked_per_group.get(k, 0) >= per_group:
                continue
            cand = _best_candidate_for_group(k)
            if cand is None:
                continue
            exhausted = False
            selected.append(cand)
            picked_per_group[k] = picked_per_group.get(k, 0) + 1
            pbar.update(1)

    # if still short, fill with any remaining rows (random)
    if len(selected) < target_rows:
        all_remaining = [i for i in df.index if i not in selected]
        rng.shuffle(all_remaining)
        needed = target_rows - len(selected)
        selected.extend(all_remaining[:needed])
    pbar.close()

    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input training CSV")
    p.add_argument("--out-dir", required=True, help="Directory where subsets are written")
    p.add_argument(
        "--group-col",
        default=None,
        help="Column used as group key (default: auto-detect _group_key/input_reactants/input_sulfonamide)",
    )
    p.add_argument(
        "--sizes",
        default="",
        help="Comma-separated target subset sizes in number of groups (e.g. 10,50,100)",
    )
    p.add_argument(
        "--sizes-rows",
        default="",
        help="Comma-separated target subset sizes in number of rows (will be converted to groups using --per-group)",
    )
    p.add_argument(
        "--sample-mode",
        choices=["representative", "all-group-rows", "diverse", "spread"],
        default="representative",
        help=(
            "representative = 1 random row per group (no duplicate group key rows); "
            "all-group-rows = include every row from selected groups; "
            "diverse = choose representative rows to maximise chemical diversity (requires RDKit); "
            "spread = round-robin per-group selection to spread group occurrences across subsets"
        ),
    )
    p.add_argument(
        "--diversity-col",
        default="product_1_canonical_smiles",
        help="Column used to compute diversity fingerprints (SMILES).",
    )
    p.add_argument(
        "--per-group",
        type=int,
        default=1,
        help="Number of rows to sample per group when using 'diverse' mode (default 1).",
    )
    p.add_argument(
        "--auto-per-group",
        action="store_true",
        help="Automatically choose the minimal per-group count per target rows (ceil(rows / groups_selected)).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--no-canonicalise",
        action="store_true",
        help="Do not canonicalise SMILES in group column before grouping",
    )
    p.add_argument(
        "--exclude-full",
        action="store_true",
        help="Do not force-add full dataset size as the final subset",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    group_col = detect_group_col(df, args.group_col)

    group_key = build_group_key_series(
        df,
        group_col=group_col,
        canonicalise=not args.no_canonicalise,
    )

    df2 = df.copy()
    if "_group_key" in df2.columns:
        # Preserve original for traceability, but use computed key for sampling.
        df2["_group_key_sampled"] = group_key
        effective_group_col = "_group_key_sampled"
    else:
        df2["_group_key"] = group_key
        effective_group_col = "_group_key"

    groups = build_groups(df2[effective_group_col])
    keys = list(groups.keys())

    rnd = random.Random(args.seed)
    rnd.shuffle(keys)

    # Parse requested sizes (groups) or sizes as rows
    requested_sizes = []
    requested_rows = None
    if args.sizes_rows and args.sizes_rows.strip():
        requested_rows = [int(x) for x in parse_sizes(args.sizes_rows)]
    else:
        requested_sizes = sorted(set(parse_sizes(args.sizes)))

    n_groups = len(keys)

    # If not using auto-per-group and sizes-rows provided, convert rows->groups using args.per_group
    skipped_sizes = []
    if requested_rows is not None and not args.auto_per_group:
        import math

        if args.per_group <= 0:
            raise SystemExit("--per-group must be > 0")
        requested_sizes = sorted(set(max(1, math.ceil(r / args.per_group)) for r in requested_rows))

    if not args.exclude_full and n_groups not in requested_sizes:
        requested_sizes.append(n_groups)
        requested_sizes = sorted(requested_sizes)

    valid_sizes = [s for s in requested_sizes if s <= n_groups]
    skipped_sizes = [s for s in requested_sizes if s > n_groups]

    if not valid_sizes and not (requested_rows is not None and args.auto_per_group):
        raise SystemExit(
            f"No valid sizes. Requested sizes are all larger than available groups ({n_groups})."
        )

    summary_rows = []

    # If auto-per-group and sizes-rows provided, iterate over requested_rows directly
    if requested_rows is not None and args.auto_per_group:
        import math
        requested_rows_sorted = sorted(requested_rows)
        summary_rows = []
        for r in tqdm(requested_rows_sorted, desc="Creating progressive row subsets"):
            if r <= n_groups:
                n = r
                selected_keys = keys[:n]
                target_rows = r
                per_group_for_n = 1
            else:
                n = n_groups
                selected_keys = keys[:n]
                target_rows = r
                per_group_for_n = max(1, math.ceil(r / max(1, n)))

            if args.sample_mode == "representative":
                indices = choose_representative_indices(groups, selected_keys, seed=args.seed)
                subset = df2.loc[indices].copy()
                subset = subset.sort_index()
            elif args.sample_mode == "diverse":
                indices = choose_diverse_indices(
                    groups=groups,
                    ordered_keys=selected_keys,
                    df=df2,
                    diversity_col=args.diversity_col,
                    per_group=max(1, args.per_group),
                    seed=args.seed,
                )
                subset = df2.loc[indices].copy()
                subset = subset.sort_index()
            elif args.sample_mode == "spread":
                indices = choose_spread_indices(
                    groups=groups,
                    ordered_keys=selected_keys,
                    df=df2,
                    diversity_col=args.diversity_col,
                    per_group=per_group_for_n,
                    target_rows=target_rows,
                    seed=args.seed,
                )
                subset = df2.loc[indices].copy()
                subset = subset.sort_index()
            else:
                indices = []
                for k in selected_keys:
                    indices.extend(groups[k])
                subset = df2.loc[indices].copy()
                subset = subset.sort_index()

            out_name = f"train_rows_{r}.csv"
            out_path = os.path.join(args.out_dir, out_name)
            subset.to_csv(out_path, index=False)

            row = {
                "requested_rows": r,
                "requested_groups": n,
                "actual_groups": subset[effective_group_col].nunique(),
                "rows": len(subset),
                "sample_mode": args.sample_mode,
                "file": out_name,
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.out_dir, "subsets_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        key_order_path = os.path.join(args.out_dir, "group_key_order.csv")
        pd.DataFrame({effective_group_col: keys}).to_csv(key_order_path, index=False)

        print(f"Input rows: {len(df2)}")
        print(f"Group column: {group_col}")
        print(f"Effective sampling key column: {effective_group_col}")
        print(f"Unique groups available: {n_groups}")
        print(f"Wrote {len(summary_rows)} subset file(s) to: {args.out_dir}")
        print(f"Summary: {summary_path}")
        print(f"Group key order: {key_order_path}")
        return

    for n in tqdm(valid_sizes, desc="Creating progressive group subsets"):
        selected_keys = keys[:n]

        if args.sample_mode == "representative":
            indices = choose_representative_indices(groups, selected_keys, seed=args.seed)
            subset = df2.loc[indices].copy()
            subset = subset.sort_index()
        elif args.sample_mode == "diverse":
            # choose up to `per_group` rows per group using diversity-driven greedy selection
            indices = choose_diverse_indices(
                groups=groups,
                ordered_keys=selected_keys,
                df=df2,
                diversity_col=args.diversity_col,
                per_group=max(1, args.per_group),
                seed=args.seed,
            )
            subset = df2.loc[indices].copy()
            subset = subset.sort_index()
        elif args.sample_mode == "spread":
            # target_rows: if sizes were provided as rows, map to this group count; otherwise use n * per_group
            if requested_rows is not None:
                matching_rows = [r for r in requested_rows if max(1, -(-r // args.per_group)) == n]
                target_rows = max(matching_rows) if matching_rows else n * max(1, args.per_group)
            else:
                target_rows = n * max(1, args.per_group)

            # if auto-per-group requested, compute minimal per-group for this n
            if args.auto_per_group:
                import math

                per_group_for_n = max(1, math.ceil(target_rows / max(1, n)))
            else:
                per_group_for_n = max(1, args.per_group)

            indices = choose_spread_indices(
                groups=groups,
                ordered_keys=selected_keys,
                df=df2,
                diversity_col=args.diversity_col,
                per_group=per_group_for_n,
                target_rows=target_rows,
                seed=args.seed,
            )
            subset = df2.loc[indices].copy()
            subset = subset.sort_index()
        else:
            indices = []
            for k in selected_keys:
                indices.extend(groups[k])
            subset = df2.loc[indices].copy()
            subset = subset.sort_index()

        out_name = f"train_groups_{n}.csv"
        out_path = os.path.join(args.out_dir, out_name)
        subset.to_csv(out_path, index=False)

        row = {
            "requested_groups": n,
            "actual_groups": subset[effective_group_col].nunique(),
            "rows": len(subset),
            "sample_mode": args.sample_mode,
            "file": out_name,
        }
        if requested_rows is not None:
            # map back to the requested rows that correspond to this group count (may be multiple)
            matching_rows = [r for r in requested_rows if max(1, -(-r // args.per_group)) == n]
            row["requested_rows"] = ",".join(str(r) for r in matching_rows) if matching_rows else ""

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "subsets_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    key_order_path = os.path.join(args.out_dir, "group_key_order.csv")
    pd.DataFrame({effective_group_col: keys}).to_csv(key_order_path, index=False)

    print(f"Input rows: {len(df2)}")
    print(f"Group column: {group_col}")
    print(f"Effective sampling key column: {effective_group_col}")
    print(f"Unique groups available: {n_groups}")
    print(f"Wrote {len(valid_sizes)} subset file(s) to: {args.out_dir}")

    if skipped_sizes:
        print(f"Skipped sizes larger than available groups: {skipped_sizes}")

    print(f"Summary: {summary_path}")
    print(f"Group key order: {key_order_path}")


if __name__ == "__main__":
    main()
