"""
Push all `train_groups_*.csv` (or other CSV split files) from a directory
to a Hugging Face Dataset repo as separate splits in a DatasetDict.

Usage examples:

# Push all train_groups_*.csv in a directory to a dataset repo
python mt_training/push_splits_to_hf.py \
  --input-dir datasets_final/chanlam_final/progressive_spread_rows \
  --repo-id your-username/chanlam_progressive_splits \
  --private

# Provide a token explicitly (optional)
python mt_training/push_splits_to_hf.py --input-dir ... --repo-id ... --token HF_TOKEN

The script will collect CSVs matching `--pattern`, sort them by numeric
component if possible (e.g. train_groups_10.csv -> 10) and push them as splits.
"""

from pathlib import Path
import argparse
import re
import sys
import logging

import pandas as pd

try:
    from datasets import Dataset, DatasetDict
except Exception as e:
    raise RuntimeError("This script requires the 'datasets' package. Install via `pip install datasets[torch]`.") from e

logger = logging.getLogger("push_splits_to_hf")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

NUM_EXTRACT_RE = re.compile(r"(\d+)")


def _sortable_key(fname: str):
    """Extract a numeric key from filename for sensible ordering, fall back to name."""
    m = NUM_EXTRACT_RE.search(fname)
    if m:
        return int(m.group(1))
    return fname


def gather_csv_files(directory: Path, pattern: str):
    files = sorted(directory.glob(pattern), key=lambda p: _sortable_key(p.name))
    return files


def make_split_name(path: Path):
    # use stem as split name, but ensure it is a valid dataset split identifier
    name = path.stem
    name = re.sub(r"[^0-9a-zA-Z_\-]", "_", name)
    return name


def main():
    p = argparse.ArgumentParser(description="Push CSV split files to a Hugging Face dataset repo")
    p.add_argument("--input-dir", required=True, help="Directory containing CSV split files")
    p.add_argument("--pattern", default="train_rows_*.csv", help="Glob pattern to find split CSVs (default: train_groups_*.csv)")
    p.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/repo_name")
    p.add_argument("--token", default=None, help="Hugging Face token (optional, uses local cache if omitted)")
    p.add_argument("--private", action="store_true", help="Create the dataset repo as private")
    p.add_argument("--description", default=None, help="Optional dataset description to include in the dataset card")
    p.add_argument("--max-rows-preview", type=int, default=5, help="Number of rows to include as preview in card metadata")

    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(2)

    files = gather_csv_files(input_dir, args.pattern)
    if not files:
        logger.error("No files found using pattern '%s' in %s", args.pattern, input_dir)
        sys.exit(2)

    logger.info("Found %d CSV files", len(files))

    split_map = {}
    for fp in files:
        split_name = make_split_name(fp)
        logger.info("Reading %s -> split '%s'", fp.name, split_name)
        df = pd.read_csv(fp)
        # drop unnamed index column if present
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        df = df.reset_index(drop=True)
        # Convert to Hugging Face Dataset
        ds = Dataset.from_pandas(df)
        split_map[split_name] = ds

    dataset_dict = DatasetDict(split_map)

    # optional dataset card metadata
    card = {
        "name": args.repo_id,
    }
    if args.description:
        card["description"] = args.description
    else:
        card["description"] = (
            f"Progressive split CSVs pushed from {input_dir} — files: {', '.join(p.name for p in files)}"
        )

    # include a tiny preview in the dataset_info if possible
    try:
        preview_rows = pd.read_csv(files[0], nrows=args.max_rows_preview)
        card["preview_csv_head"] = preview_rows.to_csv(index=False)
    except Exception:
        pass

    logger.info("Pushing dataset to Hub: %s (private=%s)", args.repo_id, args.private)

    try:
        dataset_dict.push_to_hub(args.repo_id, token=args.token, private=args.private)
    except Exception as e:
        logger.error("Failed to push dataset: %s", e)
        logger.info("If the repo does not exist, create it on the Hub (or ensure you have permissions).")
        raise

    logger.info("Dataset pushed successfully. You can `load_dataset('%s')` or visit https://huggingface.co/%s", args.repo_id, args.repo_id)


if __name__ == "__main__":
    main()
