#!/usr/bin/env python3
"""
Push a final model checkpoint to the Hugging Face Hub.

Steps performed:
1. Take the provided final checkpoint folder (argument `--checkpoint_dir`).
2. Search for a `trainer_state.json` inside the checkpoint folder (or its subfolders).
   If found in a subfolder, copy it into the checkpoint root (so it lives next to model files).
3. Locate a tokenizer directory under `--tokenizers_root` (defaults to ../tokenizers).
   Tries to pick the best match automatically; you can override with `--tokenizer_name`.
4. Assemble model + tokenizer + trainer_state into a temporary folder and upload to HF Hub.

Usage:
  python push_model_to_hf.py --checkpoint_dir ./models/mymodel_final --repo_id username/mymodel

Requires: `huggingface_hub` installed and an HF token available via env `HF_TOKEN` or `--token`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi
except Exception as e:
    print("Error: huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


def find_trainer_state(checkpoint_dir: Path) -> Optional[Path]:
    # Look for trainer_state.json in checkpoint_dir first, then recursively choose the newest one
    candidate = checkpoint_dir / "trainer_state.json"
    if candidate.exists():
        return candidate

    found = list(checkpoint_dir.rglob("trainer_state.json"))
    if not found:
        return None
    # pick the most recently modified one
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0]


def choose_tokenizer_dir(tokenizers_root: Path, checkpoint_dir: Path, override: Optional[str] = None) -> Optional[Path]:
    if override:
        cand = tokenizers_root / override
        if cand.exists():
            return cand
        # allow direct path
        cand2 = Path(override)
        if cand2.exists():
            return cand2

    if not tokenizers_root.exists():
        return None

    subdirs = [p for p in tokenizers_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None

    name = checkpoint_dir.name.lower()
    # heuristic: look for a subdir name that shares tokens with checkpoint name
    tokens = [t for t in re_split(name) if t]
    for sub in subdirs:
        s = sub.name.lower()
        if any(tok in s for tok in tokens):
            return sub

    # fallback: if exactly one, return it
    if len(subdirs) == 1:
        return subdirs[0]

    # final fallback: pick the first one
    return subdirs[0]


def re_split(s: str):
    import re

    return re.split(r"[^0-9a-zA-Z]+", s)


def copy_tree(src: Path, dst: Path, ignore_large_files: bool = False, exclude_tokens: Optional[list[str]] = None):
    # copytree-like but merges into dst if exists
    # exclude_tokens: list of lowercase substrings; if any token present in filename, skip that file/dir
    if exclude_tokens is None:
        exclude_tokens = []
    for item in src.iterdir():
        name_l = item.name.lower()
        if any(tok in name_l for tok in exclude_tokens):
            # skip optimizer/scheduler/rng related files
            print(f"Skipping excluded file/dir: {item}")
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def upload_folder_to_hf(folder: Path, repo_id: str, token: str):
    api = HfApi()
    print(f"Uploading {folder} -> {repo_id} ...")
    try:
        api.upload_folder(folder_path=str(folder), path_in_repo="", repo_id=repo_id, token=token)
    except Exception as e:
        print(f"Error uploading folder to {repo_id}: {e}")
        raise
    print("Upload finished.")


def main():
    parser = argparse.ArgumentParser(description="Push final checkpoint + tokenizer to Hugging Face Hub")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to final checkpoint folder")
    parser.add_argument("--repo_id", required=True, help="HF repo id (username/model_name)")
    parser.add_argument("--tokenizers_root", default=os.path.join("..", "tokenizers"), help="Parent folder containing tokenizer subfolders")
    parser.add_argument("--tokenizer_name", default=None, help="Optional tokenizer subfolder name or path override")
    parser.add_argument("--private", action="store_true", help="Create the HF repo as private")
    parser.add_argument("--no-private", dest="private", action="store_false", help="Create the HF repo as public (opposite of --private)")
    parser.set_defaults(private=True)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (env HF_TOKEN if not provided)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        print(f"Error: checkpoint_dir does not exist: {checkpoint_dir}")
        sys.exit(2)

    token = args.token
    # If token not provided via arg or env, try reading .apikey in repo root (cwd)
    if not token:
        apikey_path = Path(".apikey")
        if apikey_path.exists():
            try:
                token = apikey_path.read_text(encoding="utf-8").strip()
                if token:
                    print(f"Read HF token from {apikey_path}")
            except Exception as e:
                print(f"Warning: failed to read {apikey_path}: {e}")

    if not token:
        print("Error: No HF token provided. Set HF_TOKEN env var, create a .apikey file, or pass --token.")
        sys.exit(2)

    # validate token and create repo before uploading
    api = HfApi()
    try:
        who = api.whoami(token=token)
        print(f"Authenticated as: {who.get('name') or who.get('user', {}).get('name', 'unknown')}")
    except Exception as e:
        print(f"Error: HF token is invalid or network issue: {e}")
        sys.exit(2)

    try:
        print(f"Creating repo {args.repo_id} (private={args.private}) if needed...")
        api.create_repo(repo_id=args.repo_id, token=token, private=bool(args.private), exist_ok=True)
    except Exception as e:
        print(f"Warning: create_repo failed or repo may already exist: {e}")

    # 1/2: find trainer_state.json and ensure it's present in checkpoint root
    ts = find_trainer_state(checkpoint_dir)
    if ts is None:
        print("Warning: trainer_state.json not found under checkpoint tree.")
    else:
        if ts.parent.resolve() != checkpoint_dir.resolve():
            # copy into root
            dest = checkpoint_dir / "trainer_state.json"
            print(f"Copying trainer_state.json from {ts} -> {dest}")
            shutil.copy2(ts, dest)
        else:
            print(f"trainer_state.json already present at {ts}")

    # 3: find tokenizer
    tokenizers_root = Path(args.tokenizers_root).expanduser().resolve()
    tokenizer_dir = choose_tokenizer_dir(tokenizers_root, checkpoint_dir, override=args.tokenizer_name)
    if tokenizer_dir is None:
        print(f"Warning: no tokenizer dir found under {tokenizers_root}; continuing without tokenizer files.")
    else:
        print(f"Using tokenizer dir: {tokenizer_dir}")

    # 4: assemble temp folder
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dest_model = tmp / "model"
        dest_model.mkdir(parents=True, exist_ok=True)

        # copy checkpoint contents into dest_model, excluding optimizer/scheduler/rng artifacts
        print(f"Copying checkpoint files from {checkpoint_dir} to temporary folder (excluding optimizer/scheduler/rng)...")
        exclude_tokens = ["optimizer", "optim", "scheduler", "rng_state", "rng"]
        copy_tree(checkpoint_dir, dest_model, exclude_tokens=exclude_tokens)

        if tokenizer_dir is not None:
            print(f"Copying tokenizer files from {tokenizer_dir} to temporary folder...")
            copy_tree(tokenizer_dir, dest_model)

        # Finally upload
        upload_folder_to_hf(dest_model, args.repo_id, token)


if __name__ == "__main__":
    main()
