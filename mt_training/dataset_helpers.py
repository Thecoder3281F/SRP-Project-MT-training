from datasets import load_dataset, concatenate_datasets, DatasetDict


def load_custom_mit_dataset(
    type: str = "normal",
    format: str = "separated",
    split: str | None = None,
):
    """Load MIT datasets with the same interface as `load_chanlam_dataset`.

    - `type`: dataset config (e.g. 'normal' or 'augmented')
    - `format`: 'separated', 'mixed', or 'combined'
    - `split`: optional split name ('train','val','test'). If provided, returns a Dataset for that split.

    Returns a DatasetDict when split is None (for separated/mixed), or a concatenated Dataset/ DatasetDict for combined.
    """
    name_sep = f"Thecoder3281f/MIT_separated_final"
    name_mix = f"Thecoder3281f/MIT_mixed_final"

    if format == "separated":
        if split is not None:
            return load_dataset(name_sep, type, split=split)
        return load_dataset(name_sep, type)

    if format == "mixed":
        if split is not None:
            return load_dataset(name_mix, type, split=split)
        return load_dataset(name_mix, type)

    # combined -> concatenate separated + mixed
    if split is not None:
        ds_sep = load_dataset(name_sep, type, split=split)
        ds_mix = load_dataset(name_mix, type, split=split)
        return concatenate_datasets([ds_sep, ds_mix])

    # return a DatasetDict combining splits
    ds_sep = load_dataset(name_sep, type)
    ds_mix = load_dataset(name_mix, type)

    combined = DatasetDict()
    for split_name in ds_sep.keys():
        combined[split_name] = concatenate_datasets([ds_sep[split_name], ds_mix[split_name]])

    return combined


def load_chanlam_dataset(ds_type: str = "default", format: str = "separated", split: str | None = None):
    """Load the Chanlam major/minor product datasets.

    Parameters
    - ds_type: HF dataset config name (commonly 'default', or 'normal'/'augmented' if available)
    - format: one of 'separated', 'mixed', or 'combined'
    - split: optional split name (e.g. 'train','val','test'). If provided and format!='combined', returns that split.

    Returns either a DatasetDict (when split is None and format!='combined') or a Dataset/split or a concatenated Dataset.
    """
    name_sep = "Thecoder3281f/chanlam-majorminorproduct-separated-final"
    name_mix = "Thecoder3281f/chanlam-majorminorproduct-mixed-final"

    if format == "separated":
        if split is not None:
            return load_dataset(name_sep, ds_type, split=split)
        return load_dataset(name_sep, ds_type)

    if format == "mixed":
        if split is not None:
            return load_dataset(name_mix, ds_type, split=split)
        return load_dataset(name_mix, ds_type)

    # combined -> concatenate separated + mixed
    if split is not None:
        ds_sep = load_dataset(name_sep, ds_type, split=split)
        ds_mix = load_dataset(name_mix, ds_type, split=split)
        return concatenate_datasets([ds_sep, ds_mix])

    # return a DatasetDict with concatenated splits
    ds_sep = load_dataset(name_sep, ds_type)
    ds_mix = load_dataset(name_mix, ds_type)

    combined = DatasetDict()
    for split_name in ds_sep.keys():
        combined[split_name] = concatenate_datasets([ds_sep[split_name], ds_mix[split_name]])

    return combined


def load_chanlam_hf(format: str = "separated", split: str | None = None, dataset_name: str = "Thecoder3281f/chanlam-dataset"):
    """Load the new unified Chanlam HF dataset (Thecoder3281f/chanlam-dataset).

    Returns the requested split (if `split` provided) or the full DatasetDict.
    The dataset contains `input_reactants`, `input_reagents`, and product columns
    (the canonical product column is `product_1_canonical_smiles`).
    """
    ds = load_dataset(dataset_name)
    # standardize split naming
    if "validation" in ds and "val" not in ds:
        ds["val"] = ds["validation"]

    if split is not None:
        return ds.get(split)
    return ds


def combine_reactants_reagents(ds_split, sep: str = ".", target_col: str = "product_1_canonical_smiles"):
    """Return a mapped dataset split that contains `_combined_input` and `_target_col`.

    `_combined_input` is `input_reactants` + `sep` + `input_reagents` (or the non-empty one).
    """
    def _comb(batch):
        reactants = batch.get("input_reactants", [""] * len(batch[next(iter(batch))]))
        reagents = batch.get("input_reagents", [""] * len(batch[next(iter(batch))]))
        combined = []
        for r, q in zip(reactants, reagents):
            r = r or ""
            q = q or ""
            if r and q:
                combined.append(f"{r}{sep}{q}")
            else:
                combined.append(r or q)
        batch["_combined_input"] = combined
        batch["_target_col"] = batch.get(target_col, [""] * len(combined))
        return batch

    return ds_split.map(_comb, batched=True)


def pick_split(ds_dict_or_dataset, preferred: str = "test"):
    """Return a Dataset for the requested split name, tolerant of common split keys.

    If `ds_dict_or_dataset` is a Dataset (not a dict), it is returned directly.
    If it's a DatasetDict, the function will try the following keys in order:
    [preferred, 'test', 'val', 'validation', 'train'] and return the first found split.
    Raises KeyError if no matching split is present.
    """
    # if already a Dataset (not a dict-like), return as-is
    try:
        keys = list(ds_dict_or_dataset.keys())
    except Exception:
        return ds_dict_or_dataset

    candidates = [preferred, "test", "val", "validation", "train"]
    for c in candidates:
        if c in ds_dict_or_dataset:
            return ds_dict_or_dataset[c]

    raise KeyError(f"None of the expected split names found in dataset. Tried: {candidates}")