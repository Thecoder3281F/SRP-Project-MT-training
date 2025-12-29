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