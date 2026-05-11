import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import load_dataset
from dataset_helpers import load_custom_mit_dataset, pick_split
import torch

from training_utils import train_t5_model

print("Imports complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Train full T5/FLAN-T5 models on Chanlam or MIT datasets")

    parser.add_argument("--base_model_name", type=str, required=True,
                        help="Base model name from HuggingFace")

    parser.add_argument("--output_model_dir", type=str, required=True,
                        help="Folder to store checkpoints & logs")

    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help=(
            "Dataset identifier. Use: `chanlam_separated`, `chanlam_mixed`, `chanlam_combined`, or "
            "`mit_<separated|mixed|combined>_<normal|augmented>` (e.g. mit_separated_normal)."
        ),
        choices=[
            "chanlam_separated",
            "chanlam_mixed",
            "chanlam_combined",
            "mit_separated_normal",
            "mit_separated_augmented",
            "mit_mixed_normal",
            "mit_mixed_augmented",
            "mit_combined_normal",
            "mit_combined_augmented",
        ],
    )

    train_length_group = parser.add_mutually_exclusive_group()
    train_length_group.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps. Mutually exclusive with --num_train_epochs (default mode: steps=20000)",
    )
    train_length_group.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="Number of training epochs. Mutually exclusive with --max_steps",
    )

    parser.add_argument("--lr", type=float, default=7e-4,
                        help="Learning rate")

    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "constant"],
                        help="LR scheduler type")

    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")

    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")

    parser.add_argument("--resume_checkpoint", action="store_true",
                        help="Whether to resume from last checkpoint")
    parser.add_argument("--use_early_stopping", action="store_true",
                        help="Whether to use early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--early_stopping_threshold", type=float, default=0,
                        help="Early stopping threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default 42)")

    return parser.parse_args()


def build_preprocess(tokenizer, input_col: str, target_col: str):
    def preprocess(batch):
        inputs = batch[input_col]
        targets = batch[target_col]

        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=350,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                padding="max_length",
                truncation=True,
                max_length=350,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


if __name__ == "__main__":
    args = parse_args()

    if args.max_steps is None and args.num_train_epochs is None:
        args.max_steps = 20000

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0")
    if args.num_train_epochs is not None and args.num_train_epochs <= 0:
        raise ValueError("--num_train_epochs must be > 0")

    print("Launching full finetune training with:")
    print(args)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # Parse dataset identifier formats
    if args.dataset_type.startswith("chanlam_"):
        _, fmt = args.dataset_type.split("_", 1)
        # separator: '>' for separated, '.' for mixed, default ' ' for combined
        sep = ">" if fmt == "separated" else "." if fmt == "mixed" else " "

        # Load single HF dataset that contains chanlam splits
        ds = load_dataset("Thecoder3281f/chanlam-dataset")
        # normalize split names if needed
        if "validation" in ds and "val" not in ds:
            ds["val"] = ds["validation"]

        # use canonical product column
        target_col = "product_1_canonical_smiles"

        # build preprocess that joins input_reactants and input_reagents
        def chanlam_preprocess(tokenizer, max_len=350):
            def _pre(batch):
                reactants = batch.get("input_reactants", [""] * len(batch[next(iter(batch))]))
                reagents = batch.get("input_reagents", [""] * len(batch[next(iter(batch))]))
                inputs = []
                for r, q in zip(reactants, reagents):
                    r = r or ""
                    q = q or ""
                    if r and q:
                        inp = f"{r}{sep}{q}"
                    else:
                        inp = r or q
                    inputs.append(inp)

                model_inputs = tokenizer(
                    inputs,
                    padding="max_length",
                    truncation=True,
                    max_length=350,
                )
                targets = batch.get(target_col, [""] * len(inputs))
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        targets,
                        padding="max_length",
                        truncation=True,
                        max_length=350,
                    )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            return _pre

        preprocess = chanlam_preprocess(tokenizer)

        # map preprocess across splits
        ds = {s: ds[s] for s in ds.keys()}
        for s in ds:
            ds[s] = ds[s].map(preprocess, batched=True)

        ds_train = pick_split(ds, preferred="train")
        ds_val = pick_split(ds, preferred="val")

    elif args.dataset_type.startswith("mit_"):
        parts = args.dataset_type.split("_")
        if len(parts) != 3:
            raise ValueError("MIT dataset identifier must be mit_<format>_<type>")
        _, fmt, ds_type = parts
        input_col = "input"
        target_col = "target"
        preprocess = build_preprocess(tokenizer, input_col, target_col)

        ds = load_custom_mit_dataset(type=ds_type, format=fmt)
        ds = ds.map(preprocess, batched=True, remove_columns=[input_col, target_col])
        ds_train = pick_split(ds, preferred="train")
        ds_val = pick_split(ds, preferred="val")

        try:
            val_n = max(1, len(ds_val) // 10)
            ds_val = ds_val.shuffle(seed=42).select(range(val_n))
            print(f"MIT dataset detected: using 10% of validation set -> val={len(ds_val)}")
        except Exception as e:
            print(f"Warning: failed to reduce validation set size: {e}")

    else:
        raise ValueError("Unsupported dataset_type; see --help for allowed identifiers")

    print(f"Training dataset size: {len(ds_train)}")
    print(f"Validation dataset size: {len(ds_val)}")

    # instantiate model/tokenizer and call shared training utility
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    train_t5_model(
        model=model,
        tokenizer=tokenizer,
        output_model=args.output_model_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        lr=args.lr,
        train_ds=ds_train,
        val_ds=ds_val,
        scheduler=args.scheduler,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        resume_from_checkpoint=args.resume_checkpoint,
        use_early_stopping=args.use_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        seed=args.seed,
    )

    torch.cuda.empty_cache()
    print("Full finetune training complete!")
