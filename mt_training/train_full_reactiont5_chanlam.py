import argparse
from typing import cast

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from dataset_helpers import pick_split
from training_utils import train_t5_model

print("Imports complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train full T5/FLAN-T5 models on Chanlam with ReactionT5 prompt formatting"
    )

    parser.add_argument("--base_model_name", type=str, required=True, help="Base model name from HuggingFace")
    parser.add_argument("--output_model_dir", type=str, required=True, help="Folder to store checkpoints & logs")

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

    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant"],
        help="LR scheduler type",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    parser.add_argument("--resume_checkpoint", action="store_true", help="Whether to resume from last checkpoint")
    parser.add_argument("--use_early_stopping", action="store_true", help="Whether to use early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--early_stopping_threshold", type=float, default=0, help="Early stopping threshold")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default 42)")

    return parser.parse_args()


def build_chanlam_reactiont5_preprocess(tokenizer, target_col: str, max_length: int = 350):
    def preprocess(batch):
        reactants = batch.get("input_reactants", [""] * len(batch[next(iter(batch))]))
        reagents = batch.get("input_reagents", [""] * len(batch[next(iter(batch))]))

        inputs = []
        for r, q in zip(reactants, reagents):
            r = r or ""
            q = q or ""
            inputs.append(f"REACTANT:{r}REAGENT:{q}")

        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        targets = batch.get(target_col, [""] * len(inputs))
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                padding="max_length",
                truncation=True,
                max_length=max_length,
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

    print("Launching full ReactionT5-style finetune training with:")
    print(args)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    ds = cast(DatasetDict, load_dataset("Thecoder3281f/chanlam-dataset"))

    target_col = "product_1_canonical_smiles"
    preprocess = build_chanlam_reactiont5_preprocess(tokenizer, target_col=target_col)

    ds_train = cast(Dataset, pick_split(ds, preferred="train")).map(preprocess, batched=True)
    ds_val = cast(Dataset, pick_split(ds, preferred="val")).map(preprocess, batched=True)

    print(f"Training dataset size: {len(ds_train)}")
    print(f"Validation dataset size: {len(ds_val)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name)

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
    print("Full ReactionT5-style finetune training complete!")
