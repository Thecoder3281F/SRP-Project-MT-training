"""TODO: Add module docstring describing this training script."""

import argparse
import sys
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from utils.training_utils import train_t5_model

print("Imports complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train T5 models with LoRA on Chanlam with ReactionT5 prompt formatting"
    )

    parser.add_argument("--base_model_name", type=str, required=True, help="Base model name from HuggingFace")
    parser.add_argument("--output_model_dir", type=str, required=True, help="Folder to store LoRA adapters & logs")
    parser.add_argument(
        "--train_rows",
        type=int,
        default=None,
        choices=[10, 50, 100, 500, 1000, 2000, 4000],
        help="Row-count subset to train on from the HF parquet files (e.g. 10 -> train_rows_10-00000-of-00001.parquet). Required unless --use_full_train_split is set.",
    )
    parser.add_argument(
        "--use_full_train_split",
        action="store_true",
        help="Train on the full 'train' split from Thecoder3281f/chanlam-dataset instead of a train_rows subset parquet file",
    )

    train_length_group = parser.add_mutually_exclusive_group()
    train_length_group.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps. Mutually exclusive with --num_train_epochs (default mode: steps=10000)",
    )
    train_length_group.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="Number of training epochs. Mutually exclusive with --max_steps",
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default 1e-4 for LoRA)")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant"],
        help="LR scheduler type",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--bf16", dest="bf16", action="store_true", help="Use bfloat16 training")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable bfloat16 training")
    parser.set_defaults(bf16=True)

    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default 0.1)")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q,v",
        help="Comma-separated list of target modules for LoRA (e.g. q,v)",
    )

    parser.add_argument("--resume_checkpoint", action="store_true", help="Whether to resume from last checkpoint")
    parser.add_argument("--use_early_stopping", action="store_true", help="Whether to use early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--early_stopping_threshold", type=float, default=0, help="Early stopping threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default 42)")

    parser.add_argument(
        "--save_merged_model",
        action="store_true",
        help="Whether to save the merged model after training (default: False)",
    )
    parser.add_argument(
        "--debug_data_only",
        action="store_true",
        help="Load train/validation datasets, print first 10 rows of each, and exit without training",
    )

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


def _truncate_text(value: Any, max_len: int = 120) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _print_dataset_preview(name: str, preview_df: Any, n_rows: int = 10) -> None:
    preferred_cols = [
        "_group_key",
        "input_reactants",
        "input_reagents",
        "product_1_canonical_smiles",
        "product_2_canonical_smiles",
        "input_base",
        "input_catalyst",
        "input_solvent",
    ]
    cols = [c for c in preferred_cols if c in getattr(preview_df, "columns", [])]
    if not cols:
        cols = list(getattr(preview_df, "columns", []))

    print(f"\n=== {name} (first {n_rows} rows, compact preview) ===")
    print(f"Columns shown: {', '.join(cols)}")

    for i, (_, row) in enumerate(preview_df[cols].head(n_rows).iterrows(), start=1):
        print(f"\n[{i}]")
        for col in cols:
            print(f"  {col}: {_truncate_text(row[col])}")


def train_lora_model(
    base_model: str,
    output_model: str,
    max_steps: int | None,
    num_train_epochs: float | None,
    train_ds,
    val_ds,
    lr: float,
    scheduler: str,
    use_early_stopping: bool,
    early_stopping_patience: int,
    resume_from_checkpoint: bool,
    warmup_ratio: float,
    weight_decay: float,
    early_stopping_threshold: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules=None,
    save_merged_model: bool = False,
    bf16: bool = True,
    seed: int = 42,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    set_seed(seed)

    if isinstance(target_modules, str):
        target_modules_list = [t.strip() for t in target_modules.split(",") if t.strip()]
    else:
        target_modules_list = list(target_modules) if target_modules is not None else ["q", "v"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules_list,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if bf16:
        try:
            model.bfloat16()
        except Exception as e:
            print(f"Warning: model.bfloat16() failed; continuing in full precision: {e}")

    def _post_train(trainer):
        run_suffix = f"{max_steps}steps" if max_steps is not None else f"{num_train_epochs:g}epochs"
        adapter_dir = f"./models/{output_model}_adapter_{run_suffix}"
        merged_dir = f"./models/{output_model}_merged_{run_suffix}"

        try:
            if hasattr(trainer.model, "save_pretrained"):
                try:
                    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
                except TypeError:
                    trainer.model.save_pretrained(adapter_dir)
        except Exception as e:
            print(f"Warning: saving adapter failed: {e}")

        if save_merged_model:
            try:
                if hasattr(trainer.model, "merge_and_unload"):
                    merged = trainer.model.merge_and_unload()
                    if merged is not None:
                        trainer.model = merged
            except Exception as e:
                print(f"Warning: merge_and_unload failed: {e}")

            try:
                trainer.save_model(merged_dir)
            except Exception as e:
                print(f"Warning: saving merged model failed: {e}")

    train_t5_model(
        model=model,
        tokenizer=tokenizer,
        output_model=output_model,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        train_ds=train_ds,
        val_ds=val_ds,
        lr=lr,
        scheduler=scheduler,
        use_early_stopping=use_early_stopping,
        early_stopping_patience=early_stopping_patience,
        resume_from_checkpoint=resume_from_checkpoint,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        early_stopping_threshold=early_stopping_threshold,
        bf16=bf16,
        seed=seed,
        post_train_callback=_post_train,
    )


if __name__ == "__main__":
    args = parse_args()

    if args.max_steps is None and args.num_train_epochs is None:
        args.max_steps = 10000

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0")
    if args.num_train_epochs is not None and args.num_train_epochs <= 0:
        raise ValueError("--num_train_epochs must be > 0")
    if not args.use_full_train_split and (args.train_rows is None or args.train_rows <= 0):
        raise ValueError("--train_rows must be provided and > 0 unless --use_full_train_split is set")

    print("Launching LoRA ReactionT5-style finetune training with:")
    print(args)

    set_seed(args.seed)

    if args.use_full_train_split:
        print("Loading full training split from: Thecoder3281f/chanlam-dataset")
        ds_train = cast(Dataset, load_dataset("Thecoder3281f/chanlam-dataset", split="train"))
    else:
        train_data_file = (
            f"hf://datasets/Thecoder3281f/chanlam-dataset-splits/data/"
            f"train_rows_{args.train_rows}-00000-of-00001.parquet"
        )
        print(f"Loading training subset file: {train_data_file}")
        ds_train = cast(Dataset, load_dataset("parquet", data_files=train_data_file, split="train"))
    print("Loading validation split from: Thecoder3281f/chanlam-dataset")
    ds_val = cast(Dataset, load_dataset("Thecoder3281f/chanlam-dataset", split="validation"))

    if args.debug_data_only:
        train_preview_source = cast(Any, ds_train.to_pandas(batched=True, batch_size=10))
        val_preview_source = cast(Any, ds_val.to_pandas(batched=True, batch_size=10))
        train_preview = next(train_preview_source) if hasattr(train_preview_source, "__next__") else train_preview_source.head(10)
        val_preview = next(val_preview_source) if hasattr(val_preview_source, "__next__") else val_preview_source.head(10)
        _print_dataset_preview("TRAIN", train_preview, n_rows=10)
        _print_dataset_preview("VALIDATION", val_preview, n_rows=10)
        print("\nDebug mode enabled: exiting before preprocessing/training.")
        sys.exit(0)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    target_col = "product_1_canonical_smiles"
    preprocess = build_chanlam_reactiont5_preprocess(tokenizer, target_col=target_col)

    ds_train = ds_train.map(preprocess, batched=True)
    ds_val = ds_val.map(preprocess, batched=True)

    def _safe_len(x):
        try:
            return len(x)
        except Exception:
            return "unknown"

    print(f"Training dataset size: {_safe_len(ds_train)}")
    print(f"Validation dataset size: {_safe_len(ds_val)}")

    train_lora_model(
        base_model=args.base_model_name,
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        save_merged_model=args.save_merged_model,
        bf16=args.bf16,
        seed=args.seed,
    )

    torch.cuda.empty_cache()
    print("LoRA ReactionT5-style finetune training complete!")
