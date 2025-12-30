import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from dataset_helpers import load_custom_mit_dataset, load_chanlam_dataset
import torch
from peft import LoraConfig, get_peft_model, TaskType

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from training_utils import train_t5_model

print("Imports complete")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1), labels


def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 models with LoRA on Chanlam or MIT datasets")

    parser.add_argument("--base_model_name", type=str, required=True,
                        help="Base model name from HuggingFace")

    parser.add_argument("--output_model_dir", type=str, required=True,
                        help="Folder to store LoRA adapters & logs")

    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help=(
            "Dataset identifier. Use the formats: "
            "`chanlam_separated`, `chanlam_mixed`, `chanlam_combined`, or "
            "`mit_<separated|mixed>_<normal|augmented>` (e.g. mit_separated_normal)."
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

    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Max training steps (default 10000)")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default 1e-4 for LoRA)")

    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "constant"],
                        help="LR scheduler type")

    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")

    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")

    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default 16)")

    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default 32)")

    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout (default 0.1)")

    parser.add_argument("--target_modules", type=str, default="q,v",
                        help="Comma-separated list of target modules for LoRA (e.g. q,v)")

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


def train_lora_model(
    base_model: str,
    output_model: str,
    max_steps: int,
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
    seed: int = 42,
):
    # instantiate base model/tokenizer
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

    # define post-train callback to save adapter and merged model while trainer is available
    def _post_train(trainer):
        adapter_dir = f"./models/{output_model}_adapter_{max_steps}"
        merged_dir = f"./models/{output_model}_merged_{max_steps}"

        try:
            if hasattr(trainer.model, "save_pretrained"):
                try:
                    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
                except TypeError:
                    trainer.model.save_pretrained(adapter_dir)
        except Exception as e:
            print(f"Warning: saving adapter failed: {e}")

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

    # Call shared training utility; pass post-train callback so adapter can be saved/merged
    train_t5_model(
        model=model,
        tokenizer=tokenizer,
        output_model=output_model,
        max_steps=max_steps,
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
        seed=seed,
        post_train_callback=_post_train,
    )


if __name__ == "__main__":
    args = parse_args()
    print("Launching LoRA training with:")
    print(args)
    # Set global seed
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

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

    # Determine columns and dataset loading strategy similar to final_trainscript
    # New identifier formats:
    # - chanlam_<separated|mixed|combined>
    # - mit_<separated|mixed|combined>_<normal|augmented>
    if args.dataset_type.startswith("chanlam_"):
        _, fmt = args.dataset_type.split("_", 1)
        input_col = "reactants"
        target_col = "product1"
        preprocess = build_preprocess(tokenizer, input_col, target_col)

        ds = load_chanlam_dataset(ds_type="default", format=fmt)
        # ds is a DatasetDict; map each split
        ds = ds.map(preprocess, batched=True, remove_columns=[input_col, target_col])
        ds_train = ds["train"]
        ds_val = ds["val"]

    elif args.dataset_type.startswith("mit_"):
        parts = args.dataset_type.split("_")
        if len(parts) != 3:
            raise ValueError("MIT dataset identifier must be of the form mit_<format>_<type>")
        _, fmt, ds_type = parts
        input_col = "input"
        target_col = "target"
        preprocess = build_preprocess(tokenizer, input_col, target_col)

        ds = load_custom_mit_dataset(type=ds_type, format=fmt)
        ds = ds.map(preprocess, batched=True, remove_columns=[input_col, target_col])
        ds_train = ds["train"]
        ds_val = ds["val"]

        # For MIT datasets, shrink validation for quicker runs
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

    train_lora_model(
        base_model=args.base_model_name,
        output_model=args.output_model_dir,
        max_steps=args.max_steps,
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
        seed=args.seed,
    )
    torch.cuda.empty_cache()
    print("Training complete!")
