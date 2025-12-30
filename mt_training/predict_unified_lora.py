import argparse
import logging
import os
from typing import Iterable, List, Tuple

import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import PeftModel

from dataset_helpers import load_chanlam_dataset, load_custom_mit_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_chunks_spec(spec: str, total: int | None = None) -> List[int]:
    if spec.lower() == "all":
        if total is None:
            raise ValueError("total must be provided when spec is 'all'")
        return list(range(total))

    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return sorted(selected)


def build_preprocess(tokenizer, max_length: int, input_col: str, target_col: str, prompt_mode: str = "none", prompt_template: str | None = None):
    def preprocess(batch):
        inputs = batch[input_col]
        targets = batch[target_col]

        # Apply prompt transformations
        if prompt_template is not None:
            inputs = [prompt_template.format(input=inp) for inp in inputs]
        elif prompt_mode == "mtct5":
            inputs = [f"Predict the product of the following reaction: {inp}" for inp in inputs]
        elif prompt_mode == "reactiont5":
            transformed = []
            for inp in inputs:
                if isinstance(inp, str) and ">" in inp:
                    parts = inp.split(">", 1)
                    transformed.append(f"REACTANT:{parts[0]}REAGENT:{parts[1]}")
                else:
                    transformed.append(inp)
            inputs = transformed

        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
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


def get_dataset_chunks(dataset, chunk_size: int) -> List[Tuple[int, any]]:
    chunks: List[Tuple[int, any]] = []
    total = len(dataset)
    n_chunks = (total + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, total)
        subset = dataset.select(range(s, e))
        chunks.append((i, subset))
    return chunks


def predict_chunk(
    chunk_id: int,
    subset,
    trainer: Seq2SeqTrainer,
    tokenizer,
    output_dir: str,
    csv_prefix: str,
):
    logger.info(f"Running prediction on chunk {chunk_id} ({len(subset)} rows)")

    out = trainer.predict(subset)
    preds = out.predictions
    labels = out.label_ids

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels_original = tokenizer.batch_decode(labels, skip_special_tokens=True)

    num_return_sequences = trainer.model.generation_config.num_return_sequences

    decoded_preds_reshaped = [
        decoded_preds[j : j + num_return_sequences]
        for j in range(0, len(decoded_preds), num_return_sequences)
    ]

    data: dict[str, Iterable[str]] = {
        "chunk": [chunk_id] * len(decoded_labels_original),
        "label": decoded_labels_original,
    }

    for i in range(num_return_sequences):
        data[f"prediction_{i+1}"] = [
            preds_for_label[i] for preds_for_label in decoded_preds_reshaped
        ]

    df = pd.DataFrame(data)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{csv_prefix}chunk_{chunk_id}.csv" if csv_prefix else f"chunk_{chunk_id}.csv"
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")


def run_selected_chunks(
    test_dataset,
    trainer: Seq2SeqTrainer,
    tokenizer,
    chunk_size: int,
    selected_chunk_ids: Iterable[int],
    output_dir: str,
    csv_prefix: str,
):
    chunks = get_dataset_chunks(test_dataset, chunk_size)
    selected_set = set(selected_chunk_ids)

    for chunk_id, subset in chunks:
        if chunk_id in selected_set:
            predict_chunk(chunk_id, subset, trainer, tokenizer, output_dir, csv_prefix)


def main():
    parser = argparse.ArgumentParser(description="Chunked predictions for base+LoRA adapter models (merge for inference).")
    parser.add_argument("--tokenizer_name", required=True, help="Tokenizer name or path")
    parser.add_argument("--base_model_name", required=True, help="Base model name (used to load base weights)")
    parser.add_argument("--lora_adapter_path", required=True, help="Path to LoRA adapter weights (local dir)")
    parser.add_argument(
        "--dataset_type",
        required=True,
        help=(
            "Dataset identifier: chanlam_<separated|mixed|combined> or "
            "mit_<separated|mixed|combined>_<normal|augmented>"
        ),
    )
    parser.add_argument("--max_length", type=int, default=350)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument(
        "--chunks",
        default="all",
        help="Chunk selection, e.g. 'all' or '0-3,5,7'",
    )
    parser.add_argument(
        "--output_dir",
        default="pred_unified_lora",
        help="Directory to save CSV outputs",
    )
    parser.add_argument(
        "--csv_prefix",
        default="",
        help="Prefix to prepend to saved CSV filenames",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument(
        "--prompt_mode",
        choices=["none", "mtct5", "reactiont5"],
        default="none",
        help="Built-in prompt modes: mtct5 or reactiont5; 'none' applies no special prompt",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Optional Python-format template to transform input text. Use '{input}' placeholder.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    logger.info("Loading tokenizer and base model...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # Determine dataset and preprocessing
    if args.dataset_type.startswith("chanlam_"):
        _, fmt = args.dataset_type.split("_", 1)
        input_col = "reactants"
        target_col = "product1"
        preprocess = build_preprocess(
            tokenizer,
            args.max_length,
            input_col,
            target_col,
            prompt_mode=args.prompt_mode,
            prompt_template=args.prompt_template,
        )
        eval_dataset = load_chanlam_dataset(ds_type="default", format=fmt, split="test")
    elif args.dataset_type.startswith("mit_"):
        parts = args.dataset_type.split("_")
        if len(parts) != 3:
            raise ValueError("MIT dataset identifier must be mit_<format>_<type>")
        _, fmt, ds_type = parts
        input_col = "input"
        target_col = "target"
        preprocess = build_preprocess(
            tokenizer,
            args.max_length,
            input_col,
            target_col,
            prompt_mode=args.prompt_mode,
            prompt_template=args.prompt_template,
        )
        eval_dataset = load_custom_mit_dataset(type=ds_type, format=fmt, split="test")
    else:
        raise ValueError("Unsupported dataset_type; see --help for allowed identifiers")

    eval_dataset = eval_dataset.map(preprocess, batched=True)

    logger.info(f"Using {args.dataset_type} dataset for evaluation (size={len(eval_dataset)})")

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, "tmp_trainer"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_eval=True,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        predict_with_generate=True,
        bf16=args.bf16,
    )

    logger.info(f"Loading base model from {args.base_model_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name)

    logger.info(f"Loading LoRA adapter from {args.lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)

    logger.info("Merging LoRA weights with base model...")
    model = model.merge_and_unload()

    if args.bf16:
        try:
            model.bfloat16()
        except Exception:
            logger.warning("Model.bfloat16() failed; continuing in full precision.")

    gen_cfg = model.generation_config
    gen_cfg.num_beams = args.num_beams
    gen_cfg.max_new_tokens = args.max_new_tokens
    gen_cfg.num_return_sequences = args.num_return_sequences
    gen_cfg.use_cache = True

    trainer = Seq2SeqTrainer(model=model, args=training_args)

    total = (len(eval_dataset) + args.chunk_size - 1) // args.chunk_size
    selected_chunk_ids = parse_chunks_spec(args.chunks, total=total)

    os.makedirs(args.output_dir, exist_ok=True)

    run_selected_chunks(
        test_dataset=eval_dataset,
        trainer=trainer,
        tokenizer=tokenizer,
        chunk_size=args.chunk_size,
        selected_chunk_ids=selected_chunk_ids,
        output_dir=args.output_dir,
        csv_prefix=args.csv_prefix,
    )


if __name__ == "__main__":
    main()
