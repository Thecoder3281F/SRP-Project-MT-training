# Training & Prediction Scripts — Quick Reference

This file documents the CLI arguments and example usage for the training and prediction scripts in this repository.

**Training scripts**

- `train_full_unified.py` (path: readme/../train_full_unified.py)
  - Purpose: Full finetuning (no LoRA) using `training_utils.train_t5_model`.
  - Important args:
    - `--base_model_name` (str, required): HF model id (e.g., `google/flan-t5-base`).
    - `--output_model_dir` (str, required): Output directory for checkpoints/logs.
    - `--dataset_type` (str, required): One of `chanlam_separated`, `chanlam_mixed`, `chanlam_combined`, `mit_separated_normal`, `mit_separated_augmented`, `mit_mixed_normal`, `mit_mixed_augmented`, `mit_combined_normal`, `mit_combined_augmented`.
    - `--max_steps` (int, default=20000)
    - `--lr` (float, default=7e-4)
    - `--scheduler` (str, default=`constant`, choices: `linear`,`cosine`,`constant`)
    - `--weight_decay` (float, default=0.01)
    - `--warmup_ratio` (float, default=0.1)
    - `--resume_checkpoint` (flag)
    - `--use_early_stopping` (flag)
    - `--early_stopping_patience` (int, default=3)
    - `--early_stopping_threshold` (float, default=0)
    - `--seed` (int, default=42)

  - Example:

```bash
python train_full_unified.py \
  --base_model_name google/flan-t5-base \
  --output_model_dir my_full_finetune_run \
  --dataset_type chanlam_separated \
  --max_steps 100 \
  --lr 5e-4 \
  --scheduler linear
```

- `train_lora_unified.py` (path: readme/../train_lora_unified.py)
  - Purpose: Train LoRA adapters on top of a base model; saves adapter and merged model.
  - Important args:
    - `--base_model_name` (str, required)
    - `--output_model_dir` (str, required)
    - `--dataset_type` (str, same choices as full training)
    - `--max_steps` (int, default=10000)
    - `--lr` (float, default=1e-4)
    - `--scheduler` (str, default=`linear`)
    - `--weight_decay` (float, default=0.01)
    - `--warmup_ratio` (float, default=0.1)
    - `--lora_r` (int, default=16)
    - `--lora_alpha` (int, default=32)
    - `--lora_dropout` (float, default=0.1)
    - `--target_modules` (str, default=`q,v`) — comma-separated modules to apply LoRA to.
    - `--resume_checkpoint` (flag)
    - `--use_early_stopping` (flag)
    - `--early_stopping_patience` (int, default=3)
    - `--early_stopping_threshold` (float, default=0)
    - `--seed` (int, default=42)

  - Example:

```bash
python train_lora_unified.py \
  --base_model_name google/flan-t5-base \
  --output_model_dir my_lora_run \
  --dataset_type mit_separated_augmented \
  --max_steps 5000 \
  --lr 1e-4 \
  --lora_r 8 \
  --lora_alpha 32
```


**Prediction scripts**

- `predict_unified.py` (full finetune models)
  - Purpose: Chunked predictions for full finetuned models. Supports prompt modes and custom prompt templates.
  - Important args:
    - `--tokenizer_name` (str, required)
    - `--model_path` (str, required)
    - `--dataset_type` (str, same choices as training)
    - `--max_length` (int, default=350)
    - `--num_beams` (int, default=10)
    - `--num_return_sequences` (int, default=1)
    - `--max_new_tokens` (int, default=220)
    - `--per_device_eval_batch_size` (int, default=32)
    - `--chunk_size` (int, default=1000)
    - `--chunks` (str, default=`all`) — e.g. `0-3,5,7` or `all`
    - `--output_dir` (str, default=`pred_unified`)
    - `--csv_prefix` (str)
    - `--prompt_mode` (choices: `none`, `mtct5`, `reactiont5`, default=`none`)
    - `--prompt_template` (str, optional) — Python-format string using `{input}`
    - `--seed` (int, default=42)
    - `--bf16`/`--no-bf16` flags (default bf16)

  - Example:

```bash
python predict_unified.py \
  --tokenizer_name google/flan-t5-base \
  --model_path ./models/my_full_finetune_run_final_100 \
  --dataset_type chanlam_combined \
  --chunks 0-2,5 \
  --output_dir preds/full_run
```

- `predict_unified_lora.py` (base model + LoRA adapter)
  - Purpose: Load base model and LoRA adapter, merge weights for inference, and run chunked predictions.
  - Important args:
    - `--tokenizer_name` (str, required)
    - `--base_model_name` (str, required)
    - `--lora_adapter_path` (str, required) — local dir where adapter was saved
    - The rest are the same as `predict_unified.py` (`--dataset_type`, `--max_length`, `--num_beams`, `--chunk_size`, `--chunks`, `--output_dir`, `--csv_prefix`, `--prompt_mode`, `--prompt_template`, `--bf16`, `--seed`, etc.)

  - Example:

```bash
python predict_unified_lora.py \
  --tokenizer_name google/flan-t5-base \
  --base_model_name google/flan-t5-base \
  --lora_adapter_path ./models/my_lora_run_adapter_5000 \
  --dataset_type mit_mixed_normal \
  --chunks all \
  --output_dir preds/lora_run
```


**Notes & tips**

- Dataset identifiers:
  - Chanlam: `chanlam_separated`, `chanlam_mixed`, `chanlam_combined`.
  - MIT: `mit_<separated|mixed|combined>_<normal|augmented>` (e.g. `mit_separated_augmented`).

- For quick smoke tests set `--max_steps 1` (training) or `--chunks 0` / `--chunk_size 1` (prediction) to verify end-to-end behavior without large runs.

- The training entrypoints now share the central training helper `training_utils.train_t5_model(model, tokenizer, ...)`. LoRA training wraps the base model with PEFT and passes a `post_train_callback` to save the adapter and merged model.

- Saved artifacts:
  - Full finetune final model: `./models/<output_model>_final_<max_steps>`
  - LoRA adapter: `./models/<output_model>_adapter_<max_steps>`
  - LoRA merged model: `./models/<output_model>_merged_<max_steps>`


**Questions?**
If you want, I can:
- Run a smoke test now (`--max_steps 1`) for either LoRA or full finetune.
- Add example `*.bat` wrappers for these commands.
- Extend README with environment/setup instructions (requirements).

