from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
    set_seed,
)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger


# Silence RDKit logs (warnings, info, parse errors)
RDLogger.DisableLog('rdApp.*')



def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1), labels


def canonicalize(smiles):
    """Join tokens, parse to molecule, return canonical SMILES or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def tanimoto(a, b):
    """Compute Tanimoto similarity between two SMILES."""
    try:
        ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
        if not ma or not mb:
            return 0
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2) # type: ignore
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2) # type: ignore
        return DataStructs.TanimotoSimilarity(fa, fb)
    except Exception:
        return 0
    

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    # Simple accuracy
    acc = (preds == labels).mean()

    return {"accuracy": acc}




def train_t5_model(
    model,
    tokenizer,
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
    *,
    per_device_eval_batch_size: int = 16,
    auto_find_batch_size: bool = True,
    bf16: bool = True,
    fp16: bool = False,
    save_total_limit: int = 3,
    predict_with_generate: bool = False,
    seed: int = 42,
    dataloader_num_workers: int = 4,
    dataloader_pin_memory: bool = True,
    max_grad_norm: float = 1.0,
    logging_steps: int | None = None,
    post_train_callback=None,
):
    """
    Train a provided Seq2Seq model with flexible settings.

    - `model` and `tokenizer` should be instantiated by caller (supports LoRA-wrapped models).
    - Other parameters control training arguments; sensible defaults are provided.
    """
    set_seed(seed)

    # ---- 1. Create callbacks list ----
    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    # compute sensible default logging_steps if not provided
    if logging_steps is None:
        logging_steps = max(1, max_steps // 40) if max_steps and max_steps > 0 else 50

    # ---- 2. TrainingArguments ----
    args = Seq2SeqTrainingArguments(
        output_dir=f"./models/{output_model}",
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=lr,
        auto_find_batch_size=auto_find_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        save_steps=max(1, max_steps // 20) if max_steps and max_steps > 0 else 500,
        eval_steps=max(1, max_steps // 20) if max_steps and max_steps > 0 else 500,
        logging_strategy="steps",
        logging_steps=logging_steps,
        report_to="tensorboard",
        weight_decay=weight_decay,
        logging_dir=f"./logs/{output_model}",
        run_name=output_model,
        greater_is_better=False,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        gradient_checkpointing=False,
        eval_accumulation_steps=128,
        fp16=fp16,
        fp16_full_eval=False,
        bf16=bf16,
        save_total_limit=save_total_limit,
        lr_scheduler_type=scheduler,
        seed=seed,
        predict_with_generate=predict_with_generate,
        max_grad_norm=max_grad_norm,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_num_workers=dataloader_num_workers,
    )

    print(f"Per-device train batch (auto): {args.per_device_train_batch_size}")

    # ---- 3. Trainer ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # type: ignore
        callbacks=callbacks,
    )

    # ---- 4. Train ----
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # optional callback to allow callers to save adapters/merge/unload while trainer is available
    if post_train_callback is not None:
        try:
            post_train_callback(trainer)
        except Exception as e:
            print(f"Warning: post_train_callback raised: {e}")

    # ---- 5. Save final model ----
    trainer.save_model(f"./models/{output_model}_final_{max_steps}")

    del trainer
