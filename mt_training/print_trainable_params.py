import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    # PEFT may not be installed; fall back to flagging LoRA unavailable
    LoraConfig = None
    get_peft_model = None
    TaskType = None


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return total, trainable, pct


def parse_args():
    p = argparse.ArgumentParser(description="Print trainable parameter counts for full or LoRA models (no training)")
    p.add_argument("--base_model_name", required=True, help="HuggingFace model name or path")
    p.add_argument("--mode", choices=["full", "lora", "both"], default="both",
                   help="Which counts to print: full model, LoRA-adapted, or both")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", type=str, default="q,v",
                   help="Comma-separated target modules for LoRA (e.g. q,v)")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                   help="Device to load model on (default: cpu). Use `cuda` if available.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    if args.mode in ("full", "both"):
        print(f"Loading base model for full-finetune: {args.base_model_name} (device={device})")
        model_full = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name)
        model_full.to(device)
        total, trainable, pct = count_params(model_full)
        print("Full model:")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,} ({pct:.4f}%)")

    if args.mode in ("lora", "both"):
        if get_peft_model is None:
            print("PEFT (LoRA) is not available in this environment. Install `peft` to get LoRA counts.")
            sys.exit(1)

        print(f"Loading base model for LoRA: {args.base_model_name} (device={device})")
        model_lora = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name)
        model_lora.to(device)

        if isinstance(args.target_modules, str):
            target_modules_list = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        else:
            target_modules_list = list(args.target_modules)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules_list,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        model_lora = get_peft_model(model_lora, lora_config)

        # Many PEFT wrappers provide a helper to print trainable params; compute explicitly too
        try:
            model_lora.print_trainable_parameters()
        except Exception:
            pass

        total, trainable, pct = count_params(model_lora)
        print("LoRA-adapted model:")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,} ({pct:.6f}%)")


if __name__ == "__main__":
    main()
