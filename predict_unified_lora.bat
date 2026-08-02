@echo off
REM Activate virtualenv and run unified LoRA prediction script
@REM call "C:\Users\user\Desktop\srp\Scripts\Activate.ps1"


@REM python predict_unified_lora.py ^
@REM   --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
@REM   --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
@REM   --lora_adapter_path Thecoder3281f/ReactionT5-lora-10rows-2k ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds/Thecoder3281f/ReactionT5-lora-10rows-2k ^
@REM   --prompt_mode "reactiont5"

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/ReactionT5-lora-10rows-2k ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-10rows-2k ^
  --prompt_mode "reactiont5" ^
  --no-bf16