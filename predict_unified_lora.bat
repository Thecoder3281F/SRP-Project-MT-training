@echo off
REM Activate virtualenv and run unified LoRA prediction script
@REM call "C:\Users\user\Desktop\srp\Scripts\Activate.ps1"


python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/Lora-250rows-123 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-Lora-250rows-123 ^
  --prompt_mode "reactiont5"
