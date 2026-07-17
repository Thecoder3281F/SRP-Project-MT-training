@echo off
REM Activate virtualenv and run unified LoRA prediction script
@REM call "C:\Users\user\Desktop\srp\Scripts\Activate.ps1"


python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/ReactionT5-lora-10rows-30epochs ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-10rows-30epochs ^
  --prompt_mode "reactiont5"

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/ReactionT5-lora-50rows-30epochs ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-50rows-30epochs ^
  --prompt_mode "reactiont5"

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/ReactionT5-lora-100rows-30epochs ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-100rows-30epochs ^
  --prompt_mode "reactiont5"

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/ReactionT5-lora-500rows-30epochs ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-500rows-30epochs ^
  --prompt_mode "reactiont5"

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/ReactionT5-lora-1000rows-30epochs ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-1000rows-30epochs ^
  --prompt_mode "reactiont5"

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/ReactionT5-lora-10rows-50epochs ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-10rows-50epochs ^
  --prompt_mode "reactiont5"