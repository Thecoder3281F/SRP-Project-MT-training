@echo off
REM Activate virtualenv and run unified LoRA prediction script
@REM call "C:\Users\user\Desktop\srp\Scripts\Activate.ps1"


@REM python predict_unified_lora.py ^
@REM   --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
@REM   --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
@REM   --lora_adapter_path Thecoder3281f/lora-10 ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks 8 ^
@REM   --output_dir preds/Thecoder3281f/ReactionT5-lora-10 ^
@REM   --prompt_mode "reactiont5" && ^

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-50 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-50 ^
  --prompt_mode "reactiont5"

echo .

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-100 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-100 ^
  --prompt_mode "reactiont5"

echo .

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-250 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-250 ^
  --prompt_mode "reactiont5"

echo .

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-500 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-500 ^
  --prompt_mode "reactiont5"

echo .

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-1000 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-1000 ^
  --prompt_mode "reactiont5"

echo .

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-2000 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-2000 ^
  --prompt_mode "reactiont5"

echo .

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-4000 ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-4000 ^
  --prompt_mode "reactiont5"

echo .

python predict_unified_lora.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --lora_adapter_path Thecoder3281f/lora-full ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ReactionT5-lora-full ^
  --prompt_mode "reactiont5"