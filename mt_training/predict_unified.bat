@echo off
REM Activate virtualenv and run unified normal prediction script
python predict_unified.py ^
  --tokenizer_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --model_path sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds/sagawa/ReactionT5v2-forward-USPTO_MIT ^
  --prompt_mode "reactiont5" ^