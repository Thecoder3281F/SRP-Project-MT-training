@echo off
REM Activate virtualenv and run unified normal prediction script
python predict_unified.py ^
  --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-standard ^
  --model_path GT4SD/multitask-text-and-chemistry-t5-small-standard ^
  --dataset_type mit_separated_normal ^
  --chunks 1-3 ^
  --output_dir preds/GT4SD/multitask-text-and-chemistry-t5-small-standard ^
  --prompt_mode mtct5

python predict_unified.py ^
  --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-augm ^
  --model_path GT4SD/multitask-text-and-chemistry-t5-small-augm ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/GT4SD/multitask-text-and-chemistry-t5-small-augm ^
  --prompt_mode mtct5

python predict_unified.py ^
  --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-standard ^
  --model_path GT4SD/multitask-text-and-chemistry-t5-small-standard ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds_chanlam/GT4SD/multitask-text-and-chemistry-t5-small-standard ^
  --prompt_mode mtct5

python predict_unified.py ^
  --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-augm ^
  --model_path GT4SD/multitask-text-and-chemistry-t5-small-augm ^
  --dataset_type chanlam_separated ^
  --chunks all ^
  --output_dir preds_chanlam/GT4SD/multitask-text-and-chemistry-t5-small-augm ^
  --prompt_mode mtct5
