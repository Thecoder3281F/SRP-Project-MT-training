@echo off
REM Activate virtualenv and run full finetune trainer

python train_full_unified.py ^
  --base_model_name Thecoder3281f/ManganumT5v1_1-small-separated-augmented ^
  --output_model_dir Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-full-finetuned-10000 ^
  --dataset_type chanlam_separated ^
  --max_steps 10000 ^
  --lr 7e-4 ^
  --use_early_stopping ^
  --early_stopping_patience 3 ^
  --early_stopping_threshold 0.0

python train_full_unified.py ^
  --base_model_name Thecoder3281f/ManganumT5v1_1-base-separated-standard ^
  --output_model_dir Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-full-finetuned-10000 ^
  --dataset_type chanlam_separated ^
  --max_steps 10000 ^
  --lr 7e-4 ^
  --use_early_stopping ^
  --early_stopping_patience 3 ^
  --early_stopping_threshold 0.0

python train_full_unified.py ^
  --base_model_name Thecoder3281f/ManganumT5v1_1-base-separated-augmented ^
  --output_model_dir Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-full-finetuned-10000 ^
  --dataset_type chanlam_separated ^
  --max_steps 10000 ^
  --lr 7e-4 ^
  --use_early_stopping ^
  --early_stopping_patience 3 ^
  --early_stopping_threshold 0.0
