@echo off
REM Activate virtualenv and run unified normal prediction script
python predict_unified.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-2500 ^
  --model_path Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-2500 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-2500

@REM REM reevaluate with proper format of data on MIT

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-standard ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-small-standard ^
@REM   --dataset_type mit_separated_normal ^
@REM   --chunks all ^
@REM   --output_dir preds/GT4SD/multitask-text-and-chemistry-t5-small-standard

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-augm ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-small-augm ^
@REM   --dataset_type mit_separated_normal ^
@REM   --chunks all ^
@REM   --output_dir preds/GT4SD/multitask-text-and-chemistry-t5-small-augm

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-base-standard ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-base-standard ^
@REM   --dataset_type mit_separated_normal ^
@REM   --chunks all ^
@REM   --output_dir preds/GT4SD/multitask-text-and-chemistry-t5-base-standard

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-base-augm ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-base-augm ^
@REM   --dataset_type mit_separated_normal ^
@REM   --chunks all ^
@REM   --output_dir preds/GT4SD/multitask-text-and-chemistry-t5-base-augm



@REM REM reevaluate with proper format of data on chan lam

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-standard ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-small-standard ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds_chanlam/GT4SD/multitask-text-and-chemistry-t5-small-standard

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-small-augm ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-small-augm ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds_chanlam/GT4SD/multitask-text-and-chemistry-t5-small-augm

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-base-standard ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-base-standard ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds_chanlam/GT4SD/multitask-text-and-chemistry-t5-base-standard

@REM python predict_unified.py ^
@REM   --tokenizer_name GT4SD/multitask-text-and-chemistry-t5-base-augm ^
@REM   --model_path GT4SD/multitask-text-and-chemistry-t5-base-augm ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds_chanlam/GT4SD/multitask-text-and-chemistry-t5-base-augm
