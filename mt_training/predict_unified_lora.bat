@echo off
REM Activate virtualenv and run unified LoRA prediction script
@REM call "C:\Users\user\Desktop\srp\Scripts\Activate.ps1"

@REM python predict_unified_lora.py ^
@REM   --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-standard ^
@REM   --base_model_name Thecoder3281f/ManganumT5v1_1-small-separated-standard ^
@REM   --lora_adapter_path Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-adapter-10000 ^
@REM   --dataset_type mit_separated_normal ^
@REM   --chunks all ^
@REM   --output_dir preds/Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-adapter-10000

@REM python predict_unified_lora.py ^
@REM   --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-augmented ^
@REM   --base_model_name Thecoder3281f/ManganumT5v1_1-small-separated-augmented ^
@REM   --lora_adapter_path Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-adapter-10000 ^
@REM   --dataset_type mit_separated_normal ^
@REM   --chunks all ^
@REM   --output_dir preds/Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-adapter-10000

python predict_unified_lora.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-standard ^
  --base_model_name Thecoder3281f/ManganumT5v1_1-base-separated-standard ^
  --lora_adapter_path Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-adapter-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks 2-3 ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-adapter-10000

python predict_unified_lora.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-augmented ^
  --base_model_name Thecoder3281f/ManganumT5v1_1-base-separated-augmented ^
  --lora_adapter_path Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-adapter-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-adapter-10000

pause