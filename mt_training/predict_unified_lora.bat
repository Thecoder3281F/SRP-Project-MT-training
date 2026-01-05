@echo off
REM Activate virtualenv and run unified LoRA prediction script
@REM call "C:\Users\user\Desktop\srp\Scripts\Activate.ps1"


python predict_unified_lora.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-augmented ^
  --base_model_name Thecoder3281f/ManganumT5v1_1-small-separated-augmented ^
  --lora_adapter_path Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-adapter-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-adapter-10000

python predict_unified_lora.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-augmented ^
  --base_model_name Thecoder3281f/ManganumT5v1_1-base-separated-augmented ^
  --lora_adapter_path Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-adapter-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-adapter-10000

pause