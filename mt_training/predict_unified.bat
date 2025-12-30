@echo off
REM Activate virtualenv and run unified normal prediction script
python predict_unified.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-2500 ^
  --model_path Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-2500 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-2500
