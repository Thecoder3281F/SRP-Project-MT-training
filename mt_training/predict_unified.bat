@echo off
REM Activate virtualenv and run unified normal prediction script
python predict_unified.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000 ^
  --model_path Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000

python predict_unified.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-10000 ^
  --model_path Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-small-separated-standard-chanlam-fullft-10000

python predict_unified.py ^
  --tokenizer_name google/t5-v1_1-small ^
  --model_path google/t5-v1_1-small ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/google_t5-v1_1-small

python predict_unified.py ^
  --tokenizer_name google/t5-v1_1-base ^
  --model_path google/t5-v1_1-base ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/google_t5-v1_1-base