@echo off

@REM python predict_unified.py ^
@REM   --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000 ^
@REM   --model_path Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000 ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds_chanlam/Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000

@REM python predict_unified.py ^
@REM   --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000 ^
@REM   --model_path Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000 ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds_chanlam/Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000
@REM python predict_unified.py ^
@REM   --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000 ^
@REM   --model_path Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000 ^
@REM   --dataset_type chanlam_separated ^
@REM   --chunks all ^
@REM   --output_dir preds_chanlam/Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000


@REM python predict_unified.py ^
@REM   --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000 ^
@REM   --model_path Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000 ^
@REM   --dataset_type mit_separated_normal ^
@REM   --chunks all ^
@REM   --output_dir preds/Thecoder3281f/ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000

python predict_unified.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000 ^
  --model_path Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-base-separated-standard-chanlam-fullft-10000
python predict_unified.py ^
  --tokenizer_name Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000 ^
  --model_path Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000 ^
  --dataset_type mit_separated_normal ^
  --chunks all ^
  --output_dir preds/Thecoder3281f/ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000
