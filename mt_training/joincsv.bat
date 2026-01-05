@echo off
python join_csvs.py ^
--folder "model_predictions_42\my_models\afterft\full\mit\Thecoder3281f\ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000" ^
--output "model_predictions_42\my_models\afterft\full\mit\Thecoder3281f\ManganumT5v1_1-base-separated-augmented-chanlam-fullft-10000\finalpreds.csv"

python join_csvs.py ^
--folder "model_predictions_42\my_models\afterft\full\mit\Thecoder3281f\ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000" ^
--output "model_predictions_42\my_models\afterft\full\mit\Thecoder3281f\ManganumT5v1_1-small-separated-augmented-chanlam-fullft-10000\finalpreds.csv"