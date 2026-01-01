@echo off
python join_csvs.py ^
--folder "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-base-separated-augmented_separated_beam10_r5" ^
--output "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-base-separated-augmented_separated_beam10_r5\finalpreds.csv"

python join_csvs.py ^
--folder "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-base-separated-standard_separated_beam10_r5" ^
--output "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-base-separated-standard_separated_beam10_r5\finalpreds.csv"

python join_csvs.py ^
--folder "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-small-separated-augmented_separated_beam10_r5" ^
--output "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-small-separated-augmented_separated_beam10_r5\finalpreds.csv"

python join_csvs.py ^
--folder "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-small-separated-standard_separated_beam10_r5" ^
--output "model_predictions\my_models\beforeft\mit\pred_chunks_Thecoder3281F\ManganumT5v1_1-small-separated-standard_separated_beam10_r5\finalpreds.csv"