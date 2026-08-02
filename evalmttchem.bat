
python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-50rows-2k\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-50rows-2k\evaluation_results.json"

@REM python utils/metrics_json_from_preds_csv.py ^
@REM     --csv_path "preds\Thecoder3281f\ReactionT5-lora-50rows-30epochs\chunk_0.csv" ^
@REM     --gt_column "label" ^
@REM     --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
@REM     --output_file "preds\Thecoder3281f\ReactionT5-lora-50rows-30epochs\evaluation_results.json"

@REM python utils/metrics_json_from_preds_csv.py ^
@REM     --csv_path "preds\Thecoder3281f\ReactionT5-lora-100rows-30epochs\chunk_0.csv" ^
@REM     --gt_column "label" ^
@REM     --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
@REM     --output_file "preds\Thecoder3281f\ReactionT5-lora-100rows-30epochs\evaluation_results.json"

@REM python utils/metrics_json_from_preds_csv.py ^
@REM     --csv_path "preds\Thecoder3281f\ReactionT5-lora-500rows-30epochs\chunk_0.csv" ^
@REM     --gt_column "label" ^
@REM     --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
@REM     --output_file "preds\Thecoder3281f\ReactionT5-lora-500rows-30epochs\evaluation_results.json"

@REM python utils/metrics_json_from_preds_csv.py ^
@REM     --csv_path "preds\Thecoder3281f\ReactionT5-lora-1000rows-30epochs\chunk_0.csv" ^
@REM     --gt_column "label" ^
@REM     --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
@REM     --output_file "preds\Thecoder3281f\ReactionT5-lora-1000rows-30epochs\evaluation_results.json"

@REM python utils/metrics_json_from_preds_csv.py ^
@REM     --csv_path "preds\Thecoder3281f\ReactionT5-lora-10rows-50epochs\chunk_0.csv" ^
@REM     --gt_column "label" ^
@REM     --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
@REM     --output_file "preds\Thecoder3281f\ReactionT5-lora-10rows-50epochs\evaluation_results.json"