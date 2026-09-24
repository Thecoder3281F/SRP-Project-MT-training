
python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\seed123\Lora-10rows-123\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\seed123\Lora-10rows-123\evaluation_results.json"


python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\seed123\Lora-50rows-123\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\seed123\Lora-50rows-123\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\seed123\Lora-100rows-123\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\seed123\Lora-100rows-123\evaluation_results.json"

@REM python utils/metrics_json_from_preds_csv.py ^
@REM     --csv_path "preds\Thecoder3281f\seed123\Lora-250rows-123\finalpreds.csv" ^
@REM     --gt_column "label" ^
@REM     --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
@REM     --output_file "preds\Thecoder3281f\seed123\Lora-250rows-123\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\seed123\Lora-250rows-123\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\seed123\Lora-250rows-123\evaluation_results.json"


