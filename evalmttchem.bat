
python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-Lora-10rows-67\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-Lora-10rows-67\evaluation_results.json"


python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-Lora-50rows-67\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-Lora-50rows-67\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-Lora-100rows-67\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-Lora-100rows-67\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-Lora-250rows-67\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-Lora-250rows-67\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-Lora-500rows-67\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-Lora-500rows-67\evaluation_results.json"

