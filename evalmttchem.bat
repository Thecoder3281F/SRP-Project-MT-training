
python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-10\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-10\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-50\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-50\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-100\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-100\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-250\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-250\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-500\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-500\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-1000\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-1000\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-2000\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-2000\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-4000\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-4000\evaluation_results.json"

python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-full\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-full\evaluation_results.json"
