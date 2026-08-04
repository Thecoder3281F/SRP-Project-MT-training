
python utils/metrics_json_from_preds_csv.py ^
    --csv_path "preds\Thecoder3281f\ReactionT5-lora-10rows-2k\finalpreds.csv" ^
    --gt_column "label" ^
    --pred_columns "prediction_1,prediction_2,prediction_3,prediction_4,prediction_5" ^
    --output_file "preds\Thecoder3281f\ReactionT5-lora-10rows-2k\evaluation_results.json"
