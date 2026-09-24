@echo off
python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-10rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-10rows-123\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-50rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-50rows-123\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-100rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-100rows-123\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-250rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-250rows-123\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-500rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-500rows-123\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-1000rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-1000rows-123\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-2000rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-2000rows-123\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\seed123\Lora-4000rows-123" ^
--output "preds\Thecoder3281f\seed123\Lora-4000rows-123\finalpreds.csv"
