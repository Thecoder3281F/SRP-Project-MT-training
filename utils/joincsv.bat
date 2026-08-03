@echo off
python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-100rows-2k" ^
--output "preds\Thecoder3281f\ReactionT5-lora-100rows-2k\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-500rows-2k" ^
--output "preds\Thecoder3281f\ReactionT5-lora-500rows-2k\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-1000rows-2k" ^
--output "preds\Thecoder3281f\ReactionT5-lora-1000rows-2k\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-2000rows-2k" ^
--output "preds\Thecoder3281f\ReactionT5-lora-2000rows-2k\finalpreds.csv"