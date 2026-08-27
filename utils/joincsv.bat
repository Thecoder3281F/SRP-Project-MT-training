@echo off
python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-Lora-10rows-67" ^
--output "preds\Thecoder3281f\ReactionT5-Lora-10rows-67\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-Lora-50rows-67" ^
--output "preds\Thecoder3281f\ReactionT5-Lora-50rows-67\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-Lora-100rows-67" ^
--output "preds\Thecoder3281f\ReactionT5-Lora-100rows-67\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-Lora-250rows-67" ^
--output "preds\Thecoder3281f\ReactionT5-Lora-250rows-67\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-Lora-500rows-67" ^
--output "preds\Thecoder3281f\ReactionT5-Lora-500rows-67\finalpreds.csv"
