@echo off
python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-10" ^
--output "preds\Thecoder3281f\ReactionT5-lora-10\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-50" ^
--output "preds\Thecoder3281f\ReactionT5-lora-50\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-100" ^
--output "preds\Thecoder3281f\ReactionT5-lora-100\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-250" ^
--output "preds\Thecoder3281f\ReactionT5-lora-250\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-500" ^
--output "preds\Thecoder3281f\ReactionT5-lora-500\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-1000" ^
--output "preds\Thecoder3281f\ReactionT5-lora-1000\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-2000" ^
--output "preds\Thecoder3281f\ReactionT5-lora-2000\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-4000" ^
--output "preds\Thecoder3281f\ReactionT5-lora-4000\finalpreds.csv"

python utils/join_csvs.py ^
--folder "preds\Thecoder3281f\ReactionT5-lora-full" ^
--output "preds\Thecoder3281f\ReactionT5-lora-full\finalpreds.csv"
