python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora2000rows3epochs ^
    --num_train_epochs 3 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 2000

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora2000rows5epochs ^
    --num_train_epochs 5 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 2000

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora2000rows10epochs ^
    --num_train_epochs 10 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 2000

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora2000rows15epochs ^
    --num_train_epochs 15 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 2000
