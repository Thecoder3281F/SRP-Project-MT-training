python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora1000rows5epochs ^
    --num_train_epochs 5 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 1000

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora1000rows10epochs ^
    --num_train_epochs 10 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 1000

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora1000rows20epochs ^
    --num_train_epochs 20 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 1000

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora1000rows30epochs ^
    --num_train_epochs 30 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 1000

