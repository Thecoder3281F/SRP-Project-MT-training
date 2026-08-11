python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lorafull10epochs ^
    --num_train_epochs 10 ^
    --use_early_stopping ^
    --seed 42 ^
    --use_full_train_split
