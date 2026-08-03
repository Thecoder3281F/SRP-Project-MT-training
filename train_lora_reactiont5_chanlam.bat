python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lorachanlamfull2000steps ^
    --max_steps 2000 ^
    --use_early_stopping ^
    --seed 42 ^
    --use_full_train_split

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora4000rows2000steps ^
    --max_steps 2000 ^
    --use_early_stopping ^
    --seed 42 ^
    --train_rows 4000 ^
    --resume_checkpoint