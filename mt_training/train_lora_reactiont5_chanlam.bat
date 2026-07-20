python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora50rows50epochs ^
    --num_train_epochs 50 ^
    --use_early_stopping