python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora1000rows40epochsseed67 ^
    --num_train_epochs 40 ^
    --use_early_stopping ^
    --dataset_seed 67 ^
    --train_rows 1000

