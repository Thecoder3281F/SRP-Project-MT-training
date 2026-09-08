python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lora2000rows15epochsseed123 ^
    --num_train_epochs 15 ^
    --use_early_stopping ^
    --dataset_seed 123 ^
    --train_rows 2000
