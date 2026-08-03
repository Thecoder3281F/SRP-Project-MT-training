python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lorafull2epochs ^
    --num_train_epochs 2 ^
    --use_early_stopping ^
    --seed 42 ^
    --use_full_train_split

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lorafull3epochs ^
    --num_train_epochs 3 ^
    --use_early_stopping ^
    --seed 42 ^
    --use_full_train_split

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lorafull5epochs ^
    --num_train_epochs 5 ^
    --use_early_stopping ^
    --seed 42 ^
    --use_full_train_split

python train_lora_reactiont5_chanlam.py ^
    --base_model_name sagawa/ReactionT5v2-forward-USPTO_MIT ^
    --output_model_dir reactiont5lorafull10epochs ^
    --num_train_epochs 10 ^
    --use_early_stopping ^
    --seed 42 ^
    --use_full_train_split
