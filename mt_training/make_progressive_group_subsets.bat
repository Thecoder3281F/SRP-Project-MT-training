python mt_training/make_progressive_group_subsets.py ^
    --input datasets_final/chanlam_final/train_final.csv ^
    --out-dir datasets_final/chanlam_final/progressive_by_reactants ^
    --group-col input_reactants ^
    --sizes-rows 10,50,100,250,500,1000,2000,4000 ^
    --seed 42 ^
    --sample-mode spread ^
    --auto-per-group ^
    --exclude-full