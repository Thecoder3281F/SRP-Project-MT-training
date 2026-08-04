python make_progressive_group_subsets.py ^
    --input _datasets_final/chanlam_final/train_final.csv ^
    --out-dir _datasets_final/chanlam_final/progressive_splits/seed123 ^
    --group-col input_reactants ^
    --sizes-rows 10,50,100,250,500,1000,2000,4000 ^
    --seed 123 ^
    --sample-mode spread ^
    --auto-per-group ^
    --exclude-full