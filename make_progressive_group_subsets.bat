python make_progressive_group_subsets.py ^
    --input _datasets_final/buchwald_hartwig_final/train.csv ^
    --out-dir _datasets_final/buchwald_hartwig_final/progressive_splits/seed1111 ^
    --group-col reactants ^
    --sizes-rows 10,50,100,250,500,1000,2000,3000 ^
    --seed 1111 ^
    --sample-mode spread ^
    --auto-per-group ^
    --exclude-full