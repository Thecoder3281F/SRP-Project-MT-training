---
configs:
- config_name: normal
  data_files:
  - split: train
    path: normal/MIT_separated_train_final.csv
  - split: test
    path: normal/MIT_separated_test_final.csv
  - split: val
    path: normal/MIT_separated_val_final.csv
- config_name: augmented
  data_files:
  - split: train
    path: augmented/MIT_separated_augm_train_final.csv
  - split: test
    path: augmented/MIT_separated_augm_test_final.csv
  - split: val
    path: augmented/MIT_separated_augm_val_final.csv
license: mit
task_categories:
- translation
language:
- en
tags:
- chemistry
pretty_name: MIT Dataset with differentiation between reactants and reagents, no spaces
size_categories:
- 1M<n<10M
---