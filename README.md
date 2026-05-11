## SRP Project MT Training

This repository contains the code, preprocessing utilities, training scripts, and evaluation helpers used for molecular transformation training on the Chan-Lam and MIT datasets.

## Scope

- Sequence-to-sequence training and inference for reaction prediction tasks
- Dataset preparation for Chan-Lam and MIT variants
- Progressive subset generation and diversity-based sampling
- Evaluation with canonical accuracy, SMILES validity, and Tanimoto similarity

## Recommended repository layout

- [mt_training/](mt_training/) – training, evaluation, sampling, and utility scripts
- [datasets_final/](datasets_final/) – final split CSVs and small reproducible dataset artifacts
- [_datasets_unprocessed/](_datasets_unprocessed/) – local/raw preprocessing inputs, kept out of version control
- [data_visualisation/](data_visualisation/) – notebooks and final figures
- [README.md](README.md) – setup and workflow notes

## Data handling

- Keep raw or source datasets outside the public repository when licensing or size is a concern.
- Commit only the final splits or lightweight metadata needed to reproduce experiments.
- Store large model checkpoints, logs, and prediction dumps outside GitHub release history.

## Training and evaluation workflow

1. Preprocess reactions into the expected input format, such as reactants > reagents.
2. Canonicalize inputs and outputs.
3. Train the selected model variant.
4. Evaluate on validation and test splits.
5. Run inference on novel or held-out data.

## Metrics

- Top-1 canonical accuracy on validation and test sets
- Mean Tanimoto similarity on validation and test sets
- Mean SMILES validity on validation and test sets
- Training, validation, and test loss

## Reproducibility notes

- Record the exact command line, random seed, and dataset split used for each run.
- Keep one canonical training configuration for the paper.
- Prefer small, well-named output directories for each experiment.
- Remove notebook outputs before publishing.

## Before publishing

- Confirm that no secrets, checkpoints, or large generated artifacts are tracked.
- Add a license and citation file.
- Document the environment and any required downloads.
- Tag the exact repository state used for the manuscript.

## Example focus of the existing notebooks and scripts

- Dataset comparison between separated and combined settings
- Progressive subset construction for diversity and spread experiments
- Inference and post-processing for predicted SMILES strings

## Citation

If you use this repository in academic work, please cite the project using the provided CITATION file.
