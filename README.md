## SRP Project MT Training

This repository contains the code, preprocessing utilities, training scripts, and evaluation helpers used for molecular transformation training on the Chan-Lam and MIT datasets.

## Scope

- Sequence-to-sequence training and inference for reaction prediction tasks
- Dataset preparation for Chan-Lam and MIT variants
- Progressive subset generation and diversity-based sampling
- Evaluation with canonical accuracy, SMILES validity, and Tanimoto similarity

## Recommended repository layout

- [utils/](utils/) – shared helpers, evaluation utilities, and support scripts
- [visualisation_scripts/](visualisation_scripts/) – UMAP and Streamlit visualisation tools
- [train_lora_reactiont5_chanlam.py](train_lora_reactiont5_chanlam.py) – LoRA training entry point
- [predict_unified.py](predict_unified.py) – standard inference entry point
- [predict_unified_lora.py](predict_unified_lora.py) – LoRA inference entry point
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

## Suggested training schedule

For comparing row-count sweeps, use a fixed epoch budget and let steps scale with dataset size.
This is closer to how fine-tuning is usually reported in papers than using the same `--max_steps`
for every subset size.

The table below assumes an effective batch size of 8. If auto batch sizing lowers your actual
batch size, multiply the step counts by `8 / effective_batch_size`.

| train_rows | recommended epochs | approx. max_steps |
| --- | --- | --- |
| 10 | 20 | 40 |
| 50 | 20 | 140 |
| 100 | 20 | 260 |
| 500 | 20 | 1260 |
| 1000 | 20 | 2500 |
| 2000 | 20 | 5000 |
| 4000 | 20 | 10000 |

If you want a more aggressive overfit baseline for very small subsets, you can push the 10-row
and 50-row runs higher, but keep the larger subsets on the same epoch budget so the comparison
stays meaningful.

## How to use early stopping

Early stopping still needs a ceiling, either `--max_steps` or `--num_train_epochs`. In practice,
set that ceiling high enough that the model can converge, then let validation stop the run early.

That means:

- the ceiling is a safety limit, not the thing you are trying to hit;
- the best checkpoint should come from validation loss, not the final step;
- if the run always ends exactly at the ceiling, the ceiling is probably too low;
- if the run stops very early and validation never improves again, the ceiling was probably high enough.

For your sweeps, a better pattern is to use the epoch counts above as the maximum budget, turn on early
stopping, and compare the best validation checkpoint from each run.

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
