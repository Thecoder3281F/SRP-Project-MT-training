Following code contains the code for finetuning on both MIT and Chan-Lam dataset

Models use their default tokenizer (T5)


- Preprocessing
-- Arrange inputs in format: reactants>reagents
-- Canonicalise both inputs and outputs

Use LR=3e-4


- Metrics
-- Top 1 Canonical Accuracy (Val/Test)
-- Mean Tanimoto Similarity (Val/Test)
-- Mean Validity of SMILES (Val/Test)
-- Loss (Train/Val/Test)

- Training of models
- Evaluation
- Inference on novel data (other amines)

- t5_small_nospaces_sepvscomb_trial1d.ipynb: test separated dataset vs combined dataset to see which performs better
