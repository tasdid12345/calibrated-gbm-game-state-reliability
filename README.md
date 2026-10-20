# Calibrated LightGBM vs. Random Forest on Tic-Tac-Toe Endgame States

Motivation: "A logic for reasoning about game descriptions" (2018) studies how formal logic can describe the rules of games, so that an agent can reason about win and loss conditions. Checking whether a model can reliably judge win/loss from a board position is a simple, measurable proxy for that kind of rule understanding. This repository runs a small stand-in experiment on game states rather than reproducing the paper's logic.

Question: Can a class-balanced, isotonic-calibrated LightGBM model give more reliable win/loss judgements than a standard random forest on game states?

Dataset: Tic-Tac-Toe Endgame from OpenML (data_id 50, https://www.openml.org/d/50). It has 958 board states with nine categorical cells and a positive/negative win label. This is not the paper's own dataset.

Baseline: a standard 300-tree random forest on one-hot encoded cells, using raw probabilities.

Improved model: a class-balanced LightGBM with a small 5-fold cross-validated grid search on the training set, wrapped in isotonic calibration. Both models share the same split, preprocessing and evaluation.

How to run:

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and a chart to figures/).

## Results (latest run)

On the Tic-Tac-Toe Endgame dataset (OpenML data_id 50), a standard 300-tree random forest on raw probabilities reached 96.8% test accuracy, while my class-balanced, isotonic-calibrated LightGBM reached 99.9% on the same 240-row split, a modest gain on a single small split.

| Metric | Value |
|---|---|
| n_seeds | 3 |
| baseline_score | 0.9681 |
| baseline_score_std | 0.003928 |
| baseline_f1 | 0.9762 |
| baseline_brier | 0.05823 |
| baseline_logloss | 0.2416 |
| improved_score | 0.9986 |
| improved_score_std | 0.001964 |
| improved_f1 | 0.9989 |
| improved_brier | 0.001502 |
| improved_logloss | 0.007565 |
| improvement | 0.03056 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in experiment, not a replication of the paper's logic-based method. Only one stratified split was used (seed 42) per the split-once rule; seeds 42-44 vary the models only, so the std reflects model randomness, not split variability. The test set (about 240 rows) is small, so accuracy differences of a few points are within noise. Grid search and calibration use only training data. fetch_openml takes no timeout argument, so a socket-level 60 s timeout is set globally. Compute is matched only roughly: the improved model runs a small grid search plus calibration.

## Reference

This project is motivated by the following paper:

> D Zhang (2018). *A logic for reasoning about game descriptions*. Australasian Joint Conference on Artificial Intelligence. https://link.springer.com/chapter/10.1007/978-3-030-03991-2_4

Paper author: Prof. Assoc. Prof. Dongmo Zhang, Western Sydney University (d.zhang@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
