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

## Reference

This project is motivated by the following paper:

> D Zhang (2018). *A logic for reasoning about game descriptions*. Australasian Joint Conference on Artificial Intelligence. https://link.springer.com/chapter/10.1007/978-3-030-03991-2_4

Paper author: Prof. Assoc. Prof. Dongmo Zhang, Western Sydney University (d.zhang@westernsydney.edu.au)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
