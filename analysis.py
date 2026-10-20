"""Stand-in experiment: class-balanced, isotonic-calibrated LightGBM (improved)
vs. a standard 300-tree random forest (baseline) on tic-tac-toe endgame board states.
One stratified 75/25 split (seed 42); randomized models run with seeds 42, 43, 44."""
import json
import socket
import sys
import time
from urllib.error import HTTPError, URLError

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
SEEDS = [42, 43, 44]
TEST_SIZE = 0.25
DATA_ID = 50
N_ROWS_MAX = 200_000  # subsample cap for very large data (not triggered by this dataset)
SOURCE = "https://www.openml.org/d/50"
socket.setdefaulttimeout(60)  # timeout=60s for every urllib download inside fetch_openml


def load_data(retries=4):
    """Download OpenML data_id=50; retry with exponential backoff on HTTP 429/5xx and network errors."""
    for attempt in range(retries + 1):
        try:
            return fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto")
        except HTTPError as e:  # must precede URLError (HTTPError subclasses it)
            if not (e.code == 429 or e.code >= 500) or attempt == retries:
                raise
            reason = f"HTTP {e.code}"
        except (URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise
            reason = type(e).__name__
        wait = 2 ** attempt
        print(f"  download failed ({reason}); retrying in {wait}s")
        time.sleep(wait)


def evaluate(model, X, y_true):
    """Shared evaluation: accuracy (main), F1, Brier score and log loss on the positive (win) class."""
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {"acc": accuracy_score(y_true, pred), "f1": f1_score(y_true, pred),
            "brier": brier_score_loss(y_true, p), "logloss": log_loss(y_true, p)}


def improved_model(X_tr, y_tr, seed):
    """Class-balanced LightGBM, small 5-fold CV grid search on training data only, then isotonic calibration."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = {"num_leaves": [7, 15], "learning_rate": [0.05, 0.1], "n_estimators": [100, 300]}
    base = LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1, n_jobs=1)
    search = GridSearchCV(base, grid, cv=cv, scoring="neg_log_loss", n_jobs=-1)
    search.fit(X_tr, y_tr)
    print(f"  seed {seed}: best LightGBM params {search.best_params_}")
    # Isotonic calibration fitted with internal CV on the training rows only
    cal = CalibratedClassifierCV(clone(search.best_estimator_), method="isotonic", cv=cv)
    return cal.fit(X_tr, y_tr)


def main():
    print("Downloading OpenML data_id=50 ...")
    bunch = load_data()
    X_raw = bunch.data
    y = (bunch.target.astype(str) == "positive").astype(int).to_numpy()  # 1 = win for x
    if len(y) > N_ROWS_MAX:
        idx = np.random.RandomState(SEED).choice(len(y), N_ROWS_MAX, replace=False)
        X_raw, y = X_raw.iloc[idx], y[idx]
    print(f"Rows: {len(y)}, positive share: {y.mean():.3f}")

    # ONE stratified split, seed 42
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    # Shared preprocessing: one-hot encode the 9 board cells, fitted on training rows only
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(X_tr_raw)
    X_te = enc.transform(X_te_raw)
    print(f"Train: {len(y_tr)}, test: {len(y_te)}, features after one-hot: {X_tr.shape[1]}")

    runs = {"baseline": [], "improved": []}
    for s in SEEDS:
        print(f"Seed {s}: fitting baseline random forest")
        rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1).fit(X_tr, y_tr)
        print(f"Seed {s}: fitting improved calibrated LightGBM")
        imp = improved_model(X_tr, y_tr, s)
        runs["baseline"].append(evaluate(rf, X_te, y_te))
        runs["improved"].append(evaluate(imp, X_te, y_te))

    def agg(name, key):
        vals = np.array([r[key] for r in runs[name]])
        return float(vals.mean()), float(vals.std())

    m = {"n_seeds": len(SEEDS)}
    for name in runs:
        m[f"{name}_score"], m[f"{name}_score_std"] = agg(name, "acc")
        m[f"{name}_f1"], _ = agg(name, "f1")
        m[f"{name}_brier"], _ = agg(name, "brier")
        m[f"{name}_logloss"], _ = agg(name, "logloss")
    m["improvement"] = m["improved_score"] - m["baseline_score"]  # accuracy: higher is better

    # Sanity check: refuse to write meaningless metrics
    if not all(np.isfinite(v) for v in m.values()):
        print("ERROR: non-finite metric produced")
        sys.exit(1)

    # Optional chart: accuracy and log loss, baseline vs improved (mean +/- std over seeds)
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    names = ["baseline", "improved"]
    ax[0].bar(names, [m["baseline_score"], m["improved_score"]],
              yerr=[m["baseline_score_std"], m["improved_score_std"]], capsize=5)
    ax[0].set_title("Test accuracy (higher is better)")
    ax[1].bar(names, [m["baseline_logloss"], m["improved_logloss"]])
    ax[1].set_title("Test log loss (lower is better)")
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can a class-balanced, isotonic-calibrated LightGBM model give more reliable "
                    "win/loss judgements than a standard random forest on game states, as a small "
                    "stand-in for checking a player's reading of game rules?",
        "method": "Stratified 75/25 split (seed 42) of OpenML tic-tac-toe endgame data; one-hot encoded "
                  "cells; baseline 300-tree random forest vs. class-balanced LightGBM with 5-fold CV grid "
                  "search and isotonic calibration; seeds 42-44; accuracy main, F1/Brier/log loss secondary.",
        "dataset": {
            "name": "Tic-Tac-Toe Endgame (OpenML data_id 50)",
            "source_url": SOURCE,
            "n_rows": int(len(y)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "test accuracy",
        "higher_is_better": True,
        "baseline": "Standard RandomForestClassifier (300 trees) on one-hot cells, raw probabilities.",
        "improved": "LightGBM with class_weight='balanced', hyper-parameters chosen by 5-fold CV grid "
                    "search on the training set (neg. log loss), wrapped in isotonic calibration "
                    "(5-fold internal CV on training data).",
        "metrics": m,
        "notes": "Stand-in experiment, not a replication of the paper's logic-based method. Only one "
                 "stratified split was used (seed 42) per the split-once rule; seeds 42-44 vary the "
                 "models only, so the std reflects model randomness, not split variability. The test set "
                 "(about 240 rows) is small, so accuracy differences of a few points are within noise. "
                 "Grid search and calibration use only training data. fetch_openml takes no timeout "
                 "argument, so a socket-level 60 s timeout is set globally. Compute is matched only "
                 "roughly: the improved model runs a small grid search plus calibration.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
