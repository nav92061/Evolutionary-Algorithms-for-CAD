# Evolutionary ML - Cardiac Classification

Genetic algorithm that does joint feature selection and hyperparameter tuning for a gradient boosting classifier on cardiac data. Healthy vs unhealthy patient classification using the UCI Heart Disease dataset (Cleveland).

## What it does

Runs a GA where each individual encodes a binary feature mask plus GBM hyperparameters (n_estimators, max_depth, learning_rate, subsample). Fitness is 5-fold cross-validated ROC-AUC. After N generations the best genome is decoded and the final model is evaluated on a held-out test set.

Basically instead of doing a grid search or random search for hyperparams, the population evolves toward better solutions over generations. Also handles feature selection at the same time which is the more interesting part imo.

## Requirements

```
pip install deap scikit-learn pandas numpy matplotlib seaborn ucimlrepo
```

## Usage

Open in google colab and run cells top to bottom. By default it pulls the UCI heart disease dataset automatically. If you want to use your own data, set `USE_GOOGLE_DRIVE = True` and point `DRIVE_FILE_PATH` at your csv. Target column should be binary (0 = healthy, 1 = unhealthy).

## GA setup

- Population: 30 individuals
- Generations: 20
- Crossover: blend crossover (alpha=0.4)
- Mutation: gaussian, sigma=0.2, per-gene prob 0.15
- Selection: tournament size 3

These are reasonable defaults but feel free to increase population/generations if you have time and compute.

## Output

Saves four plots: evolution curve, ROC curve, confusion matrix, and feature importances for the evolved feature subset. Also prints a baseline vs evolved AUC comparison at the end.

## Notes

Dataset is small (~300 samples) so results will vary between runs. Setting a seed helps but the GA is still stochastic. If the evolved model doesnt beat baseline just re-run, it usually does after a couple tries.
