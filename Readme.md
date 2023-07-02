# Target Permutation Importances

## Overview
This method aims at lower the feature attribution due to the variance of a feature.
If a feature is important after the target vector is shuffled, it is fitting to noise.

By default, this package 

1. Fit the given model class on the given dataset M times to compute the mean actual feature importances ($A$).
2. Fit the given model class on the given dataset with shuffled targets N times to compute mean random feature importances ($R$).
3. Compute the final importances by either $A - R$ or $A / (MinMaxScale(R) + 1)$

Not to be confused with [sklearn.inspection.permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance),
this sklearn method is about feature permutation instead of target permutation.

This method were originally proposed/implemented by:
- [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
- [Feature Selection with Null Importances
](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances/notebook)



## Basic Usage

### With Scikit Learn Models

```python
# Import the function
from target_permutation_importances import compute

# Prepare a dataset
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = load_breast_cancer()

# Compute permutation importances with default settings
result_df = compute(
    model_cls=RandomForestClassifier,
    model_cls_params={ # The params for the model class construction
        "n_estimators": 1,
    },
    model_fit_params={}, # The params for model.fit
    X=Xpd,
    y=data.target,
    num_actual_runs=2,
    num_random_runs=10,
)
```

### With XGBoost

### With LightGBM

### With CatBoost

## Advance Usage

## Feature Selection Examples

## Benchmarks

### Datasets
Benchmark has been done with some tabular datasets from the Tabular data learning benchmark
- [Github](https://github.com/LeoGrin/tabular-benchmark/tree/main)
- [Hugging Face](https://huggingface.co/datasets/inria-soda/tabular-benchmark)

### Models
The following models with their default params are used:
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.RandomForestRegressor`
- `xgboost.XGBClassifier`
- `xgboost.XGBRegressor`
- `catboost.CatBoostClassifier`
- `catboost.CatBoostRegressor`
- `lightgbm.LGBMClassifier`
- `lightgbm.LGBMRegressor`

### Evaluation
For binary classification task, `sklearn.metrics.f1_score` is used for evaluation.
For regression task, `sklearn.metrics.mean_squared_error` is used for evaluation.

The download datasets are divided into 3 sections: `train`: 50%, `val`: 10%, `test`: 40%

Feature importance is calculated from the `train` set. Feature selection is done on the `val` set. 
The final benchmark is evaluated on the `test` set. Therefore the `test` set is unseen to both the feature importance and selection process.

## Development Setup and Contribution Guide
### Python Version
You can find the suggested development Python version in `.python-version`.
You might consider setting up `Pyenv` if you want to have multiple Python versions in your machine.

### Python packages
This repository is setup with `Poetry`. If you are not familiar with Poetry, you can find packages requirements are listed in `pyproject.toml`. 
Otherwise, you can just set up with `poetry install`

### Run Benchmarks
To run benchmark locally on your machine, run `make benchmark` or `python -m benchmarks.run`



