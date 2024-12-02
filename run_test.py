from sklearn.ensemble import RandomForestRegressor
import target_permutation_importances as tpi
from target_permutation_importances import TargetPermutationImportancesWrapper
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Generate correlated targets
y_correlated_1 = y + np.random.normal(scale=0.1, size=len(y))
y_correlated_2 = y * 1.1 + np.random.normal(scale=0.2, size=len(y))

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
targets = [y, y_correlated_1, y_correlated_2]

# Compute permutation importances using RandomForestRegressor
result_df = tpi.compute(
    model_cls=RandomForestRegressor,  # Use regression model for continuous targets
    model_cls_params={"n_jobs": -1},
    model_fit_params={},
    X=X_df,
    y=targets,
    num_actual_runs=len(targets),
    num_random_runs=len(targets),
)

# Display the results
df = result_df.sort_values("importance", ascending=False).head()
