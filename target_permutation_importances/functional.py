"""
The core APIs of this library.
"""

import gc
from functools import partial

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Dict, List, Union, Tuple
from scipy.stats import wasserstein_distance  # type: ignore
from tqdm import tqdm

from target_permutation_importances.typing import (
    ModelBuilderType,
    ModelFitParamsBuilderType,
    ModelFitterType,
    ModelImportanceGetter,
    PermutationImportanceCalculatorType,
    PositiveInt,
    XBuilderType,
    XType,
    YBuilderType,
    YType,
)


def compute_permutation_importance_by_subtraction(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Given a list of actual importance DataFrames and a list of random importance compute
    the permutation importance by I_f = Avg(A_f) - Avg(R_f)

    Args:
        actual_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]
        random_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]

    Returns:
        pd.DataFrame: The return DataFrame with columns ["feature", "importance"]
    """
    # Calculate the mean importance
    actual_importance_df = pd.concat(actual_importance_dfs)
    mean_actual_importance_df = actual_importance_df.groupby("feature").mean()
    std_actual_importance_df = actual_importance_df.groupby("feature").std()

    # Calculate the mean random importance
    random_importance_df = pd.concat(random_importance_dfs)
    mean_random_importance_df = random_importance_df.groupby("feature").mean()
    std_random_importance_df = random_importance_df.groupby("feature").std()

    # Sort by feature name to make sure the order is the same
    mean_actual_importance_df = mean_actual_importance_df.sort_index()
    std_actual_importance_df = std_actual_importance_df.sort_index()
    mean_random_importance_df = mean_random_importance_df.sort_index()
    std_random_importance_df = std_random_importance_df.sort_index()
    assert (mean_random_importance_df.index == mean_actual_importance_df.index).all()

    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["std_actual_importance"] = std_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]
    mean_actual_importance_df["importance"] = (
        mean_actual_importance_df["importance"] - mean_random_importance_df["importance"]
    )
    mean_actual_importance_df["std_random_importance"] = std_random_importance_df[
        "importance"
    ]
    return mean_actual_importance_df[
        [
            "importance",
            "mean_actual_importance",
            "mean_random_importance",
            "std_actual_importance",
            "std_random_importance",
        ]
    ].reset_index()


def compute_permutation_importance_by_division(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Given a list of actual importance DataFrames and a list of random importance compute
    the permutation importance by I_f = Avg(A_f) / (Avg(R_f) + 1)

    Args:
        actual_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]
        random_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]

    Returns:
        pd.DataFrame: The return DataFrame with columns ["feature", "importance"]
    """
    # Calculate the mean importance
    actual_importance_df = pd.concat(actual_importance_dfs)
    mean_actual_importance_df = actual_importance_df.groupby("feature").mean()
    std_actual_importance_df = actual_importance_df.groupby("feature").std()

    # Calculate the mean random importance
    random_importance_df = pd.concat(random_importance_dfs)
    mean_random_importance_df = random_importance_df.groupby("feature").mean()
    std_random_importance_df = random_importance_df.groupby("feature").std()

    # Sort by feature name to make sure the order is the same
    mean_actual_importance_df = mean_actual_importance_df.sort_index()
    std_actual_importance_df = std_actual_importance_df.sort_index()
    mean_random_importance_df = mean_random_importance_df.sort_index()
    std_random_importance_df = std_random_importance_df.sort_index()

    assert (mean_random_importance_df.index == mean_actual_importance_df.index).all()

    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["std_actual_importance"] = std_actual_importance_df[
        "importance"
    ]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df[
        "importance"
    ]
    mean_actual_importance_df["importance"] = mean_actual_importance_df["importance"] / (
        mean_random_importance_df["importance"] + 1
    )
    mean_actual_importance_df["std_random_importance"] = std_random_importance_df[
        "importance"
    ]
    return mean_actual_importance_df[
        [
            "importance",
            "mean_actual_importance",
            "mean_random_importance",
            "std_actual_importance",
            "std_random_importance",
        ]
    ].reset_index()


def compute_permutation_importance_by_wasserstein_distance(
    actual_importance_dfs: List[pd.DataFrame], random_importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Given a list of actual importance DataFrames and a list of random importance compute
    the permutation importance by I_f = wasserstein_distance(A_f, R_f)

    Args:
        actual_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]
        random_importance_dfs (List[pd.DataFrame]): list of random importance DataFrames with columns ["feature", "importance"]

    Returns:
        pd.DataFrame: The return DataFrame with columns ["feature", "importance"]
    """
    # Calculate the mean importance
    actual_importance_df = pd.concat(actual_importance_dfs)
    mean_actual_importance_df = actual_importance_df.groupby("feature").mean()
    std_actual_importance_df = actual_importance_df.groupby("feature").std()

    # Calculate the mean random importance
    random_importance_df = pd.concat(random_importance_dfs)
    mean_random_importance_df = random_importance_df.groupby("feature").mean()
    std_random_importance_df = random_importance_df.groupby("feature").std()

    # Calculate the wasserstein_distance for each feature
    distances = {}
    for f in random_importance_df["feature"].unique():
        distances[f] = wasserstein_distance(
            actual_importance_df[actual_importance_df["feature"] == f]["importance"].to_numpy(),
            random_importance_df[random_importance_df["feature"] == f]["importance"].to_numpy(),
        )
    mean_actual_importance_df["wasserstein_distance"] = mean_actual_importance_df.index.map(distances)

    # Sort by feature name to make sure the order is the same
    mean_actual_importance_df = mean_actual_importance_df.sort_index()
    std_actual_importance_df = std_actual_importance_df.sort_index()
    mean_random_importance_df = mean_random_importance_df.sort_index()
    std_random_importance_df = std_random_importance_df.sort_index()

    assert (mean_random_importance_df.index == mean_actual_importance_df.index).all()

    mean_actual_importance_df["mean_actual_importance"] = mean_actual_importance_df["importance"]
    mean_actual_importance_df["std_actual_importance"] = std_actual_importance_df["importance"]
    mean_actual_importance_df["mean_random_importance"] = mean_random_importance_df["importance"]
    mean_actual_importance_df["std_random_importance"] = std_random_importance_df["importance"]
    mean_actual_importance_df["importance"] = mean_actual_importance_df["wasserstein_distance"]
    return mean_actual_importance_df[
        [
            "importance",
            "mean_actual_importance",
            "mean_random_importance",
            "std_actual_importance",
            "std_random_importance",
            "wasserstein_distance",
        ]
    ].reset_index()


def _compute_one_run(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    importance_getter: ModelImportanceGetter,
    X_builder: XBuilderType,
    y_builder: YBuilderType,
    is_random_run: bool,
    run_idx: int,
):
    model = model_builder(is_random_run=is_random_run, run_idx=run_idx)
    X = X_builder(is_random_run=is_random_run, run_idx=run_idx)
    y = y_builder(is_random_run=is_random_run, run_idx=run_idx)

    model = model_fitter(model, X, y)
    gc.collect()
    return importance_getter(model, X, y)


@beartype
def generic_compute(
    model_builder: ModelBuilderType,
    model_fitter: ModelFitterType,
    importance_getter: ModelImportanceGetter,
    permutation_importance_calculator: Union[
        PermutationImportanceCalculatorType, List[PermutationImportanceCalculatorType]
    ],
    X_builder: XBuilderType,
    y_builder: YBuilderType,
    num_actual_runs: PositiveInt = 2,
    num_random_runs: PositiveInt = 10,
    native_importance: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], List[pd.DataFrame]]:
    """
    The generic compute function allows customization of the computation. It is used by the `compute` function.

    If native_importance=True, returns a tuple: (result_df, native_df).
    Otherwise returns result_df as a DataFrame (or List[pd.DataFrame] if a list of calculators was provided).

    Args:
        model_builder (ModelBuilderType): A function that return a model.
        model_fitter (ModelFitterType): A function that fits a model.
        importance_getter (ModelImportanceGetter): A function that computes the importance of a model.
        permutation_importance_calculator (Union[ PermutationImportanceCalculatorType, List[PermutationImportanceCalculatorType] ]):
            A function or list of functions that compute the final permutation importance.
        X_builder (XBuilderType): A function that returns the X data.
        y_builder (YBuilderType): A function that returns the y data.
        num_actual_runs (PositiveInt, optional): Number of actual runs. Defaults to 2.
        num_random_runs (PositiveInt, optional): Number of random runs. Defaults to 10.
        native_importance (bool, optional): Whether to also return the averaged native importance of
            the actual (non-random) runs. Defaults to False.

    Returns:
        If `permutation_importance_calculator` is a single function and `native_importance=False`,
            returns a single DataFrame with columns ["feature", "importance"].
        If `permutation_importance_calculator` is a single function and `native_importance=True`,
            returns a tuple of (result_df, native_importance_df).
        If `permutation_importance_calculator` is a list of functions and `native_importance=False`,
            returns a list of DataFrames.
        If `permutation_importance_calculator` is a list of functions and `native_importance=True`,
            returns a tuple of (list of DataFrames, native_importance_df).
    """
    run_params = {
        "model_builder": model_builder,
        "model_fitter": model_fitter,
        "importance_getter": importance_getter,
        "X_builder": X_builder,
        "y_builder": y_builder,
    }
    partial_compute_one_run = partial(_compute_one_run, **run_params)

    print(f"Running {num_actual_runs} actual runs and {num_random_runs} random runs")

    actual_importance_dfs = []
    for run_idx in tqdm(range(num_actual_runs)):
        actual_importance_dfs.append(
            partial_compute_one_run(
                is_random_run=False,
                run_idx=run_idx,
            )
        )

    random_importance_dfs = []
    for run_idx in tqdm(range(num_random_runs)):
        random_importance_dfs.append(
            partial_compute_one_run(
                is_random_run=True,
                run_idx=run_idx,
            )
        )

    # Calculate the permutation importance
    if isinstance(permutation_importance_calculator, list):
        result_df_list = [
            calc(actual_importance_dfs, random_importance_dfs)
            for calc in permutation_importance_calculator
        ]
        if native_importance:
            # Compute the average native importances across the actual runs
            all_actual_importance_df = pd.concat(actual_importance_dfs)
            native_importance_df = (
                all_actual_importance_df.groupby("feature")["importance"]
                .mean()
                .reset_index()
            )
            return (result_df_list, native_importance_df)
        return result_df_list

    result_df = permutation_importance_calculator(
        actual_importance_dfs, random_importance_dfs
    )

    if native_importance:
        # Compute the average native importances across the actual runs
        all_actual_importance_df = pd.concat(actual_importance_dfs)
        native_importance_df = (
            all_actual_importance_df.groupby("feature")["importance"].mean().reset_index()
        )
        return result_df, native_importance_df

    return result_df


def _get_feature_names_attr(model: Any):
    feature_attr = "feature_names_in_"
    if "LGBM" in str(model.__class__):
        feature_attr = "feature_name_"
    elif "Cat" in str(model.__class__):
        feature_attr = "feature_names_"
    return feature_attr


def _get_model_importances_attr(model: Any):
    if hasattr(model, "feature_importances_"):
        return "feature_importances_"
    if hasattr(model, "coef_"):
        return "coef_"
    raise NotImplementedError(  # pragma: no cover
        "Model does not have feature importances method"
    )


@beartype
def compute(
    model_cls: Any,
    model_cls_params: Dict,
    model_fit_params: Union[ModelFitParamsBuilderType, Dict],
    X: XType,
    y: YType,
    num_actual_runs: PositiveInt = 2,
    num_random_runs: PositiveInt = 10,
    shuffle_feature_order: bool = False,
    permutation_importance_calculator: Union[
        PermutationImportanceCalculatorType, List[PermutationImportanceCalculatorType]
    ] = compute_permutation_importance_by_subtraction,
    native_importance: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], List[pd.DataFrame]]:
    """
    Compute the permutation importance of a model given a dataset.

    If native_importance=True, returns a tuple: (result_df, native_df).
    Otherwise returns just result_df or a list of result DataFrames.

    Args:
        model_cls: The constructor/class of the model.
        model_cls_params: The parameters to pass to the model constructor.
        model_fit_params: A Dict or a function that returns parameters to pass to the model fit method.
        X: The input data.
        y: The target vector.
        num_actual_runs: Number of actual runs. Defaults to 2.
        num_random_runs: Number of random runs. Defaults to 10.
        shuffle_feature_order: Whether to shuffle the feature order for each run (only for X being pd.DataFrame). Defaults to False.
        permutation_importance_calculator: The function(s) to compute the final importance. Defaults to compute_permutation_importance_by_subtraction.
        native_importance (bool, optional): If True, also returns a DataFrame of averaged native importances across actual runs.

    Returns:
        If `permutation_importance_calculator` is a single function and `native_importance=False`:
            A DataFrame with columns ["feature", "importance"].
        If `permutation_importance_calculator` is a single function and `native_importance=True`:
            A tuple (result_df, native_importance_df).
        If `permutation_importance_calculator` is a list of functions and `native_importance=False`:
            A list of DataFrames.
        If `permutation_importance_calculator` is a list of functions and `native_importance=True`:
            A tuple (list of DataFrames, native_importance_df).

    Example:
        ```python
        # import the package
        import target_permutation_importances as tpi

        # Prepare a dataset
        import pandas as pd
        from sklearn.datasets import load_breast_cancer

        # Models
        from sklearn.ensemble import RandomForestClassifier

        data = load_breast_cancer()
        Xpd = pd.DataFrame(data.data, columns=data.feature_names)

        # Compute permutation importances with default settings
        result_df = tpi.compute(
            model_cls=RandomForestClassifier,
            model_cls_params={"n_jobs": -1},
            model_fit_params={},
            X=Xpd,
            y=data.target,
            num_actual_runs=2,
            num_random_runs=10,
            permutation_importance_calculator=tpi.compute_permutation_importance_by_subtraction,
        )
        print(result_df[["feature", "importance"]].sort_values("importance", ascending=False).head())
        ```
    """

    def _x_builder(is_random_run: bool, run_idx: int) -> XType:
        if shuffle_feature_order:
            if isinstance(X, pd.DataFrame):
                rng = np.random.default_rng(seed=run_idx)
                shuffled_columns = rng.permutation(X.columns)
                return X[shuffled_columns]
            raise NotImplementedError(  # pragma: no cover
                "Only support pd.DataFrame when shuffle_feature_order=True"
            )
        return X

    def _y_builder(is_random_run: bool, run_idx: int) -> YType:
        rng = np.random.default_rng(seed=run_idx)
        if is_random_run:
            # Only shuffle the target for random runs
            return rng.permutation(y)
        return y

    def _model_builder(is_random_run: bool, run_idx: int) -> Any:
        _model_cls_params = model_cls_params.copy()
        if "MultiOutput" not in model_cls.__name__:
            _model_cls_params["random_state"] = run_idx
        else:
            _model_cls_params["estimator"].random_state = run_idx
        return model_cls(**_model_cls_params)

    def _model_fitter(model: Any, X: XType, y: YType) -> Any:
        if isinstance(model_fit_params, dict):  # pragma: no cover
            _model_fit_params = model_fit_params.copy()
        else:
            _model_fit_params = model_fit_params(
                list(X.columns) if isinstance(X, pd.DataFrame) else None,
            )
        if "Cat" in str(model.__class__):
            _model_fit_params["verbose"] = False
        return model.fit(X, y, **_model_fit_params)

    def _importance_getter(model: Any, X: XType, y: YType) -> pd.DataFrame:
        # This returns the native importances of the model for a single run
        feature_names_attr = _get_feature_names_attr(model)
        is_pd = isinstance(X, pd.DataFrame)

        if "MultiOutput" not in str(model.__class__):
            if is_pd:
                features = getattr(model, feature_names_attr)
            else:
                features = list(range(0, X.shape[1]))

            model_importances_attr = _get_model_importances_attr(model)
            importances = np.abs(getattr(model, model_importances_attr))
            if len(importances.shape) > 1:
                importances = importances.mean(axis=0)
            return pd.DataFrame({"feature": features, "importance": importances})

        # MultiOutput model
        features = []
        feature_importances = np.zeros(X.shape[1])
        for est in model.estimators_:
            if is_pd:
                feature_names_attr = _get_feature_names_attr(est)
                features = getattr(est, feature_names_attr)
            else:
                features = list(range(0, X.shape[1]))

            model_importances_attr = _get_model_importances_attr(est)
            importances = np.abs(getattr(est, model_importances_attr))
            if len(importances.shape) > 1:  # pragma: no cover
                importances = importances.mean(axis=0)
            feature_importances += importances
        return pd.DataFrame(
            {
                "feature": features,
                "importance": feature_importances / len(model.estimators_),
            }
        )

    result = generic_compute(
        model_builder=_model_builder,
        model_fitter=_model_fitter,
        importance_getter=_importance_getter,
        permutation_importance_calculator=permutation_importance_calculator,
        X_builder=_x_builder,
        y_builder=_y_builder,
        num_actual_runs=num_actual_runs,
        num_random_runs=num_random_runs,
        native_importance=native_importance,
    )
    return result
