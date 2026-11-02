"""Model registry and hyperparameter search helpers for tabular baselines."""

from __future__ import annotations

import itertools
from importlib.util import find_spec

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

try:
    if find_spec("xgboost") is not None:
        from xgboost import XGBRegressor
    else:
        XGBRegressor = None
except Exception:
    XGBRegressor = None


SCALING_REQUIRED = {"linear_regression", "ridge", "svr", "xgboost", "mlp"}


def parse_hidden_layer_sizes(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return (100,)
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, (int, float)):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        parsed = tuple(int(item) for item in value)
        return parsed if parsed else (100,)
    raise ValueError(f"Unsupported hidden_layer_sizes value: {value!r}")


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def namespace_to_dict(section, exclude=None):
    """Convert a torchpack namespace-like object into a plain dict."""
    exclude = set(exclude or [])
    values = {}
    if section is None or not hasattr(section, "keys"):
        return values
    for key in section.keys():
        if str(key) in exclude:
            continue
        value = getattr(section, key)
        if hasattr(value, "keys"):
            values[str(key)] = namespace_to_dict(value)
        else:
            values[str(key)] = to_builtin(value)
    return values


def model_dependency_available(model_name: str):
    if model_name == "xgboost":
        return XGBRegressor is not None
    return True


def missing_dependency_message(model_name: str):
    if model_name == "xgboost":
        return "xgboost package is not installed in the current environment."
    return None


def build_model_candidates(model_name: str, model_config):
    """Enumerate all hyperparameter candidates for one model."""
    base_params = namespace_to_dict(model_config, exclude={"search_space"})
    search_space = getattr(model_config, "search_space", None)
    if search_space is None or not hasattr(search_space, "keys"):
        return [{"candidate_id": 0, "params": base_params}]

    search_values = {}
    for key in search_space.keys():
        value = getattr(search_space, key)
        if isinstance(value, (list, tuple)):
            search_values[str(key)] = [to_builtin(item) for item in value]
        else:
            search_values[str(key)] = [to_builtin(value)]

    if not search_values:
        return [{"candidate_id": 0, "params": base_params}]

    candidate_keys = sorted(search_values.keys())
    candidates = []
    for candidate_id, combo in enumerate(
        itertools.product(*(search_values[key] for key in candidate_keys))
    ):
        params = dict(base_params)
        params.update(
            {
                candidate_keys[position]: combo[position]
                for position in range(len(candidate_keys))
            }
        )
        candidates.append({"candidate_id": candidate_id, "params": params})
    return candidates


def metric_higher_is_better(metric_name: str):
    return metric_name.lower() in {"r2"}


def score_candidate_result(candidate_result, metric_name: str):
    metric_value = float(candidate_result["mean_valid_metrics"][metric_name])
    if metric_higher_is_better(metric_name):
        return metric_value
    return -metric_value


def create_estimator(model_name: str, params: dict, seed: int):
    """Instantiate one scikit-learn-style regressor."""
    params = dict(params)

    if model_name == "linear_regression":
        return LinearRegression(
            fit_intercept=bool(params.get("fit_intercept", True)),
        )

    if model_name == "ridge":
        return Ridge(
            alpha=float(params.get("alpha", 1.0)),
            fit_intercept=bool(params.get("fit_intercept", True)),
        )

    if model_name == "random_forest":
        max_depth = params.get("max_depth", None)
        if max_depth in ["null", "None"]:
            max_depth = None
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=None if max_depth is None else int(max_depth),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(seed),
            n_jobs=int(params.get("n_jobs", 1)),
        )

    if model_name == "svr":
        return SVR(
            kernel=str(params.get("kernel", "rbf")),
            C=float(params.get("C", 1.0)),
            epsilon=float(params.get("epsilon", 0.1)),
            gamma=params.get("gamma", "scale"),
        )

    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=parse_hidden_layer_sizes(params.get("hidden_layer_sizes", "128")),
            activation=str(params.get("activation", "relu")),
            alpha=float(params.get("alpha", 1.0e-4)),
            learning_rate_init=float(params.get("learning_rate_init", 1.0e-3)),
            max_iter=int(params.get("max_iter", 300)),
            random_state=int(seed),
        )

    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError(missing_dependency_message(model_name))
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            random_state=int(seed),
            n_jobs=int(params.get("n_jobs", 1)),
            tree_method="hist",
        )

    raise NotImplementedError(f"Unsupported model: {model_name}")
