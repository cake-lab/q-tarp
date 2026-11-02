"""Tabular dataset helpers for classical baseline models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold


def load_processed_graphs(dataset_path: Path):
    """Load processed PyTorch Geometric Data objects from a pickle/npy file."""
    with open(dataset_path, "rb") as file:
        return pickle.load(file)


def prepare_graphs(graphs, max_nodes: Optional[int] = None, max_samples: Optional[int] = None):
    """Apply optional graph-count / node-count filtering before extraction."""
    graphs = list(graphs)
    if max_nodes is not None:
        graphs = [data for data in graphs if int(data.x.shape[0]) <= int(max_nodes)]
    if max_samples is not None:
        graphs = graphs[: int(max_samples)]
    return graphs


def _to_numpy_feature_row(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1).astype(float)
    return np.asarray(value, dtype=float).reshape(-1)


def _to_float_target(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.asarray(value, dtype=float).reshape(-1)[0])
    return float(value)


def extract_tabular_dataset(graphs):
    """Extract X (global features) and y from the pickled graph dataset."""
    x_rows = []
    y_rows = []
    for graph in graphs:
        if not hasattr(graph, "global_features"):
            raise ValueError("Graph sample is missing global_features; cannot build tabular baseline dataset.")
        if not hasattr(graph, "y"):
            raise ValueError("Graph sample is missing y; cannot build tabular baseline dataset.")
        x_rows.append(_to_numpy_feature_row(graph.global_features))
        y_rows.append(_to_float_target(graph.y))

    if not x_rows:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)

    feature_dims = {row.shape[0] for row in x_rows}
    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent global feature dimension across samples: {sorted(feature_dims)}")

    x = np.vstack(x_rows).astype(float)
    y = np.asarray(y_rows, dtype=float)
    return x, y


def split_lengths(num_items: int, split_ratio: Sequence[float]):
    total_ratio = float(sum(split_ratio))
    train_ratio = float(split_ratio[0]) / total_ratio
    train_len = int(train_ratio * num_items)

    if len(split_ratio) == 2:
        valid_len = 0
        test_len = num_items - train_len
        return train_len, valid_len, test_len

    if len(split_ratio) == 3:
        valid_ratio = float(split_ratio[1]) / total_ratio
        valid_len = int(valid_ratio * num_items)
        test_len = num_items - train_len - valid_len
        return train_len, valid_len, test_len

    raise ValueError("split_ratio must have length 2 or 3")


def make_index_split(num_items: int, split_ratio: Sequence[float], seed: int = 1234):
    """Create a split over integer indices."""
    train_len, valid_len, test_len = split_lengths(num_items, split_ratio)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator).tolist()

    train_idx = perm[:train_len]
    if len(split_ratio) == 2:
        test_idx = perm[train_len : train_len + test_len]
        return {"train": train_idx, "test": test_idx}

    valid_start = train_len
    valid_end = valid_start + valid_len
    valid_idx = perm[valid_start:valid_end]
    test_idx = perm[valid_end : valid_end + test_len]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def make_kfold_splits(train_indices: Sequence[int], num_folds: int = 5, seed: int = 1234):
    """Create train/valid index maps for k-fold CV inside an outer train split."""
    train_indices = np.asarray(list(train_indices), dtype=int)
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    if len(train_indices) < num_folds:
        raise ValueError(
            f"num_folds={num_folds} is larger than the available train set size {len(train_indices)}"
        )

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_splits = []
    for fold_id, (inner_train_pos, inner_valid_pos) in enumerate(kfold.split(train_indices)):
        fold_splits.append(
            {
                "fold_id": fold_id,
                "train": train_indices[inner_train_pos].tolist(),
                "valid": train_indices[inner_valid_pos].tolist(),
            }
        )
    return fold_splits


def _derive_workload_family(name: str) -> str:
    name = Path(str(name)).name
    if "_indep_" in name:
        return name.split("_indep_", maxsplit=1)[0]
    stem = Path(name).stem
    parts = stem.split("_")
    return parts[0] if parts else stem


def load_split_metadata(metadata_csv: Path, expected_len: int):
    """Load metadata aligned 1:1 with the dataset rows."""
    metadata_df = pd.read_csv(metadata_csv).copy()
    if len(metadata_df) != int(expected_len):
        raise ValueError(
            f"Metadata row count mismatch: metadata={len(metadata_df)} dataset={expected_len}"
        )
    if "name" in metadata_df.columns and "workload_family" not in metadata_df.columns:
        metadata_df["workload_family"] = metadata_df["name"].map(_derive_workload_family)
    return metadata_df


def infer_balance_features(
    metadata_df: pd.DataFrame,
    categorical_features: Optional[Sequence[str]] = None,
    numeric_features: Optional[Sequence[str]] = None,
):
    """Infer or validate metadata features used for split-balance scoring."""
    if categorical_features is None:
        categorical_features = [
            column
            for column in ["device", "workload_family"]
            if column in metadata_df.columns
        ]
    else:
        categorical_features = [column for column in categorical_features if column in metadata_df.columns]

    if numeric_features is None:
        candidate_numeric = [
            "time_taken",
            "source_num_qubits",
            "source_depth",
            "source_total_ops",
            "source_two_qubit_ops",
            "transpiled_depth",
            "transpiled_total_ops",
            "transpiled_two_qubit_ops",
        ]
        numeric_features = [
            column
            for column in candidate_numeric
            if column in metadata_df.columns and pd.api.types.is_numeric_dtype(metadata_df[column])
        ]
    else:
        numeric_features = [
            column
            for column in numeric_features
            if column in metadata_df.columns and pd.api.types.is_numeric_dtype(metadata_df[column])
        ]

    if not categorical_features and not numeric_features:
        raise ValueError(
            "No valid balance features were found in the metadata CSV. "
            "Provide columns such as device/name/time_taken/source_depth."
        )

    return list(categorical_features), list(numeric_features)


def _categorical_balance_score(train_series: pd.Series, test_series: pd.Series) -> float:
    train_dist = train_series.fillna("__MISSING__").value_counts(normalize=True)
    test_dist = test_series.fillna("__MISSING__").value_counts(normalize=True)
    categories = train_dist.index.union(test_dist.index)
    total_variation = 0.5 * float(
        sum(abs(float(train_dist.get(category, 0.0)) - float(test_dist.get(category, 0.0))) for category in categories)
    )
    return total_variation


def _numeric_balance_score(train_series: pd.Series, test_series: pd.Series, full_series: pd.Series) -> float:
    train_values = pd.to_numeric(train_series, errors="coerce").dropna().to_numpy(dtype=float)
    test_values = pd.to_numeric(test_series, errors="coerce").dropna().to_numpy(dtype=float)
    full_values = pd.to_numeric(full_series, errors="coerce").dropna().to_numpy(dtype=float)

    if len(train_values) == 0 or len(test_values) == 0 or len(full_values) == 0:
        return 0.0

    scale = float(np.std(full_values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(full_values) - np.min(full_values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    mean_score = abs(float(np.mean(train_values)) - float(np.mean(test_values))) / scale
    std_score = abs(float(np.std(train_values)) - float(np.std(test_values))) / scale

    quantile_levels = [0.1, 0.5, 0.9]
    quantile_score = float(
        np.mean(
            [
                abs(
                    float(np.quantile(train_values, level))
                    - float(np.quantile(test_values, level))
                )
                / scale
                for level in quantile_levels
            ]
        )
    )
    return float(np.mean([mean_score, std_score, quantile_score]))


def score_split_balance(
    metadata_df: pd.DataFrame,
    split_idx: Dict[str, Sequence[int]],
    categorical_features: Optional[Sequence[str]] = None,
    numeric_features: Optional[Sequence[str]] = None,
):
    """Measure how similar train/test metadata distributions are."""
    categorical_features, numeric_features = infer_balance_features(
        metadata_df,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )
    train_df = metadata_df.iloc[list(split_idx["train"])]
    test_df = metadata_df.iloc[list(split_idx["test"])]

    categorical_scores = {}
    for column in categorical_features:
        categorical_scores[column] = _categorical_balance_score(train_df[column], test_df[column])

    numeric_scores = {}
    for column in numeric_features:
        numeric_scores[column] = _numeric_balance_score(train_df[column], test_df[column], metadata_df[column])

    all_scores = list(categorical_scores.values()) + list(numeric_scores.values())
    balance_score = float(np.mean(all_scores)) if all_scores else 0.0

    return {
        "balance_score": balance_score,
        "categorical_scores": categorical_scores,
        "numeric_scores": numeric_scores,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "train_size": len(split_idx["train"]),
        "test_size": len(split_idx["test"]),
    }


def select_best_balanced_split(
    metadata_df: pd.DataFrame,
    split_ratio: Sequence[float],
    num_candidates: int = 1000,
    seed_start: int = 0,
    categorical_features: Optional[Sequence[str]] = None,
    numeric_features: Optional[Sequence[str]] = None,
):
    """Search many candidate train/test splits and keep the most balanced one."""
    if len(split_ratio) != 2:
        raise ValueError("Best-balanced split selection expects a two-way split ratio such as [0.8, 0.2].")

    candidates = []
    best_candidate = None
    for offset in range(int(num_candidates)):
        candidate_seed = int(seed_start) + offset
        split_idx = make_index_split(len(metadata_df), split_ratio, seed=candidate_seed)
        score = score_split_balance(
            metadata_df,
            split_idx,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )
        candidate = {
            "seed": candidate_seed,
            "train_indices": list(split_idx["train"]),
            "test_indices": list(split_idx["test"]),
            **score,
        }
        candidates.append(candidate)
        if best_candidate is None or candidate["balance_score"] < best_candidate["balance_score"]:
            best_candidate = candidate

    candidates_sorted = sorted(candidates, key=lambda item: item["balance_score"])
    return best_candidate, candidates_sorted


def fit_tabular_transform(
    x_train: np.ndarray,
    drop_zero_sum_global_features: bool = True,
    scale_features: bool = False,
):
    """Fit train-only feature filtering and optional standardization."""
    if x_train.ndim != 2 or x_train.shape[0] == 0:
        raise ValueError(f"x_train must be a non-empty 2D array, got shape={x_train.shape}")

    keep_mask = np.ones(x_train.shape[1], dtype=bool)
    if drop_zero_sum_global_features:
        keep_mask = np.sum(x_train, axis=0) > 0
        if not np.any(keep_mask):
            keep_mask = np.ones(x_train.shape[1], dtype=bool)

    x_train_kept = x_train[:, keep_mask]
    mean = None
    std = None
    if scale_features:
        mean = np.mean(x_train_kept, axis=0)
        std = np.std(x_train_kept, axis=0)
        std = np.where(std <= 1e-12, 1.0, std)

    return {
        "keep_mask": keep_mask,
        "scale_features": bool(scale_features),
        "mean": mean,
        "std": std,
        "input_dim": int(x_train.shape[1]),
        "output_dim": int(x_train_kept.shape[1]),
    }


def apply_tabular_transform(x: np.ndarray, transform_bundle):
    """Apply train-fitted feature filtering and optional scaling."""
    keep_mask = np.asarray(transform_bundle["keep_mask"], dtype=bool)
    transformed = x[:, keep_mask]
    if bool(transform_bundle.get("scale_features", False)):
        transformed = (transformed - transform_bundle["mean"]) / (1e-8 + transform_bundle["std"])
    return transformed
