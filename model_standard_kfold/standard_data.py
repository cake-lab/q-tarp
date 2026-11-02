

import copy
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader


def load_processed_graphs(dataset_path: Path):
    """Load processed PyTorch Geometric Data objects from a pickle/npy file."""
    with open(dataset_path, "rb") as file:
        return pickle.load(file)


def prepare_graphs(graphs, max_nodes: Optional[int] = None, max_samples: Optional[int] = None):
    """Apply optional filtering and shape cleanup to loaded graph objects."""
    graphs = list(graphs)
    if max_nodes is not None:
        graphs = [data for data in graphs if int(data.x.shape[0]) <= int(max_nodes)]
    if max_samples is not None:
        graphs = graphs[: int(max_samples)]
    for data in graphs:
        if hasattr(data, "global_features") and not isinstance(data.global_features, torch.Tensor):
            data.global_features = torch.tensor(data.global_features, dtype=torch.float32)
        if hasattr(data, "global_features") and data.global_features.dim() == 1:
            data.global_features = data.global_features.unsqueeze(0)
    return graphs


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


def make_subsets(graphs, split_ratio: Sequence[float], seed: int = 1234, shuffle: bool = True):
    """Backward-compatible single-split helper retained for old experiments."""
    train_len, valid_len, test_len = split_lengths(len(graphs), split_ratio)
    generator = torch.Generator().manual_seed(seed)

    if len(split_ratio) == 2:
        train_set, heldout_set = random_split(
            graphs,
            [train_len, test_len],
            generator=generator,
        )
        subsets = {
            "train": train_set,
            "valid": heldout_set,
            "test": heldout_set,
        }
        split_idx = {
            "train": list(train_set.indices),
            "valid": list(heldout_set.indices),
            "test": list(heldout_set.indices),
        }
        return subsets, split_idx

    train_set, valid_set, test_set = random_split(
        graphs,
        [train_len, valid_len, test_len],
        generator=generator,
    )
    subsets = {
        "train": train_set,
        "valid": valid_set,
        "test": test_set,
    }
    split_idx = {
        "train": list(train_set.indices),
        "valid": list(valid_set.indices),
        "test": list(test_set.indices),
    }
    return subsets, split_idx


def make_subsets_from_indices(graphs, split_idx: Dict[str, Sequence[int]]):
    """Rebuild subsets from previously computed indices."""
    subsets = {"train": Subset(graphs, list(split_idx["train"]))}
    if "valid" in split_idx:
        subsets["valid"] = Subset(graphs, list(split_idx["valid"]))
    if "test" in split_idx:
        subsets["test"] = Subset(graphs, list(split_idx["test"]))
    return subsets


def make_index_split(num_items: int, split_ratio: Sequence[float], seed: int = 1234):
    """Create a split over integer indices instead of graph objects."""
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
    """Load metadata aligned 1:1 with the graph dataset rows."""
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


def _fit_train_only_transforms(graphs, train_indices: Sequence[int]):
    """Fit feature selection and standardization statistics on train only."""
    train_graphs = [graphs[int(index)] for index in train_indices]
    if not train_graphs:
        raise ValueError("Train split is empty; cannot fit train-only transforms.")

    x = torch.cat([data.x for data in train_graphs], dim=0)
    global_features = torch.cat(
        [data.global_features.reshape(1, -1) for data in train_graphs],
        dim=0,
    )

    keep_mask = global_features.sum(0) > 0
    filtered_global_features = global_features[:, keep_mask]

    continuous_x_cols = slice(173, 178)
    continuous_x = x[:, continuous_x_cols]

    return {
        "keep_mask": keep_mask,
        "continuous_x_cols": continuous_x_cols,
        "x_mean": continuous_x.mean(0),
        "x_std": continuous_x.std(0),
        "gf_mean": filtered_global_features.mean(0),
        "gf_std": filtered_global_features.std(0),
    }


def apply_train_only_transforms(graphs, split_idx: Dict[str, Sequence[int]]):
    """Apply train-fitted feature filtering and standardization to every split."""
    transforms = _fit_train_only_transforms(graphs, split_idx["train"])
    transformed_graphs = copy.deepcopy(list(graphs))

    for data in transformed_graphs:
        cols = transforms["continuous_x_cols"]
        data.x[:, cols] = (data.x[:, cols] - transforms["x_mean"]) / (
            1e-8 + transforms["x_std"]
        )

        global_features = data.global_features.reshape(1, -1)[:, transforms["keep_mask"]]
        data.global_features = (global_features - transforms["gf_mean"]) / (
            1e-8 + transforms["gf_std"]
        )

    return transformed_graphs


def make_loaders(subsets: Dict[str, Subset], batch_size: int, num_workers: int = 0):
    """Create PyG DataLoaders for train, validation, and test splits."""
    loaders = {
        "train": DataLoader(
            subsets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    }
    if "valid" in subsets:
        loaders["valid"] = DataLoader(
            subsets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    if "test" in subsets:
        loaders["test"] = DataLoader(
            subsets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return loaders


def infer_global_feature_dim(graphs, default: int = 41):
    """Infer the global feature dimension from the first graph object."""
    if len(graphs) == 0:
        return default
    sample = graphs[0]
    if hasattr(sample, "global_features"):
        return int(sample.global_features.reshape(-1).shape[0])
    return default
