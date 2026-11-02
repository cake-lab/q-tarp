import argparse
import copy
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()) / "data_preparation"))

from torchpack.utils.config import configs
from helper import get_path_training_data

from baseline_metrics import regression_metrics
from baseline_models import (
    build_model_candidates,
    create_estimator,
    metric_higher_is_better,
    missing_dependency_message,
    model_dependency_available,
    score_candidate_result,
)
from tabular_data import (
    apply_tabular_transform,
    extract_tabular_dataset,
    fit_tabular_transform,
    load_processed_graphs,
    load_split_metadata,
    make_kfold_splits,
    prepare_graphs,
    select_best_balanced_split,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_git_output(repo_root: Path, *args: str):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "keys"):
        return {str(key): to_builtin(getattr(value, key)) for key in value.keys()}
    return value


def save_json(path: Path, payload):
    with open(path, "w") as file:
        json.dump(to_builtin(payload), file, indent=2)


def resolve_path(repo_root: Path, value):
    if value in [None, ""]:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path


def build_run_output_dir(output_root: Path, seed: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_tag = "-".join(str(int(float(x) * 100)) for x in configs.dataset.split_ratio)
    num_folds = int(configs.cv.num_folds)
    run_name = f"{timestamp}_split{split_tag}_kfold{num_folds}_seed{int(seed)}"
    return output_root / "baseline_runs" / run_name


def namespace_to_builtin_dict(section, exclude=None):
    exclude = set(exclude or [])
    values = {}
    if section is None or not hasattr(section, "keys"):
        return values
    for key in section.keys():
        if str(key) in exclude:
            continue
        value = getattr(section, key)
        if hasattr(value, "keys"):
            values[str(key)] = namespace_to_builtin_dict(value)
        else:
            values[str(key)] = to_builtin(value)
    return values


def get_enabled_models(cli_models: Optional[str]):
    if cli_models:
        return [item.strip() for item in cli_models.split(",") if item.strip()]
    enabled = getattr(configs.models, "enabled", [])
    return [str(item) for item in to_builtin(enabled)]


def scale_model_names():
    scale_models = getattr(configs.preprocessing, "scale_models", [])
    return {str(item) for item in to_builtin(scale_models)}


def fit_predict_fold(
    model_name: str,
    candidate_params: dict,
    x: np.ndarray,
    y: np.ndarray,
    metadata_df: Optional[pd.DataFrame],
    fold_split,
    fold_seed: int,
):
    train_indices = np.asarray(fold_split["train"], dtype=int)
    valid_indices = np.asarray(fold_split["valid"], dtype=int)

    x_train_raw = x[train_indices]
    y_train = y[train_indices]
    x_valid_raw = x[valid_indices]
    y_valid = y[valid_indices]

    transform_bundle = fit_tabular_transform(
        x_train_raw,
        drop_zero_sum_global_features=bool(getattr(configs.preprocessing, "drop_zero_sum_global_features", True)),
        scale_features=model_name in scale_model_names(),
    )
    x_train = apply_tabular_transform(x_train_raw, transform_bundle)
    x_valid = apply_tabular_transform(x_valid_raw, transform_bundle)

    estimator = create_estimator(model_name, candidate_params, seed=fold_seed)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_valid)
    metrics = regression_metrics(y_valid, y_pred)

    return {
        "metrics": {**metrics, "pred": y_pred, "target": y_valid},
        "transform_bundle": transform_bundle,
        "train_size": len(train_indices),
        "valid_size": len(valid_indices),
        "train_indices": train_indices.tolist(),
        "valid_indices": valid_indices.tolist(),
    }


def summarize_fold_metrics(fold_records):
    metric_names = ["mse", "rmse", "nmse", "r2"]
    mean_metrics = {}
    std_metrics = {}
    for metric_name in metric_names:
        values = [float(record["valid_metrics"][metric_name]) for record in fold_records]
        mean_metrics[metric_name] = float(np.mean(values))
        std_metrics[metric_name] = float(np.std(values))
    return mean_metrics, std_metrics


def save_prediction_npz(path: Path, y_true, y_pred):
    np.savez(path, pred=np.asarray(y_pred, dtype=float), target=np.asarray(y_true, dtype=float))


def save_predictions_csv(path: Path, indices, metadata_df: Optional[pd.DataFrame], y_true, y_pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_df is not None:
        df = metadata_df.iloc[list(indices)].copy().reset_index(drop=True)
    else:
        df = pd.DataFrame({"dataset_idx": list(indices)})
    df["dataset_idx"] = list(indices)
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_pred"] = np.asarray(y_pred, dtype=float)
    df["abs_err"] = np.abs(df["y_pred"] - df["y_true"])
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def build_top_level_run_metadata(
    repo_root: Path,
    config_root: Path,
    args,
    dataset_path: Path,
    metadata_path: Optional[Path],
    output_dir: Path,
    enabled_models,
    selected_split_info,
):
    git_status = safe_git_output(repo_root, "status", "--short")
    return {
        "commit_hash": safe_git_output(repo_root, "rev-parse", "HEAD"),
        "dirty": bool(git_status),
        "git_status": git_status,
        "config_text": (config_root / "config.yaml").read_text(),
        "protocol": "outer train/test split selection plus inner k-fold cross-validation for tabular baselines",
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()) if metadata_path is not None else None,
        "output_dir": str(output_dir.resolve()),
        "seed": int(args.seed),
        "split_ratio": list(configs.dataset.split_ratio),
        "num_candidate_splits": int(configs.cv.num_candidate_splits),
        "split_seed_start": int(configs.cv.split_seed_start),
        "num_folds": int(configs.cv.num_folds),
        "selection_metric": str(configs.cv.selection_metric),
        "enabled_models": enabled_models,
        "selected_split_seed": selected_split_info.get("seed"),
        "selected_split_balance_score": selected_split_info.get("balance_score"),
        "selected_split_train_size": selected_split_info.get("train_size"),
        "selected_split_test_size": selected_split_info.get("test_size"),
    }


def build_model_run_metadata(
    repo_root: Path,
    model_name: str,
    output_dir: Path,
    dataset_path: Path,
    metadata_path: Optional[Path],
    args,
    selected_split_info,
    best_candidate,
):
    git_status = safe_git_output(repo_root, "status", "--short")
    return {
        "commit_hash": safe_git_output(repo_root, "rev-parse", "HEAD"),
        "dirty": bool(git_status),
        "git_status": git_status,
        "model_name": model_name,
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()) if metadata_path is not None else None,
        "output_dir": str(output_dir.resolve()),
        "seed": int(args.seed),
        "split_ratio": list(configs.dataset.split_ratio),
        "num_candidate_splits": int(configs.cv.num_candidate_splits),
        "split_seed_start": int(configs.cv.split_seed_start),
        "num_folds": int(configs.cv.num_folds),
        "selection_metric": str(configs.cv.selection_metric),
        "selected_split_seed": selected_split_info.get("seed"),
        "selected_split_balance_score": selected_split_info.get("balance_score"),
        "selected_split_train_size": selected_split_info.get("train_size"),
        "selected_split_test_size": selected_split_info.get("test_size"),
        "best_hyperparameters": best_candidate["params"],
        "cv_mean_valid_metrics": best_candidate["mean_valid_metrics"],
        "cv_std_valid_metrics": best_candidate["std_valid_metrics"],
    }


def load_outer_split(split_path: Path):
    with open(split_path, "r") as file:
        payload = json.load(file)
    if "train" not in payload or "test" not in payload:
        raise ValueError("Split file must contain 'train' and 'test' index lists.")
    return {
        "seed": payload.get("seed"),
        "balance_score": payload.get("balance_score"),
        "categorical_scores": payload.get("categorical_scores", {}),
        "numeric_scores": payload.get("numeric_scores", {}),
        "categorical_features": payload.get("categorical_features", []),
        "numeric_features": payload.get("numeric_features", []),
        "train_size": len(payload["train"]),
        "test_size": len(payload["test"]),
        "train_indices": list(payload["train"]),
        "test_indices": list(payload["test"]),
        "source": str(split_path.resolve()),
    }


def select_best_candidate(candidate_results, metric_name: str):
    if not candidate_results:
        raise ValueError("No hyperparameter candidates were evaluated.")
    return max(candidate_results, key=lambda item: score_candidate_result(item, metric_name))


def sanitize_candidate_rankings(candidate_results):
    sanitized = []
    for item in candidate_results:
        sanitized.append(
            {
                "candidate_id": item["candidate_id"],
                "params": item["params"],
                "mean_valid_metrics": item["mean_valid_metrics"],
                "std_valid_metrics": item["std_valid_metrics"],
                "output_dir": item["output_dir"],
            }
        )
    return sanitized


def final_train_test(
    model_name: str,
    best_params: dict,
    x: np.ndarray,
    y: np.ndarray,
    outer_split_idx,
    final_seed: int,
):
    train_indices = np.asarray(outer_split_idx["train"], dtype=int)
    test_indices = np.asarray(outer_split_idx["test"], dtype=int)

    x_train_raw = x[train_indices]
    y_train = y[train_indices]
    x_test_raw = x[test_indices]
    y_test = y[test_indices]

    transform_bundle = fit_tabular_transform(
        x_train_raw,
        drop_zero_sum_global_features=bool(getattr(configs.preprocessing, "drop_zero_sum_global_features", True)),
        scale_features=model_name in scale_model_names(),
    )
    x_train = apply_tabular_transform(x_train_raw, transform_bundle)
    x_test = apply_tabular_transform(x_test_raw, transform_bundle)

    estimator = create_estimator(model_name, best_params, seed=int(final_seed))
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    metrics = regression_metrics(y_test, y_pred)
    return {
        "metrics": {**metrics, "pred": y_pred, "target": y_test},
        "transform_bundle": transform_bundle,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_size": len(train_indices),
        "test_size": len(test_indices),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed for CV and final training.")
    parser.add_argument("--dataset-file", default="training_data_raw_with_transpiled.npy", help="Processed graph dataset filename.")
    parser.add_argument("--metadata-csv", default=None, help="CSV aligned with dataset rows for split-balance scoring.")
    parser.add_argument("--split-file", default=None, help="Optional JSON file with precomputed train/test indices.")
    parser.add_argument("--models", default=None, help="Optional comma-separated subset of models to run.")
    args, opts = parser.parse_known_args()

    set_seed(args.seed)
    repo_root = Path(os.getcwd())
    config_root = repo_root / "model_tabular_baselines"
    output_root = repo_root / "model_tabular_baselines"
    configs.load(str(config_root / "config.yaml"), recursive=False)
    configs.update(opts)

    enabled_models = get_enabled_models(args.models)
    output_dir = build_run_output_dir(output_root, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(get_path_training_data()) / args.dataset_file
    metadata_path = resolve_path(
        repo_root,
        args.metadata_csv if args.metadata_csv is not None else getattr(configs.dataset, "metadata_csv", None),
    )
    split_path = resolve_path(repo_root, args.split_file)

    print(
        f"[setup] dataset={dataset_path} seed={int(args.seed)} split_ratio={list(configs.dataset.split_ratio)} "
        f"folds={int(configs.cv.num_folds)} models={enabled_models}",
        flush=True,
    )

    graphs = load_processed_graphs(dataset_path)
    graphs = prepare_graphs(
        graphs,
        max_nodes=getattr(getattr(configs, "dataset", None), "max_nodes", None),
        max_samples=getattr(getattr(configs, "dataset", None), "max_samples", None),
    )
    x, y = extract_tabular_dataset(graphs)
    print(f"[data] loaded {len(graphs)} graphs -> X={x.shape} y={y.shape}", flush=True)

    if split_path is not None:
        outer_split_info = load_outer_split(split_path)
        metadata_df = (
            load_split_metadata(metadata_path, expected_len=len(graphs))
            if metadata_path is not None
            else None
        )
        save_json(
            output_dir / "split_selection.json",
            {
                "source": "provided_split_file",
                "path": str(split_path.resolve()),
                "best_split": {
                    key: value
                    for key, value in outer_split_info.items()
                    if key not in ["train_indices", "test_indices"]
                },
                "train": outer_split_info["train_indices"],
                "test": outer_split_info["test_indices"],
            },
        )
    else:
        if metadata_path is None:
            raise ValueError("metadata CSV is required unless --split-file is provided.")
        metadata_df = load_split_metadata(metadata_path, expected_len=len(graphs))
        outer_split_info, ranked_candidates = select_best_balanced_split(
            metadata_df,
            split_ratio=configs.dataset.split_ratio,
            num_candidates=int(configs.cv.num_candidate_splits),
            seed_start=int(configs.cv.split_seed_start),
            categorical_features=getattr(configs.dataset, "categorical_balance_features", None),
            numeric_features=getattr(configs.dataset, "numeric_balance_features", None),
        )
        save_json(
            output_dir / "split_selection.json",
            {
                "source": "balanced_split_search",
                "best_split": {
                    key: value
                    for key, value in outer_split_info.items()
                    if key not in ["train_indices", "test_indices"]
                },
                "train": outer_split_info["train_indices"],
                "test": outer_split_info["test_indices"],
                "top_candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in ["train_indices", "test_indices"]
                    }
                    for candidate in ranked_candidates[: min(20, len(ranked_candidates))]
                ],
            },
        )

    outer_split_idx = {
        "train": list(outer_split_info["train_indices"]),
        "test": list(outer_split_info["test_indices"]),
    }
    save_json(output_dir / "split_indices.json", outer_split_idx)

    fold_splits = make_kfold_splits(
        outer_split_idx["train"],
        num_folds=int(configs.cv.num_folds),
        seed=int(args.seed),
    )
    save_json(output_dir / "fold_indices.json", fold_splits)

    top_level_metadata = build_top_level_run_metadata(
        repo_root=repo_root,
        config_root=config_root,
        args=args,
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        enabled_models=enabled_models,
        selected_split_info=outer_split_info,
    )
    save_json(output_dir / "run_metadata.json", top_level_metadata)

    summary_rows = []
    selection_metric = str(configs.cv.selection_metric)
    skip_unavailable = bool(getattr(configs.runtime, "skip_unavailable_models", True))

    for model_name in enabled_models:
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[model] starting {model_name}", flush=True)

        if not hasattr(configs.models, model_name):
            raise ValueError(f"Model '{model_name}' is not defined in model_tabular_baselines/config.yaml")

        if not model_dependency_available(model_name):
            reason = missing_dependency_message(model_name)
            payload = {
                "model_name": model_name,
                "status": "skipped_unavailable_dependency",
                "reason": reason,
            }
            save_json(model_dir / "run_metadata.json", payload)
            summary_rows.append({"model_name": model_name, "status": "skipped_unavailable_dependency", "reason": reason})
            print(f"[model] skipping {model_name}: {reason}", flush=True)
            if not skip_unavailable:
                raise ImportError(reason)
            continue

        model_config = getattr(configs.models, model_name)
        model_candidates = build_model_candidates(model_name, model_config)
        save_json(
            model_dir / "search_space.json",
            {
                "model_name": model_name,
                "num_candidates": len(model_candidates),
                "candidates": model_candidates,
            },
        )

        candidate_results = []
        for candidate in model_candidates:
            candidate_id = int(candidate["candidate_id"])
            candidate_dir = model_dir / "cv_runs" / f"candidate_{candidate_id:03d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            print(f"[cv] model={model_name} candidate={candidate_id} params={candidate['params']}", flush=True)

            fold_records = []
            for fold_split in fold_splits:
                fold_id = int(fold_split["fold_id"])
                fold_seed = int(args.seed) + fold_id
                fold_dir = candidate_dir / f"fold_{fold_id:02d}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                fold_result = fit_predict_fold(
                    model_name=model_name,
                    candidate_params=candidate["params"],
                    x=x,
                    y=y,
                    metadata_df=metadata_df,
                    fold_split=fold_split,
                    fold_seed=fold_seed,
                )
                safe_valid_metrics = {
                    key: value
                    for key, value in fold_result["metrics"].items()
                    if key not in ["pred", "target"]
                }
                save_json(fold_dir / "valid_metrics.json", safe_valid_metrics)
                save_prediction_npz(
                    fold_dir / "valid_predictions.npz",
                    fold_result["metrics"]["target"],
                    fold_result["metrics"]["pred"],
                )
                if metadata_df is not None:
                    save_predictions_csv(
                        fold_dir / "valid_predictions.csv",
                        fold_result["valid_indices"],
                        metadata_df,
                        fold_result["metrics"]["target"],
                        fold_result["metrics"]["pred"],
                    )
                save_json(
                    fold_dir / "run_metadata.json",
                    {
                        "model_name": model_name,
                        "candidate_id": candidate_id,
                        "hyperparameters": candidate["params"],
                        "fold_id": fold_id,
                        "fold_seed": fold_seed,
                        "train_size": fold_result["train_size"],
                        "valid_size": fold_result["valid_size"],
                        "transform_bundle": fold_result["transform_bundle"],
                    },
                )
                fold_records.append(
                    {
                        "fold_id": fold_id,
                        "fold_seed": fold_seed,
                        "train_size": fold_result["train_size"],
                        "valid_size": fold_result["valid_size"],
                        "valid_metrics": safe_valid_metrics,
                        "output_dir": str(fold_dir.resolve()),
                    }
                )

            mean_valid_metrics, std_valid_metrics = summarize_fold_metrics(fold_records)
            candidate_result = {
                "candidate_id": candidate_id,
                "params": candidate["params"],
                "mean_valid_metrics": mean_valid_metrics,
                "std_valid_metrics": std_valid_metrics,
                "fold_records": fold_records,
                "output_dir": str(candidate_dir.resolve()),
            }
            save_json(candidate_dir / "candidate_summary.json", candidate_result)
            candidate_results.append(candidate_result)
            print(
                f"[cv] model={model_name} candidate={candidate_id} "
                f"mean_valid_{selection_metric}={mean_valid_metrics[selection_metric]:.6f}",
                flush=True,
            )

        best_candidate = select_best_candidate(candidate_results, selection_metric)
        sorted_candidate_results = sorted(
            candidate_results,
            key=lambda item: score_candidate_result(item, selection_metric),
            reverse=True,
        )
        save_json(model_dir / "cv_results.json", sanitize_candidate_rankings(sorted_candidate_results))
        save_json(model_dir / "best_hparams.json", best_candidate)

        print(
            f"[final] model={model_name} selected candidate={best_candidate['candidate_id']} "
            f"mean_valid_{selection_metric}={best_candidate['mean_valid_metrics'][selection_metric]:.6f}",
            flush=True,
        )

        final_result = final_train_test(
            model_name=model_name,
            best_params=best_candidate["params"],
            x=x,
            y=y,
            outer_split_idx=outer_split_idx,
            final_seed=int(args.seed),
        )
        safe_test_metrics = {
            key: value
            for key, value in final_result["metrics"].items()
            if key not in ["pred", "target"]
        }
        save_json(model_dir / "test_metrics.json", safe_test_metrics)
        save_prediction_npz(
            model_dir / "test_predictions.npz",
            final_result["metrics"]["target"],
            final_result["metrics"]["pred"],
        )
        if bool(getattr(configs.output, "save_test_predictions_csv", True)):
            save_predictions_csv(
                model_dir / "test_predictions.csv",
                final_result["test_indices"],
                metadata_df,
                final_result["metrics"]["target"],
                final_result["metrics"]["pred"],
            )
        save_json(
            model_dir / "run_metadata.json",
            build_model_run_metadata(
                repo_root=repo_root,
                model_name=model_name,
                output_dir=model_dir,
                dataset_path=dataset_path,
                metadata_path=metadata_path,
                args=args,
                selected_split_info=outer_split_info,
                best_candidate=best_candidate,
            ),
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "status": "completed",
                "candidate_id": int(best_candidate["candidate_id"]),
                **safe_test_metrics,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "test_metrics_summary.csv", index=False)
    save_json(output_dir / "test_metrics_summary.json", summary_rows)


if __name__ == "__main__":
    main()
