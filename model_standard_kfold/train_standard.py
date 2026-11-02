import argparse
import copy
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()) / "model"))
sys.path.append(str(Path(os.getcwd()) / "data_preparation"))

from torchpack.utils.config import configs
from helper import get_path_training_data

from standard_data import (
    apply_train_only_transforms,
    infer_global_feature_dim,
    load_processed_graphs,
    load_split_metadata,
    make_kfold_splits,
    make_loaders,
    make_subsets_from_indices,
    prepare_graphs,
    select_best_balanced_split,
)
from standard_transformer_model import StandardTransformerRuntimeModel
from standard_trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_device(device_name: str):
    if device_name == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def flatten_config_values(section, prefix: str = ""):
    values = {}
    if section is None or not hasattr(section, "keys"):
        return values
    for key in section.keys():
        value = getattr(section, key)
        path = f"{prefix}.{key}" if prefix else str(key)
        if hasattr(value, "keys"):
            values.update(flatten_config_values(value, prefix=path))
        else:
            values[path] = to_builtin(value)
    return values


def flatten_search_space(section, prefix: str = ""):
    values = {}
    if section is None or not hasattr(section, "keys"):
        return values
    for key in section.keys():
        value = getattr(section, key)
        path = f"{prefix}.{key}" if prefix else str(key)
        if hasattr(value, "keys"):
            values.update(flatten_search_space(value, prefix=path))
        else:
            if isinstance(value, (list, tuple)):
                values[path] = [to_builtin(item) for item in value]
            else:
                values[path] = [to_builtin(value)]
    return values


def config_to_namespace(section):
    data = {}
    if section is None or not hasattr(section, "keys"):
        return SimpleNamespace()
    for key in section.keys():
        value = getattr(section, key)
        if hasattr(value, "keys"):
            data[str(key)] = config_to_namespace(value)
        else:
            data[str(key)] = to_builtin(value)
    return SimpleNamespace(**data)


def set_namespace_attr(namespace_obj, path: str, value):
    parts = path.split(".")
    current = namespace_obj
    for part in parts[:-1]:
        if not hasattr(current, part) or getattr(current, part) is None:
            setattr(current, part, SimpleNamespace())
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def section_values(flat_values, section_name: str):
    prefix = f"{section_name}."
    return {
        key[len(prefix):]: value
        for key, value in flat_values.items()
        if key.startswith(prefix)
    }


def build_hparam_candidates():
    base_values = {
        "batch_size": int(configs.batch_size),
        "num_epochs": int(configs.num_epochs),
    }
    base_values.update(flatten_config_values(configs.model, prefix="model"))
    base_values.update(flatten_config_values(configs.optimizer, prefix="optimizer"))
    base_values.update(flatten_config_values(configs.scheduler, prefix="scheduler"))

    search_values = flatten_search_space(getattr(configs, "search_space", None))
    if not search_values:
        return [{"candidate_id": 0, "overrides": {}, "values": base_values}]

    override_keys = sorted(search_values.keys())
    candidates = []
    for candidate_id, combo in enumerate(
        itertools.product(*(search_values[key] for key in override_keys))
    ):
        overrides = {
            key: combo[position]
            for position, key in enumerate(override_keys)
        }
        resolved = copy.deepcopy(base_values)
        resolved.update(overrides)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "overrides": overrides,
                "values": resolved,
            }
        )
    return candidates


def build_model_config(candidate_values, debug_trace: bool, debug_max_forwards: int):
    model_config = config_to_namespace(configs.model)
    for key, value in section_values(candidate_values, "model").items():
        set_namespace_attr(model_config, key, value)
    model_config.debug_trace = bool(debug_trace)
    model_config.debug_max_forwards = int(debug_max_forwards)
    return model_config


def make_optimizer(model, optimizer_values):
    optimizer_name = str(optimizer_values["name"])
    lr = optimizer_values["lr"]
    if isinstance(lr, str):
        lr = eval(lr)
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=float(optimizer_values.get("weight_decay", 0.0)),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(optimizer_values.get("weight_decay", 0.0)),
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optimizer_values.get("momentum", 0.0)),
            weight_decay=float(optimizer_values.get("weight_decay", 0.0)),
        )
    raise NotImplementedError(f"Unsupported optimizer: {optimizer_name}")


def make_scheduler(optimizer, scheduler_values, num_epochs: int):
    scheduler_name = scheduler_values.get("name")
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(num_epochs),
            eta_min=0,
        )
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0,
        )
    if scheduler_name in ["none", None]:
        return None
    raise NotImplementedError(f"Unsupported scheduler: {scheduler_name}")


def build_run_output_dir(output_root: Path, seed: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_tag = "-".join(str(int(float(x) * 100)) for x in configs.dataset.split_ratio)
    num_folds = int(configs.cv.num_folds)
    run_name = f"{timestamp}_split{split_tag}_kfold{num_folds}_seed{int(seed)}"
    return output_root / "standard_runs" / run_name


def metric_higher_is_better(metric_name: str):
    return metric_name.lower() in {"r2"}


def summarize_fold_metrics(fold_records):
    metric_names = ["mse", "rmse", "nmse", "r2"]
    mean_metrics = {}
    std_metrics = {}
    for metric_name in metric_names:
        values = [float(record["valid_metrics"][metric_name]) for record in fold_records]
        mean_metrics[metric_name] = float(np.mean(values))
        std_metrics[metric_name] = float(np.std(values))
    return mean_metrics, std_metrics


def score_candidate_result(candidate_result, metric_name: str):
    metric_value = float(candidate_result["mean_valid_metrics"][metric_name])
    if metric_higher_is_better(metric_name):
        return metric_value
    return -metric_value


def save_prediction_npz(path: Path, metrics):
    np.savez(path, pred=metrics["pred"], target=metrics["target"])


def build_run_metadata(
    repo_root: Path,
    config_root: Path,
    args,
    dataset_path: Path,
    metadata_path: Path,
    output_dir: Path,
    device,
    selected_split_info,
    best_candidate,
):
    git_status = safe_git_output(repo_root, "status", "--short")
    return {
        "commit_hash": safe_git_output(repo_root, "rev-parse", "HEAD"),
        "dirty": bool(git_status),
        "git_status": git_status,
        "config_text": (config_root / "config.yaml").read_text(),
        "protocol": "outer train/test split selection plus inner k-fold cross-validation",
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()) if metadata_path is not None else None,
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "split_ratio": list(configs.dataset.split_ratio),
        "num_candidate_splits": int(configs.cv.num_candidate_splits),
        "split_seed_start": int(configs.cv.split_seed_start),
        "num_folds": int(configs.cv.num_folds),
        "selection_metric": str(configs.cv.selection_metric),
        "split_source": selected_split_info.get("source", "balanced_split_search"),
        "selected_split_seed": selected_split_info.get("seed"),
        "selected_split_balance_score": selected_split_info.get("balance_score"),
        "selected_split_train_size": selected_split_info.get("train_size"),
        "selected_split_test_size": selected_split_info.get("test_size"),
        "best_hyperparameters": best_candidate["values"],
        "cv_mean_valid_metrics": best_candidate["mean_valid_metrics"],
        "cv_std_valid_metrics": best_candidate["std_valid_metrics"],
    }


def write_fold_metadata(
    fold_dir: Path,
    repo_root: Path,
    dataset_path: Path,
    metadata_path: Path,
    device,
    outer_split_seed,
    candidate,
    fold_split,
    fold_seed,
):
    git_status = safe_git_output(repo_root, "status", "--short")
    payload = {
        "commit_hash": safe_git_output(repo_root, "rev-parse", "HEAD"),
        "dirty": bool(git_status),
        "git_status": git_status,
        "dataset_path": str(dataset_path.resolve()),
        "metadata_path": str(metadata_path.resolve()) if metadata_path is not None else None,
        "device": str(device),
        "outer_split_seed": outer_split_seed,
        "candidate_id": int(candidate["candidate_id"]),
        "hyperparameters": candidate["values"],
        "fold_id": int(fold_split["fold_id"]),
        "fold_seed": int(fold_seed),
        "train_size": len(fold_split["train"]),
        "valid_size": len(fold_split["valid"]),
    }
    save_json(fold_dir / "run_metadata.json", payload)


def run_cv_candidate(
    graphs,
    dataset_path: Path,
    metadata_path: Path,
    repo_root: Path,
    device,
    output_dir: Path,
    outer_split_info,
    fold_splits,
    candidate,
    args,
):
    candidate_id = int(candidate["candidate_id"])
    candidate_dir = output_dir / "cv_runs" / f"candidate_{candidate_id:03d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    fold_records = []
    for fold_split in fold_splits:
        fold_id = int(fold_split["fold_id"])
        fold_seed = int(args.seed) + fold_id
        fold_dir = candidate_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[cv] candidate={candidate_id} fold={fold_id} "
            f"train={len(fold_split['train'])} valid={len(fold_split['valid'])}",
            flush=True,
        )

        set_seed(fold_seed)
        transformed_graphs = apply_train_only_transforms(graphs, fold_split)
        subsets = make_subsets_from_indices(transformed_graphs, fold_split)
        loaders = make_loaders(
            subsets,
            batch_size=int(candidate["values"]["batch_size"]),
            num_workers=0,
        )
        length_of_gf = infer_global_feature_dim(transformed_graphs)
        model_config = build_model_config(
            candidate["values"],
            debug_trace=args.debug_trace,
            debug_max_forwards=args.debug_max_forwards,
        )
        model = StandardTransformerRuntimeModel(
            model_config,
            length_of_gf=length_of_gf,
        ).to(device)
        optimizer = make_optimizer(model, section_values(candidate["values"], "optimizer"))
        scheduler = make_scheduler(
            optimizer,
            section_values(candidate["values"], "scheduler"),
            int(candidate["values"]["num_epochs"]),
        )
        # Per-fold TensorBoard event files are expensive on the cluster file system
        # and are not needed for split selection or final reporting.
        writer = None
        trainer = Trainer(
            model,
            torch.nn.MSELoss(),
            optimizer,
            scheduler,
            device,
            fold_dir,
            writer=writer,
            debug_trace=args.debug_trace,
        )
        write_fold_metadata(
            fold_dir=fold_dir,
            repo_root=repo_root,
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            device=device,
            outer_split_seed=outer_split_info.get("seed"),
            candidate=candidate,
            fold_split=fold_split,
            fold_seed=fold_seed,
        )
        trainer.fit(loaders["train"], loaders["valid"], int(candidate["values"]["num_epochs"]))
        valid_metrics = trainer.evaluate(loaders["valid"])
        safe_valid_metrics = {
            key: value
            for key, value in valid_metrics.items()
            if key not in ["pred", "target"]
        }
        save_json(fold_dir / "valid_metrics.json", safe_valid_metrics)
        save_prediction_npz(fold_dir / "valid_predictions.npz", valid_metrics)
        trainer.close_writer()

        fold_records.append(
            {
                "fold_id": fold_id,
                "fold_seed": fold_seed,
                "train_size": len(fold_split["train"]),
                "valid_size": len(fold_split["valid"]),
                "valid_metrics": safe_valid_metrics,
                "output_dir": str(fold_dir.resolve()),
            }
        )

    mean_valid_metrics, std_valid_metrics = summarize_fold_metrics(fold_records)
    result = {
        "candidate_id": candidate_id,
        "overrides": candidate["overrides"],
        "values": candidate["values"],
        "mean_valid_metrics": mean_valid_metrics,
        "std_valid_metrics": std_valid_metrics,
        "fold_records": fold_records,
        "output_dir": str(candidate_dir.resolve()),
    }
    save_json(candidate_dir / "candidate_summary.json", result)
    return result


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
                "values": item["values"],
                "mean_valid_metrics": item["mean_valid_metrics"],
                "std_valid_metrics": item["std_valid_metrics"],
                "output_dir": item["output_dir"],
            }
        )
    return sanitized


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed for CV and final training.")
    parser.add_argument("--dataset-file", default="training_data_raw_with_transpiled.npy", help="Processed graph dataset filename.")
    parser.add_argument("--metadata-csv", default=None, help="CSV aligned with dataset rows for split-balance scoring.")
    parser.add_argument("--split-file", default=None, help="Optional JSON file with precomputed train/test indices.")
    parser.add_argument("--debug-trace", action="store_true", help="Print detailed model/data traces.")
    parser.add_argument("--debug-max-forwards", type=int, default=2, help="Max forward passes to trace.")
    args, opts = parser.parse_known_args()

    set_seed(args.seed)
    repo_root = Path(os.getcwd())
    config_root = repo_root / "model_standard_kfold"
    output_root = repo_root / "model_standard_kfold"
    configs.load(str(config_root / "config.yaml"), recursive=False)
    configs.update(opts)

    output_dir = build_run_output_dir(output_root, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = make_device(configs.device)
    dataset_path = Path(get_path_training_data()) / args.dataset_file
    metadata_path = resolve_path(
        repo_root,
        args.metadata_csv if args.metadata_csv is not None else getattr(configs.dataset, "metadata_csv", None),
    )
    split_path = resolve_path(repo_root, args.split_file)

    print(
        f"[setup] device={device} dataset={dataset_path} seed={int(args.seed)} "
        f"split_ratio={list(configs.dataset.split_ratio)} folds={int(configs.cv.num_folds)}",
        flush=True,
    )

    graphs = load_processed_graphs(dataset_path)
    graphs = prepare_graphs(
        graphs,
        max_nodes=getattr(getattr(configs, "dataset", None), "max_nodes", None),
        max_samples=getattr(getattr(configs, "dataset", None), "max_samples", None),
    )
    print(f"[data] loaded and prepared {len(graphs)} graphs", flush=True)

    if split_path is not None:
        outer_split_info = load_outer_split(split_path)
        metadata_df = None
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
            raise ValueError(
                "metadata CSV is required unless --split-file is provided."
            )
        metadata_df = load_split_metadata(metadata_path, expected_len=len(graphs))
        outer_split_info, ranked_candidates = select_best_balanced_split(
            metadata_df,
            split_ratio=configs.dataset.split_ratio,
            num_candidates=int(configs.cv.num_candidate_splits),
            seed_start=int(configs.cv.split_seed_start),
            categorical_features=getattr(configs.dataset, "categorical_balance_features", None),
            numeric_features=getattr(configs.dataset, "numeric_balance_features", None),
        )
        split_selection_payload = {
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
        }
        save_json(output_dir / "split_selection.json", split_selection_payload)

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

    hyperparameter_candidates = build_hparam_candidates()
    save_json(
        output_dir / "search_space.json",
        {
            "num_candidates": len(hyperparameter_candidates),
            "candidates": hyperparameter_candidates,
        },
    )

    selection_metric = str(configs.cv.selection_metric)
    candidate_results = []
    for candidate in hyperparameter_candidates:
        print(
            f"[cv] candidate={candidate['candidate_id']} values={candidate['values']}",
            flush=True,
        )
        candidate_result = run_cv_candidate(
            graphs=graphs,
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            repo_root=repo_root,
            device=device,
            output_dir=output_dir,
            outer_split_info=outer_split_info,
            fold_splits=fold_splits,
            candidate=candidate,
            args=args,
        )
        print(
            f"[cv] candidate={candidate['candidate_id']} "
            f"mean_valid_{selection_metric}={candidate_result['mean_valid_metrics'][selection_metric]:.6f}",
            flush=True,
        )
        candidate_results.append(candidate_result)

    best_candidate = select_best_candidate(candidate_results, selection_metric)
    sorted_candidate_results = sorted(
        candidate_results,
        key=lambda item: score_candidate_result(item, selection_metric),
        reverse=True,
    )
    save_json(output_dir / "cv_results.json", sanitize_candidate_rankings(sorted_candidate_results))
    save_json(output_dir / "best_hparams.json", best_candidate)

    print(
        f"[final] selected candidate={best_candidate['candidate_id']} "
        f"mean_valid_{selection_metric}={best_candidate['mean_valid_metrics'][selection_metric]:.6f}",
        flush=True,
    )

    final_split = {
        "train": list(outer_split_idx["train"]),
        "test": list(outer_split_idx["test"]),
    }
    set_seed(args.seed)
    final_graphs = apply_train_only_transforms(graphs, final_split)
    final_subsets = make_subsets_from_indices(final_graphs, final_split)
    final_loaders = make_loaders(
        final_subsets,
        batch_size=int(best_candidate["values"]["batch_size"]),
        num_workers=0,
    )
    length_of_gf = infer_global_feature_dim(final_graphs)
    final_model_config = build_model_config(
        best_candidate["values"],
        debug_trace=args.debug_trace,
        debug_max_forwards=args.debug_max_forwards,
    )
    model = StandardTransformerRuntimeModel(
        final_model_config,
        length_of_gf=length_of_gf,
    ).to(device)
    optimizer = make_optimizer(model, section_values(best_candidate["values"], "optimizer"))
    scheduler = make_scheduler(
        optimizer,
        section_values(best_candidate["values"], "scheduler"),
        int(best_candidate["values"]["num_epochs"]),
    )
    writer = (
        SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if SummaryWriter is not None
        else None
    )
    trainer = Trainer(
        model,
        torch.nn.MSELoss(),
        optimizer,
        scheduler,
        device,
        output_dir,
        writer=writer,
        debug_trace=args.debug_trace,
    )
    trainer.fit_fixed_epochs(final_loaders["train"], int(best_candidate["values"]["num_epochs"]))
    trainer.test_and_save(final_loaders["test"])
    trainer.close_writer()

    run_metadata = build_run_metadata(
        repo_root=repo_root,
        config_root=config_root,
        args=args,
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        device=device,
        selected_split_info=outer_split_info,
        best_candidate=best_candidate,
    )
    save_json(output_dir / "run_metadata.json", run_metadata)


if __name__ == "__main__":
    main()
