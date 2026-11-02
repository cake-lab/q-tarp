#!/usr/bin/env python3

"""Build a PyG dataset from transpiled QASM files.

This builder creates graph samples where:
- the graph comes from the transpiled circuit DAG; and
- global features are computed from that same transpiled circuit.

It does not modify the existing source-graph pipeline. It produces a new
pickled dataset file that can be consumed by the copied k-fold trainer in
``model_standard_kfold_transpiled_graph``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def parse_backend_qasm_dirs(repo_root: Path, items):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --backend-qasm-dir value '{item}'. Expected format backend=path."
            )
        backend, raw_path = item.split("=", 1)
        backend = backend.strip().lower()
        if not backend:
            raise ValueError(f"Invalid backend name in '{item}'.")
        qasm_dir = resolve_path(repo_root, raw_path.strip())
        if not qasm_dir.exists():
            raise FileNotFoundError(f"Transpiled QASM directory not found for backend '{backend}': {qasm_dir}")
        mapping[backend] = qasm_dir
    return mapping


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a transpiled-graph GNN dataset from metadata CSV + transpiled QASM directories."
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/combined_backends_timeout900_1024_optlevel2_with_name_and_transpiled.csv",
        help="CSV with at least name/device/time_taken columns, aligned to the intended dataset rows.",
    )
    parser.add_argument(
        "--dataset-output",
        default="data/training_data_transpiled_graph_timeout900_1024_optlevel2_with_transpiled.npy",
        help="Output pickle path for the transpiled-graph dataset.",
    )
    parser.add_argument(
        "--filtered-metadata-output",
        default=None,
        help=(
            "Optional CSV path to save the subset of metadata rows that were actually "
            "used in the dataset. If omitted, a sidecar file is written next to the "
            "dataset output."
        ),
    )
    parser.add_argument(
        "--timing-output",
        default=None,
        help=(
            "Optional CSV path for per-circuit graph-construction timings. Rows are "
            "flushed after each completed graph so completed measurements survive a later failure."
        ),
    )
    parser.add_argument(
        "--backend-qasm-dir",
        nargs="+",
        default=[
            "washington=data/washington_automatic_timeout900_1024shots_transpiled_optlevel2_qasm",
            "sherbrooke=data/sherbrooke_automatic_timeout900_1024shots_transpiled_optlevel2_qasm",
        ],
        help=(
            "One or more backend=path mappings for transpiled QASM folders. "
            "Example: washington=data/... sherbrooke=data/..."
        ),
    )
    parser.add_argument("--name-column", default="name", help="Metadata column containing the original circuit filename.")
    parser.add_argument("--device-column", default="device", help="Metadata column containing the backend/device name.")
    parser.add_argument("--target-column", default="time_taken", help="Metadata column containing the regression target.")
    parser.add_argument("--n-qubit", type=int, default=127, help="Maximum qubit slots encoded in each graph sample.")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional cap on the built graph node count. Samples above the cap are skipped.",
    )
    parser.add_argument(
        "--max-transpiled-total-ops",
        type=int,
        default=None,
        help=(
            "Optional metadata-level cap using the transpiled_total_ops column. "
            "Rows above the cap are skipped before loading QASM."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process for a smoke test.")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows whose transpiled QASM file is missing instead of raising an error.",
    )
    return parser


def validate_metadata(df: pd.DataFrame, name_column: str, device_column: str, target_column: str):
    missing_columns = [
        column
        for column in [name_column, device_column, target_column]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Metadata CSV is missing required columns: {missing_columns}")

    if df[name_column].isna().any():
        bad_rows = df.index[df[name_column].isna()].tolist()[:10]
        raise ValueError(f"Metadata contains missing circuit names at rows {bad_rows}")
    if df[device_column].isna().any():
        bad_rows = df.index[df[device_column].isna()].tolist()[:10]
        raise ValueError(f"Metadata contains missing device names at rows {bad_rows}")

    numeric_target = pd.to_numeric(df[target_column], errors="coerce")
    if numeric_target.isna().any():
        bad_rows = df.index[numeric_target.isna()].tolist()[:10]
        raise ValueError(f"Metadata contains non-numeric targets at rows {bad_rows}")
    df[target_column] = numeric_target.astype(float)


def default_filtered_metadata_path(dataset_output: Path) -> Path:
    stem = dataset_output.stem
    if dataset_output.suffix:
        return dataset_output.with_name(f"{stem}_metadata.csv")
    return dataset_output.with_name(f"{dataset_output.name}_metadata.csv")


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_circ_dag_converter(repo_root: Path):
    candidates = [
        repo_root / "data_preparation" / "circ_dag_converter_afterfixes.py",
        repo_root / "data_preparation" / "circ_dag_converter.py",
    ]
    for module_path in candidates:
        if module_path.exists():
            return load_module_from_path(module_path.stem, module_path)
    raise FileNotFoundError(
        "Could not find either circ_dag_converter_afterfixes.py or circ_dag_converter.py "
        f"under {repo_root / 'data_preparation'}"
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.append(str(repo_root / "data_preparation"))

    import helper  # type: ignore
    circ_dag_converter = load_circ_dag_converter(repo_root)

    parser = build_parser(repo_root)
    args = parser.parse_args()

    metadata_path = resolve_path(repo_root, args.metadata_csv)
    output_path = resolve_path(repo_root, args.dataset_output)
    filtered_metadata_path = (
        resolve_path(repo_root, args.filtered_metadata_output)
        if args.filtered_metadata_output is not None
        else default_filtered_metadata_path(output_path)
    )
    timing_path = (
        resolve_path(repo_root, args.timing_output)
        if args.timing_output is not None
        else None
    )
    backend_qasm_dirs = parse_backend_qasm_dirs(repo_root, args.backend_qasm_dir)

    metadata_df = pd.read_csv(metadata_path).copy()
    validate_metadata(
        metadata_df,
        name_column=args.name_column,
        device_column=args.device_column,
        target_column=args.target_column,
    )

    if args.limit is not None:
        metadata_df = metadata_df.head(int(args.limit)).copy()

    if args.max_transpiled_total_ops is not None and "transpiled_total_ops" not in metadata_df.columns:
        raise ValueError(
            "--max-transpiled-total-ops was provided but the metadata CSV does not contain "
            "a transpiled_total_ops column."
        )

    training_data = []
    kept_rows = []
    skipped_missing = []
    skipped_max_total_ops = []
    skipped_max_nodes = []
    timing_file = None
    timing_writer = None
    if timing_path is not None:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_file = timing_path.open("w", newline="")
        timing_writer = csv.DictWriter(
            timing_file,
            fieldnames=[
                "row_idx",
                "name",
                "device",
                "transpiled_total_ops",
                "graph_nodes",
                "qasm_load_s",
                "feature_extraction_s",
                "dag_construction_s",
                "total_graph_build_s",
            ],
        )
        timing_writer.writeheader()
        timing_file.flush()

    for row_idx, row in enumerate(metadata_df.itertuples(index=False), start=1):
        name = str(getattr(row, args.name_column))
        device = str(getattr(row, args.device_column)).strip().lower()
        target = float(getattr(row, args.target_column))

        if args.max_transpiled_total_ops is not None:
            transpiled_total_ops = int(float(getattr(row, "transpiled_total_ops")))
            if transpiled_total_ops > int(args.max_transpiled_total_ops):
                skipped_max_total_ops.append(
                    {
                        "row_idx": row_idx,
                        "name": name,
                        "device": device,
                        "transpiled_total_ops": transpiled_total_ops,
                    }
                )
                print(
                    f"[skip] row={row_idx} name={name} device={device} "
                    f"reason=max_transpiled_total_ops value={transpiled_total_ops}",
                    flush=True,
                )
                continue

        if device not in backend_qasm_dirs:
            message = (
                f"No transpiled QASM directory was provided for backend '{device}'. "
                f"Available backends: {sorted(backend_qasm_dirs.keys())}"
            )
            if args.skip_missing:
                skipped_missing.append(
                    {
                        "row_idx": row_idx,
                        "name": name,
                        "device": device,
                        "reason": "missing_backend_mapping",
                        "path": None,
                    }
                )
                print(f"[skip] row={row_idx} name={name} device={device} reason=missing_backend_mapping", flush=True)
                continue
            raise KeyError(message)

        transpiled_qasm_path = backend_qasm_dirs[device] / f"{Path(name).stem}_transpiled.qasm"
        if not transpiled_qasm_path.exists():
            if args.skip_missing:
                skipped_missing.append(
                    {
                        "row_idx": row_idx,
                        "name": name,
                        "device": device,
                        "reason": "missing_transpiled_qasm",
                        "path": str(transpiled_qasm_path),
                    }
                )
                print(
                    f"[skip] row={row_idx} name={name} device={device} "
                    f"reason=missing_transpiled_qasm path={transpiled_qasm_path}",
                    flush=True,
                )
                continue
            raise FileNotFoundError(
                f"Missing transpiled QASM for row {row_idx}: "
                f"name={name} device={device} path={transpiled_qasm_path}"
            )

        graph_build_start = time.perf_counter()
        qasm_load_start = graph_build_start
        qc_t = QuantumCircuit.from_qasm_file(str(transpiled_qasm_path))
        qasm_load_s = time.perf_counter() - qasm_load_start

        feature_extraction_start = time.perf_counter()
        global_features = helper.create_feature_dict(qc_t)
        feature_extraction_s = time.perf_counter() - feature_extraction_start

        dag_construction_start = time.perf_counter()
        graph = circ_dag_converter.circ_to_dag_with_data(
            qc_t,
            device,
            list(global_features.values()),
            n_qubit=int(args.n_qubit),
        )
        dag_construction_s = time.perf_counter() - dag_construction_start
        total_graph_build_s = time.perf_counter() - graph_build_start
        graph_nodes = int(graph.x.shape[0])
        if args.max_nodes is not None and graph_nodes > int(args.max_nodes):
            skipped_max_nodes.append(
                {
                    "row_idx": row_idx,
                    "name": name,
                    "device": device,
                    "graph_nodes": graph_nodes,
                }
            )
            print(
                f"[skip] row={row_idx} name={name} device={device} "
                f"reason=max_nodes value={graph_nodes}",
                flush=True,
            )
            continue
        graph.y = float(target)
        training_data.append(graph)
        kept_rows.append(row_idx - 1)

        if timing_writer is not None and timing_file is not None:
            timing_writer.writerow(
                {
                    "row_idx": row_idx,
                    "name": name,
                    "device": device,
                    "transpiled_total_ops": getattr(row, "transpiled_total_ops", None),
                    "graph_nodes": graph_nodes,
                    "qasm_load_s": qasm_load_s,
                    "feature_extraction_s": feature_extraction_s,
                    "dag_construction_s": dag_construction_s,
                    "total_graph_build_s": total_graph_build_s,
                }
            )
            timing_file.flush()

        if row_idx % 100 == 0:
            print(f"[build] processed {row_idx} rows", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if timing_file is not None:
        timing_file.close()
    with open(output_path, "wb") as file:
        pickle.dump(training_data, file)

    filtered_metadata = metadata_df.iloc[kept_rows].copy().reset_index(drop=True)
    filtered_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_metadata.to_csv(filtered_metadata_path, index=False)

    print(f"metadata_csv = {metadata_path}")
    print(f"dataset_output = {output_path}")
    print(f"filtered_metadata_output = {filtered_metadata_path}")
    if timing_path is not None:
        print(f"timing_output = {timing_path}")
    print(f"graphs = {len(training_data)}")
    print(f"skipped_missing = {len(skipped_missing)}")
    print(f"skipped_max_total_ops = {len(skipped_max_total_ops)}")
    print(f"skipped_max_nodes = {len(skipped_max_nodes)}")
    if skipped_missing:
        print(f"first_skipped = {skipped_missing[:10]}")
    if skipped_max_total_ops:
        print(f"first_skipped_max_total_ops = {skipped_max_total_ops[:10]}")
    if skipped_max_nodes:
        print(f"first_skipped_max_nodes = {skipped_max_nodes[:10]}")
    if training_data:
        print(f"global_features_len = {len(training_data[0].global_features)}")
        print(f"node_feature_dim = {int(training_data[0].x.shape[1])}")


if __name__ == "__main__":
    main()
