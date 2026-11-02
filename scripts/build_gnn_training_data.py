#!/usr/bin/env python3
"""Build Source and Hybrid GNN datasets directly from tidy execution CSVs.

``run_bulk_execution.py`` writes one tidy CSV per backend and optimization
level.  Those files already contain the source circuit filename, measured
runtime, and transpiled features.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from qiskit import QuantumCircuit


TRANSPILED_FEATURE_COLUMNS = [
    "transpiled_depth",
    "transpiled_total_ops",
    "transpiled_two_qubit_ops",
    "transpiled_op_ecr",
    "transpiled_op_cx",
    "transpiled_op_rz",
    "transpiled_op_sx",
    "transpiled_op_x",
    "transpiled_supermarq_program_communication",
    "transpiled_supermarq_critical_depth",
    "transpiled_supermarq_entanglement_ratio",
    "transpiled_supermarq_parallelism",
    "transpiled_supermarq_liveness",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    parser = argparse.ArgumentParser(
        description=(
            "Create aligned Source and Hybrid GNN metadata CSVs and "
            ".npy datasets from run_bulk_execution tidy CSVs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tidy-csv",
        nargs="+",
        required=True,
        metavar="BACKEND=PATH",
        help=(
            "One tidy execution CSV per backend, for example "
            "washington=data/extractedfiles/washington_execution_tidy_...csv."
        ),
    )
    parser.add_argument(
        "--source-qasm-dir",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing the original source QASM files named in the tidy CSVs.",
    )
    parser.add_argument(
        "--source-metadata-output",
        type=Path,
        default=data_root / "source_metadata.csv",
        help="Metadata CSV aligned with the Source dataset.",
    )
    parser.add_argument(
        "--hybrid-metadata-output",
        type=Path,
        default=data_root / "hybrid_metadata.csv",
        help="Metadata CSV aligned with the Hybrid dataset and reusable by the Transpiled builder.",
    )
    parser.add_argument(
        "--source-dataset-output",
        type=Path,
        default=data_root / "training_data_source.npy",
        help=" .npy output for the Source GNN.",
    )
    parser.add_argument(
        "--hybrid-dataset-output",
        type=Path,
        default=data_root / "training_data_hybrid.npy",
        help=" .npy output for the Hybrid GNN.",
    )
    return parser.parse_args()


def parse_backend_paths(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen_backends: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --tidy-csv value {value!r}; expected BACKEND=PATH.")
        backend, path_text = value.split("=", maxsplit=1)
        backend = backend.strip().lower()
        if not backend or not path_text:
            raise ValueError(f"Invalid --tidy-csv value {value!r}; expected BACKEND=PATH.")
        if backend in seen_backends:
            raise ValueError(f"Backend {backend!r} was provided more than once.")
        seen_backends.add(backend)
        parsed.append((backend, Path(path_text).expanduser().resolve()))
    return parsed


def build_qasm_lookup(qasm_dirs: list[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for qasm_dir in qasm_dirs:
        resolved_dir = qasm_dir.expanduser().resolve()
        if not resolved_dir.is_dir():
            raise FileNotFoundError(f"Source QASM directory not found: {resolved_dir}")
        for path in resolved_dir.glob("*.qasm"):
            existing = lookup.get(path.name)
            if existing is not None and existing.resolve() != path.resolve():
                raise ValueError(
                    f"Duplicate source QASM filename {path.name!r} in {existing.parent} and {path.parent}. "
                    "Use directories with unique circuit names."
                )
            lookup[path.name] = path.resolve()
    if not lookup:
        raise ValueError("No .qasm files were found in --source-qasm-dir.")
    return lookup


def load_tidy_metadata(backend_paths: list[tuple[str, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for backend, path in backend_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Tidy CSV not found: {path}")
        frame = pd.read_csv(path)
        required = {"name", "time_taken"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Tidy CSV {path} is missing required columns: {missing}")

        frame = frame.copy()
        frame = frame.dropna(subset=["name"])
        frame["name"] = frame["name"].astype(str).str.strip().map(Path).map(lambda value: value.name)
        frame["device"] = backend
        frame["time_taken"] = pd.to_numeric(frame["time_taken"], errors="coerce")
        frame = frame.dropna(subset=["name", "time_taken"])
        frame = frame[frame["name"].ne("")]
        if frame.empty:
            raise ValueError(f"Tidy CSV has no usable successful rows: {path}")

        # Older execution outputs may not contain every gate-specific feature.
        for column in TRANSPILED_FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        if frame.duplicated(["device", "name"]).any():
            duplicates = frame.loc[frame.duplicated(["device", "name"], keep=False), ["device", "name"]]
            raise ValueError(
                f"Tidy CSV has duplicate device/name rows: {duplicates.drop_duplicates().to_dict('records')[:5]}"
            )
        frames.append(frame)

    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata.sort_values(["device", "name"], kind="stable").reset_index(drop=True)
    return metadata


def load_repo_modules(repo_root: Path) -> tuple[Any, Any]:
    os.chdir(repo_root)
    preparation_dir = str(repo_root / "data_preparation")
    if preparation_dir not in sys.path:
        sys.path.insert(0, preparation_dir)
    import circ_dag_converter_afterfixes as circ_dag_converter  # type: ignore
    import helper  # type: ignore

    return helper, circ_dag_converter


def build_graphs(
    metadata: pd.DataFrame,
    qasm_lookup: dict[str, Path],
    helper: Any,
    circ_dag_converter: Any,
    include_transpiled_features: bool,
) -> list[Any]:
    graphs: list[Any] = []
    representation = "Hybrid" if include_transpiled_features else "Source"

    for row_number, row in enumerate(metadata.itertuples(index=False), start=1):
        qasm_path = qasm_lookup.get(row.name)
        if qasm_path is None:
            raise FileNotFoundError(
                f"No source QASM named {row.name!r} for {row.device}; "
                "check --source-qasm-dir against the tidy CSV."
            )
        qc = QuantumCircuit.from_qasm_file(str(qasm_path))
        global_features = dict(helper.create_feature_dict(qc))
        if include_transpiled_features:
            global_features.update(
                {column: float(getattr(row, column)) for column in TRANSPILED_FEATURE_COLUMNS}
            )

        graph = circ_dag_converter.circ_to_dag_with_data(
            qc,
            row.device,
            list(global_features.values()),
            n_qubit=127,
        )
        graph.y = float(row.time_taken)
        graphs.append(graph)

        if row_number % 100 == 0:
            print(f"[{representation}] built {row_number}/{len(metadata)} graphs")
    return graphs


def write_csv(metadata: pd.DataFrame, output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False)


def write_dataset(graphs: list[Any], output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    backend_paths = parse_backend_paths(list(args.tidy_csv))
    qasm_lookup = build_qasm_lookup(list(args.source_qasm_dir))
    full_metadata = load_tidy_metadata(backend_paths)

    helper, circ_dag_converter = load_repo_modules(repo_root)
    source_metadata = full_metadata.drop(columns=TRANSPILED_FEATURE_COLUMNS)

    write_csv(source_metadata, args.source_metadata_output)
    write_csv(full_metadata, args.hybrid_metadata_output)

    source_graphs = build_graphs(
        source_metadata, qasm_lookup, helper, circ_dag_converter, include_transpiled_features=False
    )
    hybrid_graphs = build_graphs(
        full_metadata, qasm_lookup, helper, circ_dag_converter, include_transpiled_features=True
    )
    write_dataset(source_graphs, args.source_dataset_output)
    write_dataset(hybrid_graphs, args.hybrid_dataset_output)

    print(f"samples = {len(full_metadata)}")
    print(f"source_metadata_output = {Path(args.source_metadata_output).expanduser().resolve()}")
    print(f"hybrid_metadata_output = {Path(args.hybrid_metadata_output).expanduser().resolve()}")
    print(f"source_dataset_output = {Path(args.source_dataset_output).expanduser().resolve()}")
    print(f"hybrid_dataset_output = {Path(args.hybrid_dataset_output).expanduser().resolve()}")
    print(f"source_global_features_len = {len(source_graphs[0].global_features)}")
    print(f"hybrid_global_features_len = {len(hybrid_graphs[0].global_features)}")


if __name__ == "__main__":
    main()
