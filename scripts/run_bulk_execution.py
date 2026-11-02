#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

def safe_tail(value, limit: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-limit:]
    return str(value)[-limit:]


def build_repo_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    return repo_root, data_root


def process_usage_snapshot() -> dict[str, float | int]:
    """Capture the current worker process CPU time and peak RSS."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_s": float(usage.ru_utime),
        "system_cpu_s": float(usage.ru_stime),
        "max_rss_kb": int(usage.ru_maxrss),
    }


def process_usage_delta(start: dict[str, float | int]) -> dict[str, float | int]:
    """Compute worker CPU deltas and report current peak RSS."""
    end = process_usage_snapshot()
    return {
        "user_cpu_s": round(float(end["user_cpu_s"]) - float(start["user_cpu_s"]), 6),
        "system_cpu_s": round(float(end["system_cpu_s"]) - float(start["system_cpu_s"]), 6),
        "max_rss_kb": int(end["max_rss_kb"]),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(sanitize_for_json(payload)), encoding="utf-8")
    tmp_path.replace(path)


def load_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except BaseException:
        return {}

class ResourceUsageHeartbeat:

    def __init__(
        self,
        snapshot_path: Path | None,
        usage_start: dict[str, float | int],
        interval_s: float = 1.0,
    ) -> None:
        self.snapshot_path = snapshot_path
        self.usage_start = usage_start
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._metadata: dict[str, Any] = {}
        self._lock = threading.Lock()

    def write_snapshot(self, stage: str) -> None:
        if self.snapshot_path is None:
            return
        with self._lock:
            metadata = dict(self._metadata)
            payload = {
                **process_usage_delta(self.usage_start),
                **metadata,
                "resource_snapshot_stage": stage,
                "resource_snapshot_epoch_s": round(time.time(), 6),
            }
            write_json_atomic(self.snapshot_path, payload)

    def update(self, **metadata: Any) -> None:
        if self.snapshot_path is None or not metadata:
            return
        with self._lock:
            self._metadata.update(
                {str(key): sanitize_for_json(value) for key, value in metadata.items()}
            )
        self.write_snapshot("running")

    def start(self) -> None:
        if self.snapshot_path is None:
            return

        def _runner() -> None:
            while not self._stop_event.wait(self.interval_s):
                self.write_snapshot("running")

        self.write_snapshot("started")
        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self, stage: str) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        self.write_snapshot(stage)

def sanitize_for_json(value: Any) -> Any:
    """Convert Paths, numpy values, and other objects into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return sanitize_for_json(value.tolist())
        except BaseException:
            pass
    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except BaseException:
            pass
    return str(value)


def normalize_column_name(value: str) -> str:
    pieces: list[str] = []
    prev_sep = False
    for ch in value.lower():
        if ch.isalnum():
            pieces.append(ch)
            prev_sep = False
        elif not prev_sep:
            pieces.append("_")
            prev_sep = True
    return "".join(pieces).strip("_") or "value"


def flatten_count_ops(prefix: str, counts: dict[str, Any]) -> dict[str, int]:
    """Flatten qc.count_ops() into columns like source_op_cx or transpiled_op_rz."""
    return {
        f"{prefix}_op_{normalize_column_name(name)}": int(count)
        for name, count in sorted(counts.items())
    }

def instruction_stats(qc) -> tuple[int, int]:
    """Count two-qubit ops and all multi-qubit ops in a circuit."""
    two_qubit = 0
    multi_qubit = 0
    for item in qc.data:
        arity = int(item.operation.num_qubits)
        if arity >= 2:
            multi_qubit += 1
        if arity == 2:
            two_qubit += 1
    return two_qubit, multi_qubit


def circuit_summary(qc, prefix: str) -> dict[str, Any]:
    """Build one flat record of circuit size/depth/gate-count statistics."""
    counts = dict(qc.count_ops())
    total_ops = int(sum(int(v) for v in counts.values()))
    two_qubit, multi_qubit = instruction_stats(qc)
    return {
        f"{prefix}_num_qubits": int(qc.num_qubits),
        f"{prefix}_depth": int(qc.depth()),
        f"{prefix}_total_ops": total_ops,
        f"{prefix}_two_qubit_ops": two_qubit,
        f"{prefix}_multi_qubit_ops": multi_qubit,
        f"{prefix}_swap_count": int(counts.get("swap", 0)),
        f"{prefix}_size": int(qc.size()),
        **flatten_count_ops(prefix, counts),
    }


def provider_name(provider: Any) -> str | None:
    """Read backend/provider name whether it is stored as an attribute or method."""
    name = getattr(provider, "name", None)
    if callable(name):
        try:
            name = name()
        except BaseException:
            return None
    return str(name) if name is not None else None


def selected_metadata_fields(metadata: dict[str, Any], sim) -> dict[str, Any]:
    """Keep only the needed Aer metadata fields in raw logs."""
    active_input_qubits = metadata.get("active_input_qubits")
    active_input_qubits_count = (
        len(active_input_qubits) if isinstance(active_input_qubits, list) else None
    )

    return {
        "simulation_method": metadata.get("method", getattr(sim.options, "method", None)),
        "metadata_num_qubits": metadata.get("num_qubits"),
        "metadata_num_clbits": metadata.get("num_clbits"),
        "parallel_shots": metadata.get("parallel_shots"),
        "measure_sampling": metadata.get("measure_sampling"),
        "noise": metadata.get("noise"),
        "simulator_device_reported": metadata.get("device"),
        "batched_shots_optimization": metadata.get("batched_shots_optimization"),
        "remapped_qubits": metadata.get("remapped_qubits"),
        "active_input_qubits_count": active_input_qubits_count,
    }


def supermarq_summary(helper: Any, qc, prefix: str = "supermarq") -> dict[str, float]:
    """Compute SupermarQ features."""
    features = helper.calc_supermarq_features(qc)
    return {
        f"{prefix}_program_communication": float(features.program_communication),
        f"{prefix}_critical_depth": float(features.critical_depth),
        f"{prefix}_entanglement_ratio": float(features.entanglement_ratio),
        f"{prefix}_parallelism": float(features.parallelism),
        f"{prefix}_liveness": float(features.liveness),
    }


TIDY_EXACT_FIELDS = [
    "name",
    "provider_name",
    "shots",
    "simulation_method",
    "transpile_optimization_level",
    "transpile_wall_s",
    "run_wall_s",
    "time_taken",
    "supermarq_program_communication",
    "supermarq_critical_depth",
    "supermarq_entanglement_ratio",
    "supermarq_parallelism",
    "supermarq_liveness",
    "transpiled_supermarq_program_communication",
    "transpiled_supermarq_critical_depth",
    "transpiled_supermarq_entanglement_ratio",
    "transpiled_supermarq_parallelism",
    "transpiled_supermarq_liveness",
]

TIDY_PREFIXES = ("source_", "transpiled_")


def tidy_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("status") != "success":
        return {}

    excluded = {
        "source_op_barrier",
        "transpiled_op_barrier",
        "source_size",
        "source_multi_qubit_ops",
        "source_op_swap",
        "transpiled_size",
        "transpiled_multi_qubit_ops",
        "transpiled_op_measure",
        "transpiled_swap_count",
        "transpiled_op_swap",
    }

    row: dict[str, Any] = {}
    for key in TIDY_EXACT_FIELDS:
        if key in record:
            row[key] = record[key]
    for key, value in record.items():
        if key in excluded:
            continue
        if key.startswith(TIDY_PREFIXES) and (
            value is None or isinstance(value, (str, int, float, bool))
        ):
            row[key] = value
    return row



def tidy_fieldnames(records: list[dict[str, Any]]) -> list[str]:
    """Keep key columns first, then append any remaining source_/transpiled_ fields."""
    preferred = list(TIDY_EXACT_FIELDS)
    seen = set().union(*(record.keys() for record in records))
    ordered = [name for name in preferred if name in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def write_tidy_csv(tidy_path: Path, records: list[dict[str, Any]]) -> None:
    tidy_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in (tidy_record(record) for record in records) if row]
    if not rows:
        tidy_path.write_text("", encoding="utf-8")
        return

    fieldnames = tidy_fieldnames(rows)
    with tidy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_transpiled_qasm(output_dir: Path, qasm_file: Path, qc_t) -> Path:
    """Write the transpiled circuit as a plain QASM reference file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{qasm_file.stem}_transpiled.qasm"
    output_path.write_text(qc_t.qasm(), encoding="utf-8")
    return output_path

def resource_summary_record(record: dict[str, Any]) -> dict[str, Any]:
    """compact per-circuit resource/utilization record for JSONL analysis."""
    fields = [
        "file",
        "name",
        "backend",
        "provider_name",
        "status",
        "num_qubits",
        "shots",
        "elapsed_s",
        "transpile_wall_s",
        "run_wall_s",
        "time_taken",
        "user_cpu_s",
        "system_cpu_s",
        "max_rss_kb",
        "worker_phase",
        "worker_event",
    ]
    return {field: record.get(field) for field in fields}

def is_success_record(record: dict[str, Any]) -> bool:
    return record.get("status") == "success"

def merge_snapshot_usage(record: dict[str, Any], snapshot_path: Path | None) -> None:
    snapshot = load_json_file(snapshot_path)
    if not snapshot:
        return
    for field in [
        "provider_name",
        "shots",
        "num_qubits",
        "transpile_wall_s",
        "user_cpu_s",
        "system_cpu_s",
        "max_rss_kb",
        "worker_phase",
        "worker_event",
        "resource_snapshot_stage",
        "resource_snapshot_epoch_s",
    ]:
        if snapshot.get(field) is not None and record.get(field) is None:
            record[field] = snapshot[field]


def parse_args() -> argparse.Namespace:
    repo_root, data_root = build_repo_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Execute many QASM circuits by isolating each circuit in its own subprocess."
        )
    )
    parser.add_argument(
        "--backend",
        default="washington",
        help="Backend key from helper.provider_dict, for example washington or sherbrooke.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_root / "quantum_circuits",
        help="Directory containing input .qasm files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-circuit CSV outputs. "
            "Useful values are data/washington_1, data/washington_2, etc."
        ),
    )
    parser.add_argument(
        "--raw-status-file",
        type=Path,
        default=None,
        help="JSONL file to append full raw/debug per-circuit records to.",
    )
    parser.add_argument(
        "--tidy-file",
        type=Path,
        default=None,
        help="CSV file to write flat analysis-friendly per-circuit rows to.",
    )
    parser.add_argument(
        "--transpiled-qasm-dir",
        type=Path,
        default=None,
        help="Optional directory to save each transpiled circuit as a plain .qasm file.",
    )
    parser.add_argument(
        "--resource-usage-file",
        type=Path,
        default=None,
        help="Optional JSONL file to write compact successful-circuit resource usage records to.",
    )
    parser.add_argument(
        "--resource-failure-file",
        type=Path,
        default=None,
        help="Optional JSONL file to write compact unsuccessful-circuit resource usage records to.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2,
        help="Shot count for execute(...). Lower this if simulator runs are too slow.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Hard timeout in seconds for each circuit subprocess.",
    )
    parser.add_argument(
        "--simulation-method",
        default="automatic",
        help="Aer simulator method, for example automatic, statevector, or matrix_product_state.",
    )
    parser.add_argument(
        "--simulator-device",
        default="CPU",
        help="Aer simulator device option, usually CPU or GPU.",
    )
    parser.add_argument(
        "--max-parallel-threads",
        type=int,
        default=8,
        help="Aer max_parallel_threads option. Use 0 to let Aer decide.",
    )
    parser.add_argument(
        "--transpile-optimization-level",
        type=int,
        default=1,
        help="Optional transpiler optimization level. Leave unset for Qiskit's default.",
    )
    parser.add_argument(
        "--seed-transpiler",
        type=int,
        default=None,
        help="Optional fixed transpiler seed for reproducibility.",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Optional user qubit cap. Circuits above this are skipped before execution.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many qasm files to attempt this run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run circuits even if the per-circuit CSV already exists.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional substring filter for qasm filenames.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    # The parent uses this hidden argument when it launches the one-file worker.
    parser.add_argument(
        "--qasm-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resource-snapshot-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def import_repo_modules(repo_root: Path):
    sys.path.append(str(repo_root))
    sys.path.append(str(repo_root / "data_preparation"))
    import helper  # type: ignore
    from qiskit import QuantumCircuit  # type: ignore

    return helper, QuantumCircuit

def run_worker(args: argparse.Namespace) -> int:
    """Execute exactly one QASM file"""
    repo_root, _ = build_repo_paths()
    helper, QuantumCircuit = import_repo_modules(repo_root)

    from qiskit import transpile
    from qiskit.providers.aer import AerSimulator

    assert args.qasm_file is not None
    backend_key = args.backend.upper()
    provider = helper.provider_dict[backend_key]["provider"]
    provider_max_qubits = int(helper.provider_dict[backend_key]["max_qubits"])
    usage_start = process_usage_snapshot()
    heartbeat = ResourceUsageHeartbeat(args.resource_snapshot_file, usage_start)
    heartbeat.start()
    transpile_wall: float | None = None
    worker_phase = "worker"
    worker_event = "worker_started"

    def stop_heartbeat(stage: str) -> None:
        heartbeat.stop(stage)

    def update_heartbeat(
        *,
        phase: str | None = None,
        event: str | None = None,
        **metadata: Any,
    ) -> None:
        nonlocal worker_phase, worker_event
        if phase is not None:
            worker_phase = phase
            metadata["worker_phase"] = phase
        if event is not None:
            worker_event = event
            metadata["worker_event"] = event
        heartbeat.update(**metadata)

    def _handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
        stop_heartbeat(f"signal_{signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # These are the run settings that apply before we even load the circuit.
    record_base: dict[str, Any] = {
        "file": str(args.qasm_file),
        "name": args.qasm_file.name,
        "backend": args.backend,
        "provider_name": provider_name(provider),
        "shots": int(args.shots),
        "timeout_s": int(args.timeout),
        "requested_simulation_method": args.simulation_method,
        "requested_simulator_device": args.simulator_device,
        "transpile_optimization_level": args.transpile_optimization_level,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    update_heartbeat(
        provider_name=record_base["provider_name"],
        shots=record_base["shots"],
        phase="worker",
        event="worker_started",
    )

    try:
        qc = QuantumCircuit.from_qasm_file(str(args.qasm_file))
    except BaseException as exc:
        stop_heartbeat("qasm_load_error")
        print(
            json.dumps(
                {
                    **record_base,
                    "status": "qasm_load_error",
                    "error": str(exc),
                    **process_usage_delta(usage_start),
                }
            ),
            flush=True,
        )
        return 0

    # Add original, pre-transpile circuit statistics.
    source_record = {
        **record_base,
        **circuit_summary(qc, "source"),
        **supermarq_summary(helper, qc),
    }
    update_heartbeat(num_qubits=int(qc.num_qubits), phase="qasm", event="qasm_loaded")

    # Skip early if the circuit cannot fit on the requested backend or user cap.
    if qc.num_qubits > provider_max_qubits:
        stop_heartbeat("skipped_provider_qubits")
        print(
            json.dumps(
                {
                    **source_record,
                    "status": "skipped_provider_qubits",
                    "num_qubits": int(qc.num_qubits),
                    "provider_max_qubits": provider_max_qubits,
                    **process_usage_delta(usage_start),
                }
            ),
            flush=True,
        )
        return 0

    if args.max_qubits is not None and qc.num_qubits > args.max_qubits:
        stop_heartbeat("skipped_user_qubits")
        print(
            json.dumps(
                {
                    **source_record,
                    "status": "skipped_user_qubits",
                    "num_qubits": int(qc.num_qubits),
                    "user_max_qubits": int(args.max_qubits),
                    **process_usage_delta(usage_start),
                }
            ),
            flush=True,
        )
        return 0

    try:
        # Using from_backend(provider) gives us backend-style noisy simulation.
        sim = AerSimulator.from_backend(provider)
        sim_options: dict[str, Any] = {
            "method": args.simulation_method,
            "device": args.simulator_device,
            "max_parallel_threads": int(args.max_parallel_threads),
        }
        sim.set_options(**sim_options)

        # Measure compilation time separately from execution time.
        t_transpile = time.perf_counter()
        transpile_kwargs: dict[str, Any] = {"backend": sim}



        if args.transpile_optimization_level is not None:
            transpile_kwargs["optimization_level"] = int(args.transpile_optimization_level)
        if args.seed_transpiler is not None:
            transpile_kwargs["seed_transpiler"] = int(args.seed_transpiler)
        update_heartbeat(phase="transpile", event="transpile_started")
        qc_t = transpile(qc, **transpile_kwargs)
        transpile_wall = round(time.perf_counter() - t_transpile, 6)
        update_heartbeat(
            phase="transpile",
            event="transpile_finished",
            transpile_wall_s=transpile_wall,
        )

        if args.transpiled_qasm_dir is not None:
            write_transpiled_qasm(args.transpiled_qasm_dir, args.qasm_file, qc_t)
            update_heartbeat(phase="transpile", event="transpiled_qasm_written")

        # Summarize the actual circuit Aer will execute after mapping/transforms.
        transpiled_record = {
            **circuit_summary(qc_t, "transpiled"),
            **supermarq_summary(helper, qc_t, prefix="transpiled_supermarq"),
        }

        # Measure external wall time around sim.run(...) and job.result().
        t_run = time.perf_counter()
        update_heartbeat(phase="run", event="run_submit_started")
        job = sim.run(qc_t, shots=int(args.shots))
        update_heartbeat(phase="run", event="run_wait_result")
        result = job.result()
        run_wall = round(time.perf_counter() - t_run, 6)
        update_heartbeat(phase="run", event="run_finished")

        # Aer metadata tells us what method it actually chose under "automatic".
        metadata = result.results[0].metadata if getattr(result, "results", None) else {}
        metadata = sanitize_for_json(metadata or {})
        if isinstance(metadata, dict):
            metadata.pop("input_qubit_map", None)
        backend_time_taken = float(getattr(result, "time_taken", run_wall))
        job_id = getattr(job, "job_id", None)
        if callable(job_id):
            job_id = job_id()

        stop_heartbeat("success")
        print(
            json.dumps(
                {
                    # A success record combines input settings, original stats,
                    # transpiled stats, selected Aer metadata, and timing.
                    **source_record,
                    **transpiled_record,
                    **selected_metadata_fields(metadata, sim),
                    "status": "success",
                    "num_qubits": int(qc.num_qubits),
                    "time_taken": backend_time_taken,
                    #"backend_time_taken": backend_time_taken,
                    "transpile_wall_s": transpile_wall,
                    "run_wall_s": run_wall,
                    "worker_phase": worker_phase,
                    "worker_event": worker_event,
                    "job_id": job_id,
                    "result_status": getattr(result, "status", None),
                    "result_metadata": metadata,
                    **process_usage_delta(usage_start),
                }
            ),
            flush=True,
        )
    except BaseException as exc:
        stop_heartbeat("execution_error")
        print(
            json.dumps(
                {
                    **source_record,
                    "status": "execution_error",
                    "num_qubits": int(qc.num_qubits),
                    "transpile_wall_s": transpile_wall,
                    "worker_phase": worker_phase,
                    "worker_event": worker_event,
                    "error": str(exc),
                    **process_usage_delta(usage_start),
                }
            ),
            flush=True,
        )
    return 0


def append_status(status_path: Path, record: dict[str, Any]) -> None:
    """Append one raw JSON record to the per-run JSONL log."""
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_for_json(record)) + "\n")


def last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def normalized_qasm_text(qasm_path: Path) -> str:
    """Match the notebook pipeline by reserializing QASM through Qiskit."""
    _, QuantumCircuit = import_repo_modules(build_repo_paths()[0])
    qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    return qc.qasm()


def write_result_csv(output_csv: Path, record: dict[str, Any], backend: str) -> None:
    qasm_path = Path(record["file"])
    try:
        quantum_circuit = normalized_qasm_text(qasm_path)
    except BaseException:
        quantum_circuit = qasm_path.read_text(encoding="utf-8")
    row = {
        "quantum_circuit": quantum_circuit,
        "time_taken": record.get("time_taken", record.get("backend_time_taken")),
        "device": backend,
    }
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["quantum_circuit", "time_taken", "device"],
        )
        writer.writeheader()
        writer.writerow(row)

def run_parent(args: argparse.Namespace) -> int:
    """Batch runner: launch one worker subprocess per QASM file and collect results."""
    repo_root, data_root = build_repo_paths()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (data_root / args.backend).resolve()
    )
    raw_status_path = (
    args.raw_status_file.resolve()
    if args.raw_status_file is not None
    else output_dir / f"{args.backend}_execution_raw.jsonl"
    )

    tidy_path = (
        args.tidy_file.resolve()
        if args.tidy_file is not None
        else output_dir / f"{args.backend}_execution_tidy.csv"
    )
    resource_usage_path = (
        args.resource_usage_file.resolve()
        if args.resource_usage_file is not None
        else output_dir / f"{args.backend}_execution_resource_usage.jsonl"
    )
    resource_failure_path = (
        args.resource_failure_file.resolve()
        if args.resource_failure_file is not None
        else output_dir / f"{args.backend}_execution_resource_usage_unsuccessful.jsonl"
    )
    resource_snapshot_dir = resource_usage_path.parent / ".resource_snapshots"
    transpiled_qasm_dir = (
        args.transpiled_qasm_dir.resolve()
        if args.transpiled_qasm_dir is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_status_path.parent.mkdir(parents=True, exist_ok=True)
    tidy_path.parent.mkdir(parents=True, exist_ok=True)
    resource_usage_path.parent.mkdir(parents=True, exist_ok=True)
    resource_failure_path.parent.mkdir(parents=True, exist_ok=True)
    resource_snapshot_dir.mkdir(parents=True, exist_ok=True)
    if transpiled_qasm_dir is not None:
        transpiled_qasm_dir.mkdir(parents=True, exist_ok=True)

    # Each run starts a fresh raw log.
    raw_status_path.write_text("", encoding="utf-8")
    resource_usage_path.write_text("", encoding="utf-8")
    resource_failure_path.write_text("", encoding="utf-8")

    qasm_files = sorted(input_dir.glob("*.qasm"))
    if args.pattern:
        qasm_files = [path for path in qasm_files if args.pattern in path.name]
    if args.limit is not None:
        qasm_files = qasm_files[: int(args.limit)]

    if not qasm_files:
        print(f"No .qasm files found in {input_dir}")
        return 1

    print(f"backend = {args.backend}")
    print(f"input_dir = {input_dir}")
    print(f"output_dir = {output_dir}")
    print(f"raw_status_file = {raw_status_path}")
    print(f"tidy_file = {tidy_path}")
    print(f"resource_usage_file = {resource_usage_path}")
    print(f"resource_failure_file = {resource_failure_path}")
    print(f"resource_snapshot_dir = {resource_snapshot_dir}")
    print(f"transpiled_qasm_dir = {transpiled_qasm_dir}")
    print(f"shots = {args.shots}")
    print(f"timeout_s = {args.timeout}")
    print(f"simulation_method = {args.simulation_method}")
    print(f"simulator_device = {args.simulator_device}")
    print(f"transpile_optimization_level = {args.transpile_optimization_level}")
    print(f"max_qubits = {args.max_qubits}")
    print(f"qasm_files = {len(qasm_files)}")

    success = 0
    skipped_existing = 0
    failures = 0
    timeouts = 0
    skipped_qubits = 0
    tidy_records: list[dict[str, Any]] = []

    for index, qasm_file in enumerate(qasm_files, start=1):
        output_csv = output_dir / f"{qasm_file.stem}.csv"
        resource_snapshot_path = resource_snapshot_dir / f"{qasm_file.stem}.json"
        if resource_snapshot_path.exists():
            resource_snapshot_path.unlink()
        if output_csv.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"[{index}/{len(qasm_files)}] skip existing {qasm_file.name}")
            continue

        print(f"[{index}/{len(qasm_files)}] running {qasm_file.name}")
        started = time.time()

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--backend",
            args.backend,
            "--qasm-file",
            str(qasm_file),
            "--shots",
            str(args.shots),
            "--timeout",
            str(args.timeout),
            "--simulation-method",
            args.simulation_method,
            "--simulator-device",
            args.simulator_device,
            "--max-parallel-threads",
            str(args.max_parallel_threads),
        ]
        if args.transpile_optimization_level is not None:
            command.extend(
                ["--transpile-optimization-level", str(args.transpile_optimization_level)]
            )
        if args.seed_transpiler is not None:
            command.extend(["--seed-transpiler", str(args.seed_transpiler)])
        if args.max_qubits is not None:
            command.extend(["--max-qubits", str(args.max_qubits)])
        if transpiled_qasm_dir is not None:
            command.extend(["--transpiled-qasm-dir", str(transpiled_qasm_dir)])
        command.extend(["--resource-snapshot-file", str(resource_snapshot_path)])

        try:
            completed = subprocess.run(
                command,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=int(args.timeout),
            )
            elapsed = time.time() - started
        except subprocess.TimeoutExpired as exc:
            timeouts += 1
            record = {
                "file": str(qasm_file),
                "name": qasm_file.name,
                "backend": args.backend,
                "shots": int(args.shots),
                "status": "timeout",
                "timeout_s": int(args.timeout),
                "elapsed_s": round(time.time() - started, 3),
                "stdout_tail": safe_tail(exc.stdout),
                "stderr_tail": safe_tail(exc.stderr),
            }
            merge_snapshot_usage(record, resource_snapshot_path)
            append_status(raw_status_path, record)
            append_status(resource_failure_path, resource_summary_record(record))
            tidy_records.append(record)
            write_tidy_csv(tidy_path, tidy_records)
            print(f"  timeout after {args.timeout}s")
            if resource_snapshot_path.exists():
                resource_snapshot_path.unlink()
            continue

        payload = last_json_line(completed.stdout)
        stderr_tail = safe_tail(completed.stderr)
        stdout_tail = safe_tail(completed.stdout)

        if completed.returncode != 0 and payload is None:
            failures += 1
            record = {
                "file": str(qasm_file),
                "name": qasm_file.name,
                "backend": args.backend,
                "shots": int(args.shots),
                "status": "crash",
                "returncode": completed.returncode,
                "elapsed_s": round(elapsed, 3),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
            merge_snapshot_usage(record, resource_snapshot_path)
            append_status(raw_status_path, record)
            append_status(resource_failure_path, resource_summary_record(record))
            tidy_records.append(record)
            write_tidy_csv(tidy_path, tidy_records)
            print(f"  crash returncode={completed.returncode}")
            if resource_snapshot_path.exists():
                resource_snapshot_path.unlink()
            continue

        if payload is None:
            failures += 1
            record = {
                "file": str(qasm_file),
                "name": qasm_file.name,
                "backend": args.backend,
                "shots": int(args.shots),
                "status": "no_json_result",
                "returncode": completed.returncode,
                "elapsed_s": round(elapsed, 3),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
            merge_snapshot_usage(record, resource_snapshot_path)
            append_status(raw_status_path, record)
            append_status(resource_failure_path, resource_summary_record(record))
            tidy_records.append(record)
            write_tidy_csv(tidy_path, tidy_records)
            print("  no structured result returned")
            if resource_snapshot_path.exists():
                resource_snapshot_path.unlink()
            continue

        record = {
            "file": str(qasm_file),
            "name": qasm_file.name,
            "backend": args.backend,
            "elapsed_s": round(elapsed, 3),
            "returncode": completed.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        record.update(payload)
        record.setdefault("error", None)
        raw_record = dict(record)

        record.pop("stdout_tail", None)
        record.pop("stderr_tail", None)

        if payload.get("status") == "success":
        
            write_result_csv(output_csv, record, args.backend)
            success += 1
            print(
                f"  success time_taken={payload['time_taken']:.6f} "
                f"transpile_wall_s={payload.get('transpile_wall_s')} "
                f"run_wall_s={payload.get('run_wall_s')} "
                f"method={payload.get('simulation_method')} "
                f"qubits={payload.get('num_qubits')}"
            )
        elif str(payload.get("status", "")).startswith("skipped_"):
            skipped_qubits += 1
            print(f"  {payload['status']} qubits={payload.get('num_qubits')}")
        else:
            failures += 1
            print(f"  {payload.get('status')} error={payload.get('error')}")

        append_status(raw_status_path, raw_record)
        if is_success_record(record):
            append_status(resource_usage_path, resource_summary_record(record))
        else:
            append_status(resource_failure_path, resource_summary_record(record))
        tidy_records.append(record)
        write_tidy_csv(tidy_path, tidy_records)
        if resource_snapshot_path.exists():
            resource_snapshot_path.unlink()

    write_tidy_csv(tidy_path, tidy_records)
    print()
    print("summary")
    print(f"  success = {success}")
    print(f"  skipped_existing = {skipped_existing}")
    print(f"  skipped_qubits = {skipped_qubits}")
    print(f"  timeouts = {timeouts}")
    print(f"  failures = {failures}")
    print(f"  raw_status_file = {raw_status_path}")
    print(f"  tidy_file = {tidy_path}")
    print(f"  resource_usage_file = {resource_usage_path}")
    print(f"  resource_failure_file = {resource_failure_path}")
    print(f"  resource_snapshot_dir = {resource_snapshot_dir}")
    print(f"  transpiled_qasm_dir = {transpiled_qasm_dir}")
    print(f"  output_dir = {output_dir}")
    return 0

def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)

if __name__ == "__main__":
    raise SystemExit(main())
