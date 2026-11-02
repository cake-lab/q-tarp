# Q-TARP

**Transpilation-Aware Runtime Prediction for Noisy Quantum Circuit Simulation.**
Q-TARP evaluates three graph neural network representations:

- **Source:** a DAG and global features from the original circuit.
- **Hybrid:** the source DAG/features plus 13 backend-specific transpiled features.
- **Transpiled:** a DAG and global features from the saved transpiled circuit.

The workflow is **execute circuits, build datasets, then train**. The
[1,402 source QASM circuits](data/quantum_circuits_original) are included. 

## Setup

Executing the pipeline requires:

- Python 3.9
- NumPy 1.23.5
- pandas 2.1.4
- PyTorch 1.13.1
- PyTorch Geometric 2.5.2
- Qiskit 0.44.2
- Qiskit Aer 0.11.2.

The exact dependency versions are listed in [`environment.yml`](./environment.yml).

## Workflow

### 1. Execute Circuits

Choose an optimization level (0-3). This runs both fake backends and saves successful times, source/transpiled features, and transpiled QASM. Repeat the workflow for each optimization level.

```bash
OPT_LEVEL=0
for BACKEND in washington sherbrooke; do
  python scripts/run_bulk_execution.py \
    --backend "$BACKEND" \
    --input-dir data/quantum_circuits_original \
    --output-dir "data/${BACKEND}_opt${OPT_LEVEL}_results" \
    --tidy-file "data/extractedfiles/${BACKEND}_opt${OPT_LEVEL}_tidy.csv" \
    --transpiled-qasm-dir "data/transpiled_qasm/${BACKEND}_opt${OPT_LEVEL}" \
    --shots 1024 --timeout 900 --simulation-method automatic \
    --transpile-optimization-level "$OPT_LEVEL" --seed-transpiler 1234
done
```

### 2. Build Source and Hybrid Datasets

Use the outputs from step 1. 
```bash
python scripts/build_gnn_training_data.py \
  --tidy-csv \
    "washington=data/extractedfiles/washington_opt${OPT_LEVEL}_tidy.csv" \
    "sherbrooke=data/extractedfiles/sherbrooke_opt${OPT_LEVEL}_tidy.csv" \
  --source-qasm-dir data/quantum_circuits_original \
  --source-metadata-output "data/source_opt${OPT_LEVEL}_metadata.csv" \
  --hybrid-metadata-output "data/hybrid_opt${OPT_LEVEL}_metadata.csv" \
  --source-dataset-output "data/training_data_source_opt${OPT_LEVEL}.npy" \
  --hybrid-dataset-output "data/training_data_hybrid_opt${OPT_LEVEL}.npy"
```

The `.npy` files contain pickled lists of PyG graphs. 

### 3. Build the Transpiled Dataset

Use the Hybrid metadata and saved transpiled QASM from the above steps.

```bash
python model_standard_kfold_transpiled_graph/build_transpiled_graph_dataset.py \
  --metadata-csv "data/hybrid_opt${OPT_LEVEL}_metadata.csv" \
  --dataset-output "data/training_data_transpiled_opt${OPT_LEVEL}.npy" \
  --filtered-metadata-output "data/transpiled_opt${OPT_LEVEL}_metadata.csv" \
  --backend-qasm-dir \
    "washington=data/transpiled_qasm/washington_opt${OPT_LEVEL}" \
    "sherbrooke=data/transpiled_qasm/sherbrooke_opt${OPT_LEVEL}"
```

### 4. Train

```bash
for REPRESENTATION in source hybrid; do
  python model_standard_kfold/train_standard.py \
    --dataset-file "training_data_${REPRESENTATION}_opt${OPT_LEVEL}.npy" \
    --metadata-csv "data/${REPRESENTATION}_opt${OPT_LEVEL}_metadata.csv" \
    --seed 1234
done

python model_standard_kfold_transpiled_graph/train_standard.py \
  --dataset-file "training_data_transpiled_opt${OPT_LEVEL}.npy" \
  --metadata-csv "data/transpiled_opt${OPT_LEVEL}_metadata.csv" \
  --seed 1234
```

Training selects a candidate 80/20 train-test split with seed 1234, performs
five-fold cross-validation within the training portion, then evaluates on the
held-out test set. ML baseline models are in `model_tabular_baselines/`;


## Recorded Timings

These [Washington](data/executions_washington.csv) and [Sherbrooke](data/executions_sherbrooke.csv)
CSVs has the exact recorded timings without requiring new executions.


| CSV column | Meaning |
| --- | --- |
| `name` | Filename in `data/quantum_circuits_original/`. |
| `device` | `washington` or `sherbrooke`, identifying the fake backend model. |
| `transpile_optimization_level` | Optimization level, 0-3. |
| `shots` | Requested shots, always 1,024. |
| `time_taken` | Aer-reported execution time in seconds; the prediction target. |
| `transpile_wall_s` | Measured Qiskit transpilation time in seconds. |


## Citation

Accepted at the [Second AAAI Symposium on Quantum Information & Machine Learning (QIML 2026)](https://aaai-qiml.github.io/), part of the AAAI Fall Symposium Series.

If you use Q-TARP or its dataset in your research, please cite
[our paper](https://arxiv.org/abs/2609.12980):

```bibtex
@misc{azizov2026transpilationaware,
  title         = {{Transpilation-Aware Runtime Prediction for Noisy Quantum Circuit Simulation}},
  author        = {Azizov, Davud and Vela-Tambo, Javier and Guo, Tian},
  year          = {2026},
  eprint        = {2609.12980},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2609.12980},
  url           = {https://arxiv.org/abs/2609.12980}
}
```

## Acknowledgements

This implementation builds on or references the following projects:

- [Quantum-Execution-Time-Prediction (MOOSE Lab)](https://github.com/mooselab/Quantum-Execution-Time-Prediction)
- [mqt-bench](https://github.com/cda-tum/mqt-bench)

