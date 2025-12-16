# Baseline Benchmark

This experiment establishes baseline performance metrics for the PDE-Selector.

## Running the Benchmark

```bash
# Full benchmark (may take 1+ hours)
python scripts/run_benchmark.py \
    --cfg experiments/baseline_benchmark/config.yaml \
    --output experiments/baseline_benchmark/outputs \
    --parallel 4

# Quick test run (~5 minutes)
python scripts/run_benchmark.py \
    --cfg experiments/baseline_benchmark/config.yaml \
    --output experiments/baseline_benchmark/outputs_quick \
    --parallel 4 \
    --quick
```

## Expected Outputs

After running, the output directory will contain:
- `benchmark_results.json` — Metrics (regret, top-1 accuracy, compute saved)
- `provenance.json` — Git hash, pip freeze, timestamp
- `regret_cdf_rf_multi.png` — CDF plot of regret values
- `confusion_matrix_rf_multi.png` — Chosen vs. oracle best
- `data/` — Generated dataset
- `models/` — Trained selector models

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Regret | E[score_chosen - score_best] | < 0.2 |
| Top-1 Accuracy | Fraction where chosen = best | > 80% |
| Compute Saved | Fraction where only 1 method run | > 50% |

## Configuration

See `config.yaml` for:
- 2 PDEs (Burgers, KdV)
- 4 noise levels (0%, 1%, 2%, 5%)
- 2 parameter sets per PDE
- RandomForest with 300 trees
