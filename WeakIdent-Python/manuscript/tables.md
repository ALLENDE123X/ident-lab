# Tables for Tower Manuscript

> Data extracted from `data/results/full_dataset_4methods.csv` and `data/results/model_comparison.csv`

---

## Table 1: Dataset Composition

| PDE Type | Window Count | Percentage |
|----------|-------------|------------|
| KdV (Korteweg-de Vries) | 1,734 | 30.0% |
| Heat Equation | 1,647 | 28.5% |
| Transport-Diffusion | 1,221 | 21.1% |
| Kuramoto-Sivashinsky | 1,184 | 20.5% |
| **Total** | **5,786** | **100%** |

---

## Table 2: Best Method Distribution

| Method | Best Count | Percentage |
|--------|-----------|------------|
| LASSO | 3,644 | 63.0% |
| STLSQ | 2,136 | 36.9% |
| WeakIDENT | 4 | 0.07% |
| RobustIDENT | 2 | 0.03% |

*Note: LASSO and STLSQ dominate because they achieve lower reconstruction error on clean synthetic data. This motivates the need for a selector—a naive baseline achieves only 63% accuracy.*

---

## Table 3: Method Performance (e2 Error)

| Method | Mean e2 | Median e2 | Min e2 | Max e2 |
|--------|---------|-----------|--------|--------|
| LASSO | 0.252 | 0.094 | 0.000006 | 1.000 |
| STLSQ | 0.436 | 0.349 | 0.000058 | 1.000 |
| RobustIDENT | 0.975 | 1.000 | 0.006 | 1.000 |
| WeakIDENT | 6978.8* | 1.003 | 0.587 | 21027850.3 |

*WeakIDENT mean is inflated by extreme outliers; median is more representative.*

---

## Table 4: Method Runtime

| Method | Mean Runtime (s) | Total Runtime (s) |
|--------|------------------|-------------------|
| STLSQ | 0.0055 | 32.11 |
| RobustIDENT | 0.202 | 1,170.22 |
| LASSO | 0.331 | 1,912.68 |
| WeakIDENT | 0.402 | 2,327.00 |

*STLSQ is ~73× faster than WeakIDENT on average.*

---

## Table 5: ML Model Comparison

| Model | Test Accuracy | 5-Fold CV Mean | CV Std Dev |
|-------|---------------|----------------|------------|
| Random Forest | **97.06%** | 87.85% | ±12.52% |
| Gradient Boosting | 95.68% | 87.64% | ±12.21% |
| KNN (k=5) | 94.99% | 87.18% | ±11.62% |
| Logistic Regression | 89.46% | 88.42% | ±10.92% |
| SVM (RBF) | 88.69% | 86.30% | ±12.73% |
| Ridge Classifier | 88.00% | 86.45% | ±10.67% |

---

## Table 6: IDENT Methods Compared

| Method | Type | Key Strength | Key Weakness |
|--------|------|--------------|--------------|
| LASSO | L1-regularized regression | Fast, good on clean data | Sensitive to noise |
| STLSQ | Thresholded least squares | Very fast, interpretable | Requires threshold tuning |
| RobustIDENT | ADMM L1 optimization | Robust to outliers | Slower, less accurate here |
| WeakIDENT | Weak formulation | Noise-robust, derivative-free | Slowest, metric outliers |

---

## Table 7: Regret Statistics (Full Dataset)

| Metric | Value |
|--------|-------|
| Zero-Regret Rate | 99.4% (5,752 / 5,786) |
| Mean Regret | 0.0002 |
| Max Regret | 0.4396 |

*Note: These statistics were computed on the full dataset after training. For rigorous evaluation, held-out test split regret should be reported (see Limitations).*

---

## Table 8: Baseline Comparison

| Strategy | Accuracy |
|----------|----------|
| Random Forest Selector | **97.06%** |
| Always LASSO (naive) | 63.0% |
| Always STLSQ | 36.9% |
| Random Choice | 25.0% |

*The selector achieves +34 percentage points over the naive baseline.*
