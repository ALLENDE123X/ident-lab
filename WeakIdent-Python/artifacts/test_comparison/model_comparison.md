# Model Comparison Results

## Summary

Comparison of 6 ML models for predicting IDENT method errors.

### WeakIDENT

| Model | MSE | MAE | R² | Train Time |
|-------|-----|-----|-----|------------|
| Ridge ** | 0.0118 | 0.0856 | 0.4897 | 0.00s |
| RandomForest | 0.0123 | 0.0874 | 0.4649 | 0.13s |
| XGBoost | 0.0130 | 0.0909 | 0.4373 | 0.11s |
| SVR | 0.0156 | 0.0961 | 0.3248 | 0.01s |
| KNN | 0.0175 | 0.1058 | 0.2414 | 0.00s |
| MLP | 0.0239 | 0.1249 | -0.0384 | 0.22s |

### PySINDy

| Model | MSE | MAE | R² | Train Time |
|-------|-----|-----|-----|------------|
| Ridge ** | 0.0148 | 0.0971 | 0.2736 | 0.00s |
| RandomForest | 0.0152 | 0.0959 | 0.2517 | 0.18s |
| SVR | 0.0167 | 0.1038 | 0.1767 | 0.01s |
| KNN | 0.0167 | 0.1014 | 0.1763 | 0.00s |
| XGBoost | 0.0179 | 0.1067 | 0.1163 | 0.08s |
| MLP | 0.0426 | 0.1589 | -1.0993 | 0.07s |

### RobustIDENT

| Model | MSE | MAE | R² | Train Time |
|-------|-----|-----|-----|------------|
| RandomForest ** | 0.0131 | 0.0917 | 0.4090 | 0.17s |
| Ridge | 0.0136 | 0.0907 | 0.3854 | 0.00s |
| XGBoost | 0.0147 | 0.0970 | 0.3332 | 0.07s |
| KNN | 0.0193 | 0.1106 | 0.1274 | 0.00s |
| SVR | 0.0198 | 0.1154 | 0.1055 | 0.01s |
| MLP | 0.0312 | 0.1439 | -0.4129 | 0.24s |

### WSINDy

| Model | MSE | MAE | R² | Train Time |
|-------|-----|-----|-----|------------|
| Ridge ** | 0.0119 | 0.0834 | 0.5329 | 0.00s |
| RandomForest | 0.0124 | 0.0860 | 0.5135 | 0.13s |
| XGBoost | 0.0127 | 0.0875 | 0.5031 | 0.07s |
| SVR | 0.0157 | 0.1007 | 0.3848 | 0.01s |
| KNN | 0.0197 | 0.1147 | 0.2281 | 0.00s |
| MLP | 0.0269 | 0.1300 | -0.0496 | 0.43s |

## Recommendation

**Best overall model: ridge** (avg MSE: 0.0130)