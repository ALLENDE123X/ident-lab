# Examples Directory

This directory contains usage examples and tutorials for the PDE-Selector framework.

## Contents

| File | Description |
|------|-------------|
| `quickstart.py` | Load a pre-trained model and select a method for new data |
| `train_from_scratch.py` | Full pipeline: generate data → train → evaluate → save |
| `notebooks/demo.ipynb` | Interactive Jupyter notebook tutorial |

## Quick Start

### Using a Pre-trained Model

```python
from src.models import PerMethodRegressor
from src.features import extract_tiny12

# Load trained model
selector = PerMethodRegressor.load("models/selector.joblib")

# Your data (shape: nt x nx)
u_data = load_your_pde_data()
dx, dt = 0.01, 0.001

# Extract features
features = extract_tiny12(u_data, dx, dt)

# Get prediction and uncertainty
pred, uncertainty = selector.predict_unc(features.reshape(1, -1))
print(f"Predicted method: {selector.best_method(pred)}")
print(f"Uncertainty: {uncertainty[0]:.3f}")
```

### Training from Scratch

```bash
# 1. Generate labeled dataset
python scripts/make_dataset.py --cfg config/default.yaml --output artifacts/

# 2. Train selector
python scripts/train_selector.py --cfg config/default.yaml --data artifacts/ --output models/

# 3. Evaluate
python scripts/evaluate_selector.py --cfg config/default.yaml --model models/selector.joblib --data artifacts/
```

## Data Format

Input data should be a 2D NumPy array with shape `(nt, nx)`:
- `nt`: number of time points
- `nx`: number of spatial points

Grid spacings `dx` and `dt` should be provided separately.

## Adding New Examples

When adding new examples:
1. Include clear docstrings explaining the purpose
2. Use the `config/default.yaml` as a starting point
3. Handle errors gracefully with informative messages
4. Add comments for non-obvious steps
