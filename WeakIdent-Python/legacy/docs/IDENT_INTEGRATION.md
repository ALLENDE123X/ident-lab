# IDENT Method Integration Guide

This document explains how to integrate new PDE identification methods into the PDE-Selector framework.

## Quick Start

1. Create a new class inheriting from `IdentMethodBase`
2. Implement `name` property and `run()` method
3. Register your method with `METHOD_REGISTRY`

```python
from src.ident_methods import IdentMethodBase, METHOD_REGISTRY
import numpy as np

class MyMethod(IdentMethodBase):
    @property
    def name(self) -> str:
        return "MyMethod"
    
    def run(self, u_win, dx, dt, **kwargs):
        # Your identification algorithm here
        # ...
        
        # Return metrics and info
        metrics = np.array([f1_score, coeff_error, residual])
        info = {"terms": [...], "coefficients": {...}}
        return metrics, info

# Register the method
METHOD_REGISTRY.register(MyMethod())
```

## Contract Details

### Input Specification

| Parameter | Type | Description |
|-----------|------|-------------|
| `u_win` | `np.ndarray (nt, nx)` | Spatiotemporal data window |
| `dx` | `float` | Spatial grid spacing |
| `dt` | `float` | Temporal grid spacing |
| `max_dx` | `int` | Maximum derivative order (kwargs) |
| `max_poly` | `int` | Maximum polynomial order (kwargs) |
| `tau` | `float` | Regularization threshold (kwargs) |
| `true_coeffs` | `dict` | Ground truth coefficients (optional) |

### Output Specification

**metrics** (required): `np.ndarray` of shape `(3,)`
- `[0]` = F1 score (structure accuracy, 0-1, higher = better)
- `[1]` = Coefficient error (relative L2, 0+, lower = better)  
- `[2]` = Residual MSE (reconstruction error, 0+, lower = better)

**info** (required): `dict` with:
- `"terms"`: `List[str]` — identified PDE terms (e.g., `["u_x", "u*u_x", "u_xx"]`)
- `"coefficients"`: `Dict[str, float]` — coefficient for each term
- `"runtime"`: `float` — execution time in seconds

## Using Registered Methods

```python
from src.ident_methods import METHOD_REGISTRY

# Check available methods
print(METHOD_REGISTRY.list_methods())  # ["WeakIDENT", "MyMethod"]

# Get and run a method
method = METHOD_REGISTRY.get_or_raise("MyMethod")
metrics, info = method.run(u_window, dx=0.01, dt=0.001)
```

## Adding to Config

After registering, add your method to `config/default.yaml`:

```yaml
methods:
  - WeakIDENT
  - MyMethod  # Your new method
```

## Testing Your Method

Create a test in `tests/test_ident_methods.py`:

```python
def test_my_method():
    from src.ident_methods import METHOD_REGISTRY
    
    u_win = np.random.randn(64, 128)
    method = METHOD_REGISTRY.get("MyMethod")
    
    metrics, info = method.run(u_win, dx=0.01, dt=0.01)
    
    assert metrics.shape == (3,)
    assert "terms" in info
    assert "coefficients" in info
```
