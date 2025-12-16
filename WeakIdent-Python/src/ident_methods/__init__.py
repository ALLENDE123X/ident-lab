"""
IDENT Methods Package

This package provides a plugin-style architecture for integrating
different PDE identification methods (WeakIDENT, RobustIDENT, WSINDy, etc.)
into the PDE-Selector framework.

Each method must implement the IdentMethodBase interface and register
itself in the METHOD_REGISTRY.

Example:
    from src.ident_methods import METHOD_REGISTRY, IdentMethodBase
    
    class MyNewMethod(IdentMethodBase):
        @property
        def name(self) -> str:
            return "MyNewMethod"
        
        def run(self, u_win, dx, dt, **kwargs):
            # Your implementation here
            metrics = np.array([f1, coeff_err, residual])
            info = {"terms": [...], "coefficients": {...}}
            return metrics, info
    
    METHOD_REGISTRY.register(MyNewMethod())
"""

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY

__all__ = ["IdentMethodBase", "METHOD_REGISTRY"]
