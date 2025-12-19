import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from scipy.fft import rfft, irfft, rfftfreq
from scipy.linalg import lstsq
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import GradientBoostingRegressor


# -----------------------------
# Features for a window
# -----------------------------
def extract_window_features(u_win: np.ndarray, dx: float, dt: float,
                            library_size: int, poly_deg: int, deriv_order_max: int,
                            K_rel: float = 0.1, tau_active: float = 0.01,
                            noise_std: Optional[float] = None) -> Dict[str, float]:
    """
    Compute 12+ predictors for a (t,x) window.
    """
    T, X = u_win.shape
    feats: Dict[str, float] = {}
    feats['N_snap'] = float(T)
    feats['grid_res'] = float(X)
    feats['dt'] = float(dt)
    feats['SNR_dB'] = float(estimate_snr_db(u_win, noise_std))

    # "IC" = first frame in this window
    u0 = u_win[0, :]
    k, E0 = energy_spectrum_1d(u0, dx=dx)
    feats['IC_total_variation'] = float(total_variation_1d(u0))
    feats['IC_spectral_entropy'] = float(spectral_entropy(E0))
    feats['IC_active_modes_tau'] = float(active_modes(E0, tau=tau_active))
    # Low-frequency fraction at K = K_rel * Nyquist
    K_index = int(np.floor(K_rel * (len(E0) - 1)))
    feats['IC_lowfreq_frac'] = float(lowfreq_fraction(E0, K_index))

    # Dynamics spectrum: average over time
    Et = []
    for ti in range(T):
        _, Ei = energy_spectrum_1d(u_win[ti, :], dx=dx)
        Et.append(Ei)
    Eavg = np.mean(np.stack(Et, axis=0), axis=0)
    # band energy ratio
    feats['band_energy_ratio'] = float(lowfreq_fraction(Eavg, K_index) / (1.0 - lowfreq_fraction(Eavg, K_index) + 1e-12))
    # spectral slope p over a mid-range (avoid DC and Nyquist)
    idx_lo = max(1, int(0.05 * len(Eavg)))
    idx_hi = max(idx_lo + 5, int(0.5 * len(Eavg)))
    feats['spec_slope_p'] = float(fit_loglog_slope(k, Eavg, idx_lo, idx_hi))

    # Library/meta
    feats['library_size'] = float(library_size)
    feats['poly_deg'] = float(poly_deg)
    feats['deriv_order_max'] = float(deriv_order_max)

    return feats

# -----------------------------
# Labels for a window
# -----------------------------
def compute_metrics(coeff_hat: np.ndarray,
                    term_names: List[str],
                    true_coeffs: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Compute e2, TPR, PPV comparing discovered coefficients to true ones.
    true_coeffs: mapping from term_name -> true_value (0 for absent terms omitted).
    """
    # Align vectors
    xi_hat = []
    xi_true = []
    for name in term_names:
        xi_hat.append(coeff_hat[term_names.index(name)])
        xi_true.append(true_coeffs.get(name, 0.0))
    xi_hat = np.array(xi_hat)
    xi_true = np.array(xi_true)

    # e2 (relative L2 coefficient error) on the *true support*
    true_support = np.where(np.abs(xi_true) > 0)[0]
    if true_support.size == 0:
        e2 = float(np.linalg.norm(xi_hat) / (np.linalg.norm(xi_true) + 1e-12))
    else:
        e2 = float(np.linalg.norm(xi_hat[true_support] - xi_true[true_support]) / (np.linalg.norm(xi_true[true_support]) + 1e-12))

    # sets for TPR/PPV
    eps = 1e-6
    S_true = set(np.where(np.abs(xi_true) > eps)[0].tolist())
    S_hat = set(np.where(np.abs(xi_hat) > eps)[0].tolist())

    tp = len(S_true & S_hat)
    fp = len(S_hat - S_true)
    fn = len(S_true - S_hat)

    TPR = tp / (tp + fn + 1e-12)  # recall
    PPV = tp / (tp + fp + 1e-12)  # precision
    return e2, float(TPR), float(PPV)

# -----------------------------
# Sliding windows & dataset build
# -----------------------------
@dataclass
class PDEData:
    name: str
    u: np.ndarray   # shape (T, X)
    dx: float
    dt: float
    noise_std: Optional[float]
    true_coeffs: Dict[str, float]  # term_name -> value for this PDE

