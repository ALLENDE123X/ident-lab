#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDE Recoverability Pipeline (single script)
------------------------------------------
End-to-end pipeline that:
  1) Takes PDE-generated data (u(x,t)) with noise.
  2) Builds overlapping space-time windows (samples).
  3) Extracts 12+ window-level predictors (spectral + meta).
  4) Runs PDE discovery (STRidge) on each window to get coefficients.
  5) Computes labels y = [e2, TPR, PPV] against ground truth (synthetic demo).
  6) Constructs X, y across multiple PDEs / trajectories.
  7) Trains multi-output models with leakage-safe grouped CV.
  8) Prints metrics and feature importances.

Dependencies: numpy, scipy, pandas, scikit-learn
(Optionally matplotlib if you extend plotting.)

NOTE: This script focuses on 1D periodic PDEs for clarity. You can extend
features and library construction to 2D/3D.

Author: ChatGPT (for Pranav / Dr. Kang meeting demo)
"""
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
# Utility math helpers
# -----------------------------
def finite_diff(u: np.ndarray, axis: int, dx: float, order: int = 1, periodic: bool = True) -> np.ndarray:
    """
    Finite difference derivative along 'axis' (1D centered, periodic by default).
    u: array (..., X, ...)
    Returns d^order u / dx^order
    """
    v = np.copy(u)
    for _ in range(order):
        if periodic:
            # centered difference with wrap
            roll_f = np.roll(v, -1, axis=axis)
            roll_b = np.roll(v,  1, axis=axis)
            v = (roll_f - roll_b) / (2.0*dx)
        else:
            # simple non-periodic (one-sided at boundaries)
            slicer_f = [slice(None)] * v.ndim
            slicer_b = [slice(None)] * v.ndim
            slicer_c = [slice(None)] * v.ndim
            slicer_f[axis] = slice(2, None)
            slicer_b[axis] = slice(None, -2)
            slicer_c[axis] = slice(1, -1)
            dv = np.zeros_like(v)
            dv[tuple(slicer_c)] = (v[tuple(slicer_f)] - v[tuple(slicer_b)]) / (2.0*dx)
            # copy edge gradients
            idx0 = [slice(None)]*v.ndim; idx0[axis]=0
            idx1 = [slice(None)]*v.ndim; idx1[axis]=1
            iden = [slice(None)]*v.ndim; iden[axis]=-1
            iden2= [slice(None)]*v.ndim; iden2[axis]=-2
            dv[tuple(idx0)] = (v[tuple(idx1)] - v[tuple(idx0)]) / dx
            dv[tuple(iden)] = (v[tuple(iden)] - v[tuple(iden2)]) / dx
            v = dv
    return v

def time_derivative(u: np.ndarray, dt: float) -> np.ndarray:
    """
    Central difference in time for u[t, x]. Drops first and last frame.
    Returns ut[t=1..T-2, x].
    """
    return (u[2:, :] - u[:-2, :]) / (2.0 * dt)

def total_variation_1d(u0: np.ndarray) -> float:
    """ Total variation of a 1D array. """
    return np.sum(np.abs(np.diff(u0)))

def energy_spectrum_1d(u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D energy spectrum E(k) of a real signal u(x).
    Returns k (1D wavenumbers for rFFT) and E(k).
    """
    n = u.shape[-1]
    # rFFT returns N//2+1 components
    U = rfft(u)
    E = (np.abs(U)**2) / n
    k = 2.0 * np.pi * rfftfreq(n, d=dx)  # angular wavenumber
    return k, E

def spectral_entropy(E: np.ndarray, eps: float = 1e-12) -> float:
    """ Entropy of (nonnegative) spectrum E. """
    S = np.maximum(E, 0.0)
    Z = np.sum(S) + eps
    p = S / Z
    return float(-(p * np.log(p + eps)).sum())

def active_modes(E: np.ndarray, tau: float = 0.01) -> int:
    """ Count modes with energy ≥ tau * max(E). """
    if E.size == 0:
        return 0
    thr = tau * float(E.max())
    return int((E >= thr).sum())

def lowfreq_fraction(E: np.ndarray, K_index: int) -> float:
    """ Fraction of energy in low band up to index K_index (inclusive). """
    K_index = int(np.clip(K_index, 0, len(E)-1))
    num = float(E[:K_index+1].sum())
    den = float(E.sum()) + 1e-12
    return num / den

def fit_loglog_slope(k: np.ndarray, E: np.ndarray, idx_lo: int, idx_hi: int) -> float:
    """ Fit slope p in log E ~ p log k on [idx_lo, idx_hi]. """
    idx_lo = int(np.clip(idx_lo, 1, len(k)-1))
    idx_hi = int(np.clip(idx_hi, idx_lo+1, len(k)-1))
    kk = k[idx_lo:idx_hi]
    EE = E[idx_lo:idx_hi]
    # filter out zeros
    mask = (kk > 0) & (EE > 0)
    if mask.sum() < 3:
        return 0.0
    X = np.log(kk[mask])
    Y = np.log(EE[mask])
    A = np.vstack([X, np.ones_like(X)]).T
    p, _ = lstsq(A, Y)[:2]  # slope, intercept
    return float(p[0])

def estimate_snr_db(u: np.ndarray, noise_std: Optional[float]) -> float:
    """
    Estimate SNR in dB. If noise_std is provided, use Var(signal)/noise^2.
    Otherwise, estimate variance via high-frequency tail as a crude proxy.
    u is a 2D window [t, x].
    """
    signal_var = float(np.var(u))
    if noise_std is not None and noise_std > 0:
        snr_linear = signal_var / float(noise_std**2 + 1e-12)
        return 10.0 * np.log10(snr_linear + 1e-12)
    # crude fallback: take spectrum of a random frame and use top 10% bins
    u0 = u[0]
    k, E = energy_spectrum_1d(u0, dx=1.0)  # dx factor cancels in ratio
    tail_start = int(0.9 * len(E))
    noise_est = float(np.mean(E[tail_start:])) if tail_start < len(E) else float(np.mean(E))
    snr_linear = signal_var / (noise_est + 1e-12)
    return 10.0 * np.log10(snr_linear + 1e-12)

# -----------------------------
# PDE library construction
# -----------------------------
def spectral_derivative(u: np.ndarray, dx: float, order: int) -> np.ndarray:
    """
    Compute spatial derivative of given order using spectral differentiation.
    u[t, x], periodic domain. Returns array same shape.
    """
    t, n = u.shape
    k = 2.0 * np.pi * rfftfreq(n, d=dx)
    U = np.fft.rfft(u, axis=1)
    deriv_factor = (1j * k) ** order
    dU = U * deriv_factor[np.newaxis, :]
    du = np.fft.irfft(dU, n=n, axis=1)
    return du.real

def build_library(u: np.ndarray, dx: float, dt: float,
                  poly_deg: int = 2, deriv_order_max: int = 2) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Construct candidate library Theta and target ut from window u[t,x].
    Uses pointwise samples at interior times for central time derivative.
    Terms: 1, u, u^2, ..., u^P; for each d=1..D include u^(p) * u_{x..x}^{(d)} for p=0..P
    (P = poly_deg, D = deriv_order_max).
    Returns:
      Theta: (M, K) design matrix (M samples across time-space)
      ut:    (M,) time derivative at same samples
      names: list of K term names (strings)
    """
    T, X = u.shape
    # Compute ut at interior times
    ut_full = time_derivative(u, dt)  # (T-2, X)
    # Align u and derivatives at t=1..T-2
    u_mid = u[1:-1, :]  # (T-2, X)

    # Prepare base powers u^p
    upows = [np.ones_like(u_mid)]
    for p in range(1, poly_deg + 1):
        upows.append(upows[-1] * u_mid)  # u^p
    # Derivative terms for d=1..D
    derivs = [None]  # placeholder for d=0
    for d in range(1, deriv_order_max + 1):
        derivs.append(spectral_derivative(u[1:-1, :], dx, d))  # align t

    # Construct Theta
    terms = []
    names = []
    # constant and polynomial terms (no derivative)
    for p in range(0, poly_deg + 1):
        terms.append(upows[p].reshape(-1, 1))
        names.append('u^%d' % p)
    # u^p * u^{(d)}
    for d in range(1, deriv_order_max + 1):
        for p in range(0, poly_deg + 1):
            terms.append((upows[p] * derivs[d]).reshape(-1, 1))
            names.append(f'u^{p} * d{d}u/dx{d}')

    Theta = np.hstack(terms)  # (M, K)
    ut = ut_full.reshape(-1)  # (M,)
    return Theta, ut, names

def stridge(Theta: np.ndarray, y: np.ndarray, lam: float = 1e-6,
            thresh: float = 1e-4, max_iter: int = 10, normalize: bool = True) -> np.ndarray:
    """
    Sequentially thresholded ridge regression.
    """
    # Optionally normalize columns
    if normalize:
        col_scale = np.linalg.norm(Theta, axis=0) + 1e-12
        Theta_n = Theta / col_scale
    else:
        col_scale = np.ones(Theta.shape[1])
        Theta_n = Theta

    # Initial ridge
    I = np.eye(Theta_n.shape[1])
    coef = np.linalg.solve(Theta_n.T @ Theta_n + lam * I, Theta_n.T @ y)

    for _ in range(max_iter):
        small = np.abs(coef) < thresh
        if not np.any(small):
            break
        keep = ~small
        if keep.sum() == 0:
            break
        Theta_k = Theta_n[:, keep]
        coef_k = np.linalg.solve(Theta_k.T @ Theta_k + lam * np.eye(keep.sum()), Theta_k.T @ y)
        coef = np.zeros_like(coef)
        coef[keep] = coef_k

    # un-normalize
    coef = coef / col_scale
    return coef

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

def sliding_windows_1d(u: np.ndarray, Wt: int, St: int) -> List[Tuple[int, int]]:
    """
    Return list of (t_start, t_end_exclusive) windows along time axis. Overlap allowed.
    """
    T = u.shape[0]
    bounds = []
    t = 0
    while t + Wt <= T:
        bounds.append((t, t + Wt))
        t += St
    return bounds

def build_dataset(pdes: List[PDEData], Wt: int, St: int,
                  poly_deg: int, deriv_order_max: int,
                  ridge_lam: float = 1e-6, ridge_thresh: float = 1e-4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    For each PDE dataset, cut windows, extract features, run discovery, compute labels.
    Returns:
      X_df: features per window
      Y_df: labels per window (e2, TPR, PPV)
      groups: group id per row for leakage-safe splitting (pde_name + block index)
    """
    X_rows: List[Dict[str, Any]] = []
    Y_rows: List[Dict[str, Any]] = []
    groups: List[str] = []

    for pde in pdes:
        windows = sliding_windows_1d(pde.u, Wt=Wt, St=St)
        # library size depends on poly/deriv caps
        # size = (poly_deg+1) + ((poly_deg+1)*deriv_order_max)
        library_size = (poly_deg + 1) * (1 + deriv_order_max)
        for w_idx, (t0, t1) in enumerate(windows):
            u_win = pde.u[t0:t1, :]
            # Extract features
            feats = extract_window_features(u_win, dx=pde.dx, dt=pde.dt,
                                            library_size=library_size, poly_deg=poly_deg,
                                            deriv_order_max=deriv_order_max,
                                            noise_std=pde.noise_std)
            feats['pde_name'] = pde.name
            feats['window_idx'] = w_idx

            # Build library and run STRidge discovery
            Theta, ut, term_names = build_library(u_win, dx=pde.dx, dt=pde.dt,
                                                  poly_deg=poly_deg, deriv_order_max=deriv_order_max)
            coef_hat = stridge(Theta, ut, lam=ridge_lam, thresh=ridge_thresh, max_iter=10)

            # Compute labels (vs ground truth coefficients)
            e2, TPR, PPV = compute_metrics(coef_hat, term_names, pde.true_coeffs)

            X_rows.append(feats)
            Y_rows.append({'e2': e2, 'TPR': TPR, 'PPV': PPV})
            groups.append(f"{pde.name}_block{w_idx}")

    X_df = pd.DataFrame(X_rows)
    Y_df = pd.DataFrame(Y_rows)
    groups_series = pd.Series(groups, name='group')
    return X_df, Y_df, groups_series

# -----------------------------
# Models & training helpers
# -----------------------------
def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def evaluate_models(X_df: pd.DataFrame, Y_df: pd.DataFrame, groups: pd.Series, numeric_cols: List[str]) -> None:
    """
    Train/eval a few multi-output models with grouped 5-fold CV.
    Prints R^2 per target and average.
    """
    X = X_df[numeric_cols].values.astype(float)
    # Transform targets
    eps = 1e-6
    Y = np.column_stack([
        np.log(Y_df['e2'].values + eps),
        logit(Y_df['TPR'].values, eps=eps),
        logit(Y_df['PPV'].values, eps=eps),
    ])

    # Models
    models = {
        'MultiTaskElasticNet': make_pipeline(StandardScaler(with_mean=True, with_std=True),
                                             MultiTaskElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000)),
        'RandomForest(multioutput)': RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        'RegChain(GBR base)': RegressorChain(base_estimator=GradientBoostingRegressor(random_state=0), order=[0,1,2])
    }

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for name, model in models.items():
        r2_scores = []
        r2_e2, r2_tpr, r2_ppv = [], [], []
        for train_idx, test_idx in gkf.split(X, Y, groups=groups.values):
            model_fit = model.fit(X[train_idx], Y[train_idx])
            Y_pred = model_fit.predict(X[test_idx])

            # Compute R^2 per target
            for j, (lst, tag) in enumerate(zip([r2_e2, r2_tpr, r2_ppv], ['e2','TPR','PPV'])):
                y_true = Y[test_idx, j]
                y_pred = Y_pred[:, j]
                ss_res = float(np.sum((y_true - y_pred)**2))
                ss_tot = float(np.sum((y_true - np.mean(y_true))**2) + 1e-12)
                lst.append(1.0 - ss_res/ss_tot)
        r2_e2_m = float(np.mean(r2_e2))
        r2_tpr_m = float(np.mean(r2_tpr))
        r2_ppv_m = float(np.mean(r2_ppv))
        r2_avg = float(np.mean([r2_e2_m, r2_tpr_m, r2_ppv_m]))
        print(f"\nModel: {name}")
        print(f"  R2 log(e2): {r2_e2_m:.3f}")
        print(f"  R2 logit(TPR): {r2_tpr_m:.3f}")
        print(f"  R2 logit(PPV): {r2_ppv_m:.3f}")
        print(f"  Avg R2: {r2_avg:.3f}")

        # Feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            print("  Feature importances (first 10):")
            for c, v in zip(numeric_cols[:10], importances[:10]):
                print(f"    {c}: {v:.4f}")
        # RandomForest in pipeline? No, but our RF isn't in a pipeline.
        if isinstance(model, RandomForestRegressor):
            fi = model.feature_importances_
            print("  RF feature importances (top 10):")
            idx = np.argsort(fi)[::-1][:10]
            for i in idx:
                print(f"    {numeric_cols[i]}: {fi[i]:.4f}")

# -----------------------------
# Synthetic PDE generators (demo)
# -----------------------------
def rk4_step(u: np.ndarray, dt: float, rhs_func, *rhs_args):
    k1 = rhs_func(u, *rhs_args)
    k2 = rhs_func(u + 0.5*dt*k1, *rhs_args)
    k3 = rhs_func(u + 0.5*dt*k2, *rhs_args)
    k4 = rhs_func(u + dt*k3, *rhs_args)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rhs_diffusion(u: np.ndarray, nu: float, dx: float) -> np.ndarray:
    return nu * finite_diff(u, axis=-1, dx=dx, order=2, periodic=True)

def rhs_burgers(u: np.ndarray, nu: float, dx: float) -> np.ndarray:
    ux = finite_diff(u, axis=-1, dx=dx, order=1, periodic=True)
    uxx = finite_diff(u, axis=-1, dx=dx, order=2, periodic=True)
    return -u * ux + nu * uxx

def simulate_pde_1d(name: str, X: int = 256, T: int = 200, L: float = 2*np.pi,
                     dt: float = 1e-3, nu: float = 0.1, seed: int = 0) -> Tuple[np.ndarray, float, float]:
    """
    Simple 1D periodic PDE simulator returning u[t,x] for a given equation.
    name: 'diffusion' or 'burgers'
    """
    rng = np.random.default_rng(seed)
    dx = L / X
    x = np.linspace(0, L, X, endpoint=False)
    # Random smooth initial condition: band-limited Fourier series
    kmax = 6
    coeffs = rng.normal(0, 1, size=kmax) * np.exp(-0.5*np.arange(1, kmax+1))
    u0 = np.zeros_like(x)
    for k in range(1, kmax+1):
        u0 += coeffs[k-1] * np.cos(2*np.pi*k*x/L)
    u = np.zeros((T, X))
    u[0] = u0
    # Time integration
    for t in range(1, T):
        if name == 'diffusion':
            u[t] = rk4_step(u[t-1], dt, rhs_diffusion, nu, dx)
        elif name == 'burgers':
            u[t] = rk4_step(u[t-1], dt, rhs_burgers, nu, dx)
        else:
            raise ValueError("Unknown PDE name")
    return u, dx, dt

def make_true_coeffs(name: str, poly_deg: int, deriv_order_max: int) -> Dict[str, float]:
    """
    Map our library term names to the true coefficients for the selected PDE.
    For our library, terms include:
      'u^0', 'u^1', ... 'u^P', and 'u^p * d{d}u/dx{d}' for d=1..D, p=0..P
    Diffusion: u_t = nu * u_xx  => coefficient for p=0,d=2 term is nu
    Burgers:   u_t = -u u_x + nu u_xx => coefficient for p=1,d=1 is -1 and p=0,d=2 is nu
    """
    coeffs: Dict[str, float] = {}
    # zero by default for all terms (we only define the nonzeros)
    if name == 'diffusion':
        coeffs['u^0 * d2u/dx2'] = 0.1  # default nu used in simulator
    elif name == 'burgers':
        coeffs['u^1 * d1u/dx1'] = -1.0
        coeffs['u^0 * d2u/dx2'] = 0.1
    # polynomial-only terms are zero in these PDEs
    for p in range(0, poly_deg+1):
        coeffs.setdefault(f'u^{p}', 0.0)
    for d in range(1, deriv_order_max+1):
        for p in range(0, poly_deg+1):
            coeffs.setdefault(f'u^{p} * d{d}u/dx{d}', coeffs.get(f'u^{p} * d{d}u/dx{d}', 0.0))
    return coeffs

# -----------------------------
# Main demo
# -----------------------------
def main_demo():
    # Config
    poly_deg = 2
    deriv_order_max = 2
    Wt = 40     # window length (time frames)
    St = 20     # stride
    noise_std = 0.02

    # Generate two PDE datasets (diffusion + burgers)
    u1, dx1, dt1 = simulate_pde_1d('diffusion', X=256, T=240, dt=1e-3, nu=0.1, seed=1)
    u2, dx2, dt2 = simulate_pde_1d('burgers',   X=256, T=240, dt=1e-3, nu=0.1, seed=2)

    # Add measurement noise
    rng = np.random.default_rng(42)
    u1n = u1 + noise_std * rng.standard_normal(u1.shape)
    u2n = u2 + noise_std * rng.standard_normal(u2.shape)

    # True coefficient maps for our library basis
    true1 = make_true_coeffs('diffusion', poly_deg, deriv_order_max)
    true2 = make_true_coeffs('burgers',   poly_deg, deriv_order_max)

    pde_list = [
        PDEData(name='diffusion', u=u1n, dx=dx1, dt=dt1, noise_std=noise_std, true_coeffs=true1),
        PDEData(name='burgers',   u=u2n, dx=dx2, dt=dt2, noise_std=noise_std, true_coeffs=true2),
    ]

    # Build dataset
    X_df, Y_df, groups = build_dataset(
        pdes=pde_list,
        Wt=Wt, St=St,
        poly_deg=poly_deg, deriv_order_max=deriv_order_max,
        ridge_lam=1e-6, ridge_thresh=5e-4
    )

    # Keep only numeric feature columns for modeling
    drop_cols = ['pde_name', 'window_idx']
    numeric_cols = [c for c in X_df.columns if c not in drop_cols]

    print("Dataset built.")
    print(f"X: {X_df.shape}, Y: {Y_df.shape}, groups: {groups.nunique()} unique blocks")
    print("First few feature columns:", numeric_cols[:8])
    print("\nSample rows:\n", pd.concat([X_df.head(2), Y_df.head(2)], axis=1))

    # Evaluate models with grouped CV
    evaluate_models(X_df, Y_df, groups, numeric_cols)

if __name__ == "__main__":
    main_demo()
