#!/usr/bin/env python3
import argparse
import re
import math
import numpy as np
from typing import Dict, Tuple

EPS = 1e-12

# ---------- parsing summary.txt ----------

SUMMARY_FIELD_RE = {
    "pde_slug": re.compile(r"^\s*pde_slug:\s*(.*)\s*$"),
    "subsample_id": re.compile(r"^\s*subsample_id:\s*(.*)\s*$"),
    "created_at": re.compile(r"^\s*created_at:\s*(.*)\s*$"),
    "index_ranges": re.compile(r"^\s*index_ranges:\s*x=(\d+)-(\d+),\s*t=(\d+)-(\d+)\s*$"),
    "shapes": re.compile(r"^\s*shapes:\s*Nx=(\d+),\s*Nt=(\d+)\s*$"),
    "config": re.compile(r"^\s*config:\s*(.*)\s*$"),
}

def parse_config_kv(config_str: str) -> Dict[str, str]:
    # config looks like: "max_dx=6, max_poly=6, use_cross_der=False, stride_x=1, stride_t=1, tau=0.05, sigma_SNR=0.0"
    kv = {}
    for part in config_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv

def parse_identification_error(lines: list) -> Tuple[float, float, float, float, float]:
    # Look for the row like:
    # |  0 | 0.00155145 | 0.0129741 | 0.00507101 |       1 |       1 |
    # robust regex: capture 5 numbers after the first " |  0 | "
    row_pattern = re.compile(
        r"^\s*\|\s*0\s*\|\s*([0-9Ee+\-\.]+)\s*\|\s*([0-9Ee+\-\.]+)\s*\|\s*([0-9Ee+\-\.]+)\s*\|\s*([0-9Ee+\-\.]+)\s*\|\s*([0-9Ee+\-\.]+)\s*\|\s*$"
    )
    for ln in lines:
        m = row_pattern.match(ln)
        if m:
            e2 = float(m.group(1))
            einf = float(m.group(2))
            eres = float(m.group(3))
            tpr = float(m.group(4))
            ppv = float(m.group(5))
            return e2, einf, eres, tpr, ppv
    raise ValueError("Could not parse identification error row from summary.txt")

def parse_summary(path: str):
    with open(path, "r") as f:
        lines = f.read().splitlines()

    meta = {}
    for ln in lines:
        for key, rx in SUMMARY_FIELD_RE.items():
            m = rx.match(ln)
            if m:
                if key == "index_ranges":
                    x0, x1, t0, t1 = map(int, m.groups())
                    meta.update({"x0": x0, "x1": x1, "t0": t0, "t1": t1})
                elif key == "shapes":
                    Nx, Nt = map(int, m.groups())
                    meta.update({"Nx": Nx, "Nt": Nt})
                elif key == "config":
                    meta["config"] = parse_config_kv(m.group(1))
                else:
                    meta[key] = m.group(1)
    # errors
    e2, einf, eres, tpr, ppv = parse_identification_error(lines)
    meta.update({"e2": e2, "einf": einf, "eres": eres, "tpr": tpr, "ppv": ppv})
    # tau
    tau = float(meta["config"].get("tau", "0.05"))
    meta["tau"] = tau
    # polys and derivative orders
    meta["max_dx"] = int(meta["config"].get("max_dx", "6"))
    meta["max_poly"] = int(meta["config"].get("max_poly", "6"))
    return meta

# ---------- core helpers ----------

def orient_to_x_t(u: np.ndarray, Nx: int, Nt: int) -> np.ndarray:
    """
    Ensure array is shape (Nx, Nt). If it's (Nt, Nx), transpose. If it already matches, return as-is.
    """
    if u.shape == (Nx, Nt):
        return u
    if u.shape == (Nt, Nx):
        return u.T
    # try squeezing extra dims
    us = np.squeeze(u)
    if us.shape == (Nx, Nt):
        return us
    if us.shape == (Nt, Nx):
        return us.T
    raise ValueError(f"Unexpected npy shape {u.shape}; cannot match Nx={Nx}, Nt={Nt}")

def spectral_derivative_x(U: np.ndarray, order: int) -> np.ndarray:
    """
    Periodic spectral derivative along x for a field U(x,t) with shape (Nx, Nt).
    Assumes unit spacing dx=1; scaling cancels after column-standardization.
    """
    if order == 0:
        return U.copy()
    Nx, Nt = U.shape
    k = 2.0 * np.pi * np.fft.fftfreq(Nx, d=1.0)  # radians per unit
    ik_pow = (1j * k) ** order
    Uhat = np.fft.fft(U, axis=0)
    Dhat = Uhat * ik_pow[:, None]
    D = np.fft.ifft(Dhat, axis=0).real
    return D

def central_time_derivative(U: np.ndarray, tau: float) -> np.ndarray:
    """
    Central difference in time: returns shape (Nx, Nt-2), aligned to times t=1..Nt-2
    """
    return (U[:, 2:] - U[:, :-2]) / (2.0 * tau)

def moving_average_along_t(A: np.ndarray, win: int) -> np.ndarray:
    """
    Simple moving average along t axis for each x row. Reflect-pad at edges.
    A shape: (Nx, T)
    """
    Nx, T = A.shape
    win = max(1, min(win, T if T % 2 == 1 else T - 1))
    if win < 1:
        return A.copy()
    radius = win // 2
    out = np.empty_like(A)
    for i in range(Nx):
        row = A[i]
        # reflect pad
        left = row[1:radius+1][::-1] if radius > 0 else np.array([])
        right = row[-radius-1:-1][::-1] if radius > 0 else np.array([])
        padded = np.concatenate([left, row, right])
        kernel = np.ones(win) / win
        sm = np.convolve(padded, kernel, mode='valid')  # length T
        out[i] = sm
    return out

def zscore_columns(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score each column (mean 0, std 1). Returns (Wz, means, stds).
    Columns with std < EPS are dropped from Wz.
    """
    means = W.mean(axis=0)
    stds = W.std(axis=0) + 0.0
    keep = stds > 1e-12
    Wz = (W[:, keep] - means[keep]) / (stds[keep] + EPS)
    return Wz, means[keep], stds[keep]

def corr_with_vector(Wz: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """
    Pearson correlation between z-scored columns Wz and z-scored vector vz.
    Since both are standardized, correlation = mean of products.
    """
    m = Wz.shape[0]
    return (Wz.T @ vz) / m

# ---------- build library W ----------

def build_library_W(u_sub: np.ndarray, max_dx: int, max_poly: int) -> Tuple[np.ndarray, int]:
    """
    Build PDE-FIND/WeakIdent-style library:
      columns: [1, u, u_x..u_x^max_dx, (u^k), (u^k)_x..(u^k)_{x^max_dx} for k=2..max_poly]
    Returns W with shape (m, p) where m = Nx*(Nt-2) and p matches the above columns.
    """
    Nx, Nt = u_sub.shape
    # time-aligned slice for spatial features (match central du/dt times)
    u_tslice = u_sub[:, 1:-1]  # shape (Nx, Nt-2)

    cols = []

    def add_col(A):
        cols.append(A.reshape(-1, order='F'))  # flatten with x fastest (not critical)

    # constant and u
    add_col(np.ones_like(u_tslice))
    add_col(u_tslice)

    # u_x .. up to order max_dx
    for n in range(1, max_dx + 1):
        add_col(spectral_derivative_x(u_tslice, n))

    # (u^k) and its x-derivatives, for k=2..max_poly
    for k in range(2, max_poly + 1):
        uk = u_tslice ** k
        add_col(uk)  # n=0
        for n in range(1, max_dx + 1):
            add_col(spectral_derivative_x(uk, n))

    W = np.column_stack(cols)  # (m, p)
    return W, W.shape[1]

# ---------- predictors (8) and y ----------

def compute_predictors_and_y(u: np.ndarray, meta: Dict) -> Dict:
    Nx = meta["Nx"]; Nt = meta["Nt"]
    x0, x1, t0, t1 = meta["x0"], meta["x1"], meta["t0"], meta["t1"]
    tau = meta["tau"]
    max_dx = meta["max_dx"]; max_poly = meta["max_poly"]

    # extract subsample and orient
    U = orient_to_x_t(u, Nx, Nt)
    Usub = U[x0:x1+1, t0:t1+1]  # (Nx_sub, Nt_sub)
    Nx_sub, Nt_sub = Usub.shape

    # time derivative (central), aligned to t indices 1..Nt_sub-2
    Ut = central_time_derivative(Usub, tau)  # (Nx_sub, Nt_sub-2)
    m = Nx_sub * (Nt_sub - 2)

    # build library W on the same time indices
    W, p_total = build_library_W(Usub, max_dx=max_dx, max_poly=max_poly)

    # zscore W, drop constant/near-constant cols automatically
    Wz, _, _ = zscore_columns(W)
    p_eff = Wz.shape[1]
    if p_eff < 2:
        raise ValueError("Too few informative columns after standardization.")

    # zscore u_t (vectorized)
    ut_vec = Ut.reshape(-1, order='F')
    ut_mean = ut_vec.mean()
    ut_std = ut_vec.std() + EPS
    utz = (ut_vec - ut_mean) / ut_std

    # 1) log kappa2(Wz)
    svals = np.linalg.svd(Wz, full_matrices=False, compute_uv=False)
    smax = svals[0]
    smin = svals[-1] if svals[-1] > 0 else EPS
    log_kappa2 = math.log((smax + EPS) / (smin + EPS))

    # 2) effective rank ratio
    ps = (svals**2) / (np.sum(svals**2) + EPS)
    H = -np.sum(ps * np.log(ps + EPS))
    eff_rank = math.exp(H)
    eff_rank_ratio = eff_rank / p_eff

    # 3) max column coherence (off-diagonal)
    G = (Wz.T @ Wz) / Wz.shape[0]  # correlation matrix since z-scored
    np.fill_diagonal(G, 0.0)
    max_coherence = float(np.max(np.abs(G)))

    # 4) projection alignment rho
    beta, *_ = np.linalg.lstsq(Wz, utz, rcond=None)
    proj = Wz @ beta
    rho = float(np.linalg.norm(proj) / (np.linalg.norm(utz) + EPS))

    # 5) top correlation and 6) margin
    r = corr_with_vector(Wz, utz)
    r_abs_sorted = np.sort(np.abs(r))[::-1]
    top_corr = float(r_abs_sorted[0])
    corr_margin = float(r_abs_sorted[0] - (r_abs_sorted[1] if len(r_abs_sorted) > 1 else 0.0))

    # 7) SNR of u_t (robust, MA smooth along t)
    win = 5 if (Nt_sub - 2) >= 5 else (3 if (Nt_sub - 2) >= 3 else 1)
    Ut_smooth = moving_average_along_t(Ut, win=win)
    signal_var = float(np.var(Ut_smooth))
    noise_var = float(np.var(Ut - Ut_smooth))
    snr_ut = (signal_var + EPS) / (noise_var + EPS)

    # 8) complexity ratio
    complexity_ratio = p_eff / m

    # y: soft-worst aggregator of the 5 metrics from summary
    e2, einf, eres, tpr, ppv = meta["e2"], meta["einf"], meta["eres"], meta["tpr"], meta["ppv"]
    m1, m2, m3 = e2, einf, eres
    m4, m5 = (1.0 - tpr), (1.0 - ppv)
    lamb = 4.0
    w = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    ms = np.array([m1, m2, m3, m4, m5])
    y = float((1.0 / lamb) * np.log(np.sum(w * np.exp(lamb * ms))))

    return {
        "subsample_id": meta.get("subsample_id", ""),
        "Nx_sub": Nx_sub, "Nt_sub": Nt_sub, "m": m, "p_total": p_total, "p_eff": p_eff,
        "log_kappa2": log_kappa2,
        "eff_rank_ratio": eff_rank_ratio,
        "max_coherence": max_coherence,
        "rho": rho,
        "top_corr": top_corr,
        "corr_margin": corr_margin,
        "snr_ut": snr_ut,
        "complexity_ratio": complexity_ratio,
        "y": y
    }

def load_u_array(path: str, Nx_expected: int, Nt_expected: int) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    def pick_2d_numeric(x):
        if isinstance(x, np.ndarray) and x.ndim == 2 and np.issubdtype(x.dtype, np.number):
            return x
        return None

    # case A: already a numeric 2D array
    cand = pick_2d_numeric(arr)
    if cand is not None:
        return cand

    # case B: object array wrapping something
    obj = None
    try:
        obj = arr.item()  # works for 0-d object arrays
    except Exception:
        obj = arr

    # dict-like?
    if isinstance(obj, dict):
        # try common keys first
        for k in ("u", "U", "field", "data", "u_xt", "Uxt", "array"):
            if k in obj:
                cand = pick_2d_numeric(np.array(obj[k]))
                if cand is not None:
                    return cand
        # else scan values
        for v in obj.values():
            cand = pick_2d_numeric(np.array(v))
            if cand is not None:
                return cand

    # tuple/list-like?
    if isinstance(obj, (list, tuple)):
        for v in obj:
            cand = pick_2d_numeric(np.array(v))
            if cand is not None:
                return cand

    # last resort: scan any nested arrays inside the original arr
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        flat = arr.ravel()
        for v in flat:
            cand = pick_2d_numeric(np.array(v))
            if cand is not None:
                return cand

    raise ValueError(
        f"Could not find a 2D numeric array in {path}. "
        f"Got type={type(arr)}, dtype={getattr(arr, 'dtype', None)}, shape={getattr(arr, 'shape', None)}"
    )


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Compute 8 predictors + y for a Burgers subsample using summary.txt metadata.")
    ap.add_argument("--npy", type=str, default="burgers_viscous.npy", help="path to burgers_viscous.npy")
    ap.add_argument("--summary", type=str, default="summary.txt", help="path to summary.txt")
    args = ap.parse_args()

    meta = parse_summary(args.summary)
    u = load_u_array(args.npy, meta["Nx"], meta["Nt"])


    results = compute_predictors_and_y(u, meta)

    print(f"subsample: {results['subsample_id']}")
    print(f"window shape: Nx={results['Nx_sub']}, Nt={results['Nt_sub']}  -> rows m={results['m']}, total cols p={results['p_total']}, effective cols p_eff={results['p_eff']}")
    print("\n8 predictors:")
    print(f"  1) log_kappa2(W)        : {results['log_kappa2']:.6f}")
    print(f"  2) eff_rank_ratio        : {results['eff_rank_ratio']:.6f}")
    print(f"  3) max_coherence         : {results['max_coherence']:.6f}")
    print(f"  4) projection_alignment  : {results['rho']:.6f}")
    print(f"  5) top_corr              : {results['top_corr']:.6f}")
    print(f"  6) corr_margin           : {results['corr_margin']:.6f}")
    print(f"  7) snr_u_t               : {results['snr_ut']:.6f}")
    print(f"  8) complexity_ratio p/m  : {results['complexity_ratio']:.8f}")
    print("\ncombined score:")
    print(f"  y (soft-worst of 5 errs) : {results['y']:.6f}")

if __name__ == "__main__":
    main()
