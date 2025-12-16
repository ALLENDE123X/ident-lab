# PDE-Selector: WeakIDENT Only - Google Colab
# =============================================
# Dedicated script for running WeakIDENT (the slowest method).
# Run this separately while other methods run elsewhere.

# ============== SETUP ==============
!pip install pyyaml joblib tqdm pandas numpy scipy==1.11.4 numpy_indexed tabulate --quiet

# Clone the repository
!git clone https://github.com/ALLENDE123X/ident-lab.git
%cd ident-lab/WeakIdent-Python

# ============== CONFIGURATION ==============
WINDOWS_PER_PDE = 1000  # Samples per PDE
OUTPUT_DIR = "/content/weakident_results"

# PDEs to process (comment out any you want to skip)
PDES_TO_RUN = [
    "burgers",
    "kdv", 
    "heat",
    "ks",
    "transport",
    "nls",
    # "duffing",      # ODEs - usually fast
    # "vanderpol",
    # "lorenz",
]

# ============== IMPORTS ==============
import sys
import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any
from tqdm import tqdm

sys.path.insert(0, ".")

# Import WeakIDENT directly from model.py
from model import weak_ident_pred

# ============== PDE CONFIGURATIONS ==============
PDE_CONFIGS = {
    "burgers": {
        "data_file": "dataset-Python/burgers_viscous.npy",
        "data_format": "object_array",
        "grid": {"dx": 0.0625, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": np.array([np.array([[1, 1, 0, -1.0], [1, 2, 0, 0.01]])], dtype=object)
    },
    "kdv": {
        "data_file": "dataset-Python/KdV.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.05, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": np.array([np.array([[1, 1, 0, -1.0], [1, 3, 0, -1.0]])], dtype=object)
    },
    "heat": {
        "data_file": "dataset-Python/heat.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.0427, "dt": 0.002},
        "windows": {"size_x": 48, "size_t": 80},
        "true_coefficients": np.array([np.array([[1, 2, 0, 1.0]])], dtype=object)
    },
    "ks": {
        "data_file": "dataset-Python/KS.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.39, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": np.array([np.array([[1, 1, 0, -1.0], [1, 2, 0, -1.0], [1, 4, 0, -1.0]])], dtype=object)
    },
    "transport": {
        "data_file": "dataset-Python/transportDiff.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.04, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": np.array([np.array([[1, 1, 0, -1.0], [1, 2, 0, 0.01]])], dtype=object)
    },
    "nls": {
        "data_file": "dataset-Python/NLS.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.1, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": np.array([np.array([[1, 2, 0, 1.0]])], dtype=object)
    },
}

# ============== HELPERS ==============
@dataclass
class Window:
    data: np.ndarray
    x_start: int
    t_start: int
    window_id: str
    dx: float
    dt: float

def load_pde_data(filepath, data_format):
    data = np.load(filepath, allow_pickle=True)
    if data_format == "nested_object":
        return np.array(data[0], dtype=np.float64)
    elif data_format == "object_array":
        u = np.array(data.tolist(), dtype=np.float64)
        return u[0] if u.ndim == 3 and u.shape[0] == 1 else u
    return data

def extract_windows(u, window_size, pde_name, target_count, dx, dt):
    if u.ndim == 3:
        u = u[:, u.shape[1]//2, :]
    nx, nt = u.shape
    size_x, size_t = window_size
    
    approx_stride = int(np.sqrt((nx * nt) / target_count))
    stride_x = max(1, min(approx_stride, (nx - size_x) // 10 + 1))
    stride_t = max(1, min(approx_stride, (nt - size_t) // 10 + 1))
    
    windows = []
    idx = 0
    for i_x in range(0, max(1, nx - size_x), stride_x):
        for i_t in range(0, max(1, nt - size_t), stride_t):
            window_data = u[i_x:i_x+size_x, i_t:i_t+size_t]
            if window_data.shape == (size_x, size_t):
                windows.append(Window(
                    data=window_data, x_start=i_x, t_start=i_t,
                    window_id=f"{pde_name}_{idx:05d}", dx=dx, dt=dt
                ))
                idx += 1
                if idx >= target_count:
                    return windows
    return windows

def run_weakident_on_window(window, true_coefficients, max_dx=4, max_poly=4, tau=0.05):
    """Run WeakIDENT on a single window."""
    u_win = window.data  # (nx, nt)
    nx, nt = u_win.shape
    
    # Prepare for weak_ident_pred
    u_hat = np.array([u_win], dtype=object)  # WeakIDENT expects this format
    x = np.linspace(0, nx * window.dx, nx).reshape(-1, 1)
    t = np.linspace(0, nt * window.dt, nt).reshape(1, -1)
    xs = np.array([x, t], dtype=object)
    
    start = time.time()
    try:
        df_errors, df_eqns, df_coeffs, run_time = weak_ident_pred(
            u_hat=u_hat,
            xs=xs,
            true_coefficients=true_coefficients,
            max_dx=max_dx,
            max_poly=max_poly,
            skip_x=4,
            skip_t=4,
            use_cross_der=False,
            tau=tau,
        )
        
        # Extract metrics
        e2 = df_errors["$e_2$"].values[0]
        tpr = df_errors["$tpr$"].values[0]
        ppv = df_errors["$ppv$"].values[0]
        e_res = df_errors["$e_{res}$"].values[0]
        
        f1 = 2 * tpr * ppv / (tpr + ppv) if (tpr + ppv) > 0 else 0.0
        
        return {
            "f1": float(f1) if np.isfinite(f1) else 0.0,
            "e2": float(e2) if np.isfinite(e2) else 1.0,
            "residual": float(e_res) if np.isfinite(e_res) else 1.0,
            "runtime": time.time() - start,
            "tpr": float(tpr),
            "ppv": float(ppv),
        }
    except Exception as e:
        return {
            "f1": 0.0, "e2": 1.0, "residual": 1.0,
            "runtime": time.time() - start,
            "error": str(e)
        }

# ============== FEATURE EXTRACTION ==============
def extract_features(window_data, dx, dt):
    u = window_data.T  # (nt, nx)
    nt, nx = u.shape
    features = np.zeros(12)
    try:
        u_x = np.gradient(u, dx, axis=1)
        u_xx = np.gradient(u_x, dx, axis=1)
        u_xxx = np.gradient(u_xx, dx, axis=1)
        features[0], features[1], features[2] = np.std(u_x), np.std(u_xx), np.std(u_xxx)
        
        u_t = np.gradient(u, dt, axis=0)
        u_tt = np.gradient(u_t, dt, axis=0)
        features[3], features[4], features[5] = np.std(u_t), np.std(u_tt), np.max(np.abs(u_t))
        
        fft_mag = np.abs(np.fft.fft2(u))
        features[6] = np.log1p(fft_mag[:nt//4, :nx//4].mean())
        features[7] = np.log1p(fft_mag[nt//4:nt//2, nx//4:nx//2].mean())
        features[8] = np.log1p(fft_mag[nt//2:, nx//2:].mean())
        
        features[9] = np.std(u)
        features[10] = np.mean(np.abs(u_xx)) / (np.std(u) + 1e-8)
        features[11] = np.max(u) - np.min(u)
    except:
        pass
    return features

# ============== PROCESS PDE ==============
def process_pde(pde_name, config):
    print(f"\n{'='*60}\n🔬 WeakIDENT: {pde_name.upper()}\n{'='*60}")
    
    if not os.path.exists(config["data_file"]):
        print(f"  ⚠️ File not found")
        return None
    
    u = load_pde_data(config["data_file"], config["data_format"])
    print(f"  Data shape: {u.shape}")
    
    w_cfg = config["windows"]
    windows = extract_windows(
        u, (w_cfg["size_x"], w_cfg["size_t"]),
        pde_name, WINDOWS_PER_PDE,
        config["grid"]["dx"], config["grid"]["dt"]
    )
    print(f"  Windows: {len(windows)}")
    
    results = []
    for w in tqdm(windows, desc=f"  {pde_name}"):
        # Run WeakIDENT
        metrics = run_weakident_on_window(w, config["true_coefficients"])
        
        # Extract features
        features = extract_features(w.data, w.dx, w.dt)
        
        result = {
            "window_id": w.window_id,
            "pde_type": pde_name,
            "window_x_start": w.x_start,
            "window_t_start": w.t_start,
        }
        for i, f in enumerate(features):
            result[f"feat_{i}"] = float(f)
        for k, v in metrics.items():
            result[f"WeakIDENT_{k}"] = v
        
        results.append(result)
    
    return pd.DataFrame(results)

# ============== RUN ==============
print("\n" + "="*60)
print("🚀 WEAKIDENT DATASET GENERATION")
print("="*60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
all_dfs = []

for pde_name in PDES_TO_RUN:
    if pde_name not in PDE_CONFIGS:
        print(f"⚠️ Unknown PDE: {pde_name}")
        continue
    
    start = time.time()
    df = process_pde(pde_name, PDE_CONFIGS[pde_name])
    elapsed = time.time() - start
    
    if df is not None and len(df) > 0:
        df.to_csv(f"{OUTPUT_DIR}/{pde_name}_weakident.csv", index=False)
        all_dfs.append(df)
        print(f"  ✅ {len(df)} samples in {elapsed:.1f}s ({elapsed/len(df):.2f}s/window)")

# Combine
if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(f"{OUTPUT_DIR}/weakident_all.csv", index=False)
    print(f"\n✅ COMPLETE: {len(combined)} samples -> {OUTPUT_DIR}/weakident_all.csv")
    
    # Download
    from google.colab import files
    files.download(f"{OUTPUT_DIR}/weakident_all.csv")
