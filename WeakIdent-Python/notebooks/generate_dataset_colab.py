# PDE-Selector Dataset Generation - Google Colab Script
# ======================================================
# Run this entire cell in Google Colab to generate the dataset.
# Note: TPU won't help here (CPU-bound), but Colab's multi-core CPU will.

#@title Setup and Install Dependencies
!git clone https://github.com/ALLENDE123X/ident-lab.git
%cd ident-lab/WeakIdent-Python

# Install dependencies
!pip install -q numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.0 pysindy==1.7.5 pyyaml tqdm joblib pandas

# Install the package
!pip install -q -e .

#@title Verify Installation
import numpy as np
import sys
sys.path.insert(0, '.')

from src.ident_methods import METHOD_REGISTRY
print(f"Registered methods: {METHOD_REGISTRY.list_methods()}")

#@title Generate Dataset (All 4 PDEs, 2500 windows each)
# This will take 1-3 hours depending on Colab resources

import os
import time
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
PDES = ["burgers", "kdv", "heat", "ks"]
WINDOWS_PER_PDE = 2500  # Set to 100 for quick test
N_JOBS = 4  # Colab usually has 2-4 cores
OUTPUT_DIR = "data/results"
METHODS = ["WeakIDENT", "PySINDy", "WSINDy", "RobustIDENT"]

# ============================================================
# PDE CONFIGURATIONS (inline)
# ============================================================
PDE_CONFIGS = {
    "burgers": {
        "data_file": "dataset-Python/burgers_viscous.npy",
        "data_format": "object_array",
        "grid": {"nx": 256, "nt": 814, "dx": 0.0625, "dt": 0.001},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": 0.01},
        "windows": {"size_x": 64, "size_t": 100, "stride_x": 8, "stride_t": 14},
        "ident_params": {"max_dx": 4, "max_poly": 4, "tau": 0.05},
    },
    "kdv": {
        "data_file": "dataset-Python/KdV.npy",
        "data_format": "nested_object",
        "grid": {"nx": 400, "nt": 601, "dx": 0.05, "dt": 0.001},
        "true_coefficients": {"u*u_x": -1.0, "u_xxx": -1.0},
        "windows": {"size_x": 64, "size_t": 100, "stride_x": 12, "stride_t": 20},
        "ident_params": {"max_dx": 4, "max_poly": 4, "tau": 0.05},
    },
    "heat": {
        "data_file": "dataset-Python/heat.npy",
        "data_format": "nested_object",
        "grid": {"nx": 234, "nt": 501, "dx": 0.0427, "dt": 0.002},
        "true_coefficients": {"u_xx": 1.0},
        "windows": {"size_x": 48, "size_t": 80, "stride_x": 7, "stride_t": 12},
        "ident_params": {"max_dx": 4, "max_poly": 4, "tau": 0.05},
    },
    "ks": {
        "data_file": "dataset-Python/KS.npy",
        "data_format": "nested_object",
        "grid": {"nx": 256, "nt": 301, "dx": 0.39, "dt": 0.01},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0},
        "windows": {"size_x": 64, "size_t": 80, "stride_x": 8, "stride_t": 8},
        "ident_params": {"max_dx": 4, "max_poly": 4, "tau": 0.05},
    },
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_pde_data(filepath: str, data_format: str) -> np.ndarray:
    """Load PDE data from .npy file."""
    data = np.load(filepath, allow_pickle=True)
    
    if data_format == "nested_object":
        return data[0]
    elif data_format == "object_array":
        u = np.array(data.tolist(), dtype=np.float64)
        if u.ndim == 3 and u.shape[0] == 1:
            u = u[0]
        return u
    return data


def extract_windows(u, window_size, stride, pde_name, target_count, dx, dt):
    """Extract overlapping windows from spatiotemporal data."""
    nx, nt = u.shape
    size_x, size_t = window_size
    stride_x, stride_t = stride
    
    n_windows_x = (nx - size_x) // stride_x + 1
    n_windows_t = (nt - size_t) // stride_t + 1
    
    # Adjust stride if needed
    if n_windows_x * n_windows_t < target_count:
        stride_x = max(1, (nx - size_x) // int(np.sqrt(target_count)) + 1)
        stride_t = max(1, (nt - size_t) // int(np.sqrt(target_count)) + 1)
        n_windows_x = (nx - size_x) // stride_x + 1
        n_windows_t = (nt - size_t) // stride_t + 1
    
    windows = []
    idx = 0
    for i_x in range(n_windows_x):
        for i_t in range(n_windows_t):
            x_start = i_x * stride_x
            t_start = i_t * stride_t
            window_data = u[x_start:x_start+size_x, t_start:t_start+size_t].T
            
            windows.append({
                "data": window_data,
                "window_id": f"{pde_name}_{idx:05d}",
                "pde_name": pde_name,
                "x_start": x_start,
                "t_start": t_start,
                "dx": dx,
                "dt": dt,
            })
            idx += 1
            if idx >= target_count:
                return windows
    return windows


def extract_features(window_data, dx, dt):
    """Extract Tiny-12 features from a window."""
    nt, nx = window_data.shape
    features = np.zeros(12)
    
    try:
        u_x = np.gradient(window_data, dx, axis=1)
        u_xx = np.gradient(u_x, dx, axis=1)
        u_xxx = np.gradient(u_xx, dx, axis=1)
        u_t = np.gradient(window_data, dt, axis=0)
        u_tt = np.gradient(u_t, dt, axis=0)
        
        features[0] = np.std(u_x)
        features[1] = np.std(u_xx)
        features[2] = np.std(u_xxx)
        features[3] = np.std(u_t)
        features[4] = np.std(u_tt)
        features[5] = np.max(np.abs(u_t))
        
        fft = np.fft.fft2(window_data)
        fft_mag = np.abs(fft)
        features[6] = np.log1p(fft_mag[:nt//4, :nx//4].mean())
        features[7] = np.log1p(fft_mag[nt//4:nt//2, nx//4:nx//2].mean())
        features[8] = np.log1p(fft_mag[nt//2:, nx//2:].mean())
        
        features[9] = np.std(window_data)
        features[10] = np.mean(np.abs(u_xx)) / (np.std(window_data) + 1e-8)
        features[11] = np.max(window_data) - np.min(window_data)
    except:
        pass
    
    return features


def run_ident_method(method_name, window_data, dx, dt, true_coeffs, ident_params):
    """Run a single IDENT method."""
    method = METHOD_REGISTRY.get(method_name)
    if method is None:
        return {"f1": 0.0, "e2": 1.0, "residual": 1.0, "runtime": 0.0}
    
    try:
        params = dict(ident_params)
        params["true_coeffs"] = true_coeffs
        metrics, info = method.run(window_data, dx, dt, **params)
        return {
            "f1": float(metrics[0]),
            "e2": float(metrics[1]),
            "residual": float(metrics[2]),
            "runtime": float(info.get("runtime", 0.0)),
        }
    except Exception as e:
        return {"f1": 0.0, "e2": 1.0, "residual": 1.0, "runtime": 0.0}


def process_window(window, methods, true_coeffs, ident_params):
    """Process a single window."""
    result = {
        "window_id": window["window_id"],
        "pde_type": window["pde_name"],
        "window_x_start": window["x_start"],
        "window_t_start": window["t_start"],
    }
    
    # Extract features
    features = extract_features(window["data"], window["dx"], window["dt"])
    for i, f in enumerate(features):
        result[f"feat_{i}"] = float(f)
    
    # Run IDENT methods
    best_e2 = float("inf")
    best_method = None
    
    for method_name in methods:
        method_result = run_ident_method(
            method_name, window["data"], window["dx"], window["dt"],
            true_coeffs, ident_params
        )
        for key, val in method_result.items():
            result[f"{method_name}_{key}"] = val
        
        if method_result["e2"] < best_e2:
            best_e2 = method_result["e2"]
            best_method = method_name
    
    result["best_method"] = best_method
    result["oracle_e2"] = best_e2
    return result


# ============================================================
# MAIN GENERATION LOOP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []

for pde_name in PDES:
    print(f"\n{'='*60}")
    print(f"Processing: {pde_name.upper()}")
    print(f"{'='*60}")
    
    config = PDE_CONFIGS[pde_name]
    
    # Load data
    print(f"Loading {config['data_file']}...")
    u = load_pde_data(config["data_file"], config["data_format"])
    print(f"Data shape: {u.shape}")
    
    # Extract windows
    print(f"Extracting {WINDOWS_PER_PDE} windows...")
    wcfg = config["windows"]
    windows = extract_windows(
        u, (wcfg["size_x"], wcfg["size_t"]), 
        (wcfg["stride_x"], wcfg["stride_t"]),
        pde_name, WINDOWS_PER_PDE,
        config["grid"]["dx"], config["grid"]["dt"]
    )
    print(f"Extracted {len(windows)} windows")
    
    # Process windows
    true_coeffs = config["true_coefficients"]
    ident_params = config["ident_params"]
    
    print(f"Processing with {N_JOBS} workers...")
    start_time = time.time()
    
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_window)(w, METHODS, true_coeffs, ident_params)
        for w in windows
    )
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s ({elapsed/len(results):.2f}s per window)")
    
    # Save per-PDE results
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/{pde_name}_results.csv", index=False)
    print(f"Saved to {OUTPUT_DIR}/{pde_name}_results.csv")
    
    all_results.extend(results)

# Combine all
print(f"\n{'='*60}")
print("Combining all results...")
combined = pd.DataFrame(all_results)
combined.to_csv(f"{OUTPUT_DIR}/full_dataset.csv", index=False)
print(f"Total samples: {len(combined)}")
print(f"Saved to {OUTPUT_DIR}/full_dataset.csv")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(combined.groupby("pde_type").size())
print("\nBest method distribution:")
print(combined["best_method"].value_counts())

#@title Download Results
from google.colab import files
files.download(f"{OUTPUT_DIR}/full_dataset.csv")
