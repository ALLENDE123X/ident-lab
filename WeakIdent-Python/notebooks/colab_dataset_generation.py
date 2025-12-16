# PDE-Selector Dataset Generation - Google Colab
# ================================================
# Run this entire cell in Google Colab to generate the full dataset.
# Note: GPU doesn't help much (IDENT methods are CPU-bound), but Colab 
# has good CPU resources and can run for hours.

# ============== SETUP ==============
!pip install pysindy==1.7.5 xgboost pyyaml joblib tqdm --quiet

# Clone the repository
!git clone https://github.com/ALLENDE123X/ident-lab.git
%cd ident-lab/WeakIdent-Python

# ============== CONFIGURATION ==============
# Adjust these settings as needed
PDES = ["burgers", "kdv", "heat", "ks"]  # 4 PDEs
WINDOWS_PER_PDE = 2500  # Total: 10,000 samples
N_JOBS = 4  # Parallel workers (Colab has ~2 cores, but can use more)
OUTPUT_DIR = "data/results"

# ============== IMPORTS ==============
import sys
import os
import time
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

# ============== WINDOW EXTRACTOR ==============
@dataclass
class Window:
    data: np.ndarray
    x_start: int
    x_end: int
    t_start: int
    t_end: int
    window_id: str
    metadata: Dict[str, Any]

def extract_windows(u, window_size, stride, pde_name, target_count=None, dx=1.0, dt=1.0):
    nx, nt = u.shape
    size_x, size_t = window_size
    stride_x, stride_t = stride
    
    n_windows_x = (nx - size_x) // stride_x + 1
    n_windows_t = (nt - size_t) // stride_t + 1
    
    if target_count and (n_windows_x * n_windows_t) < target_count:
        stride_x = max(1, (nx - size_x) // int(np.sqrt(target_count)) + 1)
        stride_t = max(1, (nt - size_t) // int(np.sqrt(target_count)) + 1)
        n_windows_x = (nx - size_x) // stride_x + 1
        n_windows_t = (nt - size_t) // stride_t + 1
    
    windows = []
    idx = 0
    for i_x in range(n_windows_x):
        for i_t in range(n_windows_t):
            x_start, x_end = i_x * stride_x, i_x * stride_x + size_x
            t_start, t_end = i_t * stride_t, i_t * stride_t + size_t
            window_data = u[x_start:x_end, t_start:t_end].T
            windows.append(Window(
                data=window_data, x_start=x_start, x_end=x_end,
                t_start=t_start, t_end=t_end,
                window_id=f"{pde_name}_{idx:05d}",
                metadata={"pde_name": pde_name, "dx": dx, "dt": dt}
            ))
            idx += 1
            if target_count and idx >= target_count:
                return windows
    return windows

def load_pde_data(filepath, data_format="nested_object"):
    data = np.load(filepath, allow_pickle=True)
    if data_format == "nested_object":
        return data[0]
    elif data_format == "object_array":
        u = np.array(data.tolist(), dtype=np.float64)
        return u[0] if u.ndim == 3 and u.shape[0] == 1 else u
    return data[0] if data.ndim == 3 and data.shape[0] == 1 else data

# ============== FEATURE EXTRACTION ==============
def extract_features(window_data, dx, dt):
    nt, nx = window_data.shape
    features = np.zeros(12)
    try:
        u_x = np.gradient(window_data, dx, axis=1)
        u_xx = np.gradient(u_x, dx, axis=1)
        u_xxx = np.gradient(u_xx, dx, axis=1)
        features[0], features[1], features[2] = np.std(u_x), np.std(u_xx), np.std(u_xxx)
        
        u_t = np.gradient(window_data, dt, axis=0)
        u_tt = np.gradient(u_t, dt, axis=0)
        features[3], features[4], features[5] = np.std(u_t), np.std(u_tt), np.max(np.abs(u_t))
        
        fft_mag = np.abs(np.fft.fft2(window_data))
        features[6] = np.log1p(fft_mag[:nt//4, :nx//4].mean())
        features[7] = np.log1p(fft_mag[nt//4:nt//2, nx//4:nx//2].mean())
        features[8] = np.log1p(fft_mag[nt//2:, nx//2:].mean())
        
        features[9] = np.std(window_data)
        features[10] = np.mean(np.abs(u_xx)) / (np.std(window_data) + 1e-8)
        features[11] = np.max(window_data) - np.min(window_data)
    except:
        pass
    return features

# ============== IDENT METHODS ==============
# Import the registered methods
sys.path.insert(0, ".")
from src.ident_methods import METHOD_REGISTRY

def run_method(method_name, window_data, dx, dt, true_coeffs=None):
    method = METHOD_REGISTRY.get(method_name)
    if method is None:
        return {"f1": 0.0, "e2": 1.0, "residual": 1.0, "runtime": 0.0}
    try:
        metrics, info = method.run(window_data, dx, dt, true_coeffs=true_coeffs)
        return {"f1": float(metrics[0]), "e2": float(metrics[1]), 
                "residual": float(metrics[2]), "runtime": float(info.get("runtime", 0))}
    except Exception as e:
        return {"f1": 0.0, "e2": 1.0, "residual": 1.0, "runtime": 0.0, "error": str(e)}

# ============== PDE CONFIGURATIONS ==============
PDE_CONFIGS = {
    "burgers": {
        "data_file": "dataset-Python/burgers_viscous.npy",
        "data_format": "object_array",
        "grid": {"dx": 0.0625, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100, "stride_x": 8, "stride_t": 14},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": 0.01}
    },
    "kdv": {
        "data_file": "dataset-Python/KdV.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.05, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100, "stride_x": 12, "stride_t": 20},
        "true_coefficients": {"u*u_x": -1.0, "u_xxx": -1.0}
    },
    "heat": {
        "data_file": "dataset-Python/heat.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.0427, "dt": 0.002},
        "windows": {"size_x": 48, "size_t": 80, "stride_x": 7, "stride_t": 12},
        "true_coefficients": {"u_xx": 1.0}
    },
    "ks": {
        "data_file": "dataset-Python/KS.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.39, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80, "stride_x": 8, "stride_t": 8},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0}
    }
}

# ============== MAIN PROCESSING ==============
def process_window(window, methods, true_coeffs):
    result = {
        "window_id": window.window_id,
        "pde_type": window.metadata["pde_name"],
        "window_x_start": window.x_start,
        "window_t_start": window.t_start,
    }
    dx, dt = window.metadata["dx"], window.metadata["dt"]
    
    features = extract_features(window.data, dx, dt)
    for i, f in enumerate(features):
        result[f"feat_{i}"] = float(f)
    
    best_e2, best_method = float("inf"), None
    for method_name in methods:
        res = run_method(method_name, window.data, dx, dt, true_coeffs)
        for k, v in res.items():
            result[f"{method_name}_{k}"] = v
        if res["e2"] < best_e2:
            best_e2, best_method = res["e2"], method_name
    
    result["best_method"] = best_method
    result["oracle_e2"] = best_e2
    return result

def process_pde(pde_name, config, max_windows, methods):
    print(f"\n{'='*60}\nProcessing: {pde_name.upper()}\n{'='*60}")
    
    u = load_pde_data(config["data_file"], config["data_format"])
    print(f"Data shape: {u.shape}")
    
    w_cfg = config["windows"]
    windows = extract_windows(
        u, (w_cfg["size_x"], w_cfg["size_t"]), 
        (w_cfg["stride_x"], w_cfg["stride_t"]),
        pde_name, max_windows,
        config["grid"]["dx"], config["grid"]["dt"]
    )
    print(f"Extracted {len(windows)} windows")
    
    results = []
    start = time.time()
    for i, w in enumerate(windows):
        if i % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(windows) - i) / rate if rate > 0 else 0
            print(f"Progress: {i+1}/{len(windows)} ({rate:.1f}/s, ETA: {eta/60:.1f}min)")
        results.append(process_window(w, methods, config.get("true_coefficients", {})))
    
    print(f"Done in {time.time() - start:.1f}s")
    return pd.DataFrame(results)

# ============== RUN ==============
print("Registered methods:", METHOD_REGISTRY.list_methods())
methods = [m for m in ["WeakIDENT", "PySINDy", "WSINDy", "RobustIDENT"] 
           if m in METHOD_REGISTRY.list_methods()]
print(f"Using: {methods}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
all_dfs = []

for pde in PDES:
    if pde in PDE_CONFIGS:
        df = process_pde(pde, PDE_CONFIGS[pde], WINDOWS_PER_PDE, methods)
        df.to_csv(f"{OUTPUT_DIR}/{pde}_results.csv", index=False)
        all_dfs.append(df)
        print(f"Saved {len(df)} samples to {OUTPUT_DIR}/{pde}_results.csv")

# Combine all
combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv(f"{OUTPUT_DIR}/full_dataset.csv", index=False)
print(f"\n✅ COMPLETE: {len(combined)} total samples -> {OUTPUT_DIR}/full_dataset.csv")

# Download the results
from google.colab import files
files.download(f"{OUTPUT_DIR}/full_dataset.csv")
