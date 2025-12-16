# PDE-Selector Dataset Generation - Google Colab (ALL PDEs)
# ===========================================================
# Generates dataset from ALL available PDE simulations.
# Includes optional GPU acceleration via CuPy.

# ============== SETUP ==============
!pip install pysindy==1.7.5 xgboost pyyaml joblib tqdm --quiet

# Optional: GPU acceleration (uncomment if using GPU runtime)
# !pip install cupy-cuda12x --quiet  # For CUDA 12.x
# USE_GPU = True

USE_GPU = False  # Set to True if you installed CuPy

# Clone the repository
!git clone https://github.com/ALLENDE123X/ident-lab.git
%cd ident-lab/WeakIdent-Python

# ============== CONFIGURATION ==============
# Set how many windows per PDE (None = auto based on data size)
WINDOWS_PER_PDE = 1000  # ~13,000 total samples for 13 PDEs

# Which methods to run (WeakIDENT is slow, can skip for faster runs)
# Options: "WeakIDENT", "PySINDy", "WSINDy", "RobustIDENT"
METHODS = ["PySINDy", "WSINDy", "RobustIDENT"]  # Fast: ~1 hour
# METHODS = ["WeakIDENT", "PySINDy", "WSINDy", "RobustIDENT"]  # Full: ~4-8 hours

OUTPUT_DIR = "/content/results"  # Colab-friendly path

# ============== IMPORTS ==============
import sys
import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# GPU array library
if USE_GPU:
    try:
        import cupy as cp
        xp = cp
        print("✅ Using CuPy (GPU)")
    except ImportError:
        xp = np
        print("⚠️ CuPy not found, using NumPy (CPU)")
else:
    xp = np
    print("Using NumPy (CPU)")

# ============== ALL PDE CONFIGURATIONS ==============
# Configs for all 13 available PDEs in dataset-Python/
ALL_PDE_CONFIGS = {
    # ===== 1D PDEs =====
    "burgers": {
        "data_file": "dataset-Python/burgers_viscous.npy",
        "data_format": "object_array",
        "grid": {"dx": 0.0625, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": 0.01}
    },
    "kdv": {
        "data_file": "dataset-Python/KdV.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.05, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": {"u*u_x": -1.0, "u_xxx": -1.0}
    },
    "heat": {
        "data_file": "dataset-Python/heat.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.0427, "dt": 0.002},
        "windows": {"size_x": 48, "size_t": 80},
        "true_coefficients": {"u_xx": 1.0}
    },
    "ks": {
        "data_file": "dataset-Python/KS.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.39, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0}
    },
    "transport": {
        "data_file": "dataset-Python/transportDiff.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.04, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": {"u_x": -1.0, "u_xx": 0.01}
    },
    "nls": {
        "data_file": "dataset-Python/NLS.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.1, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": {"u_xx": 1.0, "|u|^2*u": 1.0}
    },
    # ===== ODEs =====
    "duffing": {
        "data_file": "dataset-Python/Duffing.npy",
        "data_format": "nested_object",
        "grid": {"dx": 1.0, "dt": 0.01},
        "windows": {"size_x": 2, "size_t": 100},
        "true_coefficients": {"x": -1.0, "x^3": -1.0}
    },
    "vanderpol": {
        "data_file": "dataset-Python/VanderPol.npy",
        "data_format": "nested_object",
        "grid": {"dx": 1.0, "dt": 0.01},
        "windows": {"size_x": 2, "size_t": 100},
        "true_coefficients": {"x": 1.0, "x^3": -1.0}
    },
    "lorenz": {
        "data_file": "dataset-Python/Lorenz.npy",
        "data_format": "nested_object",
        "grid": {"dx": 1.0, "dt": 0.01},
        "windows": {"size_x": 3, "size_t": 100},
        "true_coefficients": {}
    },
    "lotka_volterra": {
        "data_file": "dataset-Python/LotkaVolterra.npy",
        "data_format": "nested_object",
        "grid": {"dx": 1.0, "dt": 0.1},
        "windows": {"size_x": 2, "size_t": 50},
        "true_coefficients": {}
    },
    "linear2d": {
        "data_file": "dataset-Python/Linear2d.npy",
        "data_format": "nested_object",
        "grid": {"dx": 1.0, "dt": 0.01},
        "windows": {"size_x": 2, "size_t": 100},
        "true_coefficients": {}
    },
    # ===== 2D PDEs (larger) =====
    "porous_medium": {
        "data_file": "dataset-Python/PM.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.05, "dt": 0.01},
        "windows": {"size_x": 40, "size_t": 40},  # 2D spatial
        "true_coefficients": {"(u^2)_xx": 0.3, "(u^2)_yy": 1.0}
    },
    "lotka_volterra_2d": {
        "data_file": "dataset-Python/LotkaVolterra2D.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.1, "dt": 0.01},
        "windows": {"size_x": 32, "size_t": 50},
        "true_coefficients": {}
    },
}

# ============== WINDOW EXTRACTOR ==============
@dataclass
class Window:
    data: np.ndarray
    x_start: int
    t_start: int
    window_id: str
    metadata: Dict[str, Any]

def extract_windows(u, window_size, pde_name, target_count=None, dx=1.0, dt=1.0):
    """Extract overlapping windows from u(x,t) data."""
    if u.ndim == 3:  # 2D spatial data
        nx, ny, nt = u.shape
        # For 2D, take slices in x dimension
        u = u[:, ny//2, :]  # Take middle slice
    
    if u.ndim != 2:
        print(f"  Warning: Unexpected shape {u.shape}, skipping")
        return []
    
    nx, nt = u.shape
    size_x, size_t = window_size
    
    # Auto-compute stride for target count
    if target_count:
        approx_stride = int(np.sqrt((nx * nt) / target_count))
        stride_x = max(1, min(approx_stride, (nx - size_x) // 10 + 1))
        stride_t = max(1, min(approx_stride, (nt - size_t) // 10 + 1))
    else:
        stride_x = max(1, size_x // 4)
        stride_t = max(1, size_t // 4)
    
    windows = []
    idx = 0
    for i_x in range(0, max(1, nx - size_x), stride_x):
        for i_t in range(0, max(1, nt - size_t), stride_t):
            window_data = u[i_x:i_x+size_x, i_t:i_t+size_t].T  # (nt, nx)
            if window_data.shape == (size_t, size_x):
                windows.append(Window(
                    data=window_data, x_start=i_x, t_start=i_t,
                    window_id=f"{pde_name}_{idx:05d}",
                    metadata={"pde_name": pde_name, "dx": dx, "dt": dt}
                ))
                idx += 1
                if target_count and idx >= target_count:
                    return windows
    return windows

def load_pde_data(filepath, data_format="nested_object"):
    """Load PDE data from .npy file."""
    data = np.load(filepath, allow_pickle=True)
    if data_format == "nested_object":
        return np.array(data[0], dtype=np.float64)
    elif data_format == "object_array":
        u = np.array(data.tolist(), dtype=np.float64)
        return u[0] if u.ndim == 3 and u.shape[0] == 1 else u
    return data[0] if data.ndim == 3 else data

# ============== FEATURE EXTRACTION (GPU-accelerated) ==============
def extract_features(window_data, dx, dt):
    """Extract Tiny-12 features, optionally on GPU."""
    # Move to GPU if available
    data = xp.asarray(window_data) if USE_GPU else window_data
    nt, nx = data.shape
    features = xp.zeros(12)
    
    try:
        u_x = xp.gradient(data, dx, axis=1)
        u_xx = xp.gradient(u_x, dx, axis=1)
        u_xxx = xp.gradient(u_xx, dx, axis=1)
        features[0], features[1], features[2] = xp.std(u_x), xp.std(u_xx), xp.std(u_xxx)
        
        u_t = xp.gradient(data, dt, axis=0)
        u_tt = xp.gradient(u_t, dt, axis=0)
        features[3], features[4], features[5] = xp.std(u_t), xp.std(u_tt), xp.max(xp.abs(u_t))
        
        fft_mag = xp.abs(xp.fft.fft2(data))
        features[6] = xp.log1p(fft_mag[:nt//4, :nx//4].mean())
        features[7] = xp.log1p(fft_mag[nt//4:nt//2, nx//4:nx//2].mean())
        features[8] = xp.log1p(fft_mag[nt//2:, nx//2:].mean())
        
        features[9] = xp.std(data)
        features[10] = xp.mean(xp.abs(u_xx)) / (xp.std(data) + 1e-8)
        features[11] = xp.max(data) - xp.min(data)
    except:
        pass
    
    # Move back to CPU for sklearn/other libs
    return np.array(features.get() if USE_GPU else features)

# ============== IDENT METHODS ==============
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
        return {"f1": 0.0, "e2": 1.0, "residual": 1.0, "runtime": 0.0}

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
    print(f"\n{'='*60}\n📊 Processing: {pde_name.upper()}\n{'='*60}")
    
    # Skip if file doesn't exist or is empty
    if not os.path.exists(config["data_file"]):
        print(f"  ⚠️ File not found: {config['data_file']}")
        return None
    if os.path.getsize(config["data_file"]) < 200:
        print(f"  ⚠️ File too small, skipping")
        return None
    
    try:
        u = load_pde_data(config["data_file"], config["data_format"])
        print(f"  Data shape: {u.shape}")
    except Exception as e:
        print(f"  ⚠️ Failed to load: {e}")
        return None
    
    w_cfg = config["windows"]
    windows = extract_windows(
        u, (w_cfg["size_x"], w_cfg["size_t"]),
        pde_name, max_windows,
        config["grid"]["dx"], config["grid"]["dt"]
    )
    
    if not windows:
        print(f"  ⚠️ No windows extracted")
        return None
    
    print(f"  Extracted {len(windows)} windows")
    
    results = []
    for w in tqdm(windows, desc=f"  {pde_name}"):
        results.append(process_window(w, methods, config.get("true_coefficients", {})))
    
    return pd.DataFrame(results)

# ============== RUN ==============
print("\n" + "="*60)
print("🚀 PDE-SELECTOR DATASET GENERATION")
print("="*60)

available_methods = METHOD_REGISTRY.list_methods()
print(f"Available methods: {available_methods}")
methods = [m for m in METHODS if m in available_methods]
print(f"Using: {methods}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
all_dfs = []
stats = []

for pde_name, config in ALL_PDE_CONFIGS.items():
    start = time.time()
    df = process_pde(pde_name, config, WINDOWS_PER_PDE, methods)
    elapsed = time.time() - start
    
    if df is not None and len(df) > 0:
        df.to_csv(f"{OUTPUT_DIR}/{pde_name}_results.csv", index=False)
        all_dfs.append(df)
        stats.append({"pde": pde_name, "samples": len(df), "time": elapsed})
        print(f"  ✅ Saved {len(df)} samples ({elapsed:.1f}s)")

# Summary
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)
for s in stats:
    print(f"  {s['pde']:20s}: {s['samples']:5d} samples in {s['time']:.1f}s")

# Combine all
if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(f"{OUTPUT_DIR}/full_dataset.csv", index=False)
    print(f"\n✅ COMPLETE: {len(combined)} total samples -> {OUTPUT_DIR}/full_dataset.csv")
    
    # Show class balance
    print("\n📈 Method selection distribution:")
    print(combined["best_method"].value_counts())
    
    # Download
    from google.colab import files
    files.download(f"{OUTPUT_DIR}/full_dataset.csv")
else:
    print("❌ No data generated!")
