#!/usr/bin/env python
"""
Quick demonstration of PDE-Selector framework in action.
Shows: Data generation → Feature extraction → Model training → Selection
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_gen import simulate_burgers, simulate_kdv, make_windows, add_noise
from src.features import extract_tiny12
from src.models import PerMethodRegressor

print("\n" + "="*70)
print("🚀 PDE-Selector Framework Demonstration")
print("="*70 + "\n")

# ============================================================================
# STEP 1: Generate Synthetic PDE Data
# ============================================================================
print("📊 STEP 1: Generating Synthetic PDE Data")
print("-" * 70)

print("  Simulating Burgers equation (u_t + u·u_x = ν·u_xx)...")
u_burgers, dx, dt = simulate_burgers(
    nu=0.01,
    L=1.0,
    T=0.5,
    nx=128,
    nt=100,
    ic_type="sine",
    ic_params={"amp": 1.0, "freq": 2.0}
)
print(f"  ✅ Burgers: shape={u_burgers.shape}, dx={dx:.4f}, dt={dt:.4f}")

print("\n  Simulating KdV equation (u_t + α·u·u_x + β·u_xxx = 0)...")
u_kdv, dx_kdv, dt_kdv = simulate_kdv(
    alpha=1.0,
    beta=0.01,
    L=2.0,
    T=0.5,
    nx=128,
    nt=100,
    ic_type="sech"
)
print(f"  ✅ KdV: shape={u_kdv.shape}, dx={dx_kdv:.4f}, dt={dt_kdv:.4f}")

# Add noise
print("\n  Adding 2% Gaussian noise...")
u_burgers_noisy = add_noise(u_burgers, noise_level=0.02)
u_kdv_noisy = add_noise(u_kdv, noise_level=0.02)
print("  ✅ Noise added")

# ============================================================================
# STEP 2: Extract Windows
# ============================================================================
print("\n📐 STEP 2: Extracting Windows")
print("-" * 70)

windows_burgers = make_windows(u_burgers_noisy, nt_win=50, nx_win=64, stride_t=25, stride_x=32)
windows_kdv = make_windows(u_kdv_noisy, nt_win=50, nx_win=64, stride_t=25, stride_x=32)

print(f"  Burgers: {len(windows_burgers)} windows of shape {windows_burgers[0].shape}")
print(f"  KdV:     {len(windows_kdv)} windows of shape {windows_kdv[0].shape}")

# ============================================================================
# STEP 3: Extract Tiny-12 Features
# ============================================================================
print("\n🔍 STEP 3: Extracting Tiny-12 Features")
print("-" * 70)

print("\n  Feature names:")
feature_names = [
    "dx", "dt", "aspect", "R_x", "R_xx", "R_t", 
    "SNR_dB", "outlier_frac", "k_centroid", "slope", "w_centroid", "rho_per"
]
for i, name in enumerate(feature_names, 1):
    print(f"    {i:2d}. {name}")

print("\n  Extracting features from first Burgers window...")
phi_burgers = extract_tiny12(windows_burgers[0], dx, dt)
print("\n  Features extracted:")
for name, value in zip(feature_names, phi_burgers):
    print(f"    {name:14s} = {value:10.4f}")

print("\n  Extracting features from first KdV window...")
phi_kdv = extract_tiny12(windows_kdv[0], dx_kdv, dt_kdv)
print("\n  Features extracted:")
for name, value in zip(feature_names, phi_kdv):
    print(f"    {name:14s} = {value:10.4f}")

# Compare features
print("\n  📊 Feature Comparison (Burgers vs KdV):")
print("  " + "-" * 50)
print(f"  {'Feature':<14s} {'Burgers':>12s} {'KdV':>12s} {'Difference':>12s}")
print("  " + "-" * 50)
for name, b_val, k_val in zip(feature_names, phi_burgers, phi_kdv):
    diff = abs(b_val - k_val)
    print(f"  {name:<14s} {b_val:12.4f} {k_val:12.4f} {diff:12.4f}")

# Extract features from ALL windows
print("\n  Extracting features from all windows...")
X_features = []
y_labels = []  # 0=Burgers, 1=KdV

for win in windows_burgers:
    phi = extract_tiny12(win, dx, dt)
    X_features.append(phi)
    y_labels.append(0)

for win in windows_kdv:
    phi = extract_tiny12(win, dx_kdv, dt_kdv)
    X_features.append(phi)
    y_labels.append(1)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

print(f"  ✅ Extracted features: X.shape = {X_features.shape}")
print(f"  ✅ Labels: {np.sum(y_labels == 0)} Burgers, {np.sum(y_labels == 1)} KdV")

# ============================================================================
# STEP 4: Train a Quick Classifier (Demo)
# ============================================================================
print("\n🤖 STEP 4: Training ML Model (Demonstration)")
print("-" * 70)

# Create synthetic "error metrics" for demonstration
# In reality, these would come from running WeakIDENT
print("\n  Creating synthetic error metrics for demo...")
print("  (In real use, these come from running IDENT methods)")

# Burgers: lower errors (WeakIDENT works well)
Y_burgers = np.random.uniform(0.0, 0.2, (len(windows_burgers), 3))
Y_burgers[:, 0] = 0.8 + np.random.uniform(0, 0.15, len(windows_burgers))  # High F1

# KdV: moderate errors
Y_kdv = np.random.uniform(0.1, 0.4, (len(windows_kdv), 3))
Y_kdv[:, 0] = 0.6 + np.random.uniform(0, 0.2, len(windows_kdv))  # Medium F1

Y_metrics = np.vstack([Y_burgers, Y_kdv])

print(f"  ✅ Created Y_metrics: shape = {Y_metrics.shape}")
print(f"     Columns: [F1, CoeffErr, ResidualMSE]")

# Train model
print("\n  Training RandomForest regressor...")
model = PerMethodRegressor(n_estimators=50, max_depth=6, random_state=42)
model.fit(X_features, Y_metrics)
print("  ✅ Model trained!")

# Make predictions
print("\n  Making predictions on test window...")
test_window = windows_burgers[0]
test_phi = extract_tiny12(test_window, dx, dt).reshape(1, -1)

y_pred = model.predict(test_phi)[0]
y_unc = model.predict_unc(test_phi)[0]

print(f"\n  Predicted metrics:")
print(f"    F1 score:       {y_pred[0]:.4f} ± {y_unc[0]:.4f}")
print(f"    Coeff Error:    {y_pred[1]:.4f} ± {y_unc[1]:.4f}")
print(f"    Residual MSE:   {y_pred[2]:.4f} ± {y_unc[2]:.4f}")

# Feature importances
print("\n  📊 Top 5 Feature Importances:")
importances = model.get_feature_importances()
sorted_idx = np.argsort(importances)[::-1]
for i in range(5):
    idx = sorted_idx[i]
    print(f"    {i+1}. {feature_names[idx]:<14s}: {importances[idx]:.4f}")

# ============================================================================
# STEP 5: Demonstrate Selection Logic
# ============================================================================
print("\n🎯 STEP 5: Selection Logic Demonstration")
print("-" * 70)

# Simulate predictions for 2 methods
print("\n  Simulating predictions for 2 IDENT methods:")

# Method 1 (WeakIDENT)
y_method1 = model.predict(test_phi)[0]
unc_method1 = np.mean(model.predict_unc(test_phi)[0])
score_method1 = 0.5 * (1 - y_method1[0]) + 0.3 * y_method1[1] + 0.2 * y_method1[2]

print(f"\n  Method 1 (WeakIDENT):")
print(f"    Predicted metrics: [F1={y_method1[0]:.3f}, CoeffErr={y_method1[1]:.3f}, ResidualMSE={y_method1[2]:.3f}]")
print(f"    Aggregated score:  {score_method1:.4f} (lower is better)")
print(f"    Uncertainty:       {unc_method1:.4f}")

# Method 2 (hypothetical)
y_method2 = y_method1 * 1.2  # Slightly worse
unc_method2 = unc_method1 * 1.1
score_method2 = 0.5 * (1 - y_method2[0]) + 0.3 * y_method2[1] + 0.2 * y_method2[2]

print(f"\n  Method 2 (RobustIDENT - hypothetical):")
print(f"    Predicted metrics: [F1={y_method2[0]:.3f}, CoeffErr={y_method2[1]:.3f}, ResidualMSE={y_method2[2]:.3f}]")
print(f"    Aggregated score:  {score_method2:.4f} (lower is better)")
print(f"    Uncertainty:       {unc_method2:.4f}")

# Selection decision
print("\n  🤔 Selection Decision:")
tau = 0.6
median_unc = (unc_method1 + unc_method2) / 2

print(f"    Safety threshold (tau): {tau}")
print(f"    Median uncertainty:     {median_unc:.4f}")

if score_method1 < score_method2:
    best = "Method 1 (WeakIDENT)"
    best_score = score_method1
    best_unc = unc_method1
else:
    best = "Method 2"
    best_score = score_method2
    best_unc = unc_method2

print(f"\n    Best method: {best} (score={best_score:.4f})")

if best_score > tau or best_unc > median_unc:
    print(f"\n    ⚠️  UNCERTAIN → Running top-2 methods (safety gate)")
    print(f"       Reason: ", end="")
    if best_score > tau:
        print(f"score ({best_score:.4f}) > tau ({tau})")
    else:
        print(f"uncertainty ({best_unc:.4f}) > median ({median_unc:.4f})")
else:
    print(f"\n    ✅ CONFIDENT → Running only {best}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✨ Framework Demonstration Complete!")
print("="*70)
print("\n📋 Summary:")
print("  ✅ Generated synthetic PDE data (Burgers + KdV)")
print("  ✅ Extracted windows from full fields")
print("  ✅ Computed Tiny-12 features (no IDENT leakage!)")
print("  ✅ Trained ML model to predict error metrics")
print("  ✅ Demonstrated selection logic with safety gate")
print("\n💡 Next Steps:")
print("  1. Add RobustIDENT implementation")
print("  2. Generate full training dataset with real IDENT runs")
print("  3. Train per-method regressors")
print("  4. Evaluate on test set")
print("  5. Use selector on new data!")
print("\n" + "="*70 + "\n")

