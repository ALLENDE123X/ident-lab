
# PDE-Selector Framework — Implementation Plan (Tiny-12, Per-Method Regressors)

> **Purpose**: This document is the single source of truth for building the **algorithm-selection meta-learner** that picks the best PDE identification (IDENT) method (e.g., WeakIDENT, RobustIDENT) for a given spatiotemporal dataset window, **without** running every method first. It defines the minimal architecture, features, interfaces, metrics, scripts, and evaluation—so a code assistant (Cursor) can audit the current repo, identify gaps, and implement the missing pieces step‑by‑step.

---

## 0) Objective & scope

- **Input**: a spatiotemporal field `u(t, x)` (or `u(t, x, y)` later) and sampling info (`dx`, `dt`).
- **Goal**: Pick the IDENT method that will produce the **lowest error** (per an aggregation of 3 error metrics) before actually running IDENT. Optionally, use a **safety gate** to run top‑2 when predictions are uncertain/high.
- **Model**: **One multi-output regressor per IDENT method**, taking a 12‑dim **characteristic feature vector** φ computed from raw data (no leakage).
- **Deliverables**:
  - Working Python package (`pde-selector/`) with CLI scripts.
  - Labeled dataset (features + per-method metric targets).
  - Trained selector models saved with `joblib`.
  - Evaluation report (regret, top‑1 accuracy, compute saved) + basic figures.
  - README instructions.

---

## 1) Repository layout

```
pde-selector/
├─ README.md
├─ pyproject.toml            # or requirements.txt
├─ config/
│  └─ default.yaml           # data gen, windowing, models, aggregation
├─ src/
│  ├─ data_gen.py            # synthetic PDE sims + windowing
│  ├─ features.py            # Tiny-12 feature extractor (this doc defines it)
│  ├─ ident_api.py           # adapters: WeakIDENT, RobustIDENT, etc.
│  ├─ metrics.py             # 3 error metrics (see §4), easily swappable
│  ├─ label_dataset.py       # loops: (φ, y_t) collection for each method
│  ├─ models.py              # per-method regressors + uncertainty
│  ├─ select_and_run.py      # selection + safety gate + run chosen IDENT
│  └─ eval.py                # regret, top-1 acc, compute-saved + plots
├─ scripts/
│  ├─ make_dataset.py        # generate & label train/val/test
│  ├─ train_selector.py      # train per-method regressors
│  ├─ evaluate_selector.py   # evaluate saved models on held-out data
│  └─ choose_and_run.py      # choose & run IDENT on a new u(t,x) .npy
└─ tests/
   ├─ test_features.py
   ├─ test_models.py
   └─ test_selection.py
```

> **If the repo already exists** under another name, keep that; create/rename files in place. Minimal disruption is fine—just ensure all modules above exist somewhere.

---

## 2) Tiny‑12 features φ(u) (no leakage)

Compute all features **within each window** `u_win ∈ R^{nt_win × nx_win}` using a **smoothed** signal `ũ` (Savitzky–Golay, order=3, fixed windows) and **central differences** for derivatives. Symbols: `‖·‖` is Frobenius/L2 on the interior to avoid edge artifacts.

**A. Sampling/geometry (3)**
1. `dx` — spatial step  
2. `dt` — time step  
3. **Aspect** `A = (nt*dt)/(nx*dx)`

**B. Derivative difficulty (3)**
4. **H1 ratio** `R_x = ‖ũ_x‖ / (‖ũ‖ + ε)`  
5. **Curvature ratio** `R_xx = ‖ũ_xx‖ / (‖ũ_x‖ + ε)`  
6. **Temporal activity** `R_t = ‖ũ_t‖ / (‖ũ‖ + ε)`

**C. Noise/outliers (2)**
7. **SNR_dB** `= 20 log10( ‖ũ‖ / (‖r‖ + ε) )` where `r = u-ũ`  
8. **Outlier fraction** `mean(|r| > 3*σ̂)`, with `σ̂ = 1.4826 * MAD(r) + ε`

**D. Spatial spectrum (2)** (FFT in x, averaged over t)
9. **k̄ (centroid)** `= Σ k P_x(k) / Σ P_x(k)`  
10. **Mid-band slope** `s_k` = slope of `log P_x(k)` vs `log k` over 10–60% of Nyquist

**E. Temporal spectrum (1)** (FFT in t, averaged over x)
11. **ω̄ (centroid)** `= Σ ω P_t(ω) / Σ P_t(ω)`

**F. Boundary/periodicity (1)**
12. **Periodicity score** `ρ_per = corr_t( u(0,t), u(L,t) )`

**Implementation notes**
- Fixed SG windows (e.g., `win_x=7`, `win_t=5`), order=3; *keep constant across dataset*.
- For FFT, mean-center across x (for spatial spectrum) and across t (for temporal spectrum).
- Return a `float64` vector with these 12 features in a fixed order.

---

## 3) IDENT adapters

Create a consistent interface in `ident_api.py`:

```python
def run_ident_and_metrics(u_win: np.ndarray, method: str, dx: float, dt: float) -> np.ndarray:
    """
    Runs the chosen IDENT method on `u_win` and returns a np.array([m1, m2, m3], dtype=float64).
    `method` ∈ {"WeakIDENT", "RobustIDENT", ...}.
    """
```

- **You will wrap your existing implementations** of WeakIDENT / RobustIDENT, etc.
- If a method requires extra kwargs (BCs, library degree, etc.), set reasonable defaults in the wrapper or read from config.
- Keep the function stateless for easy parallelization.

---

## 4) Error metrics (3)

Use/confirm the three metrics Dr. Kang specified. Default suggestions (swappable in `metrics.py`):

1. **Structure accuracy**: F1 or Jaccard of **term selection** vs. ground truth (synthetic only).  
2. **Coefficient error**: relative L2 error of nonzero coefficients vs. ground truth (synthetic).  
3. **Residual MSE**: mean squared residual on a held-out subset (applies to synthetic and real).

> In real data (no ground truth), #1–#2 are not computable; for labeling you may still compute #3 and proxy versions (e.g., cross-validation residual). The selector is trained primarily on synthetic where labels are known, and evaluated on real data via #3 and qualitative stability.

`metrics.py` must expose:
```python
def aggregate(y3: np.ndarray, w=(0.5, 0.3, 0.2)) -> float:
    """Weighted scalarization of 3 metrics into a single score."""
```

Weights are set in `config/default.yaml`. Rank-based aggregation is an optional variant.

---

## 5) Model: per‑method regressors + safety gate

- **Regressor**: `RandomForestRegressor` under `MultiOutputRegressor` (predicts 3 metrics).  
- **Targets**: train on `log1p(y)`; at inference, `expm1` back.  
- **Uncertainty**: per-metric **std across trees** → use to trigger safety.

**Safety gate** (in `select_and_run.py`):
- Predict scores `s_t = aggregate(ŷ_t)` for all methods `t`.
- Let `t* = argmin_t s_t`.
- If `s_t* > τ` **or** uncertainty of `t*` is above median across methods, **run top‑2** methods and pick best by **true** metrics. Else, run only `t*`.

---

## 6) Data generation & windowing

- **Sim PDE families** for training: start with **Burgers** and **KdV**; optional KS.  
- Sweep IC/BC, parameters, and **noise** `σ ∈ {0,1,2,5}%`.  
- **Windowing**: from full fields `(nt, nx)` extract overlapping windows `(nt_win, nx_win)` with strides `(stride_t, stride_x)`; store `dx, dt` with each window.

`data_gen.py` should expose:
```python
def simulate_burgers(...)-> np.ndarray
def simulate_kdv(...)-> np.ndarray
def make_windows(u, nt_win, nx_win, stride_t, stride_x)-> list[np.ndarray]
```

---

## 7) Labeling pipeline

`label_dataset.py`:
1. For each (PDE family × parameters × noise):
2. Simulate `u(t,x)`, window it → `u_win`.
3. Compute `φ = extract_tiny12(u_win, dx, dt)`.
4. For each method `t` in `methods`:
   - Run IDENT → metrics `y_t ∈ R^3`.
   - Append to per-method arrays.
5. Save:
   - `X_features.npy` shape `(n, 12)`
   - `Y_WeakIDENT.npy`, `Y_RobustIDENT.npy`, … each `(n, 3)`
   - Optionally CSVs for easy inspection.

**No leakage**: φ uses only raw data; no terms/coefficients from IDENT.

---

## 8) Training & serialization

`models.py`:
- Z‑score features (`StandardScaler`), `log1p` transform targets.
- Fit one pipeline per method, save with `joblib`:
  - `models/WeakIDENT.joblib`, `models/RobustIDENT.joblib`, …
- Provide:
```python
class PerMethodRegressor:
    def fit(self, X, Y): ...
    def predict(self, X)-> np.ndarray: ...      # returns (n,3)
    def predict_unc(self, X)-> np.ndarray: ...  # returns (n,3) std across trees
```

---

## 9) Selection + run (inference)

`select_and_run.py` exposes:
```python
def choose_method(phi_row: np.ndarray, models: dict, w, tau, k_fallback=2) -> list[str]
def run_pipeline(u_win: np.ndarray, dx, dt, models, w, tau):
    """
    1) φ = extract_tiny12(u_win, dx, dt)
    2) choose best (or top-2) method(s)
    3) run IDENT(s) and return best by true metrics
    """
```

---

## 10) Evaluation

`eval.py` computes and saves:
- **Regret**: `E[ aggregate(y_{t*}) - min_t aggregate(y_t) ]`
- **Top‑1 accuracy**: `Pr( t* = argmin_t aggregate(y_t) )`
- **Compute saved**: % windows where only 1 method was run
- **Plots**: regret CDF, top‑1 bars by noise level, confusion heatmap (true-best vs. chosen)

---

## 11) Config (YAML)

`config/default.yaml` (example):
```yaml
data:
  pdes: [burgers, kdv]
  noise_levels: [0.0, 0.01, 0.02, 0.05]
  nx: 256
  nt: 200
  dx: 0.00390625      # 1/256
  dt: 0.005           # 1/200
  window:
    nx_win: 128
    nt_win: 64
    stride_x: 64
    stride_t: 32

features:
  smoothing:
    sg_order: 3
    win_x: 7
    win_t: 5

methods: [WeakIDENT, RobustIDENT]

aggregation:
  weights: [0.5, 0.3, 0.2]
  safety_tau: 0.6

model:
  type: random_forest
  n_estimators: 300
  max_depth: 8
  random_state: 7
```

---

## 12) Scripts (CLI)

- `scripts/make_dataset.py --cfg config/default.yaml`
- `scripts/train_selector.py --cfg config/default.yaml`
- `scripts/evaluate_selector.py --cfg config/default.yaml --split test`
- `scripts/choose_and_run.py --npy path/to/u.npy --dx ... --dt ... --cfg ...`

Each script should log to `logs/` and write outputs to `artifacts/`.

---

## 13) Tests (minimal but real)

- `test_features.py`: synthetic wave and constant signal → assert ranges & monotonicity (e.g., `R_x` ≈ 0 for constant).
- `test_models.py`: tiny synthetic dataset → fit & predict shapes/finite numbers.
- `test_selection.py`: craft two fake models where method A is always better → selector always chooses A and skips fallback.

---

## 14) Gap analysis checklist (what Cursor should compare vs current repo)

1. Do `src/` modules above exist? If not, create them.  
2. Is `features.py` implementing **exact Tiny‑12** in §2? If not, add it.  
3. Are IDENT adapters present for **each** method in `methods`? If not, wrap/implement.  
4. Are **three metrics** implemented in `metrics.py` and a configurable `aggregate()`?  
5. Does `label_dataset.py` save `X_features.npy` and one `Y_<method>.npy` per method?  
6. Are training pipelines per method saved under `models/` with uncertainty?  
7. Is `select_and_run.py` using safety gate `τ` and top‑2 fallback?  
8. Do CLI scripts exist and run end‑to‑end?  
9. Are unit tests passing?  
10. Is `README.md` updated with quickstart commands?

Cursor should produce a short **diff report** (created files, updated files, removed files) and a **RUNLOG.md** summarizing commands run and results.

---

## 15) Step‑by‑step build order (for Cursor)

1. **Create/confirm** repo structure (§1) and Python env (`requirements.txt` with: `numpy, scipy, scikit-learn, joblib, matplotlib, pyyaml`).
2. Implement **`features.py`** (Tiny‑12) with docstrings and unit tests.
3. Add **`data_gen.py`** with Burgers + KdV simulators and `make_windows()`.
4. Implement **`ident_api.py`** wrappers for `WeakIDENT` and `RobustIDENT`. If unavailable, add TODO stubs that raise `NotImplementedError` and place integration hooks.
5. Implement **`metrics.py`** with the 3 metrics + `aggregate()`; wire config weights.
6. Implement **`label_dataset.py`** to build `X_features.npy` and `Y_*.npy` using §6 and §7.
7. Implement **`models.py`** (`PerMethodRegressor`) with `predict_unc()` via RF tree std; save pipelines.
8. Implement **`select_and_run.py`** (selector + safety + final run).
9. Implement **CLI scripts** (§12) with `argparse` and YAML loading.
10. Implement **`eval.py`** (regret, top‑1, compute-saved + figures).
11. Run **`scripts/make_dataset.py`**, **`train_selector.py`**, **`evaluate_selector.py`** and save artifacts.
12. Update **`README.md`** quickstart and add **`RUNLOG.md`** with actual outputs.
13. Ensure **tests** pass in `tests/` and include in CI if present.

---

## 16) Acceptance criteria

- Trains per-method models on synthetic (Burgers + KdV) and evaluates on held-out windows.  
- Reports **Regret < 10%** of baseline average (tunable), **Top‑1 > 70%**, **Compute saved > 40%** on validation.  
- `choose_and_run.py` successfully selects and runs IDENT on a provided `.npy` field.  
- All scripts run from a clean checkout with a single `pip install -r requirements.txt`.

---

## 17) Nice-to-have (optional if time remains)

- SHAP feature importances for interpretability.  
- Hold-out **entire PDE family** to check domain shift (train on Burgers, test on KdV or vice versa).  
- Add **KS** simulator and method support.  
- Top‑k fallback with **budget** constraint (e.g., only fallback if compute allows).

---

# Appendix A — Minimal code stubs (for quick bootstrap)

> These stubs define signatures and basic behavior so the assistant can fill in details.

**`src/features.py`**
```python
import numpy as np
from scipy.signal import savgol_filter
from numpy.fft import rfft, rfftfreq

EPS = 1e-12

def _sg(u, wx=7, wt=5, order=3):
    u1 = savgol_filter(u, window_length=wt, polyorder=order, axis=0, mode="interp")
    u2 = savgol_filter(u1, window_length=wx, polyorder=order, axis=1, mode="interp")
    return u2

def _central_diff(a, h, axis):
    return (np.roll(a, -1, axis) - np.roll(a, 1, axis)) / (2 * h)

def extract_tiny12(u, dx, dt, wx=7, wt=5):
    u = np.asarray(u, dtype=np.float64)
    nt, nx = u.shape
    A = (nt*dt) / (nx*dx + EPS)

    ut = _sg(u, wx=wx, wt=wt)
    r  = u - ut

    # derivatives on smoothed
    ux  = _central_diff(ut, dx, axis=1)
    uxx = _central_diff(ux, dx, axis=1)
    ut_t= _central_diff(ut, dt, axis=0)

    def nrm(x): return np.linalg.norm(x[1:-1,1:-1])
    Rx  = nrm(ux)/(nrm(ut) + EPS)
    Rxx = nrm(uxx)/(nrm(ux) + EPS)
    Rt  = nrm(ut_t)/(nrm(ut) + EPS)

    snr_db = 20*np.log10( (np.linalg.norm(ut)+EPS) / (np.linalg.norm(r)+EPS) )
    mad = np.median(np.abs(r - np.median(r)))
    sig = 1.4826*mad + EPS
    out_frac = np.mean(np.abs(r) > 3*sig)

    # spatial spectrum
    u_zm = u - u.mean(axis=1, keepdims=True)
    k = rfftfreq(nx, d=dx)
    Pk = np.mean(np.abs(rfft(u_zm, axis=1))**2, axis=0) + EPS
    k_centroid = float(np.sum(k*Pk)/np.sum(Pk))
    lo = max(1, int(0.10*len(k))); hi = max(lo+2, int(0.60*len(k)))
    x = np.log(k[lo:hi]+EPS); y = np.log(Pk[lo:hi])
    slope = float(np.polyfit(x, y, 1)[0])

    # temporal spectrum
    u_zm_t = u - u.mean(axis=0, keepdims=True)
    w = rfftfreq(nt, d=dt)
    Pw = np.mean(np.abs(rfft(u_zm_t, axis=0))**2, axis=1) + EPS
    w_centroid = float(np.sum(w*Pw)/np.sum(Pw))

    # periodicity
    left, right = u[:, 0], u[:, -1]
    if left.std() < 1e-12 or right.std() < 1e-12:
        rho_per = 0.0
    else:
        rho_per = float(np.corrcoef(left, right)[0, 1])

    return np.array([dx, dt, A, Rx, Rxx, Rt, snr_db, out_frac,
                     k_centroid, slope, w_centroid, rho_per], dtype=np.float64)
```

**`src/models.py`**
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class PerMethodRegressor:
    def __init__(self, n_estimators=300, max_depth=8, random_state=7):
        base = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.model  = MultiOutputRegressor(base)

    def fit(self, X, Y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, np.log1p(Y))
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Yz = self.model.predict(Xs)
        return np.expm1(Yz)

    def predict_unc(self, X):
        Xs = self.scaler.transform(X)
        stds = []
        for est in self.model.estimators_:
            trees = est.estimators_
            yk = np.column_stack([t.predict(Xs) for t in trees])
            stds.append(yk.std(axis=1))
        return np.stack(stds, axis=1)
```

**`src/select_and_run.py`**
```python
import numpy as np
from .features import extract_tiny12
from .metrics import aggregate
from .ident_api import run_ident_and_metrics

def choose_method(phi_row, models: dict, w=(0.5,0.3,0.2), tau=0.6, k_fallback=2):
    scores, uncs = {}, {}
    for name, model in models.items():
        yhat = model.predict(phi_row)[0]
        scores[name] = aggregate(yhat, w)
        uncs[name]   = model.predict_unc(phi_row)[0].mean()
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_score = ranked[0]
    if best_score > tau or uncs[best] > np.median(list(uncs.values())):
        return [r[0] for r in ranked[:k_fallback]]
    return [best]

def run_pipeline(u_win, dx, dt, models, w=(0.5,0.3,0.2), tau=0.6):
    phi = extract_tiny12(u_win, dx, dt).reshape(1, -1)
    chosen = choose_method(phi, models, w=w, tau=tau)
    # run chosen IDENT(s)
    results = []
    for m in chosen:
        y = run_ident_and_metrics(u_win, m, dx, dt)  # true metrics
        results.append((m, y))
    # pick best by true aggregate
    best = min(results, key=lambda pr: aggregate(pr[1], w))
    return best
```

---

## 18) Requirements

```
numpy
scipy
scikit-learn
joblib
matplotlib
pyyaml
```

Add SHAP if you plan to include interpretability plots.

---

## 19) Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_dataset.py --cfg config/default.yaml
python scripts/train_selector.py --cfg config/default.yaml
python scripts/evaluate_selector.py --cfg config/default.yaml

# Choosing and running IDENT on a new field u.npy (nt,nx)
python scripts/choose_and_run.py --npy data/u.npy --dx 0.0039 --dt 0.005 --cfg config/default.yaml
```
