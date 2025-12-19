import argparse
import os
import yaml
from datetime import datetime
from typing import List, Tuple

import numpy as np
from tabulate import tabulate

from utils.data import load_data, add_noise
from model import weak_ident_pred


def slugify_equation_name(equation: str) -> str:
    s = equation.strip().lower()
    for ch in [" ", "/", "\\", ",", ":", ";", "(", ")", "-", "__"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def compute_ranges(n: int, splits: int) -> List[Tuple[int, int]]:
    """Return inclusive index ranges that split [0, n-1] into `splits` contiguous parts."""
    ranges = []
    sizes = [n // splits] * splits
    for i in range(n % splits):
        sizes[i] += 1
    start = 0
    for sz in sizes:
        end = start + sz - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def main():
    parser = argparse.ArgumentParser(description="Run WeakIdent on disjoint subsamples and write summary.txt per subsample")
    parser.add_argument('--config', required=True, help='Path to YAML config (same as main.py)')
    parser.add_argument('--out-dir', default='outputs', help='Base outputs directory')
    parser.add_argument('--pde-slug', default='', help='Optional override for PDE slug directory name')
    parser.add_argument('--split-x', type=int, default=1, help='Number of disjoint splits along x (space)')
    parser.add_argument('--split-y', type=int, default=1, help='Number of disjoint splits along y (space, for 2D PDEs)')
    parser.add_argument('--split-t', type=int, default=1, help='Number of disjoint splits along t (time)')
    parser.add_argument('--override-max-dx', type=int, default=-1, help='Optional override for max_dx for small subsamples')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Flatten config to simple attributes similar to main.py
    class C: pass
    c = C()
    for key in cfg:
        for k, v in cfg[key].items():
            setattr(c, k, v)

    # Load dataset
    print('Start loading arguments and dataset', c.filename, 'for', c.equation)
    try:
        u, xs, true_coefficients = load_data('dataset-Python', c.filename)
    except Exception as e:
        print('Failed to load dataset:', repr(e))
        return

    # Noise
    sigma_SNR = getattr(c, 'sigma_SNR', 0)
    u_hat = add_noise(u, sigma_SNR)

    # Library/config params
    use_cross_der = getattr(c, 'use_cross_der', False)
    stride_x = getattr(c, 'stride_x', 5)
    stride_t = getattr(c, 'stride_t', 5)
    tau = getattr(c, 'Tau', 0.05)
    max_dx = getattr(c, 'max_dx', 6)
    max_poly = getattr(c, 'max_poly', 6)
    if args.override_max_dx >= 0:
        max_dx = args.override_max_dx

    # Determine PDE slug
    pde_slug = args.pde_slug.strip() or slugify_equation_name(getattr(c, 'equation', 'pde'))
    base_dir = os.path.join(args.out_dir, pde_slug)
    os.makedirs(base_dir, exist_ok=True)

    # Dimensionality info
    num_variables = u_hat.shape[0]
    dim_x = len(u_hat[0].shape) - 1
    if dim_x == 0:
        # ODE: time-only subsampling
        # Extract t
        try:
            # ODE datasets store only time in xs. Prefer first element if present.
            if isinstance(xs, np.ndarray) and xs.shape[0] >= 1:
                t_arr = xs[0].reshape(-1)
            else:
                t_arr = np.arange(u_hat[0].shape[-1])
        except Exception:
            t_arr = np.arange(u_hat[0].shape[-1])
        Nt = t_arr.shape[0]
        tranges = compute_ranges(Nt, max(1, args.split_t))
        for (t0, t1) in tranges:
            subsample_id = f"subsample_t{t0}-{t1}"
            out_dir = os.path.join(base_dir, subsample_id)
            os.makedirs(out_dir, exist_ok=True)
            summary_path = os.path.join(out_dir, 'summary.txt')

            u_k = np.empty((num_variables,), dtype=object)
            for i in range(num_variables):
                arr = u_hat[i]
                if arr.ndim == 2:
                    u_k[i] = arr[:, t0:t1+1]
                else:
                    u_k[i] = arr[t0:t1+1]
            t_k = t_arr[t0:t1+1].reshape(1, -1)
            xs_k = np.array([t_k], dtype=object)

            try:
                df_errs, df_eqns, df_coefs, run_time = weak_ident_pred(
                    u_k, xs_k, true_coefficients, max_dx, max_poly, stride_x, stride_t, use_cross_der, tau)
                status = 'ok'
                reason = ''
            except Exception as e:
                df_errs, df_eqns, df_coefs, run_time = None, None, None, 0.0
                status = 'failed'
                reason = repr(e)

            with open(summary_path, 'w') as f:
                f.write(f"pde_slug: {pde_slug}\n")
                f.write(f"subsample_id: {subsample_id}\n")
                f.write(f"created_at: {datetime.now().isoformat()}\n")
                f.write(f"index_ranges: t={t0}-{t1}\n")
                f.write(f"shapes: Nt={t1-t0+1}\n")
                f.write(f"config: max_dx={max_dx}, max_poly={max_poly}, use_cross_der={use_cross_der}, stride_x={stride_x}, stride_t={stride_t}, tau={tau}, sigma_SNR={sigma_SNR}\n")
                f.write(f"library: n={num_variables}, dim_x={dim_x}\n")
                f.write(f"status: {status}\n")
                if status != 'ok':
                    f.write(f"reason: {reason}\n")
                f.write("\n")
                if status == 'ok':
                    f.write(" ------------- coefficient vector overview ------noise-signal-ratio : %s  -------\n" % (sigma_SNR))
                    f.write(tabulate(df_coefs, headers=df_coefs.columns, tablefmt="grid"))
                    f.write("\n\n")
                    f.write(" ------------- equation overview ------noise-signal-ratio : %s  -------------------\n" % (sigma_SNR))
                    f.write(tabulate(df_eqns, headers=df_eqns.columns, tablefmt="grid"))
                    f.write("\n\n")
                    f.write(" ------------------------------ CPU time: %s seconds ------------------------------\n" % (round(run_time, 2)))
                    f.write("\n Identification error: \n")
                    f.write(tabulate(df_errs, headers=df_errs.columns, tablefmt="grid"))
                    f.write("\n")

            print('Wrote', summary_path)

    elif dim_x == 1:
        # 1D PDE: x and t splitting
        Nx, Nt = u_hat[0].shape
        x_arr = xs[0].reshape(-1)
        t_arr = xs[1].reshape(-1)
        xranges = compute_ranges(Nx, max(1, args.split_x))
        tranges = compute_ranges(Nt, max(1, args.split_t))
        for (x0, x1) in xranges:
            for (t0, t1) in tranges:
                subsample_id = f"subsample_x{x0}-{x1}_t{t0}-{t1}"
                out_dir = os.path.join(base_dir, subsample_id)
                os.makedirs(out_dir, exist_ok=True)
                summary_path = os.path.join(out_dir, 'summary.txt')

                u_k = np.empty((num_variables,), dtype=object)
                for i in range(num_variables):
                    u_k[i] = u_hat[i][x0:x1+1, t0:t1+1]
                x_k = x_arr[x0:x1+1].reshape(-1, 1)
                t_k = t_arr[t0:t1+1].reshape(1, -1)
                xs_k = np.array([x_k, t_k], dtype=object)

                try:
                    df_errs, df_eqns, df_coefs, run_time = weak_ident_pred(
                        u_k, xs_k, true_coefficients, max_dx, max_poly, stride_x, stride_t, use_cross_der, tau)
                    status = 'ok'
                    reason = ''
                except Exception as e:
                    df_errs, df_eqns, df_coefs, run_time = None, None, None, 0.0
                    status = 'failed'
                    reason = repr(e)

                with open(summary_path, 'w') as f:
                    f.write(f"pde_slug: {pde_slug}\n")
                    f.write(f"subsample_id: {subsample_id}\n")
                    f.write(f"created_at: {datetime.now().isoformat()}\n")
                    f.write(f"index_ranges: x={x0}-{x1}, t={t0}-{t1}\n")
                    f.write(f"shapes: Nx={x1-x0+1}, Nt={t1-t0+1}\n")
                    f.write(f"config: max_dx={max_dx}, max_poly={max_poly}, use_cross_der={use_cross_der}, stride_x={stride_x}, stride_t={stride_t}, tau={tau}, sigma_SNR={sigma_SNR}\n")
                    f.write(f"library: n={num_variables}, dim_x={dim_x}\n")
                    f.write(f"status: {status}\n")
                    if status != 'ok':
                        f.write(f"reason: {reason}\n")
                    f.write("\n")
                    if status == 'ok':
                        f.write(" ------------- coefficient vector overview ------noise-signal-ratio : %s  -------\n" % (sigma_SNR))
                        f.write(tabulate(df_coefs, headers=df_coefs.columns, tablefmt="grid"))
                        f.write("\n\n")
                        f.write(" ------------- equation overview ------noise-signal-ratio : %s  -------------------\n" % (sigma_SNR))
                        f.write(tabulate(df_eqns, headers=df_eqns.columns, tablefmt="grid"))
                        f.write("\n\n")
                        f.write(" ------------------------------ CPU time: %s seconds ------------------------------\n" % (round(run_time, 2)))
                        f.write("\n Identification error: \n")
                        f.write(tabulate(df_errs, headers=df_errs.columns, tablefmt="grid"))
                        f.write("\n")

                print('Wrote', summary_path)

    elif dim_x == 2:
        # 2D PDE: x, y, t splitting
        Nx, Ny, Nt = u_hat[0].shape
        x_arr = xs[0].reshape(-1)
        y_arr = xs[1].reshape(-1)
        t_arr = xs[2].reshape(-1)
        xranges = compute_ranges(Nx, max(1, args.split_x))
        yranges = compute_ranges(Ny, max(1, args.split_y))
        tranges = compute_ranges(Nt, max(1, args.split_t))
        for (x0, x1) in xranges:
            for (y0, y1) in yranges:
                for (t0, t1) in tranges:
                    subsample_id = f"subsample_x{x0}-{x1}_y{y0}-{y1}_t{t0}-{t1}"
                    out_dir = os.path.join(base_dir, subsample_id)
                    os.makedirs(out_dir, exist_ok=True)
                    summary_path = os.path.join(out_dir, 'summary.txt')

                    u_k = np.empty((num_variables,), dtype=object)
                    for i in range(num_variables):
                        u_k[i] = u_hat[i][x0:x1+1, y0:y1+1, t0:t1+1]
                    x_k = x_arr[x0:x1+1].reshape(-1, 1)
                    y_k = y_arr[y0:y1+1].reshape(-1, 1)
                    t_k = t_arr[t0:t1+1].reshape(1, -1)
                    xs_k = np.array([x_k, y_k, t_k], dtype=object)

                    try:
                        df_errs, df_eqns, df_coefs, run_time = weak_ident_pred(
                            u_k, xs_k, true_coefficients, max_dx, max_poly, stride_x, stride_t, use_cross_der, tau)
                        status = 'ok'
                        reason = ''
                    except Exception as e:
                        df_errs, df_eqns, df_coefs, run_time = None, None, None, 0.0
                        status = 'failed'
                        reason = repr(e)

                    with open(summary_path, 'w') as f:
                        f.write(f"pde_slug: {pde_slug}\n")
                        f.write(f"subsample_id: {subsample_id}\n")
                        f.write(f"created_at: {datetime.now().isoformat()}\n")
                        f.write(f"index_ranges: x={x0}-{x1}, y={y0}-{y1}, t={t0}-{t1}\n")
                        f.write(f"shapes: Nx={x1-x0+1}, Ny={y1-y0+1}, Nt={t1-t0+1}\n")
                        f.write(f"config: max_dx={max_dx}, max_poly={max_poly}, use_cross_der={use_cross_der}, stride_x={stride_x}, stride_t={stride_t}, tau={tau}, sigma_SNR={sigma_SNR}\n")
                        f.write(f"library: n={num_variables}, dim_x={dim_x}\n")
                        f.write(f"status: {status}\n")
                        if status != 'ok':
                            f.write(f"reason: {reason}\n")
                        f.write("\n")
                        if status == 'ok':
                            f.write(" ------------- coefficient vector overview ------noise-signal-ratio : %s  -------\n" % (sigma_SNR))
                            f.write(tabulate(df_coefs, headers=df_coefs.columns, tablefmt="grid"))
                            f.write("\n\n")
                            f.write(" ------------- equation overview ------noise-signal-ratio : %s  -------------------\n" % (sigma_SNR))
                            f.write(tabulate(df_eqns, headers=df_eqns.columns, tablefmt="grid"))
                            f.write("\n\n")
                            f.write(" ------------------------------ CPU time: %s seconds ------------------------------\n" % (round(run_time, 2)))
                            f.write("\n Identification error: \n")
                            f.write(tabulate(df_errs, headers=df_errs.columns, tablefmt="grid"))
                            f.write("\n")

                    print('Wrote', summary_path)

    else:
        print('Unsupported dimensionality dim_x =', dim_x)


if __name__ == '__main__':
    main()


