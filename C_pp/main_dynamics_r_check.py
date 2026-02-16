#!/usr/bin/env python3
"""
Python analogue of main_dynamics_BDY.cpp.
Computes conditions 3 and 4 for r=1, sign='-', over given times/gammas/size.
Writes CSV: gamma,zeta0,cond3,cond3_abs,cond4,cond4_abs,time,size
"""

import sys
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import scipy.linalg as la
import time
from joblib import Parallel, delayed


# ensure repo root (parent of C_pp) is on sys.path for ghd_module/dynamics_module
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dynamics_module import *
import ghd_module as gm
import scipy.sparse as sp

# Local charge/current helpers (same as in C++ and ghd_module)
def get_elem(C, i, j, bc):
    N = C.shape[0]
    ii = int(i)
    jj = int(j)
    if bc and bc[0].lower() == "p":
        return C[ii % N, jj % N]
    if 0 <= ii < N and 0 <= jj < N:
        return C[ii, jj]
    return 0.0 + 0.0j


# ============================================================
#  2. Symmetric Local charges
# ============================================================

def qp_symm(r, x, C, bc="pbc"):
    # Works with both even and odd hopping ranges
    if r % 2 != 0:
        R = (r - 1) // 2
        return get_elem(C, x - R, x + R + 1, bc) + get_elem(C, x + R + 1, x - R, bc)
    else:
        R =  r // 2
        return get_elem(C, x - R, x + R, bc) + get_elem(C, x + R, x - R, bc)


def qm_symm(r, x, C, bc="pbc"):
    if r % 2 != 0:
        R = (r - 1) // 2
        return 1j * (get_elem(C, x - R, x + R + 1, bc) - get_elem(C, x + R + 1, x - R, bc))
    else:
        R =  r // 2
        return 1j * (get_elem(C, x - R, x + R, bc) - get_elem(C, x + R, x - R, bc))


def jp_symm(r, x, C, bc="pbc"):

    if r % 2 != 0:
        R = (r - 1) // 2
        return 1j * (
            get_elem(C, x - R + 1, x + R + 1, bc)
            - get_elem(C, x - R, x + R + 2, bc)
            + get_elem(C, x + R + 2, x - R, bc)
            - get_elem(C, x + R + 1, x - R + 1, bc)
        )

    else:
        R =  r // 2
        return 1j * (
            get_elem(C, x - R + 1, x + R, bc)
            - get_elem(C, x - R, x + R + 1, bc)
            + get_elem(C, x + R + 1, x - R, bc)
            - get_elem(C, x + R, x - R + 1, bc)
        )


def jm_symm(r, x, C, bc="pbc"):
    if r % 2 != 0:
        R = (r - 1) // 2
        return -(
            get_elem(C, x - R + 1, x + R + 1, bc)
            - get_elem(C, x - R, x + R + 2, bc)
            - get_elem(C, x + R + 2, x - R, bc)
            + get_elem(C, x + R + 1, x - R + 1, bc)
        )
    else:
        R =  r // 2
        return -(
            get_elem(C, x - R + 1, x + R , bc)
            - get_elem(C, x - R, x + R + 1, bc)
            - get_elem(C, x + R + 1, x - R, bc)
            + get_elem(C, x + R, x - R + 1, bc)
        )
    


def Gamma0(L: int):
    """
    Inhomogeneous initial state: vacuum on the left half, Néel on the right half.
    Matches the original Python Gamma_0: right half diagonal with pattern 0,1,0,1,...
    """
    N = 2 * L
    diag = np.zeros(N, dtype=complex)
    for j in range(L):
        if j % 2 == 1:  # occupancy on odd sites of the right half
            diag[L + j] = 1.0
    return np.diag(diag)


def Gamma_beta(beta: float, h_dense: np.ndarray) -> np.ndarray:
    """
    Thermal single-particle correlator using the provided Hamiltonian block.
    rho_beta = exp(-beta h) / Tr, used directly as C.
    """
    rho_beta = la.expm(-beta * h_dense)
    z_beta = np.trace(rho_beta)
    return rho_beta / z_beta


def main():
    ap = argparse.ArgumentParser(description="Boundary conditions driver with comparison to Qags CSV.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[500], help="Half-chain sizes (L).")
    ap.add_argument("--times", type=float, nargs="+", default=[200], help="Times T.")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0, 0.001, 0.01, 0.1, 0.5, 1.0], help="Gamma values.")
    ap.add_argument("--r", type=int, default=1, help="Charge index r.")
    ap.add_argument("--sign", type=str, default="-", help="Sign for lattice ('+' or '-').")
    ap.add_argument("--l", type=float, nargs="+", default=[1.0], help="Monitored interval length")
    ap.add_argument("--qags-M", type=int, default=200, help="M used in qags runs (for filename pattern).")
    ap.add_argument("--method", choices=["expm", "rk4"], default="expm", help="Time evolution method.")
    ap.add_argument("--rk-steps", type=int, default=400, help="RK4 steps if method=rk4.")
    ap.add_argument("--rk-adapt", action="store_true", help="Use simple adaptive step-doubling for RK4.")
    ap.add_argument("--rk-tol", type=float, default=1e-6, help="Relative tolerance for adaptive RK4.")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over (time,gamma).")
    ap.add_argument("--outdir", type=str, default=".", help="Base directory; CSVs go to GHD_NELL_CSV, plots to GHD_NELL_PNG.")
    args = ap.parse_args()

    sizes = args.sizes
    r = args.r
    J1 = 1.0
    mu = 0.0
    bc = "open"
    sign = args.sign  # for filenames and lattice sign
    method = args.method
    rk_steps = args.rk_steps
    rk_adapt = args.rk_adapt
    rk_tol = args.rk_tol
    n_jobs = args.n_jobs
    l = args.l
    
    os.makedirs(args.outdir, exist_ok=True)

    


    
    
    # Build size-dependent operators and initial state (keep Gamma0 here)

    tasks = [(s, T, g, L) for s in args.sizes for T in args.times for g in args.gammas for L in l]

    def run_task(s, T, g, L):
        N = 2 * s
        center = N // 2
        Length= int(L)
        H = Hamiltonian(s, J1, mu, bc).tocsr()
        P_right = projector_RIGHT_sp(LA=s, L=Length, N=N, bc="open")
    
        C0 = Gamma0(s)
        vecC0 = mat2vec(C0)
        vecC0_flat = np.asarray(vecC0).reshape(-1)
        
        h_cond = 1j * H - g * P_right
        # matrix-free matvec: L_sup v = (h_cond⊗I + I⊗h_cond^T + 2g P⊗P^T) v
        def matvec(v):
            v = np.asarray(v).reshape(-1)
            n = int(np.sqrt(v.size))
            C = v.reshape(n, n)
            term1 = h_cond @ C
            term2 = C @ h_cond.conjugate().T
            term3 = 2.0 * g * (P_right @ C @ P_right.conjugate().T)
            return (term1 + term2 + term3).reshape(-1)

        t0 = time.perf_counter()
        print(f"Time evolution with gamma={g}, r={r}, sign={sign}, L={Length}.")
        if method == "rk4":
            def rk_step(v, dt):
                k1 = matvec(v)
                k2 = matvec(v + 0.5 * dt * k1)
                k3 = matvec(v + 0.5 * dt * k2)
                k4 = matvec(v + dt * k3)
                return v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if rk_adapt:
                t = 0.0
                v = vecC0_flat.copy()
                dt = T / float(max(rk_steps, 10))
                while t < T:
                    if t + dt > T:
                        dt = T - t
                    v_full = rk_step(v, dt)
                    v_half = rk_step(v, 0.5 * dt)
                    v_half = rk_step(v_half, 0.5 * dt)
                    err = np.linalg.norm(v_full - v_half) / (np.linalg.norm(v_half) + 1e-12)
                    if err < rk_tol or dt <= 1e-12:
                        v = v_half
                        t += dt
                        dt *= 1.5  # try to increase step
                    else:
                        dt *= 0.5  # reduce step and retry
                vecC_t = v
                t_elapsed = time.perf_counter() - t0
                print(f"  RK evolution took {t_elapsed/60:.2f} min")
            else:
                dt = T / float(rk_steps)
                v = vecC0_flat.copy()
                for _ in range(rk_steps):
                    v = rk_step(v, dt)
                vecC_t = v
        else:
            t0 = time.perf_counter()
            L_op = spla.LinearOperator((N * N, N * N), matvec=matvec, dtype=np.complex128)
            vecC_t = spla.expm_multiply(L_op * T, vecC0_flat)
            t_elapsed = time.perf_counter() - t0
            print(f"  expm_multiply took {t_elapsed:.2f} s")
        
        C_t = vecC_t.reshape(N, N)
        xs = np.arange(N)
        zetas = xs
        if sign == "-":
            q_vals = np.array([np.real(qm_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jm_symm(r, x, C_t, "open")) for x in xs])
        else:
            q_vals = np.array([np.real(qp_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jp_symm(r, x, C_t, "open")) for x in xs])

        bdy_residual = np.nan
        bdy_log10_abs = np.nan
        if sign == "-":
            idx_left = max(center - 1, 0)
            idx_right = min(center, N - 1)
            j_left = j_vals[idx_left]
            j_right = j_vals[idx_right]
            q0 = q_vals[center]
            bdy_residual = (j_right - j_left) + g * q0
            if bdy_residual != 0:
                bdy_log10_abs = np.log10(abs(bdy_residual))
            else:
                bdy_log10_abs = -np.inf

        csv_dir = os.path.join(args.outdir, "GHD_NEEL_PARTIAL_CSV")
        png_dir = os.path.join(args.outdir, "GHD_NEEL_PARTIAL_PNG")

        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        outcsv = os.path.join(
            csv_dir,
            f"GHD_VCNEEL_l{L}_r{args.r}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
        # Save data to csv file
        np.savetxt(
            outcsv,
            np.column_stack([
                np.full_like(zetas, g, dtype=float),
                np.full_like(zetas, T, dtype=float),
                zetas,
                q_vals,
                j_vals,
                np.full_like(zetas, bdy_residual, dtype=float),
                np.full_like(zetas, bdy_log10_abs, dtype=float),
            ]),
            delimiter=",",
            header="gamma,time,zeta,q,j,bdy_residual,bdy_log10_abs",
            comments="",
        )

        print("saved:"+ outcsv)

        q_plot = q_vals
        j_plot = j_vals

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes[0].scatter(zetas, q_plot, s=4, c="blue", alpha=0.6, label="q")
        axes[1].scatter(zetas, j_plot, s=4, c="red", alpha=0.6, label="J")
        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()

        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")

        l_sites = int(L) + 1
        title = rf"$l={l_sites},\ \gamma={g},\ T={T},\ N={N},\ r={args.r},\ \text{{sign}}={sign}$"
        if not np.isnan(bdy_log10_abs):
            title += rf"$\ \log_{{10}}|bdy|={bdy_log10_abs:.2f}$"
        fig.suptitle(title)

        fig.tight_layout()
        outpng = os.path.join(png_dir, os.path.basename(outcsv).replace(".csv", ".png"))
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"wrote {outcsv} and {outpng}")
        return 0

    Parallel(n_jobs=n_jobs)(
        delayed(run_task)(s, T, g, L) for (s, T, g, L) in tasks
    )
    

if __name__ == "__main__":
    main()
