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
    """ q_plus: C[x, x+r] + C[x+r, x] """
    return get_elem(C, x-r, x + r, bc) + get_elem(C, x + r, x-r, bc)



def qm_symm(r, x, C, bc="pbc"):
    """ q_minus: i*(C[x, x+r] - C[x+r, x]) """
    return 1j * (get_elem(C, x-r, x + r+1, bc) - get_elem(C, x + r+1, x-r, bc))


def jp_symm(r, x, C, bc="pbc"):
    """
    j_plus: i*( C[x+1, x+r] - C[x, x+r+1]
               + C[x+r+1, x] - C[x+r, x+1] )
    """
    return 1j * (
        get_elem(C, x -r +1,    x + r,     bc)
        - get_elem(C, x-r,      x + r + 1, bc)
        + get_elem(C, x + r + 1, x-r,      bc)
        - get_elem(C, x + r,  x -r + 1,     bc)
    )


def jm_symm(r, x, C, bc="pbc"):
    """
    j_minus: -( C[x+1, x+r] - C[x, x+r+1]
               - C[x+r+1, x] + C[x+r, x+1] )
    """
    return -(
        get_elem(C, x -r + 1,    x + r+1,     bc)
        - get_elem(C, x-r,      x + r + 2, bc)
        - get_elem(C, x + r + 2, x-r,      bc)
        + get_elem(C, x + r+1,  x -r + 1,     bc)
    )



def Gamma_odd(L: int):
    """
    Inhomogeneous initial state: odd Néel
     diagonal with pattern 0,1,0,1,...
    """
    N = 2 * L
    diag = np.zeros(N, dtype=complex)
    for j in range(N):
        if j % 2 == 1:  # occupancy on odd sites
            diag[j] = 1.0
    return np.diag(diag)


def Gamma_even(L:int):
    """
    Homogeneous initial state: even Néel.
    diagonal with pattern 1,0,1,0,...
    """
    N = 2 * L
    diag = np.zeros(N, dtype=complex)
    for j in range(N):
        if j % 2 == 0:  # occupancy on even sites 
            diag[j] = 1.0
    return np.diag(diag)



def main():
    ap = argparse.ArgumentParser(description="Boundary conditions driver with comparison to Qags CSV.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[500], help="Half-chain sizes (L).")
    ap.add_argument("--times", type=float, nargs="+", default=[200], help="Times T.")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0, 0.001, 0.01, 0.1, 0.5, 1.0], help="Gamma values.")
    ap.add_argument("--r", type=int, default=1, help="Charge index r.")
    ap.add_argument("--offset", type=int, default=1, help="Offset from center for boundary conditions.")
    ap.add_argument("--sign", type=str, default="-", help="Sign for lattice ('+' or '-').")
    ap.add_argument("--qags-sign", type=str, default="+", help="Sign used in qags CSV filenames.")
    ap.add_argument("--qags-M", type=int, default=200, help="M used in qags runs (for filename pattern).")
    ap.add_argument("--qags-dir", type=str, default=".", help="Directory containing qags CSVs.")
    ap.add_argument("--qags-pattern", type=str, default="GHD_r{r}_sign{sign}_M{M}_gamma{gamma:.6f}_qags.csv", help="Pattern for qags CSV filenames.")
    ap.add_argument("--method", choices=["expm", "rk4"], default="expm", help="Time evolution method.")
    ap.add_argument("--rk-steps", type=int, default=400, help="RK4 steps if method=rk4.")
    ap.add_argument("--rk-adapt", action="store_true", help="Use simple adaptive step-doubling for RK4.")
    ap.add_argument("--rk-tol", type=float, default=1e-6, help="Relative tolerance for adaptive RK4.")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over (time,gamma).")
    ap.add_argument("--outdir", type=str, default=".", help="Base directory; CSVs go to GHD_NELL_CSV, plots to GHD_NELL_PNG.")
    args = ap.parse_args()

    sizes = args.sizes
    times = args.times
    gammas = args.gammas
    r = args.r
    offset = args.offset
    J1 = 1.0
    mu = 0.0
    bc = "open"
    sign_char = args.sign  # for filenames and lattice sign
    qags_dir = args.qags_dir
    qags_pattern = args.qags_pattern
    qags_sign = args.qags_sign
    qags_M = args.qags_M
    method = args.method
    rk_steps = args.rk_steps
    rk_adapt = args.rk_adapt
    rk_tol = args.rk_tol
    n_jobs = args.n_jobs

    # symm functions expect step "R" where lattice r=2R (even) or r=2R-1 (odd)
    if sign_char == "+":
        R = r // 2 if r % 2 == 0 else (r + 1) // 2
    else:
        R = (r - 1) // 2 if r % 2 != 0 else r // 2
    
    # Output CSV name encodes size
    minL = min(sizes)
    maxL = max(sizes)
    if minL == maxL:
        outname = f"GHD_BDY_r{r}_sign{sign_char}_N{2*minL}_py.csv"
    else:
        outname = f"GHD_BDY_r{r}_sign{sign_char}_N{2*minL}_to_{2*maxL}_py.csv"

    

    for s in sizes:
        N = 2 * s
        center = N // 2
        i_plus = center + offset
        i_minus = center - offset

        # Build size-dependent operators and initial state (keep Gamma0 here)
        H = Hamiltonian(s, J1, mu, bc).tocsr()
        P_right = projector_prefix_RIGHT_sp(LA=s, N=N, bc=bc).tocsr()
        C0 = 0.5*(Gamma_even(s)+ Gamma_odd(s) )
        vecC0 = mat2vec(C0)
        vecC0_flat = np.asarray(vecC0).reshape(-1)

        tasks = [(T, g) for T in times for g in gammas]

        def run_task(T, g):
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
            print(f'Time evolution with gamma={g}, r={r}, sign={qags_sign}. ')
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
            C_zeta = vecC_t.reshape(N, N)

            # Overlay lattice vs qags CSV (GHD) if available
            candidates = []
            if qags_pattern:
                try:
                    candidates.append(os.path.join(
                        qags_dir,
                        qags_pattern.format(gamma=g, r=r, sign=qags_sign, M=qags_M),
                    ))
                except Exception as e:
                    print(f"could not format qags_pattern: {e}")
            # fallback filename conventions (2 decimals and 6 decimals)
            candidates.append(os.path.join(qags_dir, f"GHD_r{r}_sign{qags_sign}_M{qags_M}_gamma{g:.2f}test.csv"))
            candidates.append(os.path.join(qags_dir, f"GHD_r{r}_sign{qags_sign}_M{qags_M}_gamma{g:.6f}test.csv"))
            if g ==0:
                candidates.append(os.path.join(qags_dir, f"GHD_r{r}_sign{qags_sign}_M{qags_M}_gamma{0}test.csv"))

            qags_file = next((p for p in candidates if os.path.exists(p)), "")
            if qags_file:
                try:
                    qags_data = np.genfromtxt(qags_file, delimiter=",", names=True)
                    cols = list(qags_data.dtype.names)
                    def pick(name, fallback_idx):
                        return qags_data[name] if name in cols else qags_data[cols[fallback_idx]]
                    z_qags = pick("zeta", 0)
                    q_qags = pick("q", 1)
                    j_qags = pick("J", 2)
                    xs = np.arange(0, N - 1)  # mid-bond positions for averaging
                    center_plot = (N - 1) / 2.0
                    xis = (xs - center_plot + 0.5) / T
                    zeta_site = (np.arange(N) - center_plot) / T
                    if qags_sign == "-":
                        q_site = np.array([qm_symm(R, xx, C_zeta, bc).real for xx in range(N)])
                        J_site = np.array([jm_symm(R, xx, C_zeta, bc).real for xx in range(N)])
                    else:
                        q_site = np.array([qp_symm(R, xx, C_zeta, bc).real for xx in range(N)])
                        J_site = np.array([jp_symm(R, xx, C_zeta, bc).real for xx in range(N)])

                    csv_dir = os.path.join(args.outdir, "GHD_NELL_CSV")
                    png_dir = os.path.join(args.outdir, "GHD_NELL_PNG")
                    os.makedirs(csv_dir, exist_ok=True)
                    os.makedirs(png_dir, exist_ok=True)
                    outcsv = os.path.join(
                        csv_dir,
                        f"GHD_GHZ_r{args.r}_sign{qags_sign}_gamma{g:.2f}_N{N}.csv",
                    )
                    np.savetxt(
                        outcsv,
                        np.column_stack(
                            [
                                np.full_like(zeta_site, g, dtype=float),
                                np.full_like(zeta_site, T, dtype=float),
                                zeta_site,
                                q_site,
                                J_site,
                            ]
                        ),
                        delimiter=",",
                        header="gamma,time,zeta,q,j",
                        comments="",
                    )
                    print("saved:"+ outcsv)

                    # two-site averaging to smooth noise, align with Python convention
                    q_lat = 0.5 * (q_site[:-1] + q_site[1:])
                    J_lat = 0.5 * (J_site[:-1] + J_site[1:])
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
                    axes[0].scatter(xis, q_lat, s=4, c="blue", label="numerics: " + r"$q^-$")
                    axes[0].plot(z_qags, q_qags, c="green", label=f"GHD q (sign {qags_sign})")
                    axes[1].scatter(xis, J_lat, s=4, c="red", label="numerics: "+r"$J^-$")
                    axes[1].plot(z_qags, j_qags, c="orange", label=f"GHD J (sign {qags_sign})")
                    for ax in axes:
                        ax.set_xlabel(r"$\zeta$")
                        ax.grid(True, ls="--", alpha=0.5)
                        ax.legend()
                    axes[0].set_ylabel(r"$q^-$")
                    axes[1].set_ylabel(r"$J^-$")
                    fig.suptitle(f"gamma={g}, T={T}, N={N}, r={r}")
                    fig.tight_layout()
                    out_qags_cmp = os.path.join(png_dir, f"cmp_gamma{g}_T{T}_N{N}_r{r}_sign{sign_char}.png")
                    fig.savefig(out_qags_cmp, dpi=200)
                    plt.close(fig)
                    print(f"wrote {out_qags_cmp}")
                except Exception as e:
                    print(f"qags overlay failed (gamma={g}, T={T}, N={N}): {e}")
            else:
                print(f"No qags CSV found for gamma={g} at {qags_file}")
            return 0

        Parallel(n_jobs=n_jobs)(
            delayed(run_task)(T, g) for (T, g) in tasks
        )
        print(f"done: size={N}")



if __name__ == "__main__":
    main()
