#!/usr/bin/env python3
"""
Python module to compute Liouvillian gap

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

def main():
    ap = argparse.ArgumentParser(description="Compute slow Liouvillian mode and long-time profiles.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[500], help="Half-chain sizes (L).")
    ap.add_argument("--times", type=float, nargs="+", default=[200], help="Times T for long-time profiles.")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0], help="Gamma values.")
    ap.add_argument("--r", type=int, default=1, help="Charge index r.")
    ap.add_argument("--sign", type=str, default="-", help="Sign for lattice ('+' or '-').")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over (size,time,gamma,L).")
    ap.add_argument("--outdir", type=str, default=".", help="Base directory for outputs.")
    args = ap.parse_args()

    r = args.r
    sign = args.sign
    J1 = 1.0
    mu = 0.0
    bc = "open"
    n_jobs = args.n_jobs

    os.makedirs(args.outdir, exist_ok=True)

    tasks = [(s, T, g) for s in args.sizes for T in args.times for g in args.gammas]

    def gap_from_eigs(Lop, Lop_adj, N, k=4, tol=1e-7, maxiter=2000):
        evals_r, evecs_r = spla.eigs(Lop, k=k, which="LR", tol=tol, maxiter=maxiter)
        evals_l, evecs_l = spla.eigs(Lop_adj, k=k, which="LR", tol=tol, maxiter=maxiter)
        evals_r = np.array(evals_r)
        evals_l = np.array(evals_l)

        # Identify steady state by overlap with identity using right eigenvectors.
        overlaps = []
        for i in range(evecs_r.shape[1]):
            V = evecs_r[:, i].reshape(N, N)
            overlaps.append(abs(np.trace(V)) / N)
        steady_idx = int(np.argmax(overlaps))

        # Pick slowest non-steady mode (largest real part excluding steady).
        cand_idx = [i for i in range(len(evals_r)) if i != steady_idx]
        if not cand_idx:
            raise RuntimeError("Failed to find non-steady eigenmode.")
        slow_idx = max(cand_idx, key=lambda i: evals_r[i].real)
        gap = evals_r[slow_idx]
        vec_r = evecs_r[:, slow_idx].reshape(N, N)

        # Match left eigenvector by eigenvalue conjugacy.
        target = np.conjugate(gap)
        match_idx = int(np.argmin(np.abs(evals_l - target)))
        vec_l = evecs_l[:, match_idx].reshape(N, N)

        # Project out identity component from the right eigenvector.
        vec_r = vec_r - (np.trace(vec_r) / N) * np.eye(N, dtype=vec_r.dtype)
        return gap, vec_r, vec_l

    def run_task(s, T, g):
        N = 2 * s
        center = N // 2
        Length = int(s)
        print(f"Started diagonalization with g={g}, size={N} sites and T={T}, L={Length}")
        H = Hamiltonian(s, J1, mu, bc).tocsr()
        P_right = projector_RIGHT_sp(LA=s, L=Length, N=N, bc="open")
        C0 = Gamma0(s)
        h_cond = 1j * H - g * P_right

        def matvec_vec(v):
            C = v.reshape(N, N)
            term1 = h_cond @ C
            term2 = C @ h_cond.conjugate().T
            term3 = 2.0 * g * (P_right @ C @ P_right.conjugate().T)
            return (term1 + term2 + term3).reshape(-1)

        def matvec_adj(v):
            C = v.reshape(N, N)
            term1 = h_cond.conjugate().T @ C
            term2 = C @ h_cond
            term3 = 2.0 * g * (P_right @ C @ P_right.conjugate().T)
            return (term1 + term2 + term3).reshape(-1)

        L_op = spla.LinearOperator((N * N, N * N), matvec=matvec_vec, dtype=np.complex128)
        L_op_adj = spla.LinearOperator((N * N, N * N), matvec=matvec_adj, dtype=np.complex128)
        try:
            Delta_g, V_delta_r, V_delta_l = gap_from_eigs(L_op, L_op_adj, N)
        except Exception as e:
            err_msg = f"gap_from_eigs failed: {type(e).__name__}: {e}"
            print(err_msg)
            return {
                "error": err_msg,
                "gamma": g,
                "delta_g_real": np.nan,
                "delta_g_imag": np.nan,
                "r": r,
                "L": Length,
                "T": T,
                "bdy_ratio": np.nan,
                "bdy_log10_abs": np.nan,
            }

        # Biorthogonal projection for non-Hermitian Liouvillian.
        numerator = np.vdot(V_delta_l, C0)
        denom = np.vdot(V_delta_l, V_delta_r)
        coeff = numerator / denom if denom != 0 else 0.0
        C_delta = coeff * V_delta_r * np.exp(Delta_g * T)

        xs = np.arange(N)
        zetas = (xs - center) / T
        if sign == "-":
            q_vals = np.array([np.real(qm_symm(r, x, C_delta, "open")) for x in xs])
            j_vals = np.array([np.real(jm_symm(r, x, C_delta, "open")) for x in xs])
        else:
            q_vals = np.array([np.real(qp_symm(r, x, C_delta, "open")) for x in xs])
            j_vals = np.array([np.real(jp_symm(r, x, C_delta, "open")) for x in xs])

        bdy_ratio = np.nan
        bdy_log10_abs = np.nan
        if sign == "-":
            idx_left = max(center - 1, 0)
            idx_right = min(center, N - 1)
            j_left = j_vals[idx_left]
            j_right = j_vals[idx_right]
            q0 = q_vals[center]
            if q0 != 0:
                bdy_ratio = (j_right - j_left) / q0
                if bdy_ratio != 0:
                    bdy_log10_abs = np.log10(abs(bdy_ratio))

        csv_dir = os.path.join(args.outdir, "GHD_NELL_SLOW_CSV")
        png_dir = os.path.join(args.outdir, "GHD_NELL_SLOW_PNG")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        outcsv = os.path.join(
            csv_dir,
            f"GHD_SLOW_l{Length}_r{args.r}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
        )
        np.savetxt(
            outcsv,
            np.column_stack([
                np.full_like(zetas, g, dtype=float),
                np.full_like(zetas, T, dtype=float),
                zetas,
                q_vals,
                j_vals,
                np.full_like(zetas, Delta_g.real, dtype=float),
                np.full_like(zetas, Delta_g.imag, dtype=float),
                np.full_like(zetas, bdy_ratio, dtype=float),
                np.full_like(zetas, bdy_log10_abs, dtype=float),
            ]),
            delimiter=",",
            header="gamma,time,zeta,q,j,delta_g_real,delta_g_imag,bdy_ratio,bdy_log10_abs",
            comments="",
        )
        print(f"saved: {outcsv}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes[0].scatter(zetas, q_vals, s=4, c="blue", alpha=0.6, label="q")
        axes[1].scatter(zetas, j_vals, s=4, c="red", alpha=0.6, label="J")
        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()
        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")

        title = rf"$l={Length},\ \gamma={g},\ T={T},\ N={N},\ r={args.r},\ \text{{sign}}={sign}$"
        if not np.isnan(bdy_log10_abs):
            title += rf"$\ \log_{{10}}|bdy|={bdy_log10_abs:.2f}$"
        fig.suptitle(title)

        fig.tight_layout()
        outpng = os.path.join(png_dir, os.path.basename(outcsv).replace(".csv", ".png"))
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"wrote {outpng}")

        return {
            "gamma": g,
            "delta_g_real": Delta_g.real,
            "delta_g_imag": Delta_g.imag,
            "r": r,
            "L": Length,
            "T": T,
            "bdy_ratio": bdy_ratio,
            "bdy_log10_abs": bdy_log10_abs,
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_task)(s, T, g) for (s, T, g) in tasks
    )

    if sign == "-":
        bdy_dir = os.path.join(args.outdir, "BDY_NELL")
        os.makedirs(bdy_dir, exist_ok=True)
        outcsv = os.path.join(bdy_dir, "BDY_NELL.csv")
        with open(outcsv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gamma", "delta_g_real", "delta_g_imag", "r", "L", "T", "bdy_ratio", "bdy_log10_abs"])
            for row in results:
                if row.get("error"):
                    continue
                writer.writerow([
                    row["gamma"],
                    row["delta_g_real"],
                    row["delta_g_imag"],
                    row["r"],
                    row["L"],
                    row["T"],
                    row["bdy_ratio"],
                    row["bdy_log10_abs"],
                ])

        good = [row for row in results if not row.get("error")]
        if not good:
            print("No successful diagonalizations; BDY_NELL plot skipped.")
            return
        gammas = np.array([row["gamma"] for row in good], dtype=float)
        ratios = np.array([row["bdy_ratio"] for row in good], dtype=float)
        delta_g = np.array([row["delta_g_real"] for row in good], dtype=float)

        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(gammas, ratios, c=delta_g, cmap="viridis", s=40)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$(J^-_{0+}-J^-_{0-})/q^-_0$")
        ax.grid(True, ls="--", alpha=0.5)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\Delta_g$ (real)")
        if len(ratios) >= 3:
            A = np.column_stack([gammas, delta_g, np.ones_like(gammas)])
            coeffs, _, _, _ = np.linalg.lstsq(A, ratios, rcond=None)
            fit_gamma, fit_delta, fit_const = coeffs
            fit_path = os.path.join(bdy_dir, "BDY_NELL_fit.csv")
            with open(fit_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["fit_gamma", "fit_delta_g", "fit_const"])
                writer.writerow([fit_gamma, fit_delta, fit_const])
            ax.text(
                0.02,
                0.98,
                rf"$mathrm{{fit}}: a\gamma + b\Delta_g + c$"
                rf"$\,\ a={fit_gamma:.3e},\ b={fit_delta:.3e},\ c={fit_const:.3e}$",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            print(f"wrote {fit_path}")
        fig.tight_layout()
        outpng = os.path.join(bdy_dir, "BDY_NELL.png")
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"wrote {outpng}")
    

if __name__ == "__main__":
    main()
