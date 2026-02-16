#!/usr/bin/env python3
"""
Unitary thermal dynamics (gamma = 0).
Evolve C(t) = exp(+i H t) C0 exp(-i H t) using matrix-free expm_multiply.
Outputs CSV/PNG per (size, time, r, sign). Overlays hydrodynamic prediction (gamma=0).
Filename pattern: GHD_THERM_betaR_<betaR>_r<r>_sign<sign>_UNITARY_N<N>_TEST.(csv/png)
"""
import argparse
import os
import time
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt

# ensure repo root (parent of C_pp) is on sys.path for ghd_module/dynamics_module
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dynamics_module import Hamiltonian, projector_prefix_RIGHT_sp
from ghd_module import hyd_charge_thermal, hyd_current_thermal


def Gamma_beta(beta: float, h_dense: np.ndarray) -> np.ndarray:
    """Normalized Gibbs state exp(-beta H)/Z."""
    rho = la.expm(-beta * h_dense)
    z = np.trace(rho)
    return rho / z


def get_elem(C: np.ndarray, i: int, j: int, bc: str = "open") -> complex:
    N = C.shape[0]
    if bc.lower().startswith("p"):
        ii = ((i % N) + N) % N
        jj = ((j % N) + N) % N
        return C[ii, jj]
    if 0 <= i < N and 0 <= j < N:
        return C[i, j]
    return 0.0 + 0.0j


def qp_symm(r: int, x: int, C: np.ndarray, bc: str = "open") -> complex:
    return get_elem(C, x - r, x + r, bc) + get_elem(C, x + r, x - r, bc)


def qm_symm(r: int, x: int, C: np.ndarray, bc: str = "open") -> complex:
    return 1j * (get_elem(C, x - r, x + r + 1, bc) - get_elem(C, x + r + 1, x - r, bc))


def jp_symm(r: int, x: int, C: np.ndarray, bc: str = "open") -> complex:
    return 1j * (
        get_elem(C, x - r + 1, x + r, bc)
        - get_elem(C, x - r, x + r + 1, bc)
        + get_elem(C, x + r + 1, x - r, bc)
        - get_elem(C, x + r, x - r + 1, bc)
    )


def jm_symm(r: int, x: int, C: np.ndarray, bc: str = "open") -> complex:
    return -(
        get_elem(C, x - r + 1, x + r + 1, bc)
        - get_elem(C, x - r, x + r + 2, bc)
        - get_elem(C, x + r + 2, x - r, bc)
        + get_elem(C, x + r + 1, x - r + 1, bc)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[400], help="half-chain sizes (L); N=2L")
    ap.add_argument("--times", type=float, nargs="+", default=[200.0], help="evolution times")
    ap.add_argument("--r", type=int, required=True, help="charge index r")
    ap.add_argument("--sign", type=str, choices=["+", "-"], required=True)
    ap.add_argument("--betaL", type=float, default=0.0, help="left inverse temperature (unused if left empty)")
    ap.add_argument("--betaR", type=float, default=1.0, help="right inverse temperature")
    ap.add_argument("--outdir", type=str, default=".", help="output directory")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel jobs over (size,time)")
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    r = args.r
    sign = args.sign

    from joblib import Parallel, delayed

    tasks = [(s, T) for s in args.sizes for T in args.times]

    def run_task(s, T):
        N = 2 * s
        center = N // 2
        H = Hamiltonian(s, 1.0, 0.0, "open").tocsc()
        H_dense = H.toarray()
        H_R = H_dense[s:, s:]

        # Initial state: empty on left, thermal on right
        C_L = np.zeros((s, s), dtype=complex)
        C_R = Gamma_beta(args.betaR, H_R)
        C0 = la.block_diag(C_L, C_R)

        t0 = time.perf_counter()
        U_t = spla.expm(1j * H * T)  # sparse exp of H
        # Dynamics of covariance matrix
        C_t= U_t @ C0 @ U_t.conjugate().T
        t_elapsed = time.perf_counter() - t0
        print(f"Unitary evolution N={N}, T={T} took {t_elapsed/60:.2f} min")
        xs = np.arange(N)
        zetas = (xs - center) / T
        if sign == "-":
            q_vals = np.array([np.real(qm_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jm_symm(r, x, C_t, "open")) for x in xs])
        else:
            q_vals = np.array([np.real(qp_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jp_symm(r, x, C_t, "open")) for x in xs])

        # GHD overlay (gamma=0)
        q_ghd = np.array([hyd_charge_thermal(r, z, sign, args.betaL, args.betaR) for z in zetas])
        j_ghd = np.array([hyd_current_thermal(r, z, sign, args.betaL, args.betaR) for z in zetas])

        outbase = os.path.join(
            args.outdir,
            f"GHD_THERM_betaR_{args.betaR}_r{r}_sign{sign}_UNITARY_N{N}_TEST"
        )
        outcsv = outbase + ".csv"
        np.savetxt(
            outcsv,
            np.column_stack([np.full_like(zetas, 0.0, dtype=float), np.full_like(zetas, T, dtype=float), zetas, q_vals, j_vals]),
            delimiter=",",
            header="gamma,time,zeta,q,j",
            comments="",
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes[0].scatter(zetas, q_vals, s=4, c="blue", label="numerics q")
        axes[1].scatter(zetas, j_vals, s=4, c="red", label="numerics J")
        axes[0].plot(zetas, q_ghd, lw=3, c="green", label="ghd q")
        axes[1].plot(zetas, j_ghd, lw=3, c="orange", label="ghd J")
        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()
        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")
        if args.zmin is not None and args.zmax is not None:
            axes[0].set_xlim(args.zmin, args.zmax)
            axes[1].set_xlim(args.zmin, args.zmax)
        fig.suptitle(rf"Unitary, $\beta_L={args.betaL}, \beta_R={args.betaR}, T={T}, N={N}, r={r}, sign={sign}$")
        fig.tight_layout()
        outpng = outbase + ".png"
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"wrote {outcsv} and {outpng}")
        return 0

    Parallel(n_jobs=args.n_jobs)(
        delayed(run_task)(s, T) for (s, T) in tasks
    )


if __name__ == "__main__":
    main()
