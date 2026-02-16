#!/usr/bin/env python3
"""
Simplified Lindbladian dynamics with a two-site jump operator
P_{n,m} = delta_{n,a} delta_{m,b} + delta_{n,b} delta_{m,a}.
Computes local charge/current profiles q,J from the covariance matrix.
"""
import sys
import time
import argparse
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# ensure repo root and C_pp are on sys.path for dynamics_module
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
for p in (repo_root, here):
    if p not in sys.path:
        sys.path.append(p)

from dynamics_module import *


def get_elem(C, i, j, bc):
    N = C.shape[0]
    ii = int(i)
    jj = int(j)
    if bc and bc[0].lower() == "p":
        return C[ii % N, jj % N]
    if 0 <= ii < N and 0 <= jj < N:
        return C[ii, jj]
    return 0.0 + 0.0j


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


def proj_2sites(a: int, b: int, N: int, dtype=np.complex128) -> sp.csr_matrix:
    if a == b:
        raise ValueError("a and b must be different sites")
    if not (0 <= a < N and 0 <= b < N):
        raise ValueError(f"a,b must be in [0,{N-1}]")
    rows = np.array([a, b], dtype=int)
    cols = np.array([b, a], dtype=int)
    data = np.array([1.0, 1.0], dtype=dtype)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=dtype)


def Gamma_beta(beta: float, h_dense: np.ndarray) -> np.ndarray:
    r"""
    One-body correlation matrix for a quadratic Hamiltonian:
    C = V f(H) V^\dagger with f(eps)=1/(1+exp(beta*eps)).
    This avoids the partition-function scaling and keeps entries in [0,1].
    """
    eigvals, eigvecs = la.eigh(h_dense)
    f_occ = 1.0 / (1.0 + np.exp(beta * eigvals))
    return (eigvecs * f_occ) @ eigvecs.conjugate().T


def Gamma0(L: int):
    """
    Inhomogeneous initial state: vacuum on the left half, Neel on the right half.
    """
    N = 2 * L
    diag = np.zeros(N, dtype=complex)
    for j in range(L):
        if j % 2 == 1:  # occupancy on odd sites of the right half
            diag[L + j] = 1.0
    return np.diag(diag)


def rk4_evolve(L_sup, vec0, T, nsteps):
    dt = T / float(nsteps)
    v = vec0.copy()
    for _ in range(nsteps):
        k1 = L_sup * v
        k2 = L_sup * (v + 0.5 * dt * k1)
        k3 = L_sup * (v + 0.5 * dt * k2)
        k4 = L_sup * (v + dt * k3)
        v = v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return v



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[400], help="Half-chain sizes (L)")
    ap.add_argument("--times", type=float, nargs="+", default=[200], help="Times T")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1], help="Gamma values")
    ap.add_argument("--beta", type=float, nargs="+", default=[1], help="Inverse temperature Beta")
    ap.add_argument("--r", type=int, default=1, help="Lattice r index")
    ap.add_argument("--sign", type=str, default="-", help="+ or -")
    ap.add_argument("--a-offset", type=int, default=0, help="Offset from center for site a")
    ap.add_argument("--b-offset", type=int, default=1, help="Offset from center for site b")
    ap.add_argument("--method", choices=["expm", "rk4"], default="expm", help="Evolution method")
    ap.add_argument("--rk-steps", type=int, default=400, help="RK4 steps if method=rk4")
    ap.add_argument("--rk-adapt", action="store_true", help="Use adaptive RK4 with step-doubling")
    ap.add_argument("--rk-tol", type=float, default=1e-6, help="Relative tolerance for adaptive RK4")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory (CSV/PNG subdirs will be created)")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over (time,gamma)")

    args = ap.parse_args()

    sign = args.sign
    rk_steps = args.rk_steps
    rk_adapt = args.rk_adapt
    rk_tol = args.rk_tol
    n_jobs = args.n_jobs
    method = args.method
    r = args.r
    a_offset = args.a_offset
    b_offset = args.b_offset
    Beta= args.beta

    os.makedirs(args.outdir, exist_ok=True)

    tasks = [(s, T, g) for s in args.sizes for T in args.times for g in args.gammas]

    def run_task(s, T, g):
        N = 2 * s
        center = N // 2
        a = center + a_offset
        b = center + b_offset
        if not (0 <= a < N and 0 <= b < N):
            raise ValueError(f"a={a}, b={b} out of range for N={N}")

        H = Hamiltonian(s, 1.0, 0.0, "open")
        P_2sites = proj_2sites(a, b, N)
        
        if Beta==0:
            C0 = Gamma0(s)
            vecC0 = mat2vec(C0)
        else:
            C0 = Gamma_beta(Beta,H)
            vecC0= mat2vec(C0)
        vecC0_flat = np.asarray(vecC0).reshape(-1)

        h_cond = 1j * H 
        # matrix-free matvec: L_sup v = (h_cond⊗I + I⊗h_cond^T - 2g P⊗P^T) v
        def matvec(v):
            v = np.asarray(v).reshape(-1)
            n = int(np.sqrt(v.size))
            C = v.reshape(n, n)
            term1 = h_cond @ C
            term2 = C @ h_cond.conjugate().T
            term3 = g * (P_2sites @ C @ P_2sites.conjugate().T)
            return (term1 + term2 - term3).reshape(-1)

        t0 = time.perf_counter()
        if Beta ==0:
            print(
                f"Time evolution starting: N={N}, T={T}, gamma={g}, r={r}, sign={sign}, a={a}, b={b}"
            )
        else:
            print(
                f"Time evolution starting: Beta={Beta}, N={N}, T={T}, gamma={g}, r={r}, sign={sign}, a={a}, b={b}"
            )

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
        zetas = (xs - center) / T
        if sign == "-":
            q_vals = np.array([np.real(qm_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jm_symm(r, x, C_t, "open")) for x in xs])
        else:
            q_vals = np.array([np.real(qp_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jp_symm(r, x, C_t, "open")) for x in xs])

        csv_dir = os.path.join(args.outdir, "GHD_SIMP_NELL_CSV")
        png_dir = os.path.join(args.outdir, "GHD_SIMP_NELL_PNG")

        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        if Beta==0:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_NEEL_r{r}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )
        else:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_NEEL_Beta_{Beta}_r{r}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
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
            ]),
            delimiter=",",
            header="gamma,time,zeta,q,j",
            comments="",
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes[0].scatter(zetas, q_vals, s=4, c="blue", alpha=0.6, label="q")
        axes[1].scatter(zetas, j_vals, s=4, c="red", alpha=0.6, label="J")
        z_min, z_max = -2.5, 2.5
        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()
            ax.set_xlim(z_min, z_max)
        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")
        if Beta==0:
            fig.suptitle(rf"$a={a},\ b={b},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \text{{sign}}={sign}$")
        else:
            fig.suptitle(rf"$\Beta={Beta},\ a={a},\ b={b},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \text{{sign}}={sign}$")


        fig.tight_layout()
        outpng = os.path.join(png_dir, os.path.basename(outcsv).replace(".csv", ".png"))
        fig.savefig(outpng, dpi=200)
        plt.close(fig)
        print(f"wrote {outcsv} and {outpng}")
        return 0

    Parallel(n_jobs=n_jobs)(
        delayed(run_task)(s, T, g) for (s, T, g) in tasks
    )


if __name__ == "__main__":
    main()
