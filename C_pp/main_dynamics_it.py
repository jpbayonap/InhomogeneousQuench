#!/usr/bin/env python3
"""
Lindbladian dynamics with a row/column jump operator built from cross regions.
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


def proj_row(A: np.array, B: np.array, N: int, dtype=np.complex128) -> sp.csr_matrix:
    A = np.atleast_1d(A).astype(int)
    B = np.atleast_1d(B).astype(int)
    if A.size == 0 or B.size == 0:
        raise ValueError("A and B must be non-empty")
    if np.any(A < 0) or np.any(A >= N) or np.any(B < 0) or np.any(B >= N):
        raise ValueError(f"A,B must be in [0,{N-1}]")
    
    rows = np.concatenate([np.repeat(A, B.size), np.repeat(B, A.size)])
    cols = np.concatenate([np.tile(B, A.size), np.tile(A, B.size)])

    data = np.ones(rows.size, dtype=dtype)

    return sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=dtype)


def describe_proj_row(P_row: sp.csr_matrix, A: np.ndarray, B: np.ndarray, N: int, dense_max_n: int = 24):
    rows, cols = P_row.nonzero()
    print(f"P_row shape: {P_row.shape} | nnz={P_row.nnz}")
    print(f"A=[{A[0]}..{A[-1]}] ({A.size}) | B=[{B[0]}..{B[-1]}] ({B.size})")
    if N <= dense_max_n:
        print("P_row =")
        print(P_row.toarray().astype(int))
        return

    head = list(zip(rows[:10].tolist(), cols[:10].tolist()))
    tail = list(zip(rows[-10:].tolist(), cols[-10:].tolist()))
    print(f"first 10 nonzero pairs: {head}")
    print(f"last 10 nonzero pairs: {tail}")
    print(f"row {A[0]} nonzeros span: {B[0]}..{B[-1]} ({B.size})")
    print(f"row {A[-1]} nonzeros span: {B[0]}..{B[-1]} ({B.size})")
    print(f"row {B[0]} nonzeros span: {A[0]}..{A[-1]} ({A.size})")
    print(f"row {B[-1]} nonzeros span: {A[0]}..{A[-1]} ({A.size})")


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


def Gamma_mixed_neel(L: int):
    """
    rho = 1/2 |Neel odd><Neel odd| + 1/2 |Neel even><Neel even|
    for the full chain (N=2L). One-body covariance is diagonal with n_j=1/2.
    """
    N = 2 * L
    return 0.5 * np.eye(N, dtype=complex)

def Gamma_zero_filled(L: int):
    """
    Inhomogeneous vac/fill domain-wall state on the full chain (N=2L):
    left half vacuum (n_j=0), right half filled (n_j=1).

    For r=0 and sign='+', q(x)=2 n(x), so initial q jumps from 0 to 2.
    """
    N = 2 * L
    diag = np.zeros(N, dtype=complex)
    diag[L:] = 1.0
    return np.diag(diag)

def Gamma_PHSYMM(L: int, m: int, A: float):
    """
    Covariance matrix of the PHSYMM GGE:
        n(k) = 1/2 + A sin(2 m k),   N = 2L
    using convention c_y = (1/sqrt(N)) sum_k e^{i k y} c_k.

    In the thermodynamic-limit Fourier integral:
      C_{xy} = <c_x^dagger c_y>
             = 1/2 * delta_{xy}
               + (A/(2i)) [delta_{y-x,-2m} - delta_{y-x,2m}]
             = 1/2 * delta_{xy}
               + ( iA/2) * delta_{y-x,2m}
               + (-iA/2) * delta_{y-x,-2m}.

    We implement this directly (open-boundary embedding, no periodic wrap).
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    if m < 0:
        raise ValueError("m must be nonnegative.")

    N = 2 * int(L)
    m = int(m)
    A = float(A)

    # n(k) must lie in [0,1] for a physical free-fermion covariance.
    if abs(A) > 0.5 + 1e-12:
        raise ValueError(f"A={A} gives nonphysical n(k); require |A| <= 0.5.")

    C = 0.5 * np.eye(N, dtype=np.complex128)
    d = 2 * m
    if d > 0 and d < N:
        amp = 0.5j * A
        # y-x = +2m  -> upper diagonal k=+d
        # y-x = -2m  -> lower diagonal k=-d
        C += amp * np.eye(N, k=d, dtype=np.complex128)
        C -= amp * np.eye(N, k=-d, dtype=np.complex128)
    return C

def Gamma_PHSYMM_odd(L: int, m: int, A: float):
    """
    Covariance matrix of the odd-harmonic PHSYMM GGE:
        n(k) = 1/2 + A cos( [2 m+1] k),   N = 2L
    using convention c_y = (1/sqrt(N)) sum_k e^{i k y} c_k.

    In the thermodynamic-limit Fourier integral:
      C_{xy} = <c_x^dagger c_y>
             = 1/2 * delta_{xy}
               + (A/2) [delta_{y-x,-(2m+1)} + delta_{y-x,(2m+1)}]
             = 1/2 * delta_{xy}
               + ( A/2) * delta_{y-x,(2m+1)}
               + (A/2) * delta_{y-x,-(2m+1)}.

    We implement this directly (open-boundary embedding, no periodic wrap).
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    if m < 0:
        raise ValueError("m must be nonnegative.")

    N = 2 * int(L)
    m = int(m)
    A = float(A)

    # n(k) must lie in [0,1] for a physical free-fermion covariance.
    if abs(A) > 0.5 + 1e-12:
        raise ValueError(f"A={A} gives nonphysical n(k); require |A| <= 0.5.")

    C = 0.5 * np.eye(N, dtype=np.complex128)
    d = 2 * m + 1
    if d > 0 and d < N:
        amp = 0.5 * A
        # y-x = +(2m+1) -> upper diagonal k=+d
        # y-x = -(2m+1) -> lower diagonal k=-d
        C += amp * np.eye(N, k=d, dtype=np.complex128)
        C += amp * np.eye(N, k=-d, dtype=np.complex128)
    return C
    



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
    ap.add_argument("--times", type=float, nargs="+", default=[200], help="Times array")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1], help="Gamma values")
    ap.add_argument("--beta", type=float, default=1.0, help="Inverse temperature Beta (homogeneous)")
    ap.add_argument("--betaL", type=float, default=None, help="Left-half inverse temperature")
    ap.add_argument("--betaR", type=float, default=None, help="Right-half inverse temperature")
    ap.add_argument("--r", type=int, default=1, help="Lattice r index")
    ap.add_argument("--sign", type=str, default="-", help="+ or -")
    ap.add_argument(
        "--s-offset",
        type=int,
        nargs="+",
        default=[0],
        help="Cross-region width: A=[center-s_offset, center-1], B=[center, center+s_offset-1]",
    )
    ap.add_argument("--method", choices=["expm", "rk4"], default="expm", help="Evolution method")
    ap.add_argument("--rk-steps", type=int, default=400, help="RK4 steps if method=rk4")
    ap.add_argument("--rk-dt-max", type=float, default=None, help="Optional max RK4 step size. If set, steps scale with T to keep dt<=rk_dt_max.")
    ap.add_argument("--rk-adapt", action="store_true", help="Use adaptive RK4 with step-doubling")
    ap.add_argument("--rk-tol", type=float, default=1e-6, help="Relative tolerance for adaptive RK4")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory (CSV/PNG subdirs will be created)")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over (time,gamma)")
    ap.add_argument(
        "--check-proj-row",
        action="store_true",
        help="Build A/B and P_row, print them, then exit before time evolution.",
    )
    ap.add_argument(
        "--save-profile-png",
        dest="save_profile_png",
        action="store_true",
        help="Save per-time q/J profile PNGs (default: on).",
    )
    ap.add_argument(
        "--no-save-profile-png",
        dest="save_profile_png",
        action="store_false",
        help="Disable per-time q/J profile PNG generation (CSV is still saved).",
    )
    ap.add_argument(
        "--vac-infty",
        dest="vac_infty",
        action="store_true",
        help="Use vacuum/infinite-temperature split state (left vacuum, right beta=0).",
    )
    ap.add_argument(
        "--no-vac-infty",
        dest="vac_infty",
        action="store_false",
        help="Disable vacuum/infinite-temperature split state.",
    )
    ap.add_argument(
        "--mixed-neel",
        dest="mixed_neel",
        action="store_true",
        help="Use rho=1/2|Neel odd><Neel odd| + 1/2|Neel even><Neel even| as initial state.",
    )
    ap.add_argument(
        "--no-mixed-neel",
        dest="mixed_neel",
        action="store_false",
        help="Disable mixed-Neel initial state.",
    )
    ap.add_argument(
        "--vac-fill",
        dest="vac_fill",
        action="store_true",
        help="Use vac/fill domain-wall initial state (left vacuum, right filled).",
    )
    ap.add_argument(
        "--no-vac-fill",
        dest="vac_fill",
        action="store_false",
        help="Disable fully-filled initial state.",
    )
    ap.add_argument(
        "--phsymm",
        dest="phsymm",
        action="store_true",
        help="Use PHSYMM GGE initial state n(k)=1/2 + A*sin(2*m*k).",
    )
    ap.add_argument(
        "--no-phsymm",
        dest="phsymm",
        action="store_false",
        help="Disable PHSYMM initial state.",
    )
    ap.add_argument(
        "--phsymm-odd",
        dest="phsymm_odd",
        action="store_true",
        help="Use odd PHSYMM GGE initial state n(k)=1/2 + A*cos((2*m+1)*k).",
    )
    ap.add_argument(
        "--no-phsymm-odd",
        dest="phsymm_odd",
        action="store_false",
        help="Disable odd PHSYMM initial state.",
    )
    ap.add_argument("--phsymm-m", type=int, default=1, help="Integer m for PHSYMM initial state.")
    ap.add_argument("--phsymm-A", type=float, default=0.2, help="Amplitude A for PHSYMM initial state.")
    ap.add_argument("--z-min", type=float, default=-4.0, help="Minimum zeta shown in profile PNGs.")
    ap.add_argument("--z-max", type=float, default=4.0, help="Maximum zeta shown in profile PNGs.")
    ap.set_defaults(vac_infty=False)
    ap.set_defaults(mixed_neel=False)
    ap.set_defaults(vac_fill=False)
    ap.set_defaults(phsymm=False)
    ap.set_defaults(phsymm_odd=False)
    ap.set_defaults(save_profile_png=True)

    args = ap.parse_args()

    sign = args.sign
    rk_steps = args.rk_steps
    rk_dt_max = args.rk_dt_max
    rk_adapt = args.rk_adapt
    rk_tol = args.rk_tol
    n_jobs = args.n_jobs
    check_proj_row = args.check_proj_row
    method = args.method
    r = args.r
    Beta = args.beta
    betaL = args.betaL
    betaR = args.betaR
    s_offsets = args.s_offset
    vac_infty = args.vac_infty
    mixed_neel = args.mixed_neel
    vac_fill = args.vac_fill
    phsymm = args.phsymm
    phsymm_odd = args.phsymm_odd
    phsymm_m = args.phsymm_m
    phsymm_A = args.phsymm_A
    plot_z_min = args.z_min
    plot_z_max = args.z_max
    save_profile_png = args.save_profile_png

    if plot_z_min >= plot_z_max:
        raise ValueError("--z-min must be smaller than --z-max.")
    
    times = args.times

    os.makedirs(args.outdir, exist_ok=True)

    sizes = list(args.sizes)
    # Pair each size with an s-offset (default: same value as size).
    if len(s_offsets) == 1 and int(s_offsets[0]) == 0:
        size_offset_pairs = [(int(s), int(s)) for s in sizes]
    elif len(s_offsets) == 1:
        size_offset_pairs = [(int(s), int(s_offsets[0])) for s in sizes]
    elif len(s_offsets) == len(sizes):
        size_offset_pairs = list(zip([int(s) for s in sizes], [int(x) for x in s_offsets]))
    else:
        raise ValueError(
            "--s-offset must be a single value or have the same length as --sizes."
        )

    tasks = [(s, s_off, g, T) for (s, s_off) in size_offset_pairs for g in args.gammas for T in times]

    def run_task(s, s_off, g, T):
        N = 2 * s
        center = N // 2
        a_start = max(0, center - s_off)
        a = np.arange(a_start, center, dtype=int)
        b_end = min(N, center + s_off)
        b = np.arange(center, b_end, dtype=int)
        if s_off <= 10:
            print(f"  A sites: {a.tolist()} | B sites: {b.tolist()}")
        else:
            print(f"  A sites: [{a[0]}..{a[-1]}] ({a.size}) | B sites: [{b[0]}..{b[-1]}] ({b.size})")

        P_row = proj_row(a, b, N)
        if check_proj_row:
            describe_proj_row(P_row, a, b, N)
            return 0

        H = Hamiltonian(s, 1.0, 0.0, "open")
        pr_rows, pr_cols = P_row.nonzero()

        local_betaL = betaL
        local_betaR = betaR
        if local_betaL is not None or local_betaR is not None:
            if local_betaL is None:
                local_betaL = local_betaR
            if local_betaR is None:
                local_betaR = local_betaL
            H_dense = H.toarray()
            H_L = H_dense[:s, :s]
            H_R = H_dense[s:, s:]
            if local_betaL != local_betaR:
                C_L = Gamma_beta(local_betaL, H_L)
                C_R = Gamma_beta(local_betaR, H_R)
                C0 = la.block_diag(C_L, C_R)
            else:
                C0 = Gamma_beta(local_betaR, H_dense)
            vecC0 = mat2vec(C0)
        elif vac_fill:
            C0 = Gamma_zero_filled(s)
            vecC0 = mat2vec(C0)
        elif mixed_neel:
            C0 = Gamma_mixed_neel(s)
            vecC0 = mat2vec(C0)
        elif vac_infty:
            H_dense = H.toarray()
            H_L = H_dense[:s, :s]
            H_R = H_dense[s:, s:]
            C_L = np.zeros_like(H_L)
            C_R = Gamma_beta(0.0, H_R)
            C0 = la.block_diag(C_L, C_R)
            vecC0 = mat2vec(C0)
        elif phsymm_odd:
            C0 = Gamma_PHSYMM_odd(s, phsymm_m, phsymm_A)
            vecC0 = mat2vec(C0)
        elif phsymm:
            C0 = Gamma_PHSYMM(s, phsymm_m, phsymm_A)
            vecC0 = mat2vec(C0)
        elif Beta == 0:
            C0 = Gamma0(s)
            vecC0 = mat2vec(C0)

        else:
            C0 = Gamma_beta(Beta, H.toarray())
            vecC0= mat2vec(C0)
        vecC0_flat = np.asarray(vecC0).reshape(-1)
        

        h_cond = 1j * H 
        # matrix-free matvec: L_sup v = (h_cond⊗I + I⊗h_cond^T - 2g P∘) v
        def matvec(v):
            v = np.asarray(v).reshape(-1)
            n = int(np.sqrt(v.size))
            C = v.reshape(n, n)
            term1 = h_cond @ C
            term2 = C @ h_cond.conjugate().T
            term3 = np.zeros_like(C)
            term3[pr_rows, pr_cols] = C[pr_rows, pr_cols]
            term3 *= g
            return (term1 + term2 - term3).reshape(-1)

        t0 = time.perf_counter()
        a_label = f"[{a_start}, {center - 1}]"
        b_label = f"[{center}, {b_end - 1}]"
        if local_betaL is not None or local_betaR is not None:
            print(
                f"Time evolution starting: betaL={local_betaL}, betaR={local_betaR}, N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        elif vac_fill:
            print(
                f"Time evolution starting Vac/Fill: N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        elif mixed_neel:
            print(
                f"Time evolution starting Mixed-Neel: N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        elif vac_infty:
            print(
                f"Time evolution starting Vac/Infty: N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        elif phsymm_odd:
            print(
                f"Time evolution starting PHSYMM_ODD(m={phsymm_m}, A={phsymm_A}): N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        elif phsymm:
            print(
                f"Time evolution starting PHSYMM(m={phsymm_m}, A={phsymm_A}): N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        elif Beta ==0:
            print(
                f"Time evolution starting: N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        else:
            print(
                f"Time evolution starting: Beta={Beta}, N={N}, T={T}, gamma={g}, r={r}, sign={sign}, A={a_label}, B={b_label}"
            )
        
        if method == "rk4":
            # Keep fixed-step RK4 stable when running multiple times T:
            # start from rk_steps, then (optionally) raise steps per task so
            # dt = T / local_rk_steps never exceeds rk_dt_max.
            local_rk_steps = max(1, int(rk_steps))
            if rk_dt_max is not None:
                local_rk_steps = max(local_rk_steps, int(np.ceil(T / rk_dt_max)))

            def rk_step(v, dt):
                k1 = matvec(v)
                k2 = matvec(v + 0.5 * dt * k1)
                k3 = matvec(v + 0.5 * dt * k2)
                k4 = matvec(v + dt * k3)
                return v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if rk_adapt:
                t = 0.0
                v = vecC0_flat.copy()
                dt = T / float(max(local_rk_steps, 10))
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
                dt = T / float(local_rk_steps)
                if rk_dt_max is None and dt > 0.7:
                    print(
                        f"  WARNING: large fixed RK4 step dt={dt:.3g} (T={T}, steps={local_rk_steps}) may be unstable; "
                        "increase --rk-steps or set --rk-dt-max."
                    )
                v = vecC0_flat.copy()
                for _ in range(local_rk_steps):
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
        zetas = (xs - (center - 1)) / T
        if sign == "-":
            q_vals = np.array([np.real(qm_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jm_symm(r, x, C_t, "open")) for x in xs])
        else:
            q_vals = np.array([np.real(qp_symm(r, x, C_t, "open")) for x in xs])
            j_vals = np.array([np.real(jp_symm(r, x, C_t, "open")) for x in xs])

        csv_dir = os.path.join(args.outdir, "GHD_IT_CSV")
        png_dir = os.path.join(args.outdir, "GHD_IT_PNG")

        os.makedirs(csv_dir, exist_ok=True)
        if save_profile_png:
            os.makedirs(png_dir, exist_ok=True)

        if local_betaL is not None or local_betaR is not None:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_IT_betaL_{local_betaL}_betaR_{local_betaR}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )
        elif vac_fill:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_vac_fill_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )
        elif mixed_neel:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_IT_MixedNeel_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )
        elif vac_infty:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_IT_VacInfty_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )
        elif phsymm_odd:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_IT_PHSYMM_ODD_m{phsymm_m}_A{phsymm_A:.6g}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )
        elif phsymm:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_IT_PHSYMM_m{phsymm_m}_A{phsymm_A:.6g}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )
        elif Beta == 0:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_IT_NEEL_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
            )

        else:
            outcsv = os.path.join(
                csv_dir,
                f"GHD_IT_Beta_{Beta}_r{r}_s_{s_off}_sign{sign}_gamma{g:.2f}_T{T}_N{N}.csv",
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

        if save_profile_png:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            axes[0].scatter(zetas, q_vals, s=4, c="blue", alpha=0.6, label="q")
            axes[1].scatter(zetas, j_vals, s=4, c="red", alpha=0.6, label="J")
            for ax in axes:
                ax.set_xlabel(r"$\zeta$")
                ax.grid(True, ls="--", alpha=0.5)
                ax.legend()
                ax.set_xlim(plot_z_min, plot_z_max)
            axes[0].set_ylabel("q")
            axes[1].set_ylabel("J")
            if local_betaL is not None or local_betaR is not None:
                # Use \mathrm instead of \text for compatibility with older matplotlib mathtext (e.g. cluster Python 3.6 env).
                fig.suptitle(rf"$\beta_L={local_betaL},\ \beta_R={local_betaR},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")
            elif vac_fill:
                fig.suptitle(rf"$\mathrm{{Vac/Fill}},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")
            elif mixed_neel:
                fig.suptitle(rf"$\mathrm{{Mixed\ Neel}},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")
            elif vac_infty:
                fig.suptitle(rf"$\mathrm{{Vac/Infty}},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")
            elif phsymm_odd:
                fig.suptitle(rf"$\mathrm{{PHSYMM\ odd}},\ m={phsymm_m},\ A={phsymm_A:.6g},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")
            elif phsymm:
                fig.suptitle(rf"$\mathrm{{PHSYMM}},\ m={phsymm_m},\ A={phsymm_A:.6g},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")
            elif Beta == 0:
                fig.suptitle(rf"$A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")
            else:
                fig.suptitle(rf"$\beta={Beta},\ A={a_label},\ B={b_label},\ \gamma={g},\ T={T},\ N={N},\ r={r},\ \mathrm{{sign}}={sign}$")

            fig.tight_layout()
            outpng = os.path.join(png_dir, os.path.basename(outcsv).replace(".csv", ".png"))
            fig.savefig(outpng, dpi=200)
            plt.close(fig)
            print(f"wrote {outcsv} and {outpng}")
        else:
            print(f"wrote {outcsv}")
        return 0

    Parallel(n_jobs=n_jobs)(
        delayed(run_task)(s, s_off, g, T) for (s, s_off, g, T) in tasks
    )


if __name__ == "__main__":
    main()
