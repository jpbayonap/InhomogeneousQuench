#!/usr/bin/env python3
"""
Cluster-oriented variant of main_dynamics_it.py.
Evolves the covariance matrix and saves C_t directly, so q/j profiles for any
r, sign can be reconstructed later without re-running the dynamics.
"""
import sys
import time
import argparse
import os
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as la
from joblib import Parallel, delayed

# ensure repo root and C_pp are on sys.path
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
for p in (repo_root, here):
    if p not in sys.path:
        sys.path.append(p)

from dynamics_module import Hamiltonian, mat2vec
from main_dynamics_it import (
    proj_row,
    Gamma_beta,
    Gamma0,
    Gamma_mixed_neel,
    Gamma_zero_filled,
    Gamma_PHSYMM,
    Gamma_PHSYMM_odd,
)


def build_output_path(
    cov_dir,
    *,
    N,
    s_off,
    gamma,
    T,
    Beta,
    betaL,
    betaR,
    vac_fill,
    mixed_neel,
    vac_infty,
    phsymm,
    phsymm_odd,
    phsymm_m,
    phsymm_A,
):
    if betaL is not None or betaR is not None:
        return os.path.join(
            cov_dir,
            f"GHD_ITCOV_betaL_{betaL}_betaR_{betaR}_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
        )
    if vac_fill:
        return os.path.join(
            cov_dir,
            f"GHD_ITCOV_vac_fill_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
        )
    if mixed_neel:
        return os.path.join(
            cov_dir,
            f"GHD_ITCOV_MixedNeel_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
        )
    if vac_infty:
        return os.path.join(
            cov_dir,
            f"GHD_ITCOV_VacInfty_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
        )
    if phsymm_odd:
        return os.path.join(
            cov_dir,
            f"GHD_ITCOV_PHSYMM_ODD_m{phsymm_m}_A{phsymm_A:.6g}_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
        )
    if phsymm:
        return os.path.join(
            cov_dir,
            f"GHD_ITCOV_PHSYMM_m{phsymm_m}_A{phsymm_A:.6g}_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
        )
    if Beta == 0:
        return os.path.join(
            cov_dir,
            f"GHD_ITCOV_NEEL_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
        )
    return os.path.join(
        cov_dir,
        f"GHD_ITCOV_Beta_{Beta}_s_{s_off}_gamma{gamma:.2f}_T{T}_N{N}.npz",
    )


def infer_state_tag(*, Beta, betaL, betaR, vac_fill, mixed_neel, vac_infty, phsymm, phsymm_odd):
    if betaL is not None or betaR is not None:
        return "beta_lr"
    if vac_fill:
        return "vac_fill"
    if mixed_neel:
        return "mixed_neel"
    if vac_infty:
        return "vac_infty"
    if phsymm_odd:
        return "phsymm_odd"
    if phsymm:
        return "phsymm"
    if Beta == 0:
        return "neel"
    return "beta"


def build_initial_covariance(
    *,
    H,
    s,
    Beta,
    betaL,
    betaR,
    vac_fill,
    mixed_neel,
    vac_infty,
    phsymm,
    phsymm_odd,
    phsymm_m,
    phsymm_A,
):
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
    elif vac_fill:
        C0 = Gamma_zero_filled(s)
    elif mixed_neel:
        C0 = Gamma_mixed_neel(s)
    elif vac_infty:
        H_dense = H.toarray()
        H_L = H_dense[:s, :s]
        H_R = H_dense[s:, s:]
        C_L = np.zeros_like(H_L)
        C_R = Gamma_beta(0.0, H_R)
        C0 = la.block_diag(C_L, C_R)
    elif phsymm_odd:
        C0 = Gamma_PHSYMM_odd(s, phsymm_m, phsymm_A)
    elif phsymm:
        C0 = Gamma_PHSYMM(s, phsymm_m, phsymm_A)
    elif Beta == 0:
        C0 = Gamma0(s)
    else:
        C0 = Gamma_beta(Beta, H.toarray())

    return C0, local_betaL, local_betaR


def save_covariance(
    outpath,
    *,
    C_t,
    gamma,
    T,
    N,
    s_off,
    a_sites,
    b_sites,
    method,
    xs,
    zetas,
    state_tag,
    Beta,
    betaL,
    betaR,
    phsymm_m,
    phsymm_A,
    cov_dtype,
):
    if cov_dtype == "complex64":
        C_save = np.asarray(C_t, dtype=np.complex64)
    else:
        C_save = np.asarray(C_t, dtype=np.complex128)

    np.savez_compressed(
        outpath,
        C_t=C_save,
        gamma=float(gamma),
        time=float(T),
        N=int(N),
        L=int(N // 2),
        center=int(N // 2),
        s_offset=int(s_off),
        a_sites=np.asarray(a_sites, dtype=int),
        b_sites=np.asarray(b_sites, dtype=int),
        xs=np.asarray(xs, dtype=int),
        zetas=np.asarray(zetas, dtype=float),
        init_state=np.array(state_tag),
        beta=float(Beta) if Beta is not None else np.nan,
        betaL=float(betaL) if betaL is not None else np.nan,
        betaR=float(betaR) if betaR is not None else np.nan,
        phsymm_m=int(phsymm_m),
        phsymm_A=float(phsymm_A),
        method=np.array(method),
        cov_dtype=np.array(cov_dtype),
    )


def cov_numpy_dtype(cov_dtype_name):
    if cov_dtype_name == "complex64":
        return np.complex64
    return np.complex128


def describe_proj_row(P_row, A, B, N, dense_max_n=24):
    rows, cols = P_row.nonzero()
    print(f"P_row shape: {P_row.shape} | nnz={P_row.nnz}")
    print(f"A=[{A[0]}..{A[-1]}] ({A.size}) | B=[{B[0]}..{B[-1]}] ({B.size})")
    if N <= dense_max_n:
        print("P_row =")
        print(P_row.toarray().real.astype(int))
        return

    head = list(zip(rows[:10].tolist(), cols[:10].tolist()))
    tail = list(zip(rows[-10:].tolist(), cols[-10:].tolist()))
    print(f"first 10 nonzero pairs: {head}")
    print(f"last 10 nonzero pairs: {tail}")
    print(f"row {A[0]} nonzeros span: {B[0]}..{B[-1]} ({B.size})")
    print(f"row {A[-1]} nonzeros span: {B[0]}..{B[-1]} ({B.size})")
    print(f"row {B[0]} nonzeros span: {A[0]}..{A[-1]} ({A.size})")
    print(f"row {B[-1]} nonzeros span: {A[0]}..{A[-1]} ({A.size})")


def main():
    ap = argparse.ArgumentParser(description="Evolve and save covariance matrices C_t directly.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[400], help="Half-chain sizes (L)")
    ap.add_argument("--times", type=float, nargs="+", default=[200], help="Times array")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1], help="Gamma values")
    ap.add_argument("--beta", type=float, default=1.0, help="Inverse temperature Beta (homogeneous)")
    ap.add_argument("--betaL", type=float, default=None, help="Left-half inverse temperature")
    ap.add_argument("--betaR", type=float, default=None, help="Right-half inverse temperature")
    ap.add_argument(
        "--s-offset",
        type=int,
        nargs="+",
        default=[0],
        help="Cross-region width: A=[center-s_offset, center-1], B=[center, center+s_offset-1]",
    )
    ap.add_argument("--method", choices=["expm", "rk4"], default="expm", help="Evolution method")
    ap.add_argument("--rk-steps", type=int, default=400, help="RK4 steps if method=rk4")
    ap.add_argument("--rk-dt-max", type=float, default=None, help="Optional max RK4 step size. If set, steps scale with T so dt<=rk_dt_max.")
    ap.add_argument("--rk-adapt", action="store_true", help="Use adaptive RK4 with step-doubling")
    ap.add_argument("--rk-tol", type=float, default=1e-6, help="Relative tolerance for adaptive RK4")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over (time,gamma)")
    ap.add_argument(
        "--check-proj-row",
        action="store_true",
        help="Build A/B and P_row, print them, then exit before covariance evolution.",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip tasks whose covariance file already exists.")
    ap.add_argument("--cov-dtype", choices=["complex128", "complex64"], default="complex128", help="Stored covariance dtype.")
    ap.add_argument("--vac-infty", dest="vac_infty", action="store_true", help="Use vacuum/infinite-temperature split state.")
    ap.add_argument("--no-vac-infty", dest="vac_infty", action="store_false", help="Disable vacuum/infinite-temperature split state.")
    ap.add_argument("--mixed-neel", dest="mixed_neel", action="store_true", help="Use mixed-Neel initial state.")
    ap.add_argument("--no-mixed-neel", dest="mixed_neel", action="store_false", help="Disable mixed-Neel initial state.")
    ap.add_argument("--vac-fill", dest="vac_fill", action="store_true", help="Use vac/fill domain-wall initial state.")
    ap.add_argument("--no-vac-fill", dest="vac_fill", action="store_false", help="Disable vac/fill initial state.")
    ap.add_argument("--phsymm", dest="phsymm", action="store_true", help="Use PHSYMM GGE initial state n(k)=1/2 + A*sin(2*m*k).")
    ap.add_argument("--no-phsymm", dest="phsymm", action="store_false", help="Disable PHSYMM initial state.")
    ap.add_argument("--phsymm-odd", dest="phsymm_odd", action="store_true", help="Use odd PHSYMM GGE initial state n(k)=1/2 + A*cos((2*m+1)*k).")
    ap.add_argument("--no-phsymm-odd", dest="phsymm_odd", action="store_false", help="Disable odd PHSYMM initial state.")
    ap.add_argument("--phsymm-m", type=int, default=1, help="Integer m for PHSYMM initial state.")
    ap.add_argument("--phsymm-A", type=float, default=0.2, help="Amplitude A for PHSYMM initial state.")
    # Accepted for CLI compatibility with main_dynamics_it.py, but unused here.
    ap.add_argument("--r", type=int, default=None, help="Unused in covariance mode.")
    ap.add_argument("--sign", type=str, default=None, help="Unused in covariance mode.")

    ap.set_defaults(vac_infty=False)
    ap.set_defaults(mixed_neel=False)
    ap.set_defaults(vac_fill=False)
    ap.set_defaults(phsymm=False)
    ap.set_defaults(phsymm_odd=False)

    args = ap.parse_args()

    sizes = list(args.sizes)
    s_offsets = args.s_offset
    if len(s_offsets) == 1 and int(s_offsets[0]) == 0:
        size_offset_pairs = [(int(s), int(s)) for s in sizes]
    elif len(s_offsets) == 1:
        size_offset_pairs = [(int(s), int(s_offsets[0])) for s in sizes]
    elif len(s_offsets) == len(sizes):
        size_offset_pairs = list(zip([int(s) for s in sizes], [int(x) for x in s_offsets]))
    else:
        raise ValueError("--s-offset must be a single value or have the same length as --sizes.")

    tasks = [(s, s_off, g, T) for (s, s_off) in size_offset_pairs for g in args.gammas for T in args.times]

    cov_dir = os.path.join(args.outdir, "GHD_IT_COV")
    os.makedirs(cov_dir, exist_ok=True)
    work_dtype = cov_numpy_dtype(args.cov_dtype)
    check_proj_row = args.check_proj_row

    def run_task(s, s_off, g, T):
        N = 2 * s
        center = N // 2
        a_start = max(0, center - s_off)
        a = np.arange(a_start, center, dtype=int)
        b_end = min(N, center + s_off)
        b = np.arange(center, b_end, dtype=int)

        H = Hamiltonian(s, 1.0, 0.0, "open", dtype=work_dtype)
        P_row = proj_row(a, b, N, dtype=work_dtype)
        if check_proj_row:
            describe_proj_row(P_row, a, b, N)
            return 0
        pr_rows, pr_cols = P_row.nonzero()

        C0, local_betaL, local_betaR = build_initial_covariance(
            H=H,
            s=s,
            Beta=args.beta,
            betaL=args.betaL,
            betaR=args.betaR,
            vac_fill=args.vac_fill,
            mixed_neel=args.mixed_neel,
            vac_infty=args.vac_infty,
            phsymm=args.phsymm,
            phsymm_odd=args.phsymm_odd,
            phsymm_m=args.phsymm_m,
            phsymm_A=args.phsymm_A,
        )
        state_tag = infer_state_tag(
            Beta=args.beta,
            betaL=local_betaL,
            betaR=local_betaR,
            vac_fill=args.vac_fill,
            mixed_neel=args.mixed_neel,
            vac_infty=args.vac_infty,
            phsymm=args.phsymm,
            phsymm_odd=args.phsymm_odd,
        )
        outpath = build_output_path(
            cov_dir,
            N=N,
            s_off=s_off,
            gamma=g,
            T=T,
            Beta=args.beta,
            betaL=local_betaL,
            betaR=local_betaR,
            vac_fill=args.vac_fill,
            mixed_neel=args.mixed_neel,
            vac_infty=args.vac_infty,
            phsymm=args.phsymm,
            phsymm_odd=args.phsymm_odd,
            phsymm_m=args.phsymm_m,
            phsymm_A=args.phsymm_A,
        )
        if args.skip_existing and os.path.isfile(outpath):
            print(f"skip existing {outpath}")
            return 0

        C0 = np.asarray(C0, dtype=work_dtype)
        vecC0_flat = np.asarray(mat2vec(C0), dtype=work_dtype).reshape(-1)
        h_cond = (1j * H).astype(work_dtype, copy=False).tocsr()
        h_cond_dag = h_cond.conjugate().transpose().tocsr()
        gamma_work = work_dtype(g)

        def matvec(v):
            v = np.asarray(v, dtype=work_dtype).reshape(-1)
            n = int(np.sqrt(v.size))
            C = v.reshape(n, n)
            out = np.asarray(h_cond @ C, dtype=work_dtype)
            out += np.asarray(C @ h_cond_dag, dtype=work_dtype)
            out[pr_rows, pr_cols] -= gamma_work * C[pr_rows, pr_cols]
            return out.reshape(-1)

        a_label = f"[{a[0]}, {a[-1]}]"
        b_label = f"[{b[0]}, {b[-1]}]"
        print(
            f"Covariance evolution: state={state_tag}, N={N}, T={T}, gamma={g}, "
            f"A={a_label}, B={b_label}, dtype={np.dtype(work_dtype).name}, "
            f"out={os.path.basename(outpath)}"
        )

        t0 = time.perf_counter()
        if args.method == "rk4":
            local_rk_steps = max(1, int(args.rk_steps))
            if args.rk_dt_max is not None:
                local_rk_steps = max(local_rk_steps, int(np.ceil(T / args.rk_dt_max)))

            def rk_step(v, dt):
                k1 = matvec(v)
                k2 = matvec(v + 0.5 * dt * k1)
                k3 = matvec(v + 0.5 * dt * k2)
                k4 = matvec(v + dt * k3)
                return v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if args.rk_adapt:
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
                    if err < args.rk_tol or dt <= 1e-12:
                        v = v_half
                        t += dt
                        dt *= 1.5
                    else:
                        dt *= 0.5
                vecC_t = v
            else:
                dt = T / float(local_rk_steps)
                if args.rk_dt_max is None and dt > 0.7:
                    print(
                        f"  WARNING: large fixed RK4 step dt={dt:.3g} (T={T}, steps={local_rk_steps}) may be unstable; "
                        "increase --rk-steps or set --rk-dt-max."
                    )
                v = vecC0_flat.copy()
                for _ in range(local_rk_steps):
                    v = rk_step(v, dt)
                vecC_t = v
        else:
            L_op = spla.LinearOperator((N * N, N * N), matvec=matvec, dtype=work_dtype)
            vecC_t = spla.expm_multiply(L_op * T, vecC0_flat)

        elapsed = time.perf_counter() - t0
        print(f"  evolution took {elapsed/60:.2f} min")

        C_t = vecC_t.reshape(N, N)
        xs = np.arange(N, dtype=int)
        zetas = (xs - (center - 1)) / T
        save_covariance(
            outpath,
            C_t=C_t,
            gamma=g,
            T=T,
            N=N,
            s_off=s_off,
            a_sites=a,
            b_sites=b,
            method=args.method,
            xs=xs,
            zetas=zetas,
            state_tag=state_tag,
            Beta=args.beta,
            betaL=local_betaL,
            betaR=local_betaR,
            phsymm_m=args.phsymm_m,
            phsymm_A=args.phsymm_A,
            cov_dtype=args.cov_dtype,
        )
        print(f"wrote {outpath}")
        return 0

    Parallel(n_jobs=args.n_jobs)(
        delayed(run_task)(s, s_off, g, T) for (s, s_off, g, T) in tasks
    )


if __name__ == "__main__":
    main()
