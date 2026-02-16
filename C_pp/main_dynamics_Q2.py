#!/usr/bin/env python3
"""
Evolve an inhomogeneous half-empty, half-thermal initial state (right at betaR)
under the Lindbladian and plot q/J profiles. No analytic overlay.
"""
import sys
import time
import argparse
import os
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ensure repo root (parent of C_pp) is on sys.path for ghd_module/dynamics_module
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dynamics_module import Hamiltonian, projector_prefix_RIGHT_sp, mat2vec
from ghd_module import hyd_charge_Q2, hyd_current_Q2, hyd_charge_thermal, hyd_current_thermal



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
    
def build_q2_obc(L: int, hop: int, j: float) -> np.ndarray:
    """
    OBC Q2 matrix with hopping distance = hop:
      Q2_{i,i+hop} = Q2_{i+hop,i} = j
    so the PBC dispersion would be 2 j cos(hop k).
    """
    Q = np.zeros((L, L), dtype=float)
    for i in range(L):
        j_idx = i + hop
        if j_idx < L:
            Q[i, j_idx] += j
            Q[j_idx, i] += j
    return Q

    


def Gamma_beta_Q2(beta: float, Q2: np.ndarray) -> np.ndarray:
    """
    One-body correlation matrix for a quadratic Hamiltonian:
    C = V f(H) V^\dagger with f(eps)=1/(1+exp(beta*q_2(k))).
    This avoids the partition-function scaling and keeps entries in [0,1].
    """
    eigvals, eigvecs = la.eigh(Q2)
    f_occ = 1.0 / (1.0 + np.exp(beta * eigvals))
    return (eigvecs * f_occ) @ eigvecs.conjugate().T

def Gamma0(L: int):
    """
    Inhomogeneous initial state: vacuum on the left half.
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
    ap.add_argument("--r", type=int, default=1, help="Lattice r index")
    ap.add_argument("--sign", type=str, default="-", help="+ or -")
    ap.add_argument("--betaL", type=float, default=0.0, help="Inverse temperature left (0 => vacuum)")
    ap.add_argument("--betaR", type=float, default=0.5, help="Inverse temperature right")
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
    betaL = args.betaL
    betaR = args.betaR
    r= args.r

    os.makedirs(args.outdir, exist_ok=True)

    for s in args.sizes:
        N = 2 * s
        center = N // 2
        J = 1.0
        Q2_right = build_q2_obc(s, hop=2, j=J)
        H = Hamiltonian(s, J, 0.0, "open")
        P_right = projector_prefix_RIGHT_sp(LA=s, N=N, bc="open")
        # Left half: vacuum only when betaL == 0.0; otherwise thermal Q2
        if args.betaL == 0.0:
            C_L = np.zeros((s, s), dtype=complex)
        else:
            Q2_left = build_q2_obc(s, hop=2, j=J)
            C_L = Gamma_beta_Q2(betaL, Q2_left)
        C_R = Gamma_beta_Q2(betaR, Q2_right)
        C0 = la.block_diag(C_L, C_R)
        # C0= Gamma0(s)
        vecC0 = mat2vec(C0)
        vecC0_flat = np.asarray(vecC0).reshape(-1)

        tasks = [(T, g) for T in args.times for g in args.gammas]


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
            def rmatvec(v):
                v = np.asarray(v).reshape(-1)
                n = int(np.sqrt(v.size))
                C = v.reshape(n, n)
                h_adj = (-1j * H.conjugate().T) - g * P_right
                term1 = h_adj @ C
                term2 = C @ h_cond
                term3 = 2.0 * g * (P_right @ C @ P_right.conjugate().T)
                return (term1 + term2 + term3).reshape(-1)
            def matmat(X):
                return np.column_stack([matvec(col) for col in X.T])
            def rmatmat(X):
                return np.column_stack([rmatvec(col) for col in X.T])
            def matvec_scaled(v):
                return T * matvec(v)
            def matmat_scaled(X):
                return T * matmat(X)

            t0 = time.perf_counter()
            print(f"Time evolution with gamma={g}, r={args.r}, sign={sign}. ")
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
                L_op_scaled = spla.LinearOperator(
                    (N * N, N * N),
                    matvec=matvec_scaled,
                    matmat=matmat_scaled,
                    rmatvec=rmatvec,
                    rmatmat=rmatmat,
                    dtype=np.complex128,
                )
                vecC_t = spla.expm_multiply(L_op_scaled, vecC0_flat, traceA=0.0)
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

            # hydrodynamic overlay (only meaningful for gamma ~ 0)
            q_ghd_vals = j_ghd_vals = None
            if abs(g) < 1e-12:
                q_ghd_vals = np.array([hyd_charge_Q2(r, zeta, sign, betaR, betaL) for zeta in zetas])
                j_ghd_vals = np.array([hyd_current_Q2(r, zeta, sign, betaR, betaL) for zeta in zetas])

            csv_dir = os.path.join(args.outdir, "GHD_Q2_CSV")
            png_dir = os.path.join(args.outdir, "GHD_Q2_PNG")
            

            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

            if betaL == 0:
                outcsv = os.path.join(
                csv_dir,
                f"GHD_Q2_LVacuum_betaR_{betaR}_r{args.r}_sign{sign}_gamma{g:.2f}_N{N}_TEST.csv",
            )
            else :
                outcsv = os.path.join(
                csv_dir,
                f"GHD_Q2TH_betaL_{betaL}betaR_{betaR}_r{args.r}_sign{sign}_gamma{g:.2f}_N{N}_TEST.csv",
            )
                # Save data to csv file
            np.savetxt(
                outcsv,
                np.column_stack([np.full_like(zetas, g, dtype=float), np.full_like(zetas, T, dtype=float), zetas, q_vals, j_vals]),
                delimiter=",",
                header="gamma,time,zeta,q,j",
                comments="",
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            axes[0].scatter(zetas, q_vals, s=4, c="blue", alpha=0.6, label="q")
            axes[1].scatter(zetas, j_vals, s=4, c="red",  alpha=0.6,  label="J")
            if q_ghd_vals is not None and j_ghd_vals is not None:
                axes[0].plot(zetas, q_ghd_vals, lw=3, c="green", label="q ghd")
                axes[1].plot(zetas, j_ghd_vals, lw=3, c="orange", label="J ghd")
            for ax in axes:
                ax.set_xlabel(r"$\zeta$")
                ax.grid(True, ls="--", alpha=0.5)
                ax.legend()
            axes[0].set_ylabel("q")
            axes[1].set_ylabel("J")
            if betaL ==0:
                fig.suptitle(rf"$L= Vacuum,\ \beta_R={args.betaR},\ \gamma={g},\ T={T},\ N={N},\ r={args.r},\ \text{{sign}}={sign}$")
            else:
                fig.suptitle(rf"$\beta_L={args.betaL},\ \beta_R={args.betaR},\ \gamma={g},\ T={T},\ N={N},\ r={args.r},\ \text{{sign}}={sign}$")

            fig.tight_layout()
            outpng = os.path.join(png_dir, os.path.basename(outcsv).replace(".csv", ".png"))
            fig.savefig(outpng, dpi=200)
            plt.close(fig)
            print(f"wrote {outcsv} and {outpng}")
            return 0

        Parallel(n_jobs= n_jobs)(
            delayed(run_task)(T, g) for (T, g) in tasks
        )


if __name__ == "__main__":
    main()
