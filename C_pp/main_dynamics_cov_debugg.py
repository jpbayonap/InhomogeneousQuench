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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ensure repo root and C_pp are on sys.path
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)
for p in (repo_root, here):
    if p not in sys.path:
        sys.path.append(p)

#  Projector
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

# Initial state
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

def Hamiltonian(L, t1, mu=0.0, bc="open", dtype=np.complex128):
    """
    Nearest-neighbor Hamiltonian H = mu*I + t1*sum_j (|j><j+1| + |j+1><j|)
    on a chain of length N = 2*L.  OBC/PBC supported. Sparse CSR.
    """
    N = 2*L
    main = mu * np.ones(N, dtype=dtype)
    off  = -t1 * np.ones(N-1, dtype=dtype)

    # OBC tridiagonal
    H = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr", dtype=dtype)

    if bc.lower() == "periodic" or bc.lower() == "pbc":
        # add wrap bonds
        H = H.tolil()
        H[0,  N-1] = -t1
        H[N-1, 0]  = -t1
        H = H.tocsr()

    return H

def mat2vec(C):
    """Row-stacking vectorization: vec(C) with order='C' -> shape (N^2)."""
    return np.asarray(C).reshape(-1, order='C')

def get_elem(C, i, j, bc):
    N = C.shape[0]
    ii = int(i)
    jj = int(j)
    if bc and bc[0].lower() == "p":
        return C[ii % N, jj % N]
    if 0 <= ii < N and 0 <= jj < N:
        return C[ii, jj]
    return 0.0 + 0.0j


def qp_left(J, r, x, C, bc="pbc"):
    
    return J*(get_elem(C, x , x + r , bc) + get_elem(C, x + r, x, bc))
    


def qm_left(J, r, x, C, bc="pbc"):
    
    return 1j *J* (get_elem(C, x, x + r, bc) - get_elem(C, x + r, x, bc))


def jp_left(J, r, x, C, bc="pbc"):

    J_sq= J**2
    return 1j * J_sq*(
            get_elem(C, x + 1, x + r, bc)
            - get_elem(C, x, x + r + 1, bc)
            + get_elem(C, x + r + 1, x, bc)
            - get_elem(C, x + r, x + 1, bc)
        )


def jm_left(J, r, x, C, bc="pbc"):

    J_sq= J**2
    return -J_sq*(
        get_elem(C, x + 1, x + r , bc)
        - get_elem(C, x, x + r + 1, bc)
        - get_elem(C, x + r + 1, x, bc)
        + get_elem(C, x + r, x + 1, bc)
    )




def main():
    # System Parameters
    L= 1000
    N= 2*L
    gamma= 0.5
    J=1
    # Initial state
    C_0 = Gamma0(L)
    # Test non-unitary part only
    # C_0[-1, 0] = 1.0
    # C_0[0, -1] = 1.0
    print("C_0 shape:", C_0.shape)
    print("||C_0|| =", np.linalg.norm(C_0))
    print("C_0[:10,:10]=\n", C_0[:10, :10])

    # Dynamics parameters
    t_max= 200
    print("T:", t_max)
    RK_steps = max(800, int(np.ceil(t_max / 0.25)))
    dt= t_max/RK_steps
    # Hamiltonian
    H= Hamiltonian(L, J, 0.0, "open")
    # H= sp.csr_matrix((N,N),  dtype=np.complex128)
    print("System Hamiltonian shape", H.shape )
    # Non-unitary term P_A * P_{\bar{A}}+ P_{\bar{A}}* P_{A}
    A_bar= np.arange(0,L)
    A= np.arange(L,N)
    P_row = proj_row(A_bar, A, N)

    # non-zero values of projector
    pr_rows, pr_cols = P_row.nonzero()
    print("non-empty rows head", pr_rows[:10])
    print("non-empty cols head", pr_cols[:10])

    # matrix-free matvec: L_sup v = (h_cond⊗I + I⊗h_cond^T - g P∘) v
    def matvec(v, g, h):
        v = np.asarray(v).reshape(-1)
        n = int(np.sqrt(v.size))
        C = v.reshape(n, n, order="C" )
        h_cond= 1j * h 
        term1 = h_cond @ C
        term2 = C @ h_cond.conjugate().T
        term3 = np.zeros_like(C)
        term3[pr_rows, pr_cols] = C[pr_rows, pr_cols]
        term3 *= g
        return (term1 + term2 - term3).reshape(-1, order="C")

    def Lindbladian (N, g, h):

        I = np.eye(N, dtype=np.complex128)
        h_cond = (1j *h.toarray()).astype(np.complex128)
        mask = P_row.toarray().astype(np.complex128)

        L_sup = (
            np.kron(h_cond,I)
            + np.kron(I, h_cond.conjugate())
            -g * np.diag(mask.reshape(-1,order="C") )
        )
        return L_sup
        

    def rk_step(v, g ,h,  dt):
                    k1 = matvec(v, g ,h)
                    k2 = matvec(v + 0.5 * dt * k1, g, h)
                    k3 = matvec(v + 0.5 * dt * k2, g, h)
                    k4 = matvec(v + dt * k3, g, h)
                    return v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


    # Dynamics in doubled Hilbert space
    vecC0 = mat2vec(C_0)
    print("Vectorized C_0 tail", vecC0[-10:])

    
    # expm evolution
    # Lindladian  L= (h_cond⊗I + I⊗h_cond^T - g P∘)
    # L_sup = Lindbladian(N,gamma, H)
    # print("Lindbladian shape", L_sup.shape)
    # # C at time t
    # VecCt = la.expm(t_max*L_sup)@vecC0
    # C_t_expm= VecCt.reshape(N,N, order="C")
    # print("C_t with exponential,  shape :", C_t_expm.shape)
    # print("|| C_t_expm|| =", np.linalg.norm(C_t_expm))
    # print("C_t_expm[:10,:10]=\n", C_t_expm[:10, :10])

    # Runge kuta evolution

    # Check If norm are near at t=0
    # print("||matvec - L_sup @ v0|| =",
    #   np.linalg.norm(matvec(vecC0, gamma, H) - L_sup @ vecC0))
    # print(f"Simulation with R_K= {RK_steps} steps")

    # for gamma in [0.1, 0.25, 0.5, 1.0, 2.0]:
    v = vecC0.copy()
    for _ in range(RK_steps):
        v = rk_step(v, gamma, H,  dt)
    vecCt_rk4 = v
    C_t_rk4= vecCt_rk4.reshape(N, N, order="C")
    print("C_t with rk4,  shape:",  C_t_rk4.shape)
    print("|| C_t_rk4|| =", np.linalg.norm(C_t_rk4))
    print("C_t_rk4[:10,:10]=\n", C_t_rk4[:10, :10])

    # Compare evolution methods
    # err_abs= np.linalg.norm(C_t_expm- C_t_rk4)
    # err_rel = err_abs/ np.linalg.norm(C_t_expm)
    # err_max = np.max(np.abs(C_t_expm - C_t_rk4))
    # print("abs error = ", err_abs)
    # print("rel error = ", err_rel)
    # print(" max entrywise error =",  err_max)

    # Profiles test

    r= 3
    sgn = "-"
    # Interface
    x_0 = L -1
    for x in range(x_0-4, x_0 + 5):
        q_val= qm_left(J, r, x, C_t_rk4, "open")
        C_l= C_t_rk4[x,x+r]
        C_r= C_t_rk4[x+r,x]
        print(f"x= {x}, q_r{r}_sgn_{sgn}[{x}]={q_val}")
        print(f"x= {x}, C_l_sgn_{sgn}[{x}]={C_l}")
        print(f"x= {x}, C_r_sgn_{sgn}[{x}]={C_r}")
        print("hermiticity check =", C_r - np.conjugate(C_l))

    xs = np.arange(N)
    
    q_plot = np.array([np.real(qm_left(J, r, x, C_t_rk4, "open")) for x in xs])
    zetas = (xs - (L - 1)) / t_max

    base = f"test_q_r{r}_sgn_{sgn}_2N_{N}_g_{gamma}_T_{t_max}"

    # np.savetxt(
    #     base + ".txt",
    #     np.column_stack([xs, q_plot]),
    #     delimiter=",",
    #     header="j,q3",
    #     comments="",
    # )
    # xss= xs - (L - 1)
    # np.savetxt(
    #     base + "manuscript.txt",
    #     np.column_stack([xss, q_plot]),
    #     delimiter=",",
    #     header="j,q3",
    #     comments="",
    # )

    np.savez_compressed(
    f"debug_cov_NEEL_s_{L}_gamma{gamma:.2f}_T{t_max:.1f}_N{N}.npz",
    C_t=C_t_rk4,
    )


    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(zetas, q_plot)
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$q^{(r,-)}(\zeta)$")
    ax.grid(True, alpha=0.3)
    plt.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    
    
    





        

if __name__ == "__main__":
    main()