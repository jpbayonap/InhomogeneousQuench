import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

# sparse simulation
import scipy.sparse as sp
import scipy.sparse.linalg as spla

############################################################
# 0. FREE FERMIONIC HAMILTONIAN
############################################################

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
        H[0,  N-1] = t1
        H[N-1, 0]  = t1
        H = H.tocsr()

    return H

############################################################
# 1. INHOMOGENEOUS INITIAL STATE
############################################################

def Gamma_0(L):
    unit= L
    n1= np.array([[0,0],[0,1]])
    rho_0L= np.zeros([2*L,2*L]) #Left side is the vacuum
    nell_indices= np.array([0 if i%2 ==0 else 1 for i in range(unit)])
    rho_0R= np.kron(n1,np.diag(nell_indices)) # Right side is the Nell state
    rho_0= rho_0L +rho_0R
    return rho_0

############################################################
# 2.  SPARSE LINDBLADIAN
############################################################

# ---------- Vectorization (row-stacking) ----------
def mat2vec(C):
    """Row-stacking vectorization: vec(C) with order='C' -> shape (N^2, 1)."""
    return np.asarray(C).reshape(-1, 1, order='C')

def vec2mat(v, N):
    """Inverse of mat2vec."""
    return np.asarray(v).reshape(N, N, order='C')

# ---------- Sparse superoperators (row-stacking) ----------
def spre_sp(A):
    """
    Pre-multiplication superoperator (sparse):
        A·C  -> (A ⊗ I) vec(C)
    """
    A = sp.csr_matrix(A)
    N = A.shape[0]
    return sp.kron(A, sp.eye(N, format='csr'), format='csr')

def spost_sp(A):
    """
    Post-multiplication superoperator (sparse):
        C·A  -> (I ⊗ A^T) vec(C)
    """
    A = sp.csr_matrix(A)
    N = A.shape[0]
    return sp.kron(sp.eye(N, format='csr'), A.T, format='csr')

# ---------- Projectors ----------
def projector_centered_sp(L, size, bc='open', dtype=np.float64):
    """
    P on region A = [-|L|, |L|] centered at 'size' in a chain of length N=2*size.
    Returns sparse diagonal (CSR).
    """
    N = 2 * size
    L = abs(int(L))
    center = size

    diag = np.zeros(N, dtype=dtype)
    if bc == 'open':
        lo = max(0, center - L)
        hi = min(N - 1, center + L)
        diag[lo:hi+1] = 1.0
    elif bc == 'periodic':
        idx = (np.arange(center - L, center + L + 1) % N)
        diag[idx] = 1.0
    else:
        raise ValueError("bc must be 'open' or 'periodic'")

    return sp.diags(diag, 0, shape=(N, N), dtype=dtype, format='csr')

def projector_prefix_sp(LA, N, bc='open', dtype=np.float64):
    """
    P on region A = {0,1,..., LA-1}. (Useful for a half-chain etc.)
    Returns sparse diagonal (CSR). For PBC, same diagonal (wrap not needed).
    """
    diag = np.zeros(N, dtype=dtype)
    if LA > 0:
        if bc == 'open':
            diag[:min(LA, N)] = 1.0
        elif bc == 'periodic':
            # same diagonal (indices are already in [0,N))
            diag[:min(LA, N)] = 1.0
        else:
            raise ValueError("bc must be 'open' or 'periodic'")
    return sp.diags(diag, 0, shape=(N, N), dtype=dtype, format='csr')

def projector_prefix_RIGHT_sp(LA, N, bc='open', dtype=np.float64):
    """
    P on region A = {LA,..., N}. (Useful for a half-chain etc.)
    Returns sparse diagonal (CSR). For PBC, same diagonal (wrap not needed).
    """
    diag = np.zeros(N, dtype=dtype)
    if LA > 0:
        if bc == 'open':
            diag[max(0,LA):] = 1.0
        elif bc == 'periodic':
            # same diagonal (indices are already in [0,N))
            diag[max(0,LA):] = 1.0
        else:
            raise ValueError("bc must be 'open' or 'periodic'")
    return sp.diags(diag, 0, shape=(N, N), dtype=dtype, format='csr')

# ---------- Liouvillian (row-stacking convention) ----------
def liouvillian_sp(L, P, gamma):
    """
    Sparse Liouvillian for:
        dC/dt = L C + C L† + 2γ ( P C P  )

    Row-stacking ⇒
        L_sup = (L ⊗ I) + (I ⊗ LT ) + 2γ [ P⊗P^T ].
    """
    # ensure sparse CSR/CSC
    L = sp.csr_matrix(L)
    P = sp.csr_matrix(P)
    L_sup= spre_sp(L) + spost_sp(L.conj().T) + (2*gamma)*(spre_sp(P) @ spost_sp(P)) 
    
    return L_sup.tocsr()
    
############################################################
# 1. COVARIANCE DYNAMICS
############################################################

# ---------- Time evolution without dense expm ----------
def evolve_vec_expm_multiply(L_sup, vecC0, t_array):
    """
    Compute vec(C(t)) = exp(t*L_sup) vecC0 for all t in t_array using expm_multiply.
    L_sup : sparse (N^2 x N^2)
    vecC0 : (N^2, 1) dense vector
    t_array : 1D array of times
    Returns: list of (N^2,1) vectors (or a stacked array)
    """
    vecC0 = np.asarray(vecC0).reshape(-1)
    # expm_multiply accepts an array of times for batching:
    # returns an array with shape (len(t), N^2)
    Y = spla.expm_multiply(L_sup, vecC0, start=0.0, stop=float(t_array[-1]),
                           num=len(t_array), endpoint=True)
    # Y[k] = exp(t_k L) vecC0
    return Y.reshape(len(t_array), -1, 1)

# ---------- Time evolution without dense expm for single time step ----------
def evolve_vec_single(L_sup, vecC0, t):
    """
    Compute vec(C(t)) = exp(t * L_sup) vecC0 for a single time t.

    Parameters
    ----------
    L_sup : (N^2, N^2) sparse or dense matrix
        Liouvillian superoperator
    vecC0 : (N^2, 1) initial vector (flattened density/correlation matrix)
    t     : float
        Time point

    Returns
    -------
    vecC_t : (N^2, 1) array
        Evolved vector at time t.
    """
    vecC0 = np.asarray(vecC0).reshape(-1)
    # expm_multiply(A, v) computes exp(A) v efficiently without forming exp(A)
    vecC_t = spla.expm_multiply(L_sup * t, vecC0)
    return vecC_t.reshape(-1, 1)

############################################################
# 2. PARTICLE DENSITY DYNAMICS HEATMAP PLOT
############################################################
def plot_density_heatmap(GAMMA, t_axis=None, cmap="coolwarm",
                         vmin=0.0, vmax=1.0, cbar_label=r"$n(x,t)=C_{x,x}(t)$",
                         yticks=None):
    """
    Plot heatmap of n(x,t) = C_{x,x}(t).

    GAMMA : array of shape (T, N, N)  OR  (T, N*N) with row-stacking (index = j + i*N)
    t_axis: optional 1D array of length T for labeling the time axis
    cmap  : matplotlib colormap
    vmin, vmax: color scale limits
    cbar_label: colorbar label
    yticks: optional tick locations for time axis (only used if t_axis is given)
    """
    GAMMA = np.asarray(GAMMA)
    if GAMMA.ndim != 3 :
        raise ValueError("GAMMA must have shape (T,N,N) or (T,N*N,1).")

    T = GAMMA.shape[0]
    _, N, N2 = GAMMA.shape
    # Extract n(x,t) depending on the input format
    if GAMMA.ndim == 3:
        if N2==1:
            # (T, N*N, 1) row-stacked -> take indices i*(N+1)
            TN = GAMMA.shape[1]
            N = int(np.sqrt(TN))
            if N*N != TN:
                raise ValueError("For (T,N*N) input, second dim must be a perfect square.")
            diag_idx = np.arange(0, N*(N+1), N+1)  # i*(N+1)
            n_xt = GAMMA[:, diag_idx]  # (T,N)
       
        if N == N2:
            n_xt = np.diagonal(GAMMA, axis1=1, axis2=2)  # (T,N)

    n_xt = np.real_if_close(n_xt)

    fig, ax = plt.subplots(figsize=(9, 6))

    if t_axis is None:
        im = ax.imshow(n_xt, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel("position $x$")
        ax.set_ylabel("time index")
    else:
        t_axis = np.asarray(t_axis)
        if t_axis.shape[0] != T:
            raise ValueError("t_axis must have length T = GAMMA.shape[0].")
        im = ax.imshow(n_xt, origin="lower", aspect="auto",
                       extent=[0, N-1, t_axis[0], t_axis[-1]],
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        if yticks is not None:
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{val/np.pi:.2f}" for val in yticks])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, labelpad=15)

    plt.tight_layout()
    return fig, ax



