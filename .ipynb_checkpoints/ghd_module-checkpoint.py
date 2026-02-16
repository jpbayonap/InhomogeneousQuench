import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import csv

# sparse simulation
import scipy.sparse as sp
import scipy.sparse.linalg as spla

############################################################
# 0. TRUNCATED FOURIER SYSTEM FOR b_r  (Python version)
############################################################

def solve_truncated_system_full(M, J, gamma, use_lstsq=False):
    r"""
    Python version of Mathematica's SolveTruncatedSystemFull.

    Solve for b_r with r in [-M, ..., M] from

        4 J^2 ∑_m b_m [
           2/(1 - (r-m)^2) - 2/(1 - (r+m)^2)
        ] + π \gamma J r b_r = \gamma J (1 - (-1)^r) / (4π)

    using parity decomposition:
      - even r couple only to even m
      - odd  r couple only to odd  m

    Parameters
    ----------
    M : int
        Truncation in r (system size is 2M+1).
    J : float
        Hopping / coupling.
    gamma : float
        Dissipation parameter.
    use_lstsq : bool
        If True, always use least-squares; otherwise try solve, then
        fall back to lstsq if the matrix is singular.

    Returns
    -------
    r_full : ndarray, shape (2M+1,)
        r = -M, ..., M.
    b_full : ndarray, shape (2M+1,)
        b_r coefficients aligned with r_full.
    """
    # all r
    r_full = np.arange(-M, M + 1, dtype=int)

    # parity masks
    mask_even = (r_full % 2 == 0)
    mask_odd  = ~mask_even

    r_even = r_full[mask_even]
    r_odd  = r_full[mask_odd]

    def solve_block(r_vals, inhomogeneous):
        """
        Solve a single parity block (all even or all odd r's).

        r_vals: 1D array of r's of the same parity.
        inhomogeneous: True => RHS = γJ(1-(-1)^r)/(4π) (odd block),
                       False => RHS = 0          (even block).
        Returns
        -------
        b_block : ndarray, same shape as r_vals
        """
        dim = len(r_vals)
        if dim == 0:
            return np.zeros(0, dtype=float)

        # Broadcasting to build the kernel matrix
        R = r_vals[:, None]       # shape (dim, 1)
        Mmat = r_vals[None, :]    # shape (1, dim)

        rm = R - Mmat             # r_i - r_j
        rp = R + Mmat             # r_i + r_j

        # For same parity r,m: r±m is even, so denom never hits 1 -> safe
        kernel = 2.0 / (1.0 - rm**2) - 2.0 / (1.0 - rp**2)

        # main matrix
        A = 4.0 * (J**2) * kernel

        # add diagonal term π γ J r
        A[np.diag_indices(dim)] += np.pi * gamma * J * r_vals

        # RHS
        if inhomogeneous:
            # γ J (1 - (-1)^r) / (4π)
            rhs = gamma * J * (1.0 - (-1.0)**r_vals) / (4.0 * np.pi)
        else:
            rhs = np.zeros(dim, dtype=float)

        # Solve
        if use_lstsq:
            b_block, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        else:
            try:
                b_block = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                print(np.linalg.LinAlgError)
                b_block, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        return b_block

    # odd block (inhomogeneous), even block (homogeneous)
    b_odd  = solve_block(r_odd,  inhomogeneous=True)
    b_even = solve_block(r_even, inhomogeneous=False)

    # Merge back
    b_full = np.zeros_like(r_full, dtype=float)
    b_full[mask_even] = b_even
    b_full[mask_odd]  = b_odd

    return r_full, b_full


############################################################
# 1. LATTICE CHARGES & CURRENTS (optional)
############################################################

def _get(C, i, j, bc="pbc"):
    """
    Safe element access with boundary conditions.
    - bc="pbc": indices wrap modulo N
    - bc="obc": out-of-range -> 0
    """
    N = C.shape[0]
    if bc.lower().startswith("p"):  # PBC
        return C[i % N, j % N]
    if 0 <= i < N and 0 <= j < N:   # OBC
        return C[i, j]
    return 0.0 + 0.0j

def qp(r, x, C, bc="pbc"):
    """ q^(r,+) on lattice """
    return _get(C, x,   x + r, bc) + _get(C, x + r, x,   bc)

def qm(r, x, C, bc="pbc"):
    """ q^(r,-) on lattice """
    return 1j * (_get(C, x,   x + r, bc) - _get(C, x + r, x,   bc))

def jp(r, x, C, bc="pbc"):
    """ J^(r,+) on lattice """
    return 1j * (
        _get(C, x + 1,   x + r,   bc)
      - _get(C, x,       x + r + 1, bc)
      + _get(C, x + r + 1, x,       bc)
      - _get(C, x + r,   x + 1,   bc)
    )

def jm(r, x, C, bc="pbc"):
    """ J^(r,-) on lattice """
    return -(
        _get(C, x + 1,   x + r,   bc)
      - _get(C, x,       x + r + 1, bc)
      - _get(C, x + r + 1, x,       bc)
      + _get(C, x + r,   x + 1,   bc)
    )

def two_site_avg(arr, bc="obc"):
    """
    Two-site averaging:
      O_i -> (O_i + O_{i+1}) / 2  (with wrap for PBC or truncated for OBC).
    """
    N = len(arr)
    out = np.zeros_like(arr)
    if bc.lower().startswith("p"):
        return 0.5 * (arr + np.roll(arr, -1))
    out[:-1] = 0.5 * (arr[:-1] + arr[1:])
    out[-1]  = arr[-1]
    return out


# ============================================================
#  2. Symmetric Local charges
# ============================================================


def qp_symm(r, x, C, bc="pbc"):
    """ q_plus: C[x, x+r] + C[x+r, x] """
    return _get(C, x-r, x + r, bc) + _get(C, x + r, x-r, bc)



def qm_symm(r, x, C, bc="pbc"):
    """ q_minus: i*(C[x, x+r] - C[x+r, x]) """
    return 1j * (_get(C, x-r, x + r+1, bc) - _get(C, x + r+1, x-r, bc))


def jp_symm(r, x, C, bc="pbc"):
    """
    j_plus: i*( C[x+1, x+r] - C[x, x+r+1]
               + C[x+r+1, x] - C[x+r, x+1] )
    """
    return 1j * (
        _get(C, x -r +1,    x + r,     bc)
        - _get(C, x-r,      x + r + 1, bc)
        + _get(C, x + r + 1, x-r,      bc)
        - _get(C, x + r,  x -r + 1,     bc)
    )


def jm_symm(r, x, C, bc="pbc"):
    """
    j_minus: -( C[x+1, x+r] - C[x, x+r+1]
               - C[x+r+1, x] + C[x+r, x+1] )
    """
    return -(
        _get(C, x -r + 1,    x + r+1,     bc)
        - _get(C, x-r,      x + r + 2, bc)
        - _get(C, x + r + 2, x-r,      bc)
        + _get(C, x + r+1,  x -r + 1,     bc)
    )


############################################################
# 4. CSV export for q^- and J^- (r=1) to compare with C++ output
############################################################

def export_symm_minus_csv(Q_list, J_list, times, gamma, out_csv, bc="obc", r=1, two_site_avg=True):
    """
    Write q^-(r) and J^-(r) profiles to CSV matching the C++ format:
        gamma,time,size,x,zeta,q_minus,j_minus

    Parameters
    ----------
    Q_list : list of 1D arrays
        Each entry is q_minus profile (length N) at the corresponding time.
    J_list : list of 1D arrays
        Each entry is j_minus profile (length N) at the corresponding time.
    times : list/array
        Times aligned with Q_list/J_list.
    gamma : float
        Gamma value (single). If you have multiple gammas, call repeatedly.
    out_csv : str
        Path to write the CSV.
    bc : str
        Boundary condition ("obc" or "pbc").
    r : int
        Charge index (default 1).
    two_site_avg : bool
        If True, output two-site averaged profiles to reduce staggering.
    """
    assert len(Q_list) == len(times) == len(J_list), "profiles and times must match in length"
    if len(Q_list) == 0:
        raise ValueError("Q_list is empty")

    N = len(Q_list[0])
    center = (N - 1) // 2

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "time", "size", "x", "zeta", "q_minus", "j_minus"])
        
        
        
        for Q, J, T in zip(Q_list, J_list, Times):
            Q = np.asarray(Q)
            J = np.asarray(J)
            if Q.shape[0] != N or J.shape[0] != N:
                raise ValueError("All profiles must have length N")

            xs = range(0, N - 1) if two_site_avg else range(0, N)

            for x in xs:
                if two_site_avg:
                    zeta = (x - center + 0.5) / T
                    q_minus = 0.5 * (np.real(Q[x]) + np.real(Q[x + 1]))
                    j_minus = 0.5 * (np.real(J[x]) + np.real(J[x + 1]))
                else:
                    zeta = (x - center) / T
                    q_minus = np.real(Q[x])
                    j_minus = np.real(J[x])

                w.writerow([gamma, T, N, x, zeta, q_minus, j_minus])

    return out_csv




############################################################
# 3. CONTINUUM HYDRO: χ⁺, n_ζ
############################################################

def theta(x):
    """ Heaviside with θ(0) = 1/2 """
    if x > 0:
        return 1.0
    if x < 0:
        return 0.0
    return 0.5

# fillings at the boundaries
def n_L(k):
    return 0.0

def n_R(k):
    # your latest Mathematica used 1/(4π)
    return 1.0 / (4.0 * np.pi)

def chi_plus(k, r_vals, b_vals):
    """
    \chi⁺(k) = 1/(8π) + Σ_r b_r sin(r k)
    k : float
    r_vals, b_vals : arrays of same shape (e.g. output of solve_truncated_system_full)
    """
    return 1.0/(8.0*np.pi) + np.sum(b_vals * np.sin(r_vals * k))

def n_zeta(k, zeta, r_vals, b_vals):
    """
    n_\zeta(k) with piecewise definition.
    We use ε'(k) = 2 sin(k) (J=1 convention).
    """
    epsp = 2.0 * np.sin(k)
    chi  = chi_plus(k, r_vals, b_vals)

    if k > 0:
        # k > 0 branch
        return (theta(-zeta) * n_L(k)
                + theta(zeta) * theta(epsp - zeta) * chi
                + theta(zeta - epsp) * n_R(k))
    else:
        # k < 0 branch
        return (theta(zeta) * n_R(k)
                + theta(-zeta) * theta(-epsp + zeta) * chi
                + theta(epsp - zeta) * n_L(k))


def n_zeta_HERM (k, zeta):
    
    """
    n_\zeta(k) for unitary dynamics.
    We use ε'(k) = 2 sin(k) (J=1 convention).
    """
    epsp = 2.0 * np.sin(k)
    return theta(epsp -zeta) * n_L(k) + theta(zeta - epsp) * n_R(k)

############################################################
# 4. HYDRO CHARGES & CURRENTS
############################################################

def q_plus(k, r):
    """ single-mode q^(r,+)(k) """
    return 2.0 * np.cos(r * k)

def q_minus(k, r):
    """ single-mode q^(r,-)(k) """
    return -2.0 * np.sin(r * k)

def hyd_charge(r, zeta, sign, r_vals, b_vals):
    """
    ⟨ q^{(r,sign)} ⟩_{n_\zeta}
      = ∫ dk q^{(r,sign)}(k) n_\zeta(k)
    """
    qfun = q_minus if sign == '-' else q_plus

    def f(k):
        return qfun(k, r) * n_zeta(k, zeta, r_vals, b_vals)

    val, _ = quad(f, -np.pi, np.pi, limit=400)
    return val





def hyd_current(r, zeta, sign, r_vals, b_vals):
    """
    ⟨ J^{(r,sign)} ⟩_{n_\zeta}
      = ∫ dk ε'(k) q^{(r,sign)}(k) n_\zeta(k)
    with ε'(k) = 2 sin(k).
    """
    qfun = q_minus if sign == '-' else q_plus

    def f(k):
        return 2.0 * np.sin(k) * qfun(k, r) * n_zeta(k, zeta, r_vals, b_vals)

    val, _ = quad(f, -np.pi, np.pi, limit=400)
    return val





############################################################
# 5. COMPARISON: LATTICE vs GHD  (optional)
############################################################

def compare_lattice_vs_ghd(C_zeta, N, t, r,
                           M, J, gamma,
                           sign,
                           bc='obc',
                           out_png=None):
    """
    Compare lattice charges/currents with GHD prediction for fixed r.

    Parameters
    ----------
    C_zeta : ndarray (N,N)
        Correlation / covariance matrix at time t.
    N : int
        System size (sites = N).
    t : float
        Time (for chi = x/t).
    r : int
        Charge index.
    M, J, gamma : as in solve_truncated_system_full.
    sign : '+' or '-'
    bc : 'obc' or 'pbc'
    out_png : str or None
        Path to save PNG. If None, just show plot.
    """
    # Solve truncated Fourier system once
    R_vals, B_vals = solve_truncated_system_full(M, J, gamma)

    # ξ-axis
    x = np.arange(N)
    XIS = (x - (N - 1) / 2) / t

    # LATTICE
    if sign == '-':
        q_lat = np.array([qm_symm(r, xx, C_zeta, bc).real for xx in range(N)])
        J_lat = np.array([jm_symm(r, xx, C_zeta, bc).real for xx in range(N)])
    else:
        q_lat = np.array([qp_symm(r, xx, C_zeta, bc).real for xx in range(N)])
        J_lat = np.array([jp_symm(r, xx, C_zeta, bc).real for xx in range(N)])

    q_lat = two_site_avg(q_lat, bc)
    J_lat = two_site_avg(J_lat, bc)

    # GHD
    q_hydro = np.array([hyd_charge(r, z, sign, R_vals, B_vals) for z in XIS])
    J_hydro = np.array([hyd_current(r, z, sign, R_vals, B_vals) for z in XIS])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(XIS, q_lat, 'o', ms=3, label=f"Numerics q, r={r}")
    plt.plot(XIS, J_lat, 'o', ms=3, label=f"Numerics J, r={r}")
    # plt.plot(XIS, q_hydro, '-', lw=2, label=f"GHD q, r={r}")
    # plt.plot(XIS, J_hydro, '-', lw=2, label=f"GHD J, r={r}")
    plt.xlabel(r'$\zeta$')
    plt.title(fr'Comparison for sign={sign}, r={r}')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return q_lat, J_lat
