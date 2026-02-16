#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import jv

'''
USAGE EXAMPLE
python3 main_volterra_nsites.py \
--n_sites 5 \
--gamma 1.0 \
--tmax 200 \
--dt 0.1 \
--outdir /Users/juan/Desktop/Git/InhomogeneousQuench/C_pp/output_5sites

'''
def build_cross_pairs(A, B):
    """
    Ordered cross pairs:
      first block:  A x B
      second block: transpose of first block in the same order
    so pair i+len(AxB) is always (y_i, x_i).
    """
    ab = [(x, y) for x in A for y in B]
    ba_t = [(y, x) for (x, y) in ab]
    return ab + ba_t


def build_regions(n_sites):
    """
    Build cross-monitoring regions generalized from:
      A={1,2}, B={0,-1}
    to
      A={1,...,n_sites}, B={0,-1,...,-(n_sites-1)}.
    """
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    A = np.arange(1, n_sites + 1, dtype=int)
    B = -np.arange(0, n_sites, dtype=int)
    return A, B


def propagator(dx, t, J=1.0):
    """Free single-particle propagator on infinite tight-binding chain."""
    return np.power(1j, dx) * jv(dx, 2.0 * J * t)


def green_2body(dx, dy, t, J=1.0):
    """
    Two-body kernel entering Duhamel/Volterra form:
    G_{dx,dy}(t) = U_{dx}(t) U^*_{dy}(t).
    """
    return propagator(dx, t, J=J) * np.conjugate(propagator(dy, t, J=J))


def c_free_half_neel_pair(x, y, t, J=1.0, nmax_extra=40):
    
    """
    Free C_{x,y}(t) for half-vacuum/half-Neel initial state:
      occupied sites are odd j >= 1.

    """

    t = np.asarray(t, dtype=float)
    zmax = 2.0 * J * float(np.max(t))
    jmax = int(np.ceil(zmax + nmax_extra)) + 2

    acc = np.zeros_like(t, dtype=np.complex128)
    for j in range(1, jmax + 1, 2):
        ux = propagator(x - j, t, J=J)
        uy = propagator(y - j, t, J=J)
        acc += ux * np.conjugate(uy)
    return acc


def build_free_terms(pairs, t, free_mode="half_neel", J=1.0, nmax_extra=40):
    a = np.zeros((len(pairs), len(t)), dtype=np.complex128)
    if free_mode == "zero":
        return a
    if free_mode != "half_neel":
        raise ValueError(f"Unsupported free_mode: {free_mode}")

    for i, (x, y) in enumerate(pairs):
        a[i, :] = c_free_half_neel_pair(x, y, t, J=J, nmax_extra=nmax_extra)
    return a


def build_kernel_tensor(pairs, t, J=1.0):
    """
    K[k, i, j] = G_{x_i-m_j, y_i-n_j}(t[k]), where pair j is (m_j, n_j).
    """
    xs = np.array([p[0] for p in pairs], dtype=int)
    ys = np.array([p[1] for p in pairs], dtype=int)
    ms = xs.copy()
    ns = ys.copy()

    dx = xs[:, None] - ms[None, :]
    dy = ys[:, None] - ns[None, :]

    # Broadcast to 3D arrays:
    # - time axis:        t[:, None, None] -> (Nt,1,1)
    # - pair-difference axes: dx/dy[None,:,:] -> (1,M,M)
    # Resulting kernel has shape (Nt, M, M).
    tt = t[:, None, None]

    # Use your requested form explicitly:
    # G_{a,b}(t) = conjugate[ (-i)^a i^b J_a(2Jt) J_b(2Jt) ].
    phase = np.power(-1j, dx)[None, :, :] * np.power(1j, dy)[None, :, :]
    bess = jv(dx[None, :, :], 2.0 * J * tt) * jv(dy[None, :, :], 2.0 * J * tt)
    return np.conjugate(phase * bess)


def solve_volterra_vector_trap(t, a, K, gamma):
    """
    Solve c(t) = a(t) - gamma * (K * c)(t) with trapezoid rule.
    t: (Nt,)
    a: (M, Nt)
    K: (Nt, M, M), with K[k] = K(t[k])
    """
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    Nt = t.size
    M = a.shape[0]

    c = np.zeros((M, Nt), dtype=np.complex128)
    c[:, 0] = a[:, 0]

    A_step = np.eye(M, dtype=np.complex128) + 0.5 * gamma * dt * K[0]

    for i in range(1, Nt):
        conv_hist = np.zeros(M, dtype=np.complex128)
        if i > 1:
            # sum_{k=1}^{i-1} K(t_i-t_k) c(t_k)
            for k in range(1, i):
                conv_hist += K[i - k] @ c[:, k]

        rhs = a[:, i] - gamma * dt * (conv_hist + 0.5 * (K[i] @ c[:, 0]))
        c[:, i] = np.linalg.solve(A_step, rhs)

    return c


def symmetry_report(c, pairs, tol=1e-8):
    """
    Compare c_{x,y}(t) with +/- c_{y,x}(t) for all available transposed pairs.
    Returns table rows and per-time max errors.
    """
    
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    rows = []
    plus_over_t = []
    minus_over_t = []

    for i, (x, y) in enumerate(pairs):

        j = pair_to_idx.get((y, x))
        if j is None:
            continue

        diff_plus = c[i] - c[j]
        diff_minus = c[i] + c[j]
        diff_conj = c[i] - np.conjugate(c[j])
        err_plus_t = np.abs(diff_plus)
        err_minus_t = np.abs(diff_minus)
        err_conj_t = np.abs(diff_conj)

        max_plus = float(np.max(err_plus_t))
        max_minus = float(np.max(err_minus_t))
        max_conj = float(np.max(err_conj_t))
        if max_conj <= max_plus and max_conj <= max_minus:
            best = "conj"
        elif max_plus <= max_minus:
            best = "+"
        else:
            best = "-"
        passes = min(max_plus, max_minus, max_conj) < tol

        rows.append({
            "pair": f"({x},{y})",
            "transpose": f"({y},{x})",
            "max_abs_cxy_minus_cyx": max_plus,
            "max_abs_cxy_plus_cyx": max_minus,
            "max_abs_cxy_minus_conj_cyx": max_conj,
            "best_relation": best,
            "passes_tol": passes,
        })

        plus_over_t.append(err_plus_t)
        minus_over_t.append(err_minus_t)

    if plus_over_t:

        plus_over_t = np.max(np.vstack(plus_over_t), axis=0)
        minus_over_t = np.max(np.vstack(minus_over_t), axis=0)
    else:

        plus_over_t = np.array([])
        minus_over_t = np.array([])

    return rows, plus_over_t, minus_over_t


def run_case(
    J=1.0,
    gamma=1.0,
    tmax=100.0,
    dt=0.1,
    free_mode="half_neel",
    nmax_extra=40,
    n_sites=1,
    outdir=None,
):
    if outdir is None:
        outdir = f"output_{int(n_sites)}sites"
    A, B = build_regions(n_sites)

    pairs = build_cross_pairs(A, B)

    t = np.arange(0.0, tmax + 1e-12, dt)
    a = build_free_terms(pairs, t, free_mode=free_mode, J=J, nmax_extra=nmax_extra)
    K = build_kernel_tensor(pairs, t, J=J)
    c = solve_volterra_vector_trap(t, a, K, gamma)

    sym_rows, _, _ = symmetry_report(c, pairs)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = f"n{n_sites}_J{J}_g{gamma}_tmax{tmax}_dt{dt}_{free_mode}"
    csv_c = outdir / f"volterra_2sites_C_{tag}.csv"
    csv_sym = outdir / f"volterra_2sites_symmetry_{tag}.csv"
    pair_pngs = []

    data = {"t": t}
    for i, (x, y) in enumerate(pairs):
        data[f"Re_C_{x}_{y}"] = np.real(c[i])
        data[f"Im_C_{x}_{y}"] = np.imag(c[i])
    pd.DataFrame(data).to_csv(csv_c, index=False)

    pd.DataFrame(sym_rows).to_csv(csv_sym, index=False)

    # One figure per transpose pair: left panel Re, right panel Im.
    n_pairs = len(pairs) // 2
    for i in range(n_pairs):
        x, y = pairs[i]
        yt, xt = pairs[i + n_pairs]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

        axes[0].plot(t, np.real(c[i]), lw=1.4, label=f"Re C({x},{y})")
        axes[0].plot(t, np.real(c[i + n_pairs]), ls="--", lw=1.2, label=f"Re C({yt},{xt})")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("Re C")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=12)

        axes[1].plot(t, np.imag(c[i]), lw=1.4, label=f"Im C({x},{y})")
        axes[1].plot(t, -np.imag(c[i + n_pairs]), ls="--", lw=1.2, label=f"-Im C({yt},{xt})")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Im C")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=12)

        fig.suptitle(f"Pair comparison: ({x},{y}) vs ({yt},{xt}), gamma={gamma}, free={free_mode}")
        fig.tight_layout()

        png_pair = outdir / f"volterra_2sites_pair_{x}_{y}__{yt}_{xt}_{tag}.png"
        fig.savefig(png_pair, dpi=200)
        plt.close(fig)
        pair_pngs.append(png_pair)

    print("Saved:", csv_c)
    print("Saved:", csv_sym)
    for png_path in pair_pngs:
        print("Saved:", png_path)
    print("Pairs order:", pairs)

    for row in sym_rows:
        print(
            f"{row['pair']} vs {row['transpose']}: "
            f"max|c-cT|={row['max_abs_cxy_minus_cyx']:.3e}, "
            f"max|c+cT|={row['max_abs_cxy_plus_cyx']:.3e}, "
            f"max|c-conj(cT)|={row['max_abs_cxy_minus_conj_cyx']:.3e}, "
            f"best={row['best_relation']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--free-mode", type=str, default="half_neel", choices=["half_neel", "zero"])
    ap.add_argument("--nmax-extra", type=int, default=40)
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: output_<n_sites>sites).")
    ap.add_argument("--n_sites", type=int, default=1)
    args = ap.parse_args()

    run_case(
        J=args.J,
        gamma=args.gamma,
        tmax=args.tmax,
        dt=args.dt,
        free_mode=args.free_mode,
        nmax_extra=args.nmax_extra,
        n_sites=args.n_sites,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
