#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_elem(C, i, j, bc):
    n = C.shape[0]
    ii = int(i)
    jj = int(j)
    if bc and bc[0].lower() == "p":
        return C[ii % n, jj % n]
    if 0 <= ii < n and 0 <= jj < n:
        return C[ii, jj]
    return 0.0 + 0.0j


def qm_left(J, r, x, C, bc="pbc"):
    return 1j * J * (get_elem(C, x, x + r, bc) - get_elem(C, x + r, x, bc))


def qm_symm(r, x, C, bc="pbc"):
    if r % 2 != 0:
        R = (r - 1) // 2
        return 1j * (
            get_elem(C, x - R, x + R + 1, bc)
            - get_elem(C, x + R + 1, x - R, bc)
        )
    R = r // 2
    return 1j * (
        get_elem(C, x - R, x + R, bc)
        - get_elem(C, x + R, x - R, bc)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--J", type=float, default=1.0)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    C = data["C_t"]
    N = int(data["N"])
    T = float(data["time"])
    gamma = float(data["gamma"])
    L = int(data["L"])

    x0 = L - 1
    j_vals = np.arange(-args.window, args.window + 1)
    xs_left = x0 + j_vals
    xs_symm = L + j_vals

    q_left_vals = np.array([np.real(qm_left(args.J, args.r, x, C, "open")) for x in xs_left])
    q_symm_vals = np.array([np.real(qm_symm(args.r, x, C, "open")) for x in xs_symm])

    stem = f"compare_q_left_vs_symm_r{args.r}_N{N}_g{gamma:.2f}_T{T:.1f}_center"
    csv_path = outdir / f"{stem}.csv"
    png_path = outdir / f"{stem}.png"

    np.savetxt(
        csv_path,
        np.column_stack([j_vals, xs_left, xs_symm, q_left_vals, q_symm_vals]),
        delimiter=",",
        header="j,x_left,x_symm,q_left,q_symm",
        comments="",
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(j_vals, q_left_vals, marker="o", lw=1.5, label="left-endpoint")
    ax.plot(j_vals, q_symm_vals, marker="s", lw=1.5, label="symmetric-support")
    ax.set_xlabel("j = x - x0")
    ax.set_ylabel(r"$q^{(r,-)}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(csv_path)
    print(png_path)


if __name__ == "__main__":
    main()
