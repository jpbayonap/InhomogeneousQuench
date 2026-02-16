#!/usr/bin/env python3
"""
Plot conditions 3 and 4 vs zeta0 for varying gamma from CSV produced by main_dynamics_BDY.cpp.
Expects columns: gamma,zeta0,cond3_real,cond3_imag,cond4_real,cond4_imag,time,size
"""
import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    data = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            g = float(row["gamma"])
            z0 = float(row["zeta0"])
            c3 = float(row["cond3"])
            c3_abs = float(row.get("cond3_abs", abs(c3)))
            c4 = float(row["cond4"])
            c4_abs = float(row.get("cond4_abs", abs(c4)))
            data[g].append((z0, c3_abs, c4_abs))
    # sort by zeta0 within each gamma
    for g in data:
        data[g].sort(key=lambda t: t[0])
    return data


def main():
    ap = argparse.ArgumentParser(description="Plot boundary conditions 3/4 vs zeta0 for multiple gamma.")
    ap.add_argument("csv", help="CSV file from main_dynamics_BDY.cpp")
    ap.add_argument("--out", default=None, help="Optional output PNG; if omitted, just show.")
    args = ap.parse_args()

    data = load_csv(args.csv)
    gammas = sorted(data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for g in gammas:
        zetas = np.array([t[0] for t in data[g]])
        cond3_abs = np.array([t[1] for t in data[g]])
        cond4_abs = np.array([t[2] for t in data[g]])
        axes[0].semilogy(zetas, cond3_abs, "o-", label=f"gamma={g}")
        axes[1].semilogy(zetas, cond4_abs, "o-", label=f"gamma={g}")

    axes[0].set_title("Condition 3 (|j_delta + (2r+1) gamma q0|)")
    axes[1].set_title("Condition 4 (|j_delta_m + J1^2|)")
    for ax in axes:
        ax.set_xlabel(r"$\zeta_0$")
        ax.set_ylabel("Error magnitude")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=200)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
