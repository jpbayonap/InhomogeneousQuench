#!/usr/bin/env python3
"""
Overlay lattice numerics (main_dynamics_BDY.py CSV) with Mathematica analytics CSVs (*.test.csv).
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-csv", required=True, help="Mathematica CSV (GHD_r*_test.csv)")
    ap.add_argument("--lat-csv", required=True, help="Lattice CSV from main_dynamics_BDY.py")
    ap.add_argument("--out", default=None, help="Output png (default based on mat file)")
    args = ap.parse_args()

    mat = load_csv(args.mat_csv)
    lat = load_csv(args.lat_csv)

    z_mat = mat["zeta"]
    q_mat = mat["q"]
    j_mat = mat["J"]

    z_lat = lat["zeta0"] if "zeta0" in lat.dtype.names else lat["zeta"]
    q_lat = lat["cond3"] if "cond3" in lat.dtype.names else lat["q"]
    j_lat = lat["cond4"] if "cond4" in lat.dtype.names else lat["J"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].plot(z_mat, q_mat, c="green", label="Mat q")
    axes[0].scatter(z_lat, q_lat, s=6, c="blue", alpha=0.7, label="Lattice q")
    axes[1].plot(z_mat, j_mat, c="orange", label="Mat J")
    axes[1].scatter(z_lat, j_lat, s=6, c="red", alpha=0.7, label="Lattice J")
    for ax in axes:
        ax.set_xlabel(r"$\zeta$")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    axes[0].set_ylabel("q")
    axes[1].set_ylabel("J")
    fig.tight_layout()
    out = args.out or os.path.splitext(os.path.basename(args.mat_csv))[0] + "_cmp.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

