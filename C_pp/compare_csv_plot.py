#!/usr/bin/env python3
"""
Overlay/compare Mathematica CSVs (GHD_r{r}_sign{sign}_M{M}_gamma{g}test.csv)
with Python numerics (GHD_GHZ_r{r}_sign{sign}_gamma{g}_N*.csv) and plot q/J vs zeta.
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def find_file(patterns):
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--r-list", type=int, nargs="+", required=True)
    ap.add_argument("--sign", type=str, required=True)
    ap.add_argument("--gammas", type=float, nargs="+", required=True)
    ap.add_argument("--mat-dir", type=str, default=".")
    ap.add_argument("--py-dir", type=str, default=".")
    ap.add_argument("--zmin", type=float, default=-2.3)
    ap.add_argument("--zmax", type=float, default=2.3)
    args = ap.parse_args()

    sign = args.sign
    for r in args.r_list:
        for g in args.gammas:
            
            mat_patterns = [
                os.path.join(args.mat_dir, f"GHD_r{r}_sign{sign}_M{args.M}_gamma{g:.2f}test.csv"),
                os.path.join(args.mat_dir, f"GHD_r{r}_sign{sign}_M{args.M}_gamma{g:.6f}test.csv"),
                os.path.join(args.mat_dir, f"GHD_r{r}_sign{sign}_M{args.M}_gamma{g}test.csv"),
            ]
            mat_file = find_file(mat_patterns)

            py_patterns = [
                os.path.join(args.py_dir, f"GHD_GHZ_r{r}_sign{sign}_gamma{g:.2f}_N*.csv"),
                os.path.join(args.py_dir, f"GHD_GHZ_r{r}_sign{sign}_gamma{g:.6f}_N*.csv"),
            ]
            py_file = find_file(py_patterns)
            
                
            

            
            mat = load_csv(mat_file)
            py = load_csv(py_file)

            # Mat file: zeta,q,J (names may vary or be unlabeled)
            mcols = list(mat.dtype.names)
            if "zeta" in mcols:
                z_mat = mat["zeta"]
                q_mat = mat["q"] if "q" in mcols else mat[mcols[1]]
                j_mat = mat["J"] if "J" in mcols else mat[mcols[2]]
            elif len(mcols) >= 3:
                z_mat = mat[mcols[0]]
                q_mat = mat[mcols[1]]
                j_mat = mat[mcols[2]]
            else:
                print(f"skip r={r} g={g}: could not parse mat columns {mcols}")
                continue

            # Py file: gamma,time,zeta,q,j
            z_py = py["zeta"] if "zeta" in py.dtype.names else py[py.dtype.names[2]]
            q_py = py["q"] if "q" in py.dtype.names else py[py.dtype.names[3]]
            j_py = py["j"] if "j" in py.dtype.names else py[py.dtype.names[4]]

            # filter by z range
            mask_mat = (z_mat >= args.zmin) & (z_mat <= args.zmax)
            mask_py = (z_py >= args.zmin) & (z_py <= args.zmax)
            z_mat, q_mat, j_mat = z_mat[mask_mat], q_mat[mask_mat], j_mat[mask_mat]
            z_py, q_py, j_py = z_py[mask_py], q_py[mask_py], j_py[mask_py]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            axes[0].plot(z_mat, q_mat, c="green", lw=3, label="ghd q")
            axes[0].scatter(z_py, q_py, s=6, c="blue", alpha=0.7, label="numerics q")
            axes[1].plot(z_mat, j_mat, c="orange", lw=3, label="ghd J")
            axes[1].scatter(z_py, j_py, s=6, c="red", alpha=0.7, label="numerics J")
            for ax in axes:
                ax.set_xlabel(r"$\zeta$")
                ax.grid(True, ls="--", alpha=0.5)
                ax.legend()
                ax.set_xlim(args.zmin, args.zmax)
            axes[0].set_ylabel("q")
            axes[1].set_ylabel("J")
            fig.suptitle(fr"$r={r}, sign={sign}, \gamma={g}$")
            fig.tight_layout()
            outpng = f"cmp_mat_py_r{r}_sign{sign}_gamma{g}.png"
            fig.savefig(outpng, dpi=200)
            plt.close(fig)
            print(f"wrote {outpng}")


if __name__ == "__main__":
    main()
