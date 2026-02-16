#!/usr/bin/env python3
import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_list(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def float_in_list(val, lst, tol=1e-9):
    for x in lst:
        if abs(val - x) <= tol:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Plot q/J profiles from C++ CSV outputs.")
    ap.add_argument("--outdir", type=str, default=".", help="Base output directory.")
    ap.add_argument("--csv-dir", type=str, default=None, help="CSV directory (default: outdir/GHD_NEEL_PARTIAL_CSV_CPP).")
    ap.add_argument("--png-dir", type=str, default=None, help="PNG directory (default: outdir/GHD_NEEL_PARTIAL_PNG_CPP).")
    ap.add_argument("--r-list", type=str, default="1,3", help="Comma-separated r values.")
    ap.add_argument("--sign", type=str, default="-", help="Sign (+ or -).")
    ap.add_argument("--gammas", type=str, default="1.0,2.0,4.0", help="Comma-separated gammas.")
    ap.add_argument("--sizes", type=str, default="800", help="Comma-separated half sizes (L).")
    ap.add_argument("--times", type=str, default="250", help="Comma-separated times T.")
    ap.add_argument("--lengths", type=str, default="800", help="Comma-separated monitored lengths L.")
    ap.add_argument("--zmin", type=float, default=-2.5, help="Min zeta for x-axis.")
    ap.add_argument("--zmax", type=float, default=2.5, help="Max zeta for x-axis.")
    ap.add_argument("--verbose", action="store_true", help="Print matching/skip information.")
    args = ap.parse_args()

    csv_dir = args.csv_dir or os.path.join(args.outdir, "GHD_NEEL_PARTIAL_CSV_CPP")
    png_dir = args.png_dir or os.path.join(args.outdir, "GHD_NEEL_PARTIAL_PNG_CPP")
    os.makedirs(png_dir, exist_ok=True)

    r_list = parse_list(args.r_list, int)
    gammas = parse_list(args.gammas, float)
    sizes = parse_list(args.sizes, int)
    times = parse_list(args.times, float)
    lengths = parse_list(args.lengths, int)
    n_list = [2 * s for s in sizes]
    sign = args.sign

    pattern = re.compile(
        r"GHD_VCNEEL_l(?P<L>[-0-9.]+)_r(?P<r>[-0-9]+)_sign(?P<sign>[+-])_gamma(?P<gamma>[-0-9.]+)_T(?P<T>[-0-9.]+)_N(?P<N>[-0-9]+)\.csv"
    )

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "GHD_VCNEEL_*.csv")))
    if args.verbose:
        print(f"csv_dir={csv_dir}")
        print(f"found {len(csv_files)} csv files")
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    matched = 0
    for path in csv_files:
        name = os.path.basename(path)
        m = pattern.match(name)
        if not m:
            continue
        L = int(float(m.group("L")))
        r = int(m.group("r"))
        sgn = m.group("sign")
        gamma = float(m.group("gamma"))
        T = float(m.group("T"))
        N = int(m.group("N"))

        if r not in r_list:
            continue
        if sgn != sign:
            continue
        if not float_in_list(gamma, gammas, tol=1e-6):
            continue
        if not float_in_list(T, times, tol=1e-6):
            continue
        if L not in lengths:
            continue
        if N not in n_list:
            continue

        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        zetas = data[:, 2]
        q_vals = data[:, 3]
        j_vals = data[:, 4]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes[0].scatter(zetas, q_vals, s=4, c="blue", alpha=0.6, label="q")
        axes[1].scatter(zetas, j_vals, s=4, c="red", alpha=0.6, label="J")
        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()
            ax.set_xlim(args.zmin, args.zmax)
        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")
        fig.suptitle(rf"$L={L},\ r={r},\ \text{{sign}}={sgn},\ \gamma={gamma},\ T={T},\ N={N}$")

        fig.tight_layout()
        out_png = os.path.join(png_dir, name.replace(".csv", ".png"))
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"wrote {out_png}")
        matched += 1

    if args.verbose:
        print(f"matched {matched} files")


if __name__ == "__main__":
    main()
