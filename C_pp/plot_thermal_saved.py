#!/usr/bin/env python3
"""
Plot saved thermal CSVs from GHD_THERM_CSV and write PNGs to GHD_THERM_PNG.
Defaults to betaL=1.0, betaR=1.0, N=1600.
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="GHD_THERM_CSV")
    ap.add_argument("--png-dir", default="GHD_THERM_PNG")
    ap.add_argument("--betaL", type=float, default=1.0)
    ap.add_argument("--betaR", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=1600)
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.png_dir, exist_ok=True)

    pattern = os.path.join(
        args.csv_dir,
        f"GHD_THERM_betaL_{args.betaL}betaR_{args.betaR}_r*_sign*_gamma*_N{args.N}_TEST.csv",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched: {pattern}")
        return

    for path in files:
        data = load_csv(path)
        zeta = data["zeta"] if "zeta" in data.dtype.names else data[data.dtype.names[2]]
        q = data["q"] if "q" in data.dtype.names else data[data.dtype.names[3]]
        j = data["j"] if "j" in data.dtype.names else data[data.dtype.names[4]]
        g = data["gamma"][0] if "gamma" in data.dtype.names else None
        t = data["time"][0] if "time" in data.dtype.names else None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes[0].scatter(zeta, q, s=4, c="blue", alpha=0.6, label="q")
        axes[1].scatter(zeta, j, s=4, c="red", alpha=0.6, label="J")
        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()
            if args.zmin is not None and args.zmax is not None:
                ax.set_xlim(args.zmin, args.zmax)
        axes[0].set_ylabel("q")
        axes[1].set_ylabel("J")
        title = f"betaL={args.betaL}, betaR={args.betaR}, N={args.N}"
        if g is not None and t is not None:
            title += f", gamma={g}, T={t}"
        fig.suptitle(title)
        fig.tight_layout()

        out = os.path.join(args.png_dir, os.path.basename(path).replace(".csv", ".png"))
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
