#!/usr/bin/env python3
"""
Plot q/J from a C++ thermal CSV (gamma,time,zeta,q,j) into a PNG.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    args = ap.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
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
    title = "CPP thermal"
    if g is not None and t is not None:
        title += f" gamma={g}, T={t}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
