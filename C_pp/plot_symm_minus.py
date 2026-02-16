#!/usr/bin/env python3
"""
Plot q^- and J^- profiles from CSV produced by main_dynamics.cpp.
CSV columns: gamma,time,size,x,zeta,q_minus,j_minus
Produces plots of q_minus and j_minus vs zeta for each gamma (grouped by time).
"""
import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    data = defaultdict(lambda: defaultdict(list))  # gamma -> time -> list of (zeta, q, j)
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            g = float(row["gamma"])
            t = float(row["time"])
            z = float(row["zeta"])
            q = float(row["q_minus"])
            j = float(row["j_minus"])
            data[g][t].append((z, q, j))
    # sort by zeta
    for g in data:
        for t in data[g]:
            data[g][t].sort(key=lambda tup: tup[0])
    return data


def main():
    ap = argparse.ArgumentParser(description="Plot q^- and J^- vs zeta from main_dynamics.cpp output.")
    ap.add_argument("csv", help="CSV file from main_dynamics.cpp")
    ap.add_argument("--out-prefix", default=None, help="Optional prefix to save PNGs; if omitted, show plots.")
    args = ap.parse_args()

    data = load_csv(args.csv)
    gammas = sorted(data.keys())

    for g in gammas:
        times = sorted(data[g].keys())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        for t in times:
            arr = np.array(data[g][t])
            zeta = arr[:, 0]
            q = arr[:, 1]
            j = arr[:, 2]
            axes[0].scatter(zeta, q, label=f"t={t:g}",c="blue", s=4)
            axes[1].scatter(zeta, j, label=f"t={t:g}",c="red", s=4)
        axes[0].set_title(rf"$q^-_{{r=1}}$ (gamma={g})")
        axes[1].set_title(rf"$J^-_{{r=1}}$ (gamma={g})")
        for ax in axes:
            ax.set_xlabel(r"$\zeta$")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()
        axes[0].set_ylabel("q^-")
        axes[1].set_ylabel("J^-")
        fig.tight_layout()
        if args.out_prefix:
            fname = f"{args.out_prefix}_gamma_{g}.png"
            fig.savefig(fname, dpi=200)
            print(f"wrote {fname}")
            plt.close(fig)
    if not args.out_prefix:
        plt.show()


if __name__ == "__main__":
    main()
