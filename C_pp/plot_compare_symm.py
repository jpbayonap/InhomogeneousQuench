#!/usr/bin/env python3
"""
Compare two q^-, J^- profile CSVs (e.g., C++ vs ED) and plot them together.
Expected CSV columns: gamma,time,size,x,zeta,q_minus,j_minus
"""
import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load(csv_path):
    data = defaultdict(lambda: defaultdict(list))  # gamma -> time -> list of (zeta, q, j)
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            g = float(row["gamma"])
            t = float(row["time"])
            z = float(row["zeta"])
            q = float(row["q_minus"])
            j = float(row["j_minus"])
            data[g][t].append((z, q, j))
    for g in data:
        for t in data[g]:
            data[g][t].sort(key=lambda tup: tup[0])
    return data


def main():
    ap = argparse.ArgumentParser(description="Overlay two q^-/J^- CSVs (e.g., C++ vs ED).")
    ap.add_argument("--ref", required=True, help="Reference CSV (e.g., ED)")
    ap.add_argument("--test", required=True, help="Test CSV (e.g., C++)")
    ap.add_argument("--gamma", type=float, required=False, help="Gamma value to plot (if multiple in CSV)")
    ap.add_argument("--time", type=float, required=False, help="Time value to plot (if multiple in CSV)")
    ap.add_argument("--out", default=None, help="Output PNG (optional; else show)")
    ap.add_argument("--labels", nargs=2, default=["ref", "test"], help="Labels for ref/test series")
    args = ap.parse_args()

    ref = load(args.ref)
    test = load(args.test)

    # pick gamma/time
    gammas = sorted(set(ref.keys()) & set(test.keys())) if args.gamma is None else [args.gamma]
    if not gammas:
        raise SystemExit("No overlapping gamma between files")

    for g in gammas:
        times = sorted(set(ref[g].keys()) & set(test[g].keys())) if args.time is None else [args.time]
        if not times:
            print(f"No overlapping times for gamma={g}")
            continue
        for t in times:
            ref_arr = np.array(ref[g][t])
            test_arr = np.array(test[g][t])
            if ref_arr.size == 0 or test_arr.size == 0:
                print(f"Empty data for gamma={g}, time={t}")
                continue
            z_ref, q_ref, j_ref = ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2]
            z_tst, q_tst, j_tst = test_arr[:, 0], test_arr[:, 1], test_arr[:, 2]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            axes[0].scatter(z_ref, q_ref, s=4, c="blue", alpha=0.7, label=f"{args.labels[0]} q^-")
            axes[0].scatter(z_tst, q_tst, s=4, c="red", alpha=0.7, label=f"{args.labels[1]} q^-")
            axes[1].scatter(z_ref, j_ref, s=4, c="blue", alpha=0.7, label=f"{args.labels[0]} J^-")
            axes[1].scatter(z_tst, j_tst, s=4, c="red", alpha=0.7, label=f"{args.labels[1]} J^-")
            for ax in axes:
                ax.set_xlabel(r"$\zeta$")
                ax.grid(True, ls="--", alpha=0.5)
                ax.legend()
            axes[0].set_ylabel("q^-")
            axes[1].set_ylabel("J^-")
            fig.suptitle(f"gamma={g}, t={t}")
            fig.tight_layout()

            if args.out:
                base, ext = args.out.rsplit(".", 1) if "." in args.out else (args.out, "png")
                fname = f"{base}_gamma_{g}_t_{t}.{ext}"
                fig.savefig(fname, dpi=200)
                print(f"wrote {fname}")
                plt.close(fig)
        if not args.out:
            plt.show()


if __name__ == "__main__":
    main()
