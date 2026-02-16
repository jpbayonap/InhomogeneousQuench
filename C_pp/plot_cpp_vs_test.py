#!/usr/bin/env python3
"""
Overlay q/J from C++ CSV (_cpp.csv) against Mathematica (_test.csv).
Assumes cpp CSV has columns: gamma,time,size,x,zeta,q_sign,j_sign
and test CSV has columns: zeta,q,J (or first three unnamed columns).
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
    ap.add_argument("--cpp-csv", required=True, help="C++ CSV (e.g., GHD_SYM_r_0_sign+_gamma0.100000_N800_cpp.csv)")
    ap.add_argument("--test-csv", required=True, help="Mathematica CSV (e.g., GHD_r0_sign+_M200_gamma0.10test.csv)")
    ap.add_argument("--out", default=None, help="Output png (auto-named if omitted)")
    args = ap.parse_args()

    cpp = load_csv(args.cpp_csv)
    test = load_csv(args.test_csv)

    # pick columns from test (fallback to first three if unnamed)
    cols = list(test.dtype.names)
    def pick(name, idx):
        return test[name] if name in cols else test[cols[idx]]
    z_qags = pick("zeta", 0)
    q_qags = pick("q", 1)
    j_qags = pick("J", 2)

    z_cpp = cpp["zeta"]
    q_cpp = cpp[cpp.dtype.names[5]]
    j_cpp = cpp[cpp.dtype.names[6]]

    # gather labels from filenames
    g = cpp["gamma"][0]
    N = cpp["size"][0]
    r = None
    sign = None
    base_cpp = os.path.basename(args.cpp_csv)
    if "_r_" in base_cpp and "_sign" in base_cpp:
        # crude parse
        try:
            r_part = base_cpp.split("_r_")[1]
            r = int(r_part.split("_")[0])
            sign = r_part.split("sign")[1][0]
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].scatter(z_cpp, q_cpp, s=4, c="blue", label="numerics q")
    axes[0].plot(z_qags, q_qags, c="green", label="GHD q")
    axes[1].scatter(z_cpp, j_cpp, s=4, c="red", label="numerics J")
    axes[1].plot(z_qags, j_qags, c="orange", label="GHD J")
    for ax in axes:
        ax.set_xlabel(r"$\zeta$")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
    axes[0].set_ylabel("q")
    axes[1].set_ylabel("J")
    title = f"gamma={g}, N={N}"
    if r is not None and sign is not None:
        title += f", r={r}, sign={sign}"
    fig.suptitle(title)
    fig.tight_layout()
    out = args.out or f"cmp_cpp_vs_test_gamma{g}_N{N}_r{r if r is not None else 'X'}_sign{sign if sign else 'X'}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

